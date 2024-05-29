import json
import os
from pathlib import Path
import tempfile

from cloudpathlib import AnyPath
import fsspec
import numpy as np
import pandas as pd
import torch

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.utils.config import Config

import functions_framework


def data_prep(df,featureNames,labelNames,timeLag=10,predLead=0):
    """
    Function to pred dataframe for input into the model

    Args:
        df: dataframe with features and label
        featureNames: list of column names that will be the input features
        labelNames: list of column names that will be the output labels
    Kwargs:
        timeLag: time to lag datasets default = 10 periods
        predLead: time period as forecast ouputs
    Returns:
        x: array of input features
        y: array of output features
      
    """

    # get features
    x = df[featureNames].values

    # get the labels
    if predLead > 0:
        y = df[labelNames][timeLag:-predLead].values
    else:
         y = df[labelNames][timeLag:].values

    xshape = [y.shape[0]] + [timeLag] + [x.shape[1]] 
    yshape = [y.shape[0],predLead] if predLead > 0 else [y.shape[0], 1]
    outx = np.zeros(xshape)
    outy = np.zeros(yshape)
    for i in range(y.shape[0] - predLead):
        v = timeLag+i if i > 0 else timeLag
        u = predLead+i if i > 0 else predLead
        u = u if predLead > 0 else i + 1
        outx[i,:,:] = x[i:v,:]
        outy[i,:] = y[i:u].T

    return outx, outy


@functions_framework.http
def nh_predict(request):

    try:
        # get the data that was sent for precdiction
        request_json = request.get_json(silent=True)
        forcings = request_json['forcings']

        # get environment variables
        BUCKET = os.environ.get('BUCKET')
        EPOCH = int(os.environ.get('EPOCH',-1))
        RUN_DIR = os.environ.get('RUN_DIR')

        # specify the run directory where the model is
        run_dir = AnyPath(BUCKET) / 'runs' / RUN_DIR

        # create a temp file to move the config to
        tmp_conf = tempfile.NamedTemporaryFile()
        gcs_conf = AnyPath(run_dir / 'config.yml')

        # write the config content to the temp file
        with open(tmp_conf.name, 'w') as f:
            f.write(gcs_conf.read_bytes().decode('utf-8'))

        # load the model config
        cudalstm_config = Config(Path(tmp_conf.name))

        # if model epoch to load is not defined then use the last one from config
        if EPOCH < 0:
          EPOCH = cudalstm_config.epochs

        # load the trained weights into the new model
        with fsspec.open(run_dir / f'model_epoch{EPOCH:03d}.pt', 'rb') as f:
            weights = torch.load(f, map_location='cpu')

        # create a new model instance with random weights
        cuda_lstm = CudaLSTM(cfg=cudalstm_config)

        # load the weights from the saved model
        cuda_lstm.load_state_dict(weights)

        # load the data scaler
        scaler = load_scaler(run_dir)

        # get configuration information
        features = cudalstm_config.dynamic_inputs
        pred_labels = cudalstm_config.target_variables
        data_columns = pred_labels + features

        # create a dataframe of the forcings
        df = pd.DataFrame(forcings, columns=features)
        # create random column for the target variables (unused during inference)
        for pred_label in pred_labels:
            df[pred_label] = np.random.rand(df.shape[0])
        
        # apply scaling to the dataset
        scaled_df = (df[data_columns]- np.array(scaler['xarray_feature_center'].to_array())) / np.array(scaler['xarray_feature_scale'].to_array())

        # get the number of examples for the time series sequence from config
        seq_length = cudalstm_config.seq_length

        # create the input data with the right shape for prediction
        x, y = data_prep(
            scaled_df,
            features,
            pred_labels,
            timeLag=seq_length,
        )

        # get the data in a structure that is expected by the neuralhydro model
        in_data = {
            'x_d': torch.tensor(x.astype(np.float32)),
            'y': torch.tensor(y[np.newaxis, :, :]),
            'date': df.index.values[np.newaxis, seq_length:]
        }

        # run the prediction
        with torch.no_grad():
            out = cuda_lstm(in_data)

        # get out the prediction 
        y_pred = out['y_hat'][:, -1, :].numpy()
        y_pred = (y_pred * np.array(scaler['xarray_feature_scale'][pred_labels].to_array())) + np.array(scaler['xarray_feature_center'][pred_labels].to_array())
        y_pred[np.where(y_pred < 0)] = 0

        return json.dumps({'y_pred': y_pred.tolist()})
      
    except Exception as e:
        return json.dumps({ "errorMessage": str(e) }), 400

