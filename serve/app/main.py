import json
import logging
import os
from pathlib import Path
import tempfile

from cloudpathlib import AnyPath
import fsspec
import numpy as np
import pandas as pd
import torch

from fastapi.logger import logger
from fastapi import FastAPI, Request

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.utils.config import Config


gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)

app = FastAPI()


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


def nh_predict(forcings):
    # get environment variables
    EPOCH = int(os.environ.get('EPOCH',-1))

    model_path = Path('/model')

    config_file = model_path / 'config.yml'

    # load the model config
    cudalstm_config = Config(config_file)

    # if model epoch to load is not defined then use the last one from config
    if EPOCH < 0:
        EPOCH = cudalstm_config.epochs

    weights_ckpt = model_path / f'model_epoch{EPOCH:03d}.pt'

    # load the trained weights into the new model
    weights = torch.load(weights_ckpt, map_location='cpu')

    # create a new model instance with random weights
    cuda_lstm = CudaLSTM(cfg=cudalstm_config)

    # load the weights from the saved model
    cuda_lstm.load_state_dict(weights)

    # load the data scaler
    scaler = load_scaler(model_path)

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

    return y_pred.tolist()


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    """ health check to ensure HTTP server is ready to handle 
        prediction requests
    """
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
       
    # unfinished, returns Internal Server error
    outputs = nh_predict(instances)
    
    return {'predictions': outputs}