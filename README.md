# neuralhydrology-on-gcp
Example repo to train and deploy models from the NeuralHydrology framework with Google Cloud.

## Introduction
NeuralHydrology is a deep learning framework for training neural networks with a strong focus on hydrological applications. The focus of the framework is on research and was used in various academic publications (e.g. [Mai et al., 2022](https://doi.org/10.5194/hess-26-3537-2022) and [Lees et al., 2022](https://doi.org/10.5194/hess-26-3079-2022)). The main concept of the framework is modularity that allows easy integration of new datasets, new model architectures or any training-related aspects making it great for research. Furthermore, for users the implementations based off configuration files, which let anyone train neural networks without touching the code itself.

The models used by [Google Flood Forecasting](https://sites.research.google/floodforecasting/) initiative are based off of the open source implementations in the NeuralHydrology framework ([Nevo et al., 2022](https://doi.org/10.5194/hess-26-4013-2022)).

As mentioned, this framework has been used extensively for research. This repo serves as an example of how to train models using Google Cloud services and deploy the trained model for use to predict streamflow. This example uses [VertexAI Custom training](https://cloud.google.com/vertex-ai/docs/training/overview), [Google Cloud Functions](https://cloud.google.com/functions/docs/concepts/overview), [Earth Engine](https://developers.google.com/earth-engine/), and [Colab Enterprise](https://cloud.google.com/colab/docs/introduction) services.


## Technical steps
This next section walks through the technical steps for how to train, deploy, and use the models developed 

### Setup

Make sure all of the necessary APIs are enabled:

```
gcloud services enable \
  earthengine.googleapis.com \
  aiplatform.googleapis.com \
  run.googleapis.com 
```

Define environment variables that will be used throught:

```
PROJECT=<YOUR-PROJECT-NAME>
REGION=<GCP-REGION>
BUCKET_NAME=<YOUR-BUCKET-NAME>
```

Create a Google Cloud Storage bucket to save data to:

```
gsutil mb -l $REGION gs://$BUCKET_NAME
```

Create an Artifact Repository to save built Docker images to:

```
gcloud artifacts repositories create neuralhydrology \
 --location=$REGION \
 --repository-format=docker \
 --description="Custom image for training a hydrology LSTM"
```

### Train a model

Change directories into the `train/` subdirectory:

```
cd train
```

Update the output bucket path in `train/src/task.py` at line 41 to your bucket name that was specified.
Update `train/cloudbuild.yml` and replace "REGION" with the cloud region you are using and replace "PROJECT" with your cloud project name.
For now, the workflow is setup to read the CAMELS dataset from a public storage bucket but the `train/src/basin.yml` file at line 111 should be updated to read data from your own storage bucket.

Once those updates are complete, the Docker image can be built using the following command:

```
gcloud builds submit --config cloudbuild.yml
```

This should take a few minutes to complete.

After the Docker image is built, a custom training job can be submitted using the following command:

```
gcloud ai custom-jobs create \
  --display-name=neuralhydrology-run \
  --region=$REGION \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=$REGION-docker.pkg.dev/$PROJECT/neuralhydrology/training-image
```

Depending on how many basins being trained on and the compute resources this can take a few minutes to hours. 

After the training job has completed successfully, you should see data in bucket you specified in the `task.py` file. There should be a new `run/` directory with the model run directory that just completed.


### Deploy model for inference

Update the variables in the `inference-cf/.env.yml` file with your information. The `BUCKET` variable should be the cloud storage bucket with your model results. The `RUN_DIR` variable should be the model run directory with the save model weights.

This example uses Google Cloud Functions for the deployment to an endpoint for predictions. There are many other great options for model deployment but this is fairly straightforward and scalable which is why it is used here.

Change directories into the subdirectory with the Cloud Function code:

```
cd inference-cf
```

To deploy the functions with the following command:

```
gcloud functions deploy nh-inference \
  --entry-point=nh_predict \
  --region=$REGION \ 
  --runtime=python311 \
  --env-vars-file=env.yml \
  --memory=1GB \
  --cpu=1 \
  --allow-unauthenticated \
  --gen2 \
  --trigger-http 
```

The command will build a package container and deploy the service to Cloud Functions with an endpoint that you can use.

### Deploy model to Vertex AI

A common way to to deploy a service in Google Cloud is to use Cloud Function or one could just load the model in a notebook too. Here VertexAI is used because it allows you to run Online Predictions for quick synchronous requests and Batch Predictions for running predictions over large datasets with no change to deployment.

VertexAI has many methods for hosting models. Since NeuralHydrology is a custom framework, a Custom Container will be used for the model. Then once a model is hosted the model will need to be deployed to an endpoint. The next section will walk through this process.

Change directories into the subdirectory with the serving code:

```
cd serve
```
Update `serve/cloudbuild.yml` and replace "REGION" with the cloud region you are using and replace "PROJECT" with your cloud project name.

Move the trained model components to the `model/` subdirectory:

```
gsutil -m cp -r \ gs://$BUCKET_NAME/runs/test_run_*/model_epoch050.pt \
gs://$BUCKET_NAME/runs/test_run_*/config.yml \
gs://$BUCKET_NAME/runs/test_run_*/train_data/ \
model/
```

Build the Docker image for serving the model with the following command:

```
gcloud builds submit --config cloudbuild.yml
```

Upload the model to VertexAI Model Registry

```
gcloud ai models upload \
--display-name=neuralhydrology-model \
--container-image-uri=$REGION-docker.pkg.dev/$PROJECT/neuralhydrology/serving-image \
--container-health-route="/ping" \
--container-predict-route="/predict" \
--container-ports=7080 \
--region=$REGION \
--model-id=neuralhydrology-model
```

Create an endpoint from which to serve the model:

```
gcloud ai endpoints create \
--display-name=neurohydrology-endpoint \
--region=$REGION
```

Get the endpoint ID from VertexAI that was just created to deploy the model:

```
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region=$REGION \
  --filter=displayName:neurohydrology-endpoint \
  --format="value(ENDPOINT_ID.scope())")
```

Deploy the model to the endpoint:

```
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=neuralhydrology-model \
  --display-name=neuralhydrology-deployed \
  --machine-type=n1-standard-4
```

### Explore results

There are two notebooks provided in the `notebooks/` directory: 

1. Use Earth Engine to get forcing data and run the predictions using the Cloud Functions endpoint deployed
2. Read model weights from cloud storage and explore outputs using an updated tutorial from the NeuralHydrology ["Inspecting LSTM States and Activations"](https://neuralhydrology.readthedocs.io/en/latest/tutorials/inspect-lstm.html#) example

To use the notebooks, the easiest and quickest way to get started on Google Cloud is Colab Enterprise. You can quickly [upload the notebooks](https://cloud.google.com/vertex-ai/docs/colab/create-console-quickstart#upload) to use and [connect a runtime](https://cloud.google.com/vertex-ai/docs/colab/connect-to-runtime) to get started.

