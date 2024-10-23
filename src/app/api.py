"""
Main script: it includes our API initialization and endpoints.
"""

import os
from pathlib import Path
from http import HTTPStatus
import tempfile
import logging
from typing import Dict
from contextlib import asynccontextmanager
import numpy as np
import keras
import mlflow
from codecarbon import EmissionsTracker
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse


from src.config import METRICS_DIR
from src.features import preprocessing
MODELS_FOLDER_PATH = Path("models")

# This errors are ignored:
# pylint: disable=R0914
# pylint: disable=W0718

# Initialize the dictionary to group models by "tabular" or "image" and then by model type
model_wrappers_dict: Dict[str, Dict[str, dict]] = {"image": {}}

'''
def file_to_image(file: bytes):
    """
    Reads an image file and formats it for the model.

    Parameters
    ----------
    file:
        bytes: The image file to classify.

    Returns
    -------
    Tensor: The image formatted for the model.
    """
    image = tf.io.decode_image(file, channels=3, dtype=tf.float32)
    return tf.image.resize(image, [224, 224])
'''

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models found in `MODELS_DIR` and adds them to `models_list`"""

    model_path = MODELS_FOLDER_PATH / "model.h5"
    cv_model = keras.models.load_model(model_path)

    model_wrappers_dict["image"]["cnn"] = {
        "model": cv_model,
        "type": "cnn",
    }

    yield

    # Clear the list of models to avoid memory leaks
    # del model_wrappers_dict["tabular"]
    del model_wrappers_dict["image"]


# Define application
app = FastAPI(
    title="Landscape image classifier",
    description="Welcome to the Landscape Image Classifier API! This API allows you to classify"
    "landscape images using a Convolutional Neural Network model. Simply upload your image, and"
    "our model will provide you with classification results, including detailed prediction scores"
    "and environmental impact data related to emissions. Explore the endpoints to make the most of"
    "our powerful classification tool and contribute to a sustainable future!",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])  # path operation decorator
async def _index():
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to the landscape image classifier! Please, read the `/docs`!"},
    }
    return response


# Create and endpoint to classify an image
@app.post("/predict/image/", tags=["Prediction"])
async def _predict_image(file: UploadFile):
    """
    Classifies landscape images using a pre-trained CNN model.

    Parameters
    ----------
    file : UploadFile
        The image to classify.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Invalid file type.")
    # Read the image file and format it for the model
    image_stream = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_stream)
        tmp_path = tmp.name
    try:
        x_processed_image = preprocessing.process_images(tmp_path, [], [], 100, needs_return=True)
        x_processed_image = preprocessing.list_to_nparray(x_processed_image)
        cv_model = model_wrappers_dict["image"]["cnn"]["model"]
        '''
        with EmissionsTracker(
            project_name="image-classification",
            measure_power_secs=1,
            tracking_mode="process",
            output_dir=METRICS_DIR,
            output_file="emissions_api.csv",
            on_csv_write="append",
            default_cpu_power=45,
        ):
        '''
        predictions = cv_model.predict(x_processed_image)

        '''
        # Read the emissions file and return the latest record
        emissions_file = os.path.join(METRICS_DIR, "emissions_api.csv")
        if not os.path.exists(emissions_file):
            return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"message": "Emissions data not found."},
        )
        with open(emissions_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            last_line = lines[-1]  # Get the last recorded emissions data
            emissions_data = last_line.strip().split(",")
            emissions_response = {
                "Carbon emissions in kg": float(emissions_data[5]),
                "Energy consumed in kWh": float(emissions_data[13]),
            }
        '''
        predictions_dict = {preprocessing.getcode(i): predictions.tolist()[0][i] for i in range(6)}
    except Exception as e:
        return JSONResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content={"error": str(e)})
    await file.close()

    predicted_label = preprocessing.getcode(np.argmax(predictions))

    logging.info("Predicted class %s", predicted_label)

    response = {
        "Message": HTTPStatus.OK.phrase,
        "Status-code": HTTPStatus.OK,
        "Data": {
            "Model": "Convolutional Neural Network",
            "The prediction scores are": predictions_dict,
            "The predicted class is": predicted_label, 
        },
    }

    return response


@app.get("/training/info/", tags=["Training"])
async def training_info():
    """
    Returns model training information, including parameters, metrics, and training emissions.
    """

    mlflow_tracking_uri = "https://dagshub.com/martinaalba21/TAED2_DataExplorers.mlflow"
    mlflow.set_tracking_uri(mlflow_tracking_uri)


    # Specify the MLflow run ID
    run_id = "77e05845316e44e1959cd55bb94819ae"

    try:
        # Fetch the run data using the run ID
        run = mlflow.get_run(run_id)

        # Access parameters and metrics from the run
        params = run.data.params
        metrics = run.data.metrics

        # Read the emissions file and return the latest record
        emissions_file = os.path.join(METRICS_DIR, "emissions.csv")
        if not os.path.exists(emissions_file):
            return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={"message": "Emissions data not found."},
        )
        with open(emissions_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
            last_line = lines[-1]  # Get the last recorded emissions data
            emissions_data = last_line.strip().split(",")
            emissions_response = {
                "Carbon emissions in kg": float(emissions_data[5]),
                "Energy consumed in kWh": float(emissions_data[13]),
            }

        response_data = {
            "message": "Training Information",
            "status-code": HTTPStatus.OK,
            "data": {
                "Model parameters": params if params else "No parameters found",
                "Metrics": metrics if metrics else "No metrics found",
                "Training Emissions": emissions_response
            }
        }
        return JSONResponse(status_code=HTTPStatus.OK, content=response_data)

    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"message": f"Error retrieving training information: {str(e)}"}
        )
