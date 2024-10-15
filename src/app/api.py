"""
Main script: it includes our API initialization and endpoints.
"""

import numpy as np
import logging
import pickle
import tempfile
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict
from pathlib import Path

from codecarbon import track_emissions
from fastapi import FastAPI, UploadFile

from src.config import METRICS_DIR
from src.features import preprocessing
MODELS_FOLDER_PATH = Path("models")

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
    """Loads all pickled models found in `MODELS_DIR` and adds them to `models_list`"""

    with open(MODELS_FOLDER_PATH / "model.pkl", "rb") as pickled_model:
        cv_model = pickle.load(pickled_model)

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
    description="This API lets you classify the Intel Image Classification dataset using a CNN model.",
    version="0.1",
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
@track_emissions(
    project_name="landscape-prediction",
    measure_power_secs=1,
    save_to_file=True,
    output_dir=METRICS_DIR,
)
@app.post("/predict/image/", tags=["Prediction"])
async def _predict_image(file: UploadFile):
    """
    Classifies landscape images using a pre-trained CNN model.

    Parameters
    ----------
    file : UploadFile
        The image to classify.
    """
    # Read the image file and format it for the model
    image_stream = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_stream)
        tmp_path = tmp.name  # Obtener la ruta del archivo temporal

    try:
        x_processed_image = preprocessing.process_images(tmp_path, [], [], 100, needs_return=True)
        x_processed_image = preprocessing.list_to_nparray(x_processed_image)
        cv_model = model_wrappers_dict["image"]["cnn"]["model"]
        predictions = cv_model.predict(x_processed_image)
        predictions_list = predictions.tolist()
    except Exception as e:
        return {"error": str(e)}
    await file.close()

    predicted_label = preprocessing.getcode(np.argmax(predictions))
    print(predicted_label)

    logging.info("Predicted class %s", predicted_label)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "model-type": "cnn",
            "prediction": predictions_list,
            "predicted_class": predicted_label,
        },
    }

    return response
