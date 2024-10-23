from http import HTTPStatus
import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.app.api import app
from unittest.mock import patch, MagicMock, mock_open
from src.config import TEST_DATA_DIR
import tempfile
import os


@pytest.fixture(scope="module", autouse=True)
def client():
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        yield client 

def test_root(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["message"] == "OK"
    assert json["status-code"] == HTTPStatus.OK
    assert json["data"]["message"] == "Welcome to the landscape image classifier! Please, read the `/docs`!" 


@pytest.fixture
def test_data_dir(tmp_path):
    mock_image = tmp_path / "test_image.jpg"
    mock_image.write_bytes(b'\x89PNG\r\n\x1a\n...' * 10)  
    return tmp_path

@pytest.mark.parametrize("image_filename", ["test_image.jpg"])  # Use file names directly for mock
def test_predict_image_valid(test_data_dir, image_filename, client):
    """Test image prediction with a valid image file, including emissions tracking."""

    # Simulate the predicted values returned from the model
    with patch("tensorflow.keras.models.load_model") as mock_load_model, \
         patch("cv2.imread") as mock_image:

        img_path = test_data_dir / image_filename
        mock_model_instance = mock_load_model.return_value
        mock_model_instance.predict.return_value = [[0.0, 0.2, 0.3, 0.4, 0.5, 0.6]]  # Simulated prediction
        mock_image.return_value = np.array(np.zeros((100, 100, 3), dtype=np.uint8))

        # Simulate file upload
        with open(img_path, "rb") as f:
            response = client.post(
                "/predict/image/",
                files={"file": (image_filename, f, "image/jpeg")},
                timeout=30,
            )

    json_response = response.json()
    print("JSON Response:", json_response)

    # Assertions to validate the response
    assert response.status_code == HTTPStatus.OK
    assert json_response["Message"] == "OK"
    assert json_response["Status-code"] == HTTPStatus.OK
    assert "The predicted class is" in json_response["Data"]
    assert "The prediction scores are" in json_response["Data"]


def test_predict_image_invalid(client):
    # Test with an invalid file type
    response = client.post(
        "/predict/image/",
        files={"file": ("invalid.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY  # Expected error for invalid image input

def test_training_info(client):
    response = client.get("/training/info/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["message"] == "Training Information"
    assert json["status-code"] == HTTPStatus.OK
    assert "Model parameters" in json["data"]
    assert "Metrics" in json["data"]
    assert "Training Emissions" in json["data"]


@pytest.mark.parametrize("image_filename", ["test_image.jpg"])
def test_predict_image_exception(test_data_dir, image_filename, client):
    with patch("tensorflow.keras.models.load_model") as mock_load_model1, \
         patch("cv2.imread") as mock_image:
        mock_image.return_value = np.array(np.zeros((10, 0, 3), dtype=np.uint8))  # Assume incorrect preprocessing
        mock_load_model1.return_value.side_effect = Exception("Prediction error")

        img_path = test_data_dir / image_filename
        with open(img_path, "rb") as f:
            response = client.post(
                "/predict/image/",
                files={"file": (image_filename, f, "image/jpeg")},
                timeout=30,
            )

    # Assert: Check the response status and message
    json_response = response.json()
    print(json_response)
    assert response.status_code == 500
    assert 'error' in json_response


@patch("os.path.exists", return_value=False)  
@patch("mlflow.get_run")  
def test_training_info_no_metrics(mock_get_run, mock_exists, client):
    mock_run = MagicMock()
    mock_run.data.metrics = {}  # Simulate no metrics
    mock_run.data.params = {"param1": "value1"}  # Simulate some parameters
    mock_get_run.return_value = mock_run

    response = client.get("/training/info/")
    
    json_response = response.json()

    assert response.status_code == HTTPStatus.NOT_FOUND  # Expecting 404 as per the emissions file missing
    assert json_response["message"] == "Emissions data not found."



@patch("mlflow.get_run", side_effect=Exception("MLflow error"))
def test_training_info_exception(mock_mlflow_exception, client):
    response = client.get("/training/info/")
    assert response.status_code == 500
    assert "Error retrieving training information" in response.json()["message"]

