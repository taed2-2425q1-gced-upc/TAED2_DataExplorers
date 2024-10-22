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
    # Crea un directorio temporal y añade archivos simulados
    mock_image = tmp_path / "test_image.jpg"
    mock_image.write_bytes(b'\x89PNG\r\n\x1a\n...' * 10)  # Simula contenido de imagen
    return tmp_path

@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize("image_filename", ["test_image.jpg"])  # Use file names directly for mock
@patch("src.app.api.EmissionsTracker")  # Mock the EmissionsTracker class
@patch("builtins.open",new_callable=mock_open, read_data="col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,emissions,cpu_power\n0,0,0,0,0,0,0,0,0,0,0,0,10.0,45\n")
def test_predict_image_valid(mock_open,mock_emissions_tracker, test_data_dir, image_filename, client):
    """Test image prediction with a valid image file, including emissions tracking."""

    # Simulate the predicted values returned from the model
    with patch("tensorflow.keras.models.load_model") as mock_load_model, \
         patch("cv2.imread") as mock_image:
        
        # Mock emissions tracker to return a preset value
        mock_emissions_tracker.return_value.__enter__.return_value = {
        'col1': [0], 'col2': [0], 'col3': [0], 'col4': [0], 'col5': [0], 
                'col6': [0], 'col7': [0], 'col8': [0], 'col9': [0], 'col10': [0], 
                'emissions': [10.0], 'col11': [0],'col12': [0], # This represents an emissions metric
                'cpu_power': [45]
        }

        img_path = test_data_dir / image_filename
        mock_model_instance = mock_load_model.return_value
        mock_model_instance.predict.return_value = [[0.0, 0.2, 0.3, 0.4, 0.5, 0.6]]  # Simulated prediction
        mock_image.return_value = np.array(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # # Crear un archivo de emisiones temporal
        # emissions_file_path = os.path.join(test_data_dir, "emissions_api.csv")
        # with open(emissions_file_path, "w") as f:
        #     # Escribir datos de emisiones simulados
        #     f.write("col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,emissions,cpu_power\n")
        #     f.write("0,0,0,0,0,0,0,0,0,0,0,0,10.0,45\n")  # Última línea simulada


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
    assert "Prediction emissions " in json_response["Data"]


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

def test_training_info_no_data(client):
    # If your implementation allows for handling of a scenario where no data exists, we can test that.
    # This would require mock behavior or specific setup to create such a state.
    # This is just a placeholder for your future implementation.
    pass
