"""
Tests for the api functions.
"""
from unittest.mock import patch, MagicMock, mock_open
from http import HTTPStatus
import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.app.api import app

# pylint: disable=C0301
# pylint: disable=W0613

@pytest.fixture(scope="module", autouse=True)
def client():
    """Set up a TestClient fixture for FastAPI app testing."""
    # Use the TestClient with a `with` statement to trigger the startup and shutdown events.
    with TestClient(app) as client:
        yield client

def test_root(client):
    """Test root endpoint for proper response structure and message."""
    response = client.get("/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["message"] == "OK"
    assert json["status-code"] == HTTPStatus.OK
    assert json["data"]["message"] == """Welcome to the landscape image classifier! Please, read the `/docs`!"""


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for simulated image files."""
    # Crea un directorio temporal y añade archivos simulados
    mock_image = tmp_path / "test_image.jpg"
    mock_image.write_bytes(b'\x89PNG\r\n\x1a\n...' * 10)  # Simula contenido de imagen
    return tmp_path

@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize("image_filename", ["test_image.jpg"])  # Use file names directly for mock
@patch("src.app.api.EmissionsTracker")  # Mock the EmissionsTracker class
@patch("builtins.open",new_callable=mock_open,
    read_data="col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,emissions,cpu_power\n0,0,0,0,0,0,0,0,0,0,0,0,10.0,45\n")
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


@pytest.mark.parametrize("image_filename", ["test_image.jpg"])  # Use file names directly for mock
@patch("src.app.api.EmissionsTracker")  # Mock the EmissionsTracker class
@patch("builtins.open",new_callable=mock_open, read_data="col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,emissions,cpu_power\n0,0,0,0,0,0,0,0,0,0,0,0,10.0,45\n")
def test_predict_image_not_emissions(mock_open,mock_emissions_tracker, test_data_dir, image_filename, client):
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

    # Assertions to validate the response
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert json_response["message"] == "Emissions data not found."


def test_predict_image_invalid(client):
    """Test invalid image file upload, expecting unprocessable entity error."""
    # Test with an invalid file type
    response = client.post(
        "/predict/image/",
        files={"file": ("invalid.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY  # Expected error for invalid image input

def test_training_info(client):
    """Test the training info endpoint for proper response structure and content."""
    response = client.get("/training/info/")
    json = response.json()
    assert response.status_code == HTTPStatus.OK
    assert json["message"] == "Training Information"
    assert json["status-code"] == HTTPStatus.OK
    assert "Model parameters" in json["data"]
    assert "Metrics" in json["data"]
    assert "Training Emissions" in json["data"]



'''
@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize("image_filename", ["test_image.jpg"])  # Use file names directly for mock
@patch("src.app.api.EmissionsTracker")  # Mock the EmissionsTracker class
@patch("builtins.open",new_callable=mock_open, read_data="col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,emissions,cpu_power\n0,0,0,0,0,0,0,0,0,0,0,0,10.0,45\n")
def test_predict_image_exception(mock_open,mock_emissions_tracker, test_data_dir, image_filename, client):
    """Test image prediction with a valid image file, including emissions tracking."""

    # Simula los valores predichos que devuelve el modelo
    with patch("tensorflow.keras.models.load_model") as mock_load_model, \
         patch("cv2.imread") as mock_image:
        
        # Simular que el tracker de emisiones retorna un valor fijo
        mock_emissions_tracker.return_value.__enter__.return_value = {
        'col1': [0], 'col2': [0], 'col3': [0], 'col4': [0], 'col5': [0], 
                'col6': [0], 'col7': [0], 'col8': [0], 'col9': [0], 'col10': [0], 
                'emissions': [10.0], 'col11': [0],'col12': [0],  # Esta es la métrica de emisiones
                'cpu_power': [45]
        }

        img_path = test_data_dir / image_filename
        mock_model_instance = mock_load_model.return_value
        mock_model_instance.predict.side_effect = Exception("Error")

        # Aquí utilizamos `side_effect` para que se lance la excepción
        mock_image.return_value = np.array(np.zeros((100, 100, 3), dtype=np.uint8))

        img_path = test_data_dir / image_filename
        # Simular la carga del archivo
        with open(img_path, "rb") as f:
            response = client.post(
                "/predict/image/",
                files={"file": (image_filename, f, "image/jpeg")},
                timeout=30,
            )

    json_response = response.json()
    print(json_response)
    
    # Asegurarse de que la respuesta contenga un error
    assert "error" in json_response
    assert json_response["error"] == "Error"

@patch("src.features.preprocessing.process_images")  # Mock the process_images method
@pytest.mark.parametrize("image_filename", ["test_image.jpg"])
def test_predict_image_exception(mock_process_images, test_data_dir, image_filename, client):
    # Arrange
    mock_process_images.side_effect = Exception("Something went wrong")

    # Act
    response = client.post("/predict/image/", files={"file": image_filename})

    # Assert
    assert response.status_code == 422
    assert response.json() == {"error": "Something went wrong"}'''

@patch("os.path.exists", return_value=True)
@pytest.mark.parametrize("image_filename", ["test_image.jpg"])
@patch("src.app.api.EmissionsTracker")  # Mock the EmissionsTracker class
@patch("builtins.open",new_callable=mock_open, read_data=None)
def test_predict_image_exception(mock_open,mock_emissions_tracker1, test_data_dir, image_filename, client):
    """Test prediction when an exception occurs during emissions tracking."""
    # Arrange: Set the process_images mock to return a valid input for the model
    with patch("tensorflow.keras.models.load_model") as mock_load_model1, \
         patch("cv2.imread") as mock_image:
        mock_image.return_value = np.array(np.zeros((100, 100, 3), dtype=np.uint8))  # Assuming this is a valid processed input
        # Simular que el tracker de emisiones retorna un valor fijo
        mock_emissions_tracker1.return_value.__enter__.return_value = Exception("Prediction error")
        # Arrange: Mock the predict method to raise an exception
        mock_load_model1.return_value.side_effect = Exception("Prediction error")

        # Act: Call the endpoint with the mock image file
        img_path = test_data_dir / image_filename
        # Simular la carga del archivo
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


@patch("os.path.exists", return_value=False)  # Simulating that the emissions file does not exist
@patch("mlflow.get_run")  # Mocking mlflow.get_run to return a mock run object
def test_training_info_no_metrics(mock_get_run, mock_exists, client):
    """Test training info endpoint when metrics are missing, expecting 404 error."""
    # Mocking mlflow.get_run to return an object with empty metrics and some params
    mock_run = MagicMock()
    mock_run.data.metrics = {}  # Simulate no metrics
    mock_run.data.params = {"param1": "value1"}  # Simulate some parameters
    mock_get_run.return_value = mock_run

    # Call the endpoint
    response = client.get("/training/info/")

    # Extract the response
    json_response = response.json()

    # Assertions to validate the response
    assert response.status_code == HTTPStatus.NOT_FOUND  # Expecting 404 as per the emissions file missing
    assert json_response["message"] == "Emissions data not found."



@patch("mlflow.get_run", side_effect=Exception("MLflow error"))
def test_training_info_exception(mock_mlflow_exception, client):
    """Test training info endpoint when an MLflow error occurs, expecting 500 status."""
    response = client.get("/training/info/")
    assert response.status_code == 500
    assert "Error retrieving training information" in response.json()["message"]
