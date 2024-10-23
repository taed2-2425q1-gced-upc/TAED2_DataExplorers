from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import json
from src.models import evaluate 
from src.config import PROCESSED_DATA_DIR, MODELS_DIR


@pytest.fixture
def validation_data():
    """Fixture to provide mock validation data."""
    x_valid = np.random.rand(50, 10)  
    y_valid = np.random.randint(0, 2, size=(50, 1))
    return x_valid, y_valid


@patch('numpy.load')
def test_load_validation_data(mock_npy_load):
    """Test to ensure validation data is loaded correctly."""
    # Mock the np.load function to return dummy data
    mock_npy_load.side_effect = [
        np.random.rand(50, 10), 
        np.random.randint(0, 2, size=(50, 1))  
    ]

    x_valid, y_valid = evaluate.load_validation_data(PROCESSED_DATA_DIR)

    mock_npy_load.assert_any_call(Path(PROCESSED_DATA_DIR / "x_test.npy"))
    mock_npy_load.assert_any_call(Path(PROCESSED_DATA_DIR / "y_test.npy"))

    assert x_valid.shape == (50, 10)
    assert y_valid.shape == (50, 1)


@patch('tensorflow.keras.models.load_model')
def test_evaluate_model(mock_load_model, validation_data):
    """Test to ensure the model is evaluated correctly with validation data."""
    mock_model = MagicMock()
    mock_model.evaluate.return_value = (0.1, 0.9)
    mock_load_model.return_value = mock_model

    x_valid, y_valid = validation_data

    loss, accuracy = evaluate.evaluate_model("model.h5", x_valid, y_valid)

    mock_load_model.assert_called_once_with(Path(MODELS_DIR) / "model.h5")
    mock_model.evaluate.assert_called_once_with(x_valid, y_valid)

    assert loss == 0.1
    assert accuracy == 0.9



@pytest.fixture
def mock_paths():
    with patch("src.models.evaluate.PROCESSED_DATA_DIR", Path("/fake_path")) as processed_mock:
        with patch("src.models.evaluate.MODELS_DIR", Path("/fake_model_dir")) as models_mock:
            with patch("src.models.evaluate.METRICS_DIR", Path("/fake_metrics_dir")) as metrics_mock:
                yield processed_mock, models_mock, metrics_mock

def test_main(mock_paths):
    # Arrange
    mock_x = [np.array(np.zeros((100, 100, 3), dtype=np.uint8)), np.array(np.zeros((100, 100, 3), dtype=np.uint8)), np.array(np.zeros((100, 100, 3), dtype=np.uint8))]
    mock_y = [1, 2, 3]
    mock_accuracy = 0.95
    processed_mock, models_mock, metrics_mock = mock_paths

    # Mock functions
    with patch("src.models.evaluate.load_validation_data", return_value=(mock_x, mock_y)), \
         patch("src.models.evaluate.evaluate_model", return_value=(0.1, mock_accuracy)), \
         patch("src.models.evaluate.mlflow.start_run") as mock_start_run, \
         patch("src.models.evaluate.mlflow.set_experiment") as mock_set_mlflow_experiment, \
         patch("src.models.evaluate.mlflow.log_metrics") as mock_log_metrics, \
         patch("builtins.open", mock_open()) as mock_file:

        # Act
        evaluate.main()

        # Assert
        mock_set_mlflow_experiment.assert_called_once()  # Check if MLflow run was started
        mock_start_run.assert_called_once()
        mock_log_metrics.assert_called_once_with({"accuracy": mock_accuracy})  # Check if metrics were logged

        # Check if the JSON file was written correctly
        mock_file.assert_called_once_with(metrics_mock / "scores.json", "w", encoding="utf-8")
        mock_file().write.assert_any_call('{')
        mock_file().write.assert_any_call('\n    ')
        mock_file().write.assert_any_call('"accuracy"')
        mock_file().write.assert_any_call(': ')
        mock_file().write.assert_any_call('0.95')
        mock_file().write.assert_any_call('\n')
        mock_file().write.assert_any_call('}')