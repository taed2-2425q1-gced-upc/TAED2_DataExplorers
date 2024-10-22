from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
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
