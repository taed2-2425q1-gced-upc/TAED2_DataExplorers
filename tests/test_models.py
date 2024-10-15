import pickle
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd

import pytest

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.features import preprocessing


def test_getcode():
    assert preprocessing.getcode(0) == 'buildings'
    assert preprocessing.getcode(1) == 'forest'
    assert preprocessing.getcode(2) == 'glacier'
    assert preprocessing.getcode(3) == 'mountain'
    assert preprocessing.getcode(4) == 'sea'
    assert preprocessing.getcode(5) == 'street'
    assert preprocessing.getcode(6) is None  # A case where the key is not found

@patch('cv2.imread')
@patch('cv2.resize')
def test_process_images(mock_resize, mock_imread):
    mock_imread.return_value = MagicMock()  # Simulate image data
    mock_resize.return_value = MagicMock()  # Simulate resized image data
    
    x = []
    y = []
    preprocessing.process_images('dummy_path.jpg', x, y, 100, True, 'buildings')
    
    assert len(x) == 1  # Ensure image was appended to x
    assert y == [preprocessing.code['buildings']]  # Ensure correct label was added


@patch('glob.glob', return_value=['image1.jpg', 'image2.jpg'])
@patch('cv2.imread')
@patch('cv2.resize')
def test_read_and_prepare_predictions(mock_resize, mock_imread, mock_glob):
    mock_imread.return_value = MagicMock()  # Simulate image data
    mock_resize.return_value = MagicMock()  # Simulate resized image data

    x = []
    y = []
    result = preprocessing.read_and_prepare_predictions(Path('dummy_path'), x, y)

    assert len(result) == 2  # Expect 2 images to be processed


def test_list_to_nparray():
    data = [[1, 2], [3, 4]]
    result = preprocessing.list_to_nparray(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


@patch('numpy.save')
def test_save_preprocessing(mock_save):
    data = np.array([[1, 2], [3, 4]])
    preprocessing.save_preprocessing('/path/to/save.npy', data)
    mock_save.assert_called_once_with('/path/to/save.npy', data)


@pytest.fixture
def model():
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.mark.parametrize(
    "sample, expected",
    [
        (Path(RAW_DATA_DIR/ "seg_test" / "buildings"/ "20231.jpg"), "buildings"),
        (Path(RAW_DATA_DIR/ "seg_test" / "forest"/ "20315.jpg"), "forest"),
        (Path(RAW_DATA_DIR/ "seg_test" / "glacier"/ "20386.jpg"), "glacier"),
        (Path(RAW_DATA_DIR/ "seg_test" / "mountain"/ "20058.jpg"), "mountain"),
        (Path(RAW_DATA_DIR/ "seg_test" / "sea"/ "20236.jpg"), "sea"),
        (Path(RAW_DATA_DIR/ "seg_test" / "street"/ "20080.jpg"), "street"),
    ],
)
def test_model_results(model, sample, expected):
    x_processed_image = preprocessing.process_images(sample, [], [], 100, needs_return=True)
    x_processed_image = preprocessing.list_to_nparray(x_processed_image)
    prediction = model.predict(x_processed_image)
    prediction_mapped = preprocessing.getcode(np.argmax(prediction))

    assert prediction_mapped == expected