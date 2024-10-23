"""
Tests for preprocessing functions.
"""
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import keras
import numpy as np

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.features import preprocessing

# pylint: disable=W0621
# pylint: disable=W0108
# pylint: disable=W0612
# pylint: disable=W0613

@pytest.fixture
def experiment_path():
    """Return the path to the experiment images."""
    path_to_images = Path(RAW_DATA_DIR / "seg_train")
    return path_to_images

def test_read_and_prepare_images_valid(experiment_path):
    """Test if image preparation returns non-empty data for valid input."""
    x, y = preprocessing.read_and_prepare_images(experiment_path, [], [])
    assert len(x) > 0
    assert len(y) == len(x)

def test_read_and_prepare_images_empty(tmp_path):
    """Test if image preparation returns empty data for an empty folder."""
    empty_path = tmp_path / "empty_folder"
    empty_path.mkdir()
    x, y = preprocessing.read_and_prepare_images(empty_path, [], [])
    assert len(x) == 0
    assert len(y) == 0

@pytest.fixture
def mock_paths():
    """Mock the paths used in preprocessing functions."""
    with patch("src.features.preprocessing.prepared_folder_path",
                Path("/fake_path")) as processed_mock:
        with patch("src.features.preprocessing.train_path",
                Path("/fake_model_dir")) as train_mock:
            with patch("src.features.preprocessing.test_path",
                Path("/fake_model_dir")) as test_mock:
                with patch("src.features.preprocessing.predict_path",
                Path("/fake_model_dir")) as pred_mock:
                    yield processed_mock, train_mock, test_mock, pred_mock


def test_main(mock_paths):
    """Test the main function, ensuring correct function calls and saving."""
    processed_mock, train_mock, test_mock, pred_mock = mock_paths
    # Mocking the functions used within main
    with patch('src.features.preprocessing.read_and_prepare_images') \
            as mock_read_and_prepare_images, \
        patch('src.features.preprocessing.read_and_prepare_predictions') \
            as mock_read_and_prepare_predictions, \
        patch('src.features.preprocessing.list_to_nparray') \
            as mock_list_to_nparray, \
        patch('src.features.preprocessing.save_preprocessing') \
            as mock_save_preprocessing, \
        patch('src.config.RAW_DATA_DIR',
            new_callable=MagicMock()) as mock_raw_data_dir:  # Mocking the config variable

        # Set the return values for the mocked functions
        mock_read_and_prepare_images.side_effect = [
            ([np.array("image1"), np.array("image2")], [np.array("label1"), np.array("label2")]),
            ([np.array("image3")], [np.array("label3")]),
        ]
        mock_read_and_prepare_predictions.return_value = [np.array("image_pred")]
        mock_list_to_nparray.side_effect = lambda x: np.array(x)  # Convert lists to numpy arrays

        # Call the main function
        preprocessing.main()

        # Assert the functions were called with the correct arguments
        mock_read_and_prepare_images.assert_any_call(train_mock, [], [])
        mock_read_and_prepare_images.assert_any_call(test_mock, [], [])
        mock_read_and_prepare_predictions.assert_called_once_with(pred_mock, [], [])

        # Check if list_to_nparray was called for all data
        assert mock_list_to_nparray.call_count == 5

        # Verify that the save function was called with the correct paths and data
        assert mock_save_preprocessing.call_count == 5  # Should save all datasets

        calls = mock_save_preprocessing.call_args_list
        print(calls[0][0][0])
        for call in calls:
            print(call[0][1])

        # Use np.array to ensure consistency in types for comparisons
        expected_calls = [
            (Path(processed_mock / "x_train.npy"), ["image1", "image2"]),
            (Path(processed_mock / "y_train.npy"), ["label1", "label2"]),
            (Path(processed_mock / "x_test.npy"), ["image3"]),
            (Path(processed_mock / "y_test.npy"), ["label3"]),
            (Path(processed_mock / "x_pred.npy"), ["image_pred"])
        ]

        # Check if all expected calls were made
        for expected_call in expected_calls:
            assert any(
                call[0][0] == expected_call[0] and (call[0][1] == expected_call[1]).all()
                for call in calls
            ), f"Expected call {expected_call} not found in mock_save_preprocessing calls."

def test_getcode():
    """Test the getcode function to ensure proper class code mappings."""
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
    """Test the process_images function with mocked image processing."""
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
    """Test reading and preparing predictions with mocked image files."""
    mock_imread.return_value = MagicMock()  # Simulate image data
    mock_resize.return_value = MagicMock()  # Simulate resized image data

    x = []
    y = []
    result = preprocessing.read_and_prepare_predictions(Path('dummy_path'), x, y)

    assert len(result) == 2  # Expect 2 images to be processed


def test_list_to_nparray():
    """Test the list_to_nparray function converts lists to numpy arrays."""
    data = [[1, 2], [3, 4]]
    result = preprocessing.list_to_nparray(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


@patch('numpy.save')
def test_save_preprocessing(mock_save):
    """Test saving of preprocessing results using mocked numpy.save."""
    data = np.array([[1, 2], [3, 4]])
    preprocessing.save_preprocessing('/path/to/save.npy', data)
    mock_save.assert_called_once_with('/path/to/save.npy', data)


@pytest.fixture
def model():
    """Load and return a trained model for testing."""
    model_path = MODELS_DIR / 'model.h5'
    model = keras.models.load_model(model_path)
    return model

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
    """Test model predictions and check if they match the expected labels."""
    x_processed_image = preprocessing.process_images(sample, [], [], 100, needs_return=True)
    x_processed_image = preprocessing.list_to_nparray(x_processed_image)
    prediction = model.predict(x_processed_image)
    prediction_mapped = preprocessing.getcode(np.argmax(prediction))

    assert prediction_mapped == expected
