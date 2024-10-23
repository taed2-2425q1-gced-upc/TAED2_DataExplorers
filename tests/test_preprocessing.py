import pytest
from pathlib import Path
from src.features.preprocessing import *
from src.config import RAW_DATA_DIR
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture
def experiment_path():
    path_to_images = Path(RAW_DATA_DIR / "seg_train")
    return path_to_images

def test_read_and_prepare_images_valid(experiment_path):
    x, y = read_and_prepare_images(experiment_path, [], [])
    assert len(x) > 0  
    assert len(y) == len(x) 

def test_read_and_prepare_images_empty(tmp_path):
    empty_path = tmp_path / "empty_folder"
    empty_path.mkdir()
    x, y = read_and_prepare_images(empty_path, [], [])
    assert len(x) == 0  
    assert len(y) == 0




@pytest.fixture
def mock_paths():
    with patch("src.features.preprocessing.prepared_folder_path", Path("/fake_path")) as processed_mock:
        with patch("src.features.preprocessing.train_path", Path("/fake_model_dir")) as train_mock:
            with patch("src.features.preprocessing.test_path", Path("/fake_model_dir")) as test_mock:
                with patch("src.features.preprocessing.predict_path", Path("/fake_model_dir")) as pred_mock:
                    yield processed_mock, train_mock, test_mock, pred_mock


def test_main(mock_paths):
    processed_mock, train_mock, test_mock, pred_mock = mock_paths
    # Mocking the functions used within main
    with patch('src.features.preprocessing.read_and_prepare_images') as mock_read_and_prepare_images, \
         patch('src.features.preprocessing.read_and_prepare_predictions') as mock_read_and_prepare_predictions, \
         patch('src.features.preprocessing.list_to_nparray') as mock_list_to_nparray, \
         patch('src.features.preprocessing.save_preprocessing') as mock_save_preprocessing, \
         patch('src.config.RAW_DATA_DIR', new_callable=MagicMock()) as mock_raw_data_dir:  # Mocking the config variable
    
        # Set the return values for the mocked functions
        mock_read_and_prepare_images.side_effect = [
            ([np.array("image1"), np.array("image2")], [np.array("label1"), np.array("label2")]),  # for train
            ([np.array("image3")], [np.array("label3")]),                     # for test
        ]
        mock_read_and_prepare_predictions.return_value = [np.array("image_pred")]
        mock_list_to_nparray.side_effect = lambda x: np.array(x)  # Convert lists to numpy arrays
        
        # Call the main function
        main()

        # Assert the functions were called with the correct arguments
        mock_read_and_prepare_images.assert_any_call(train_mock, [], [])
        mock_read_and_prepare_images.assert_any_call(test_mock, [], [])
        mock_read_and_prepare_predictions.assert_called_once_with(pred_mock, [], [])
        
        # Check if list_to_nparray was called for all data
        assert mock_list_to_nparray.call_count == 5  # Called for x_train, x_test, x_pred, y_train, y_test

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

