import pytest
from pathlib import Path
from src.features.preprocessing import read_and_prepare_images
from src.config import RAW_DATA_DIR

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
