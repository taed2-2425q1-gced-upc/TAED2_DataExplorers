import pickle
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd

import pytest

from src.config import MODELS_DIR, RAW_DATA_DIR
from src.features import preprocessing

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
        (Path(RAW_DATA_DIR/ "seg_test" / "mountain"/ "20479.jpg"), "mountain"),
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
