""""
This script performs data validation and integrity checks on a dataset of images using Deepchecks.
"""
from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import data_integrity, train_test_validation

from src.config import RAW_DATA_DIR, REPORTS_DIR


OUTPUT_FILE = REPORTS_DIR / "deepchecks_validation.html"

train_ds= classification_dataset_from_directory(
    RAW_DATA_DIR / "seg_train", object_type="VisionData", image_extension="jpg"
)

test_ds= classification_dataset_from_directory(
    RAW_DATA_DIR / "seg_test", object_type="VisionData", image_extension="jpg"
)

custom_suite = data_integrity()

custom_suite.add(train_test_validation())

result = custom_suite.run(train_ds, test_ds)

# If the output file already exists, delete it to avoid duplicates
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

result.save_as_html(str(OUTPUT_FILE))
