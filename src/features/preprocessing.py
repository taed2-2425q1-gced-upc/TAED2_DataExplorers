"""
This module contains functions for data preprocessing.
"""

import glob as gb
import os
import warnings
from pathlib import Path
import numpy as np
import cv2
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
warnings.filterwarnings('ignore')

# This errors are ignored:
# pylint: disable=R0913
# pylint: disable=R0917
# pylint: disable=E1101

################################## GLOBAL VARS ##################################
input_folder_path = RAW_DATA_DIR

train_path = Path(input_folder_path / "seg_train")
test_path = Path(input_folder_path / "seg_test")
predict_path = Path(input_folder_path / "seg_pred")

prepared_folder_path = PROCESSED_DATA_DIR

code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

################################## FUNCTIONS ##################################

def getcode(n):
    """
    This function returns the label (as a string) corresponding to a given numeric code.
    """
    for x , y in code.items():
        if n == y:
            return x
    return None

def process_images(file, x, y, s, needs_y=False, folder=None, needs_return=False):
    """
    This function reads an image file, resizes it, and appends it to the provided lists.
    """
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    x.append(list(image_array))
    if needs_y:
        y.append(code[folder])
    if needs_return:
        return x
    return None

def read_and_prepare_predictions(pred_path, x, y):
    """
    This function reads and processes prediction images from the specified path.
    """
    files = gb.glob(pathname= str(pred_path / '*.jpg'))
    for file in files:
        process_images(file, x, y, 100)
    return x

def read_and_prepare_images(path, x, y):
    """
    This function eads and processes images from the specified folder, appending their labels.
    """
    for folder in os.listdir(path):
        files = gb.glob(pathname= str(path / folder / '*.jpg'))
        for file in files:
            process_images(file, x, y, 100, True, folder)
    return x, y

def list_to_nparray(data):
    """
    This function converts a list into a NumPy array.
    """
    return np.array(data)

def save_preprocessing(path, data_preprocessed):
    """
    This function saves preprocessed data as a NumPy file.
    """
    np.save(path, data_preprocessed)


################################## MAIN ##################################

def main():
    """
    Main function for preprocessing image data.
    """
    x_train, y_train = read_and_prepare_images(train_path, [], [])
    x_test, y_test = read_and_prepare_images(test_path, [], [])
    x_pred = read_and_prepare_predictions(predict_path, [], [])

    x_train_preprocessed = list_to_nparray(x_train)
    x_test_preprocessed = list_to_nparray(x_test)
    x_pred_preprocessed = list_to_nparray(x_pred)
    y_train_preprocessed = list_to_nparray(y_train)
    y_test_preprocessed = list_to_nparray(y_test)

    x_train_path = Path(prepared_folder_path / "x_train.npy")
    x_test_path = Path(prepared_folder_path / "x_test.npy")
    x_pred_path = Path(prepared_folder_path / "x_pred.npy")
    y_train_path = Path(prepared_folder_path / "y_train.npy")
    y_test_path = Path(prepared_folder_path / "y_test.npy")

    print('Data correcly processed.')

    save_preprocessing(x_train_path, x_train_preprocessed)
    save_preprocessing(x_test_path, x_test_preprocessed)
    save_preprocessing(x_pred_path, x_pred_preprocessed)
    save_preprocessing(y_train_path, y_train_preprocessed)
    save_preprocessing(y_test_path, y_test_preprocessed)

    print('Data correcly saved.')

if __name__ == "__main__":
    main()
