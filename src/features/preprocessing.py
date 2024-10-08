import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

################################## GLOBAL VARS ##################################
input_folder_path = RAW_DATA_DIR

train_path = Path(input_folder_path / "seg_train")
test_path = Path(input_folder_path / "seg_test")
predict_path = Path(input_folder_path / "seg_pred")

prepared_folder_path = PROCESSED_DATA_DIR

code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

################################## FUNCTIONS ##################################

def getcode(n) : 
    for x , y in code.items(): 
        if n == y : 
            return x   
      
def count_image_sizes(path, mode):
    size = []
    if mode == 'pred':
        files = gb.glob(pathname= str(path / '*.jpg'))
    for folder in  os.listdir(path) : 
        if mode != 'pred':
            files = gb.glob(pathname= str(path / folder / '*.jpg'))
        for file in files: 
            image = plt.imread(file)
            size.append(image.shape)
    return pd.Series(size).value_counts()


def process_images(file, x, y, s, needs_y=False, folder=None):
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    x.append(list(image_array))
    if needs_y:
        y.append(code[folder])

def read_and_prepare_predictions(predict_path, x, y):
    files = gb.glob(pathname= str(predict_path / '*.jpg'))
    for file in files: 
        process_images(file, x, y, 100) 
    return x

def read_and_prepare_images(path, x, y):
    s = 100
    for folder in os.listdir(path) : 
        files = gb.glob(pathname= str(path / folder / '*.jpg'))
        for file in files: 
            process_images(file, x, y, 100, True, folder)
    return x, y


def list_to_nparray(data):
    return np.array(data)


def save_preprocessing(path, data_preprocessed):
    np.save(path, data_preprocessed)


################################## MAIN ##################################

def main():
    #print("Train image sizes:", count_image_sizes(train_path, 'train'))
    #print("Test image sizes:", count_image_sizes(test_path, 'test'))
    #print("Predict image sizes:", count_image_sizes(predict_path, 'pred'))

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

    #loaded_array = np.load(x_train_path, allow_pickle=True)
    #print(loaded_array)


if __name__ == "__main__":
    main()
