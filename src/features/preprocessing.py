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

input_folder_path = RAW_DATA_DIR

train_path = Path(input_folder_path / "seg_train")
test_path = Path(input_folder_path / "seg_test")
predict_path = Path(input_folder_path / "seg_pred")

code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x   
      
size = []
for folder in  os.listdir(train_path) : 
    files = gb.glob(pathname= str(train_path / folder / '*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

size = []
for folder in  os.listdir(test_path) : 
    files = gb.glob(pathname= str( test_path / folder / '*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

size = []
files = gb.glob(pathname= str(predict_path  / '*.jpg'))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()


def read_and_process_predictions(predict_path, x):
    files = gb.glob(pathname= str(predict_path / '*.jpg'))
    for file in files: 
        x_data = preprocess_image(file, x) 
    return x_data

def preprocess_image(file, x, folder=None, y=None):
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    x = x.append(list(image_array))
    x_data = np.array(x)

    if y is not None:
        y = y.append(code[folder])
        y_data = np.array(x)
        return x_data, y_data
    return x_data

def read_and_process_images(file_path, x, y):
    for folder in  os.listdir(file_path) : 
        files = gb.glob(pathname= str( file_path  / folder / '*.jpg'))
        for file in files: 
            x_data, y_data = preprocess_image(file, [], folder, [])
    return x_data, y_data

s = 100
x_train = []
y_train = []
x_test = []
y_test = []
x_pred = []

x_train_preprocessed, y_train_preprocessed = read_and_process_images(train_path, x_train, y_train)
x_test_preprocessed, y_test_preprocessed = read_and_process_images(test_path, x_test, y_test)
x_pred_preprocessed = read_and_process_predictions(predict_path, x_pred)

prepared_folder_path = PROCESSED_DATA_DIR

x_train_path = Path(prepared_folder_path / "x_train.npy")
x_test_path = Path(prepared_folder_path / "x_test.npy")
x_pred_path = Path(prepared_folder_path / "x_pred.npy")
y_train_path = Path(prepared_folder_path / "y_train.npy")
y_test_path = Path(prepared_folder_path / "y_test.npy")

print('Data correcly processed.')

def save_preprocessing(path, data_preprocessed):
    np.save(path, data_preprocessed)


save_preprocessing(x_train_path, x_train_preprocessed)
save_preprocessing(x_test_path, x_test_preprocessed)
save_preprocessing(x_pred_path, x_pred_preprocessed)
save_preprocessing(y_train_path, y_train_preprocessed)
save_preprocessing(y_test_path, y_test_preprocessed)

print('Data correcly saved.')

#loaded_array = np.load(x_train_path)

