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

s = 100

x_train = []
y_train = []
for folder in  os.listdir(train_path) : 
    files = gb.glob(pathname= str( train_path  / folder / '*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        x_train.append(list(image_array))
        y_train.append(code[folder])

x_test = []
y_test = []
for folder in  os.listdir(test_path) : 
    files = gb.glob(pathname= str(test_path / folder / '*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        x_test.append(list(image_array))
        y_test.append(code[folder])

x_pred = []
files = gb.glob(pathname= str(predict_path / '*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    x_pred.append(list(image_array))     

x_train = np.array(x_train)
x_test = np.array(x_test)
x_pred = np.array(x_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)

prepared_folder_path = PROCESSED_DATA_DIR

x_train_path = Path(prepared_folder_path / "x_train.npy")
x_test_path = Path(prepared_folder_path / "x_test.npy")
x_pred_path = Path(prepared_folder_path / "x_pred.npy")
y_train_path = Path(prepared_folder_path / "y_train.npy")
y_test_path = Path(prepared_folder_path / "y_test.npy")

print('Data correcly processed.')


np.save(x_train_path, x_train)
np.save(x_test_path, x_test)
np.save(x_pred_path, x_pred)
np.save(y_train_path, y_train)
np.save(y_test_path, y_test)

print('Data correcly saved.')

#loaded_array = np.load(x_train_path)
