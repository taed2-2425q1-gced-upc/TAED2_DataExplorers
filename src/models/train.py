import numpy as np
import matplotlib.pyplot as plt
import os
import glob as gb
import cv2
import tensorflow as tf
import pickle
from pathlib import Path
import keras
import mlflow
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR

mlflow.set_experiment("image-classification")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    # Path of the parameters file
    #params_path = Path("params.yaml")

    # Path of the prepared data folder
    input_folder_path = PROCESSED_DATA_DIR

    # Read training dataset
    x_train = np.load(Path(input_folder_path / "x_train.npy"))
    y_train = np.load(Path(input_folder_path / "y_train.npy"))

    '''
    # Read data preparation parameters
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["train"]
        except yaml.YAMLError as exc:
            print(exc)
    '''

    # ============== #
    # MODEL TRAINING #
    # ============== #

    # Specify the model
    s = 100
    model = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(s,s,3)),
        keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        keras.layers.Dense(120,activation='relu') ,    
        keras.layers.Dense(100,activation='relu') ,    
        keras.layers.Dense(50,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])
    
    model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # Track the CO2 emissions of training the model
    emissions_output_folder = METRICS_DIR
    with EmissionsTracker(
        project_name="image-classification",
        measure_power_secs=1,
        tracking_mode="process",
        output_dir=emissions_output_folder,
        output_file="emissions.csv",
        on_csv_write="append",
        default_cpu_power=45,
    ):
        # Then fit the model to the training data
        model.fit(x_train, y_train, epochs=40,batch_size=64,verbose=1)

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # Save the model as a pickle file
    Path("models").mkdir(exist_ok=True)

    with open(MODELS_DIR / "model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)

