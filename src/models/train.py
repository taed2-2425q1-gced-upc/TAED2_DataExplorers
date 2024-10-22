"""
This module handles the training process for an image classification model using Keras and MLflow.
"""
from pathlib import Path
import numpy as np
import keras
import mlflow
import pandas as pd
from codecarbon import EmissionsTracker

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR

def initialize_mlflow_experiment():
    """Initializes MLflow experiment."""
    mlflow.set_experiment("image-classification")
    mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

def load_data(input_folder_path):
    """Loads training data from the specified folder."""
    x_train = np.load(Path(input_folder_path / "x_train.npy"))
    y_train = np.load(Path(input_folder_path / "y_train.npy"))
    return x_train, y_train

def build_model(input_shape):
    """Builds and compiles the Keras model."""
    model = keras.models.Sequential([
        keras.layers.Conv2D(200, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(150, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(120, kernel_size=(3,3), activation='relu'),
        keras.layers.Conv2D(80, kernel_size=(3,3), activation='relu'),
        keras.layers.Conv2D(50, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(6, activation='softmax'),
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def track_emissions(emissions_output_folder, model, x_train, y_train):
    """Tracks CO2 emissions during model training and logs the results."""
    with EmissionsTracker(
        project_name="image-classification",
        measure_power_secs=1,
        tracking_mode="process",
        output_dir=emissions_output_folder,
        output_file="emissions.csv",
        on_csv_write="append",
        default_cpu_power=45,
    ):
        model.fit(x_train, y_train, epochs=40, batch_size=64, verbose=1)

    emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    return emissions_metrics, emissions_params

def log_emissions_to_mlflow(emissions_metrics, emissions_params):
    """Logs emissions data to MLflow."""
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

def save_model(model, model_path):
    """Saves the trained model to the specified path."""
    model.save(model_path)

def train_model():
    """Main function to orchestrate the training process."""
    initialize_mlflow_experiment()

    input_folder_path = PROCESSED_DATA_DIR
    x_train, y_train = load_data(input_folder_path)

    # Build and compile the model
    S = 100
    model = build_model((S, S, 3))

    # Track CO2 emissions during training
    emissions_output_folder = METRICS_DIR
    emissions_metrics, emissions_params = track_emissions(emissions_output_folder, model, x_train, y_train)

    # Log the emissions to MLflow
    log_emissions_to_mlflow(emissions_metrics, emissions_params)

    # Save the trained model
    model_path = MODELS_DIR / "model.h5"
    save_model(model, model_path)

if __name__ == "__main__":
    train_model()
