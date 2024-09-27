import json
import pickle
from pathlib import Path
import numpy as np
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import METRICS_DIR, PROCESSED_DATA_DIR

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")


def load_validation_data(input_folder_path: Path):
    """Load the validation data from the prepared data folder.

    Args:
        input_folder_path (Path): Path to the prepared data folder.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the validation features and target.
    """
    x_valid = np.load(Path(input_folder_path / "x_test.npy"))
    y_valid = np.load(Path(input_folder_path / "y_test.npy"))


    return x_valid, y_valid


def evaluate_model(model_file_name, x, y):
    """Evaluate the model using the validation data.

    Args:
        model_file_name (str): Filename of the model to be evaluated.
        x (pd.DataFrame): Validation features.
        y (pd.DataFrame): Validation target.

    Returns:
        Tuple[float, float]: Tuple containing the MAE and MSE values.
    """

    with open(MODELS_FOLDER_PATH / model_file_name, "rb") as pickled_model:
        model = pickle.load(pickled_model)

        # Compute accuracy using the model
    accuracy = model.evaluate(x, y)
    return accuracy


if __name__ == "__main__":
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = METRICS_DIR

    x_valid, y_valid = load_validation_data(PROCESSED_DATA_DIR)

    mlflow.set_experiment("image-classification")

    with mlflow.start_run():
        # Load the model
        accuracy= evaluate_model(
            "model.pkl", x_valid, y_valid
        )

        # Save the evaluation metrics to a dictionary to be reused later
        metrics_dict = {"accuracy": accuracy}

        # Log the evaluation metrics to MLflow
        mlflow.log_metrics(metrics_dict)

        # Save the evaluation metrics to a JSON file
        with open(metrics_folder_path / "scores.json", "w") as scores_file:
            json.dump(
                metrics_dict,
                scores_file,
                indent=4,
            )

        print("Evaluation completed.")