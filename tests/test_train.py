import pytest
import pandas as pd
from unittest import mock
import numpy as np
from pathlib import Path
from src.models import train
from unittest.mock import patch, MagicMock

from src.config import PROCESSED_DATA_DIR

@pytest.fixture
def experiment_path():
    path_to_images = Path(PROCESSED_DATA_DIR / "seg_train")
    return path_to_images

@pytest.fixture
def mock_paths():
    with mock.patch("src.models.train.PROCESSED_DATA_DIR", Path("/fake_path")) as processed_mock:
        with mock.patch("src.models.train.MODELS_DIR", Path("/fake_model_dir")) as models_mock:
            with mock.patch("src.models.train.METRICS_DIR", Path("/fake_metrics_dir")) as metrics_mock:
                yield processed_mock, models_mock, metrics_mock


@pytest.fixture
def mock_training_data():
    x_train = np.random.rand(100, 100, 100, 3)  
    y_train = np.random.randint(0, 6, 100)
    return x_train, y_train


def test_initialize_mlflow_experiment():
    with mock.patch("mlflow.set_experiment") as mock_mlflow_experiment:
        train.initialize_mlflow_experiment()
        mock_mlflow_experiment.assert_called_once_with("image-classification")

def test_load_data(mock_paths):
    processed_mock, _, _ = mock_paths
    with mock.patch("numpy.load") as mock_npy_load:
        x_train, y_train = train.load_data(processed_mock)
        assert mock_npy_load.call_count == 2  

def test_build_model():
    input_shape = (100, 100, 3)
    model = train.build_model(input_shape)
    assert len(model.layers) > 0 
    assert model.input_shape == (None, 100, 100, 3) 

def test_track_emissions(mock_training_data, mock_paths):
    _, _, metrics_mock = mock_paths
    x_train, y_train = mock_training_data
    model = mock.Mock()  
    with mock.patch.object(model, 'fit', return_value=None) as mock_fit:
        with mock.patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'col1': [0], 'col2': [0], 'col3': [0], 'col4': [0], 'col5': [0], 
                'col6': [0], 'col7': [0], 'col8': [0], 'col9': [0], 'col10': [0], 
                'emissions': [10.0], 'col11': [0],'col12': [0], # This represents an emissions metric
                'cpu_power': [45]  # This represents a parameter
            })
            
            emissions_metrics, emissions_params = train.track_emissions(
                metrics_mock, model, x_train, y_train
            )

            mock_fit.assert_called_once_with(x_train, y_train, epochs=40, batch_size=64, verbose=1)
            mock_read_csv.assert_called_once_with(metrics_mock / "emissions.csv")
            #print('EMISSIONS', emissions_metrics)

            assert isinstance(emissions_metrics, dict), "emissions_metrics is not a dictionary"
            assert isinstance(emissions_params, dict), "emissions_params is not a dictionary"
            assert emissions_metrics['emissions'] == 10.0, "the value of 'emissions' is incorrect"
            assert emissions_params['cpu_power'] == 45, "the value of 'cpu_power' is incorrect"

def test_log_emissions_to_mlflow():
    emissions_metrics = {"emissions": 10.0}
    emissions_params = {"cpu_power": 45}
    with mock.patch("mlflow.log_params") as mock_log_params:
        with mock.patch("mlflow.log_metrics") as mock_log_metrics:
            train.log_emissions_to_mlflow(emissions_metrics, emissions_params)
            mock_log_params.assert_called_once_with(emissions_params)
            mock_log_metrics.assert_called_once_with(emissions_metrics)


def test_save_model(mock_paths):
    _, models_mock, _ = mock_paths
    model = mock.Mock() 
    with mock.patch.object(model, 'save') as mock_save: 
        train.save_model(model, models_mock / "model.h5")
        mock_save.assert_called_once_with(models_mock / "model.h5")

def test_train_model(mock_paths):
    # Arrange
    processed_mock, _, _ = mock_paths
    x_train = [np.array(np.zeros((100, 100, 3), dtype=np.uint8)), np.array(np.zeros((100, 100, 3), dtype=np.uint8)), np.array(np.zeros((100, 100, 3), dtype=np.uint8))]
    y_train = [1, 2, 3]

    # Mock functions
    with patch("src.models.train.initialize_mlflow_experiment") as mock_initialize_mlflow_experiment, \
         patch("src.models.train.load_data", return_value=(x_train, y_train)) as mock_load_data, \
         patch("src.models.train.build_model") as mock_build_model, \
         patch("src.models.train.track_emissions") as mock_track_emissions, \
         patch("src.models.train.log_emissions_to_mlflow") as mock_log_emissions_to_mlflow, \
         patch("src.models.train.save_model") as mock_save_model:
        
        # Simular el modelo retornado por build_model
        mock_model = mock.Mock()
        mock_build_model.return_value = mock_model
        mock_track_emissions.return_value = {"emissions": 10.0}, {"cpu_power": 45}


        # Act
        train.train_model()

        # Assert
        mock_initialize_mlflow_experiment.assert_called_once()
        mock_load_data.assert_called_once_with(mock_paths[0])  # El primer path es el de PROCESSED_DATA_DIR
        mock_build_model.assert_called_once_with((100, 100, 3))  # (S, S, 3) es (100, 100, 3)
        mock_track_emissions.assert_called_once()  # Verificar que track_emissions fue llamado
        mock_log_emissions_to_mlflow.assert_called_once()  # Verificar que log_emissions_to_mlflow fue llamado
        mock_save_model.assert_called_once_with(mock_model, mock_paths[1] / "model.h5")  # Guardar el modelo en la ruta esperada