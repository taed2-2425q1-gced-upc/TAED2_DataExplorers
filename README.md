# TAED2_DataExplorers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Introduction

This project aims to develop a machine learning system for **classifying images of natural landscapes** into categories: buildings, forests, glaciers, mountains, seas, and streets. The system can help in the detection of environmental changes linked to climate change, such as glacier melting and biodiversity loss, thereby identifying potential natural disaster risks. 

Using a dataset of approximately 25,000 images, the model will employ a **Convolutional Neural Network (CNN)** architecture via a Keras Sequential model. The system not only classifies images but also supports environmental monitoring and disaster preparedness efforts. By addressing potential biases in the dataset, the project ensures accurate predictions, contributing to **global conservation strategies and informed decision-making in environmental management**.


## Project Organization

```
├── Makefile              <- Makefile with convenience commands like `make data` or `make train`.
├── README.md             <- The top-level README for developers using this project.
├── data
│   ├── processed         <- The final, canonical data sets for modeling.
│   └── raw               <- The original, immutable data dump.
│
├── docs                  <- A default mkdocs project; see www.mkdocs.org for details.
│   ├── model_card.md     <- A document containing informaction of the model.
│   └── dataset_card.md   <- A document containing informaction of the dataset.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         taed2_dataexplorers and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── taed2_dataexplorers   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes taed2_dataexplorers a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Main Project Components

### Data Organization

- **Raw Data:** The `data/raw/` folder comprises three subdirectories: `seg_train`, `seg_test`, and `seg_pred`, containing unprocessed training, test and prediction images.
- **Processed Data:** The `data/processed/` directory contains .npy files formatted correctly for model input.

### Model developement

All these stages are included in the `src/features/` and `src/models/` foders.

- **Preprocessing:** This stage involves preparing raw data for model training, which includes transformations to ensure the data is in an optimal format and techniques such as resizing images, to enhance model robustness.
- **Train:** In this stage, the model is trained using the preprocessed data. This includes defining the model architecture, selecting hyperparameters, and running the training process.
- **Evaluate:** The model is evaluated against a validation dataset to assess its performance. A key metrics such as accuracy is computed to measure effectiveness.

### API Integration

We have developed an API that allows users to upload images and receive real-time predictions regarding the classification of natural landscapes. In addition to predictions, the API provides information about the model's training procedure, including key parameters and evaluation metrics. It also tracks the environmental impact of the predictions, reporting on carbon emissions generated during training and evaluation, highlighting our commitment to sustainability.

### Performance Evaluation and Sustainability

- **Evaluation Metrics:** Model performance metrics are documented in the `metrics/scores.json` file for comprehensive evaluation.
- **Environmental Monitoring:** The project tracks the ecological impact of model training, with emissions data in the `metrics/emissions.csv` file.

### Code Quality and Testing

- The `tests/`directory contains unit tests to ensure the code's reliability, spanning various processes from data preprocessing to model predictions.
  The project uses Pylint to enforce coding standards and improve code quality.
--------

