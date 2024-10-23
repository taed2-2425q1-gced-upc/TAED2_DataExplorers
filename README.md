# TAED2_DataExplorers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Scope

This project aims to develop a machine learning system for **classifying images of natural landscapes** into categories: buildings, forests, glaciers, mountains, seas, and streets. The system can help in the detection of environmental changes linked to climate change, such as glacier melting and biodiversity loss, thereby identifying potential natural disaster risks. 

Using a dataset of approximately 25,000 images, the model will employ a **Convolutional Neural Network (CNN)** architecture via a Keras Sequential model. The system not only classifies images but also supports environmental monitoring and disaster preparedness efforts. By addressing potential biases in the dataset, the project ensures accurate predictions, contributing to **global conservation strategies and informed decision-making in environmental management**.


## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`.
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details.
│   ├── model_card     <- A document containing informaction of the model.
│   └── dataset_card   <- A document containing informaction of the dataset.
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

--------

