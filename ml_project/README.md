

#Homework1 ML Project

##What have been done description:

Was build a prediction for the competition: https://www.kaggle.com/ronitf/heart-disease-uci

##Architectural" and tactical solutions:

####Installation:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
####Usage:

python code_source\train_pipeline.py configs\train_config.yaml
or
python code_source\train_pipeline.py configs\train_config_LogReg.yaml
####Test:

pytest -cov

####Create fake dataset:
For fake dataset generation, please type "python tests/data/create_fake_dataset.py" and size of dataset.
For example: 

python tests/data/create_fake_dataset.py 100

#### Predict
For prediction for new dataset, please provide dataset path and which model to use by setting model configuration path 

python python predict_pipeline.py --data_path data/raw/heart.csv --config_path configs/train_config.yaml


####Project Organization

├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models with its metrics and transformers
│
├── notebooks          <- Jupyter notebook with EDA
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── code_source        <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- code to download or generate data
│   │
│   ├── entities       <- code to get entities
│   │
│   ├── features       <- code to turn raw data into features for modeling
│   │
│   ├── models         <- code to train models and then use trained models to make

