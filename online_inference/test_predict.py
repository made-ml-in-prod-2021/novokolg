import os
import pandas as pd
from fastapi.testclient import TestClient
from app import app


os.environ['PATH_TO_MODEL'] = './model.pkl'
os.environ['PATH_TO_TRANSFORMER'] = './transformer.pkl'
TEST_DATASET_PATH = 'data.csv'

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "it is entry point of our predictor"

def test_predict():
    data = pd.read_csv(TEST_DATASET_PATH)
    data['target'] = 0
    with TestClient(app) as client:
        response = client.get(
            "/predict",
            json={"data": data.loc[[0]].values.tolist(), "features": data.columns.tolist()})
        assert response.status_code == 200
        assert response.json() == [1]

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/healz")
        assert response.status_code == 200
        assert response.json() == True

        response = client.get("/predict")
        assert response.status_code == 200
        assert response.json() == True