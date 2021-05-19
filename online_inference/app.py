import logging
import os
import pickle
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class Indicators(BaseModel):
    # age: int
    # sex: int
    # cp: int
    # trestbps: int
    # chol: int
    # fbs: int
    # restecg: int
    # thalach: int
    # exang: int
    # oldpeak: int
    # slope: int
    # ca: int
    # thal: int
    # target: int
    data: List[conlist(Union[int, float, None], min_items=14, max_items=14)]
    features: List[str]

class ModelResponse(BaseModel):
    id: str
    prediction: float


model: Optional[Pipeline] = None
transformer: Optional[Pipeline] = None

def make_predict(
    data: List, features: List[str], model: Pipeline, transformer: Pipeline,
) -> List[ModelResponse]:
    logger.info(f"Making prediction, please, wait..")
    data = pd.DataFrame(data, columns=features)
    ids = np.arange(len(data))
    data['target'] = 0
    logger.info(f"data columns: {data.columns}")
    logger.info(f"data type: {type(data)}")
    logger.info(f"new data shape: {data.shape}")
    logger.info(f"column id length: {len(ids)}")
    features = transformer.transform(data)
    predicts = model.predict(features)
    logger.info(f"preds shape: {predicts.shape}")

    # return [
    #     ModelResponse(id=id_, predicts=bool(pred)) for id_, pred in zip(ids, predicts)
    # ]
    return predicts.tolist()

app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model, transformer
    model_path = os.getenv("PATH_TO_MODEL")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    transformer = load_object(transformer_path)
    model = load_object(model_path)



@app.get("/healz")
def health() -> bool:
    return not (model is None) and not (transformer is None)


@app.get("/predict/", response_model=List[int])
def predict(request: Indicators):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=os.getenv("PORT", 8000))


client = TestClient(app)


def test_read_main():
    response = client.get("/predict")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
