import pickle
import logging
import sys
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from code_source.entities.train_params import TrainingParams
from code_source.entities.feat_params import FeatureParams
from code_source.feat.build_features import make_features

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

SklearnClassifier = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifier:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        logger.warning(f"Model type is is {train_params.model_type }")
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
        model: SklearnClassifier, data: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(data)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
    }


def serialize_model(model: SklearnClassifier, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output