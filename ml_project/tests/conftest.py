import os
import pytest
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from ml_project.code_source.entities import FeatureParams
from ml_project.code_source.data import read_data
from ml_project.code_source.feat.build_features import build_transformer


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "data\\train_data_faker.csv")


@pytest.fixture()
def dataset(dataset_path: str) -> pd.DataFrame:
    data = read_data(dataset_path)
    return data


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


@pytest.fixture()
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


@pytest.fixture
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return params


@pytest.fixture()
def fitted_transformer(
        dataset: pd.DataFrame, feature_params: FeatureParams
) -> ColumnTransformer:
    fitted_transformer = build_transformer(feature_params)
    fitted_transformer.fit(dataset)
    return fitted_transformer
