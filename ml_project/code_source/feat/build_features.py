from typing import Tuple, Optional

import pandas as pd

import logging
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from ml_project.code_source.entities.feat_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def cat_pipeline() -> Pipeline:
    return Pipeline([
      ('imputer', SimpleImputer(strategy='constant')),
      ('onehot', OneHotEncoder(handle_unknown='ignore'))])


def num_pipeline() -> Pipeline:
    return Pipeline([
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler', MinMaxScaler())
    ])


def column_transformer(params) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("categorical", cat_pipeline(), params.categorical_features),
            ("numerical", num_pipeline(), params.numerical_features),
        ]
    )
    return transformer

def make_features(transformer: ColumnTransformer, df: pd.DataFrame, params: FeatureParams,) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df)), df[params.target_col]

def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]


