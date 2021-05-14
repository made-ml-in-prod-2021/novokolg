import pandas as pd

import logging
import sys
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from code_source.entities.feat_params import FeatureParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def cat_pipeline() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
      ('onehot', OneHotEncoder(handle_unknown='ignore'))])


def num_pipeline() -> Pipeline:
    return Pipeline([
      ('imputer', SimpleImputer(strategy='mean')),
      ('scaler', MinMaxScaler())
    ])


def build_transformer(params) -> ColumnTransformer:
    logger.info(f"categorical features are {params.categorical_features}")
    logger.info(f"numerical features are {params.numerical_features}")
    transformer = ColumnTransformer(
        [
            ("categorical", cat_pipeline(), params.categorical_features),
            ("numerical", num_pipeline(), params.numerical_features),
        ]
    )
    return transformer


def serialize_transformer(transformer: ColumnTransformer, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(transformer, f)
    return output


def make_features(transformer: ColumnTransformer, df: pd.DataFrame, params: FeatureParams,) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def process_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))

def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]


