import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer

from code_source.data import split_train_val_data
from code_source.entities import SplittingParams
from code_source.entities.feat_params import FeatureParams
from code_source.feat.build_features import make_features


def test_make_features(
        dataset: pd.DataFrame,
        fitted_transformer,
        feature_params: FeatureParams,
        dataset_path: str,
):
    features, target = make_features(fitted_transformer, dataset, feature_params)
    print("Target column name is: ", dataset[feature_params.target_col])
    assert features.shape[1] > 2
    assert not pd.isnull(features).any().any()
    assert_allclose(
        dataset[feature_params.target_col].to_numpy(), target.to_numpy()
    )


def test_split_train_features(
        dataset: pd.DataFrame,
        fitted_transformer: ColumnTransformer,
        feature_params: FeatureParams,
        dataset_path: str,
):
    train_df, val_df = split_train_val_data(dataset, SplittingParams(val_size=0.2))

    train_features, _ = make_features(fitted_transformer, train_df, feature_params)
    val_features, _ = make_features(fitted_transformer, val_df, feature_params)

    assert train_features.columns.equals(val_features.columns)
