import logging
import pandas as pd
import sys
import pickle
import click

from code_source.feat.build_features import make_features
from code_source.entities.train_pipeline_params import read_training_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def predict(data_path: str,
            config_path: str
            ) -> pd.Series:
    data = pd.read_csv(data_path)
    logger.info(f"data columns: {data.columns}")
    logger.info(f"data columns: {type(data)}")
    logger.info(f"new data shape: {data.shape}")
    params = read_training_pipeline_params(config_path)
    feat_params = params.feature_params
    logger.info(f"params: {params}")
    data[feat_params.target_col] = 0
    transformer = pickle.load(open(params.transformer_path, "rb"))
    features, target = make_features(transformer, data, feat_params)

    model = pickle.load(open(params.output_model_path, "rb"))
    preds = pd.Series(model.predict(features))
    logger.info(f"preds shape: {preds.shape}")
    preds.to_csv("data/predictions.csv")

    return preds


@click.command(name="predict_pipeline")
@click.option("--data_path", default="data/raw/heart.csv")
@click.option("--config_path", default="configs/train_config.yaml")
def predict_pipeline_command(data_path: str, config_path: str):
    predict(data_path, config_path)


if __name__ == "__main__":
    predict_pipeline_command()
