import json
import logging
import sys

import click

from ml_project.code_source.data import read_data, split_train_val_data
from ml_project.code_source.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_project.code_source.feat import make_features
from ml_project.code_source.feat.build_features import column_transformer
from ml_project.code_source.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    feature_transformer = column_transformer(training_pipeline_params.feature_params)
    feature_transformer.fit(train_df)

    train_features, train_target = make_features(
        feature_transformer,
        train_df,
        training_pipeline_params.feature_params,
    )

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features, val_target = make_features(
        feature_transformer,
        val_df,
        training_pipeline_params.feature_params,
    )

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model, val_features
    )

    metrics = evaluate_model(
        predicts,
        val_target
            )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, metrics


if __name__ == "__main__":
    params = read_training_pipeline_params('configs/train_config.yaml')
    train_pipeline(params)
    params = read_training_pipeline_params('configs/train_config_LogReg.yaml')
    train_pipeline(params)