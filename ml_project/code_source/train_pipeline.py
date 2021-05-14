import json
import logging
import sys

import click

from code_source.data import read_data, split_train_val_data
from code_source.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from code_source.feat import make_features, extract_target, build_transformer, serialize_transformer
from code_source.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    feature_transformer = build_transformer(training_pipeline_params.feature_params)
    feature_transformer.fit(train_df)
    logger.info(f"train_df.shape after feature_transformer is {feature_transformer.transform(train_df).shape}")

    path_to_transformer = serialize_transformer(feature_transformer, training_pipeline_params.transformer_path)

    train_features = make_features(
        feature_transformer,
        train_df,
        training_pipeline_params.feature_params,
    )
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features = make_features(
        feature_transformer,
        val_df,
        training_pipeline_params.feature_params,
    )
    val_target = extract_target(val_df, training_pipeline_params.feature_params)


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

    return path_to_model, path_to_transformer, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
