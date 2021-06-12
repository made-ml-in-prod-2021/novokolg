import os
import pandas as pd
import pickle
import click


@click.command("predict")
@click.argument("data_dir")
@click.argument("preprocessed_dir")
@click.argument("model_dir")
@click.argument("prediction_dir")
def predict(data_dir: str, preprocessed_dir: str, model_dir: str, prediction_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    with open(os.path.join(preprocessed_dir, "transformer.pkl"), "rb") as f:
        transformer = pickle.load(f)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    features = transformer.transform(data)
    prediction = pd.DataFrame(model.predict(features))

    os.makedirs(prediction_dir, exist_ok=True)
    prediction.to_csv(os.path.join(prediction_dir, "prediction.csv"))



if __name__ == '__main__':
    predict()