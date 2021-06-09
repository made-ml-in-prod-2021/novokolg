import os
import pandas as pd
import pickle
import click
from sklearn.ensemble import RandomForestClassifier


@click.command("train_model")
@click.argument("model_dir")
def train_model(model_dir: str):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    data = pd.read_csv(os.path.join(model_dir, "train_data.csv"))
    features = data.drop(['target'], axis = 1)
    target = data['target']
    model.fit(features, target)

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()