import os
import pandas as pd
import pickle
import click
import json
from sklearn.metrics import roc_auc_score, accuracy_score

@click.command("validate")
@click.argument("model_dir")

def validate(model_dir: str):
    data = pd.read_csv(os.path.join(model_dir, "train_data.csv"))

    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    features = data.drop(['target'], axis = 1)
    target = data['target']
    predicts= pd.DataFrame(model.predict(features))

    model_metrics = {
        "roc_auc_score": roc_auc_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
    }

    with open(os.path.join(model_dir, "model_metrics.json"), "w") as f:
        json.dump(model_metrics, f)

if __name__ == '__main__':
    validate()