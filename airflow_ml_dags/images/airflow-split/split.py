import os
import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command("split")
@click.argument("preprocessed_dir")
@click.argument("model_dir")
def split(preprocessed_dir: str, model_dir: str):
    data = pd.read_csv(os.path.join(preprocessed_dir, "data_preprocessed.csv"))
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    os.makedirs(model_dir, exist_ok=True)
    train_data.to_csv(os.path.join(model_dir, "train_data.csv"), index = False)
    val_data.to_csv(os.path.join(model_dir, "val_data.csv"), index = False)


if __name__ == "__main__":
    split()