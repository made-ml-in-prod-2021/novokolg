import os
import pandas as pd
import click
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

@click.command("predict")
@click.argument("data_dir")
@click.argument("preprocessed_dir")
def preprocess(data_dir: str, preprocessed_dir):

    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    target = pd.read_csv(os.path.join(data_dir, "target.csv"))

    data_columns = data.columns.tolist()
    transformer = MinMaxScaler()
    data = pd.DataFrame(transformer.fit_transform(data))
    data.columns = data_columns
    df = pd.concat([pd.DataFrame(data), target], axis = 1)

    os.makedirs(preprocessed_dir, exist_ok=True)
    df.to_csv(os.path.join(preprocessed_dir, "data_preprocessed.csv"), index = False)

    with open(os.path.join(preprocessed_dir, "transformer.pkl"), "wb") as f:
        pickle.dump(transformer, f)

if __name__ == '__main__':
    preprocess()
