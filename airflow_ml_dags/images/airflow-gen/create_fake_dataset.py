import os
from faker import Faker
import pandas as pd
import click

faker = Faker()


def create_fake_data(size: int, path: str):
    rows = [{'age': faker.pyint(0, 100),
             'sex': faker.pyint(0, 1),
             'cp': faker.pyint(0, 3),
             'trestbps': faker.pyint(80, 100),
             'chol': faker.pyint(100, 600),
             'fbs': faker.pyint(0, 1),
             'restecg': faker.pyint(0, 3),
             'thalach': faker.pyint(70, 200),
             'exang': faker.pyint(0, 1),
             'oldpeak': faker.pydecimal(0, 7),
             'slope': faker.pyint(0, 2),
             'ca': faker.pyint(0, 4),
             'thal': faker.pyint(0, 3),
             'target': faker.pyint(0, 1)} for x in range(size)]
    fake_data = pd.DataFrame(rows)
    print(os.path.join(path, f"data.csv"))
    output_data = os.path.join(path, f"data.csv")
    output_target = os.path.join(path, f"target.csv")
    print(fake_data.drop('target', axis =1))
    fake_data.drop('target', axis =1 ).to_csv(output_data, index=False)
    fake_data['target'].to_csv(output_target, index=False)


@click.command(name="create_fake_data")
@click.argument("size")
@click.argument("path")
def create_fake_data_command(size: str, path:str):
    create_fake_data(int(size), path)


if __name__ == "__main__":
    create_fake_data_command()
