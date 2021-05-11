from faker import Faker
import pandas as pd
import click

faker = Faker()


def create_fake_data(size=100):
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
    fake_data.to_csv("tests/data/train_data_faker.csv", index=False)


@click.command(name="create_fake_data")
@click.argument("size")
def create_fake_data_command(size: str):
    create_fake_data(int(size))


if __name__ == "__main__":
    create_fake_data_command()
