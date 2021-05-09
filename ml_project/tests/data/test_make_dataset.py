from ml_project.code_source.data.make_dataset import read_data, split_train_val_data
from ml_project.code_source.entities.split_params import SplittingParams
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DATASET_PATH = 'data/raw/heart.csv'

def test_load_dataset(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert data.shape[0] > 10
    assert target_col in data.columns


def test_split_dataset(tmpdir, dataset_path: str):
    val_size = 0.1
    splitting_params = SplittingParams(random_state=239, val_size=val_size, )
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10
    logger.info(f'check split: {val.shape[0] / train.shape[0]}, {np.allclose(val.shape[0] / train.shape[0], 0.2, atol=0.06)}')
    assert np.allclose(val.shape[0] / train.shape[0], 0.2, atol=0.06)