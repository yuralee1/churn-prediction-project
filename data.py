"""Loading the cleaned churn table and splitting it into training and test sets."""
from pathlib import Path
from typing import NamedTuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class TrainTestData(NamedTuple):
    """The four pieces of a train/test split, kept together so they cannot be mixed up."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data(path: Union[str, Path]) -> tuple[pd.DataFrame, pd.Series]:
    """Read the cleaned CSV and return the feature table and the 0/1 churn target.

    Columns listed in ``config.COLUMNS_TO_DROP`` are removed. The target is 1 for churned
    customers and 0 for retained ones.
    """
    df = pd.read_csv(path)
    y = (df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=config.COLUMNS_TO_DROP + [config.TARGET_COLUMN])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> TrainTestData:
    """Make a stratified train/test split so both parts keep the overall churn rate."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED
    )
    return TrainTestData(X_train, X_test, y_train, y_test)
