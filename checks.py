"""Sanity checks that stop the run when something is wrong (for example, data leakage).

Each check is a small function that raises AssertionError with a clear message,
so a failure points directly at the problem.
"""
import numpy as np
import pandas as pd

from . import config
from .data import TrainTestData
from .modeling import ModelRun

CHURN_RATE_TOLERANCE = 0.005   # allowed difference between split and overall churn rate


def check_features(X: pd.DataFrame) -> None:
    """Features must be complete and must not contain dropped columns or the target."""
    assert X.isna().sum().sum() == 0, "missing values in features"
    forbidden = config.COLUMNS_TO_DROP + [config.TARGET_COLUMN]
    assert not any(col in X.columns for col in forbidden), "dropped/target column still in X"


def check_split(X: pd.DataFrame, y: pd.Series, data: TrainTestData) -> None:
    """Train and test must be disjoint, complete, and stratified."""
    assert len(data.X_train) + len(data.X_test) == len(X), "rows lost in the split"
    assert set(data.X_train.index).isdisjoint(data.X_test.index), "train/test overlap"
    for part in (data.y_train, data.y_test):
        assert abs(part.mean() - y.mean()) < CHURN_RATE_TOLERANCE, "split is not stratified"


def check_no_leakage(runs: dict[str, ModelRun], X_train: pd.DataFrame) -> None:
    """Each fitted scaler must have learned its mean from training rows only."""
    training_means = X_train[config.NUMERIC_FEATURES].mean().values
    for name, run in runs.items():
        scaler = run.search.best_estimator_.named_steps["prep"].named_transformers_["num"]
        assert np.allclose(scaler.mean_, training_means), f"{name}: scaler not fit on training data"


def check_results(results: pd.DataFrame) -> None:
    """Every model must beat chance and report valid proportions."""
    assert (results["roc_auc"] > 0.5).all(), "a model is no better than chance"
    for column in ("recall", "precision"):
        assert (results[column] <= 1).all(), f"{column} above 1"


def run_all_checks(X: pd.DataFrame, y: pd.Series, data: TrainTestData,
                   runs: dict[str, ModelRun], results: pd.DataFrame) -> None:
    """Run every check; raises AssertionError on the first failure."""
    check_features(X)
    check_split(X, y, data)
    check_no_leakage(runs, data.X_train)
    check_results(results)
