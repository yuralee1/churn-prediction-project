"""Test-set evaluation: metrics, the results table, model ranking, and feature tables."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config
from .modeling import ModelRun


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Classification metrics at one threshold (ROC-AUC does not depend on the threshold)."""
    predicted = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted),
        "recall": recall_score(y_true, predicted),
        "f1": f1_score(y_true, predicted),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def build_results_table(runs: dict[str, ModelRun], y_test: pd.Series) -> pd.DataFrame:
    """One row per model and operating point (default cut-off and tuned cut-off)."""
    rows = []
    for run in runs.values():
        operating_points = [
            (f"{config.DEFAULT_THRESHOLD:.2f}", config.DEFAULT_THRESHOLD),
            (f"tuned (F{config.F_BETA:g})", run.threshold),
        ]
        best_params = {key.split("__")[1]: value for key, value in run.search.best_params_.items()}
        for label, threshold in operating_points:
            rows.append({
                "model": run.name,
                "operating_point": label,
                **compute_metrics(y_test, run.test_proba, threshold),
                "cv_auc": run.search.best_score_,
                "train_auc": run.train_auc,
                "best_params": best_params,
            })
    return pd.DataFrame(rows)


def select_best_model(runs: dict[str, ModelRun]) -> str:
    """Name of the model with the highest cross-validated AUC (never chosen by test score)."""
    return max(runs, key=lambda name: runs[name].search.best_score_)


def feature_tables(runs: dict[str, ModelRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Logistic Regression odds ratios and Random Forest importances, strongest first."""
    forest_search = runs["Random Forest"].search.best_estimator_
    feature_names = [name.split("__", 1)[1]
                     for name in forest_search.named_steps["prep"].get_feature_names_out()]
    logistic = runs["Logistic Regression"].search.best_estimator_.named_steps["model"]
    forest = forest_search.named_steps["model"]

    odds = pd.DataFrame({
        "feature": feature_names,
        "coefficient": logistic.coef_[0],
        "odds_ratio": np.exp(logistic.coef_[0]),
    }).sort_values("coefficient", key=abs, ascending=False)
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False)
    return odds, importances
