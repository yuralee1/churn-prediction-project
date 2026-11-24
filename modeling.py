"""Preprocessing, model pipelines, hyperparameter tuning, and decision-threshold selection.

Every model is a full scikit-learn Pipeline (preprocessing + estimator). Because of this,
scaling and encoding are re-fit on the training part of each cross-validation fold, so no
information from validation or test rows leaks into training.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data import TrainTestData


@dataclass
class ModelRun:
    """Everything produced for one model: the fitted search, its threshold, and test scores."""
    name: str
    search: GridSearchCV            # fitted on the training data only
    threshold: float                # recall-favoring cut-off chosen from training data
    test_proba: np.ndarray          # predicted churn probability for each test customer
    train_auc: float                # training ROC-AUC, used to spot overfitting


# ------------------------------------------------------------------------------ preprocessing
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Standardize the numeric columns and one-hot encode all the others.

    Binary columns become a single 0/1 column; unseen categories are ignored at prediction time.
    """
    categorical = [col for col in X.columns if col not in config.NUMERIC_FEATURES]
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
    ])


# ------------------------------------------------------------------------------ model search
def make_cv() -> StratifiedKFold:
    """Cross-validation splitter shared by tuning and threshold selection."""
    return StratifiedKFold(config.N_CV_FOLDS, shuffle=True, random_state=config.SEED)


def build_searches(X: pd.DataFrame, cv: StratifiedKFold) -> dict[str, GridSearchCV]:
    """Create one (unfitted) grid search per model, keyed by model name."""
    estimators = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=config.SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST_TREES, class_weight="balanced",
            random_state=config.SEED, n_jobs=-1),
    }
    return {
        name: GridSearchCV(
            Pipeline([("prep", make_preprocessor(X)), ("model", estimator)]),
            config.PARAM_GRIDS[name], cv=cv, scoring=config.TUNING_METRIC, n_jobs=-1)
        for name, estimator in estimators.items()
    }


# ------------------------------------------------------------------------------ thresholds
def fbeta_for_thresholds(y_true: np.ndarray, proba: np.ndarray,
                         thresholds: np.ndarray, beta: float) -> np.ndarray:
    """F-beta score at every candidate threshold, computed in one vectorized pass.

    Equivalent to calling ``sklearn.metrics.fbeta_score`` once per threshold, but without
    a Python loop. A score of 0 is returned when the formula has no positive predictions.
    """
    y_true = np.asarray(y_true).astype(bool)
    predicted_positive = proba[:, None] >= thresholds[None, :]          # (customers, thresholds)
    true_pos = (predicted_positive & y_true[:, None]).sum(axis=0)
    false_pos = predicted_positive.sum(axis=0) - true_pos
    false_neg = y_true.sum() - true_pos
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * true_pos
    denominator = numerator + beta_sq * false_neg + false_pos
    return np.divide(numerator, denominator, out=np.zeros(len(thresholds)), where=denominator > 0)


def find_best_threshold(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                        cv: StratifiedKFold, beta: float = config.F_BETA) -> float:
    """Pick the cut-off that maximizes F-beta on out-of-fold TRAINING predictions.

    Using out-of-fold predictions means the threshold is chosen without touching the test set.
    """
    out_of_fold = cross_val_predict(
        estimator, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    scores = fbeta_for_thresholds(y_train, out_of_fold, config.THRESHOLD_GRID, beta)
    return float(config.THRESHOLD_GRID[int(np.argmax(scores))])


# ------------------------------------------------------------------------------ orchestration
def fit_models(data: TrainTestData, cv: StratifiedKFold) -> dict[str, ModelRun]:
    """Tune every model on the training set and score it on the test set (used once)."""
    runs = {}
    for name, search in build_searches(data.X_train, cv).items():
        search.fit(data.X_train, data.y_train)
        runs[name] = ModelRun(
            name=name,
            search=search,
            threshold=find_best_threshold(search.best_estimator_, data.X_train, data.y_train, cv),
            test_proba=search.predict_proba(data.X_test)[:, 1],
            train_auc=_train_auc(search, data),
        )
    return runs


def _train_auc(search: GridSearchCV, data: TrainTestData) -> float:
    """Training-set ROC-AUC of a fitted search (compared with test AUC to detect overfitting)."""
    return roc_auc_score(data.y_train, search.predict_proba(data.X_train)[:, 1])
