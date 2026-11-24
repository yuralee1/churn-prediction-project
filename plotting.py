"""Figures. Each function draws on a given Axes so panels can be combined or reused."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from . import config
from .modeling import ModelRun


def _clean_axes(*axes: Axes) -> None:
    """Remove the top and right borders for a cleaner look."""
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)


def plot_roc_curves(ax: Axes, y_test: pd.Series, runs: dict[str, ModelRun]) -> None:
    """ROC curve for every model, with the AUC in the legend."""
    for name, run in runs.items():
        fpr, tpr, _ = roc_curve(y_test, run.test_proba)
        auc = roc_auc_score(y_test, run.test_proba)
        ax.plot(fpr, tpr, color=config.MODEL_COLORS[name], lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)  # chance line
    ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title="ROC curves (test set)")


def plot_confusion_matrix(ax: Axes, y_test: pd.Series, run: ModelRun) -> None:
    """Confusion matrix at the model's tuned threshold: count and row percentage per cell."""
    counts = confusion_matrix(y_test, run.test_proba >= run.threshold)
    row_pct = counts / counts.sum(axis=1, keepdims=True) * 100
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{counts[row, col]}\n({row_pct[row, col]:.0f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if row_pct[row, col] > 50 else "black")
    labels = ["Retained", "Churned"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual",
           title=f"{run.name}, threshold {run.threshold:.2f}")


def plot_precision_recall(ax: Axes, y_test: pd.Series, runs: dict[str, ModelRun]) -> None:
    """Precision-recall curve for every model, with the churn rate as the no-skill baseline."""
    for name, run in runs.items():
        precision, recall, _ = precision_recall_curve(y_test, run.test_proba)
        ax.plot(recall, precision, color=config.MODEL_COLORS[name], lw=2, label=name)
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1, label="Baseline (churn rate)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    ax.legend(fontsize=8)


def plot_evaluation(y_test: pd.Series, runs: dict[str, ModelRun], best_model: str,
                    output_dir: Path) -> None:
    """Three-panel evaluation figure saved as model_evaluation.png."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    plot_roc_curves(axes[0], y_test, runs)
    plot_confusion_matrix(axes[1], y_test, runs[best_model])
    plot_precision_recall(axes[2], y_test, runs)
    _clean_axes(*axes)
    _save(fig, output_dir / "model_evaluation.png")


def plot_feature_drivers(odds: pd.DataFrame, importances: pd.DataFrame, output_dir: Path) -> None:
    """Top Random Forest importances next to top Logistic Regression coefficients."""
    fig, (ax_forest, ax_logit) = plt.subplots(1, 2, figsize=(12, 4.8))
    top_forest = importances.head(config.TOP_N_FEATURES).iloc[::-1]   # reversed: largest on top
    ax_forest.barh(top_forest["feature"], top_forest["importance"], color=config.BLUE)
    ax_forest.set_title(f"Random Forest: top {config.TOP_N_FEATURES} importances")

    top_logit = odds.head(config.TOP_N_FEATURES).iloc[::-1]
    bar_colors = [config.ORANGE if coef > 0 else config.BLUE for coef in top_logit["coefficient"]]
    ax_logit.barh(top_logit["feature"], top_logit["coefficient"], color=bar_colors)
    ax_logit.axvline(0, color="k", lw=0.8)
    ax_logit.set_title(f"Logistic Regression: top {config.TOP_N_FEATURES} coefficients\n"
                       "(orange raises churn risk)")
    _clean_axes(ax_forest, ax_logit)
    _save(fig, output_dir / "feature_drivers.png")


def _save(fig: plt.Figure, path: Path) -> None:
    """Tighten the layout, write the PNG, and free the figure's memory."""
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(fig)
