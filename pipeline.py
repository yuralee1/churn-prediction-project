"""End-to-end run: CSV file in, tables and figures out."""
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from . import checks, config, data, evaluation, modeling, plotting

logger = logging.getLogger(__name__)
DISPLAY_COLUMNS_TO_HIDE = ["best_params"]


def save_tables(results: pd.DataFrame, odds: pd.DataFrame, importances: pd.DataFrame,
                output_dir: Path) -> None:
    """Write the three CSV outputs."""
    results.to_csv(output_dir / "model_results.csv", index=False)
    odds.to_csv(output_dir / "logistic_odds_ratios.csv", index=False)
    importances.to_csv(output_dir / "rf_importances.csv", index=False)


def run(csv_path: Union[str, Path] = "data/processed/telco_clean.csv",
        output_dir: Union[str, Path] = "outputs/latest_run"
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the whole analysis and return (results, odds ratios, importances)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = data.load_data(csv_path)
    split = data.split_data(X, y)
    cv = modeling.make_cv()

    runs = modeling.fit_models(split, cv)
    results = evaluation.build_results_table(runs, split.y_test)
    checks.run_all_checks(X, y, split, runs, results)
    logger.info("All sanity checks passed.")

    logger.info("\nTest-set results (train n=%d, test n=%d, churn rate %.1f%%)",
                len(split.X_train), len(split.X_test), 100 * y.mean())
    display_table = results.drop(columns=DISPLAY_COLUMNS_TO_HIDE).round(3)
    logger.info("\n%s", display_table.to_string(index=False))

    best_model = evaluation.select_best_model(runs)
    logger.info("\nBest model by cross-validated ROC-AUC: %s", best_model)

    odds, importances = evaluation.feature_tables(runs)
    save_tables(results, odds, importances, output_dir)
    plotting.plot_evaluation(split.y_test, runs, best_model, output_dir)
    plotting.plot_feature_drivers(odds, importances, output_dir)
    logger.info("Top 5 RF importances:\n%s", importances.head(5).round(3).to_string(index=False))
    return results, odds, importances
