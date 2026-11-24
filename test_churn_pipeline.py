"""Unit and regression tests. Run from the project folder with:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from churn_pipeline import checks, cleaning, config, data, evaluation, modeling  # noqa: E402


def make_toy_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Small synthetic table with the same columns the real loader expects."""
    rng = np.random.default_rng(seed)
    churn = rng.random(n) < 0.3
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "TotalCharges": rng.random(n) * 1000,
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.random(n) * 100,
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Churn": np.where(churn, "Yes", "No"),
    })


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "toy.csv"
        make_toy_frame().to_csv(self.path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_data_drops_columns_and_encodes_target(self):
        X, y = data.load_data(self.path)
        for column in config.COLUMNS_TO_DROP + [config.TARGET_COLUMN]:
            self.assertNotIn(column, X.columns)
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(len(X), len(y))

    def test_split_is_stratified_and_disjoint(self):
        X, y = data.load_data(self.path)
        split = data.split_data(X, y)
        self.assertEqual(len(split.X_test), round(len(X) * config.TEST_SIZE))
        self.assertTrue(set(split.X_train.index).isdisjoint(split.X_test.index))
        self.assertAlmostEqual(split.y_train.mean(), split.y_test.mean(), delta=0.02)

    def test_split_is_reproducible(self):
        X, y = data.load_data(self.path)
        first, second = data.split_data(X, y), data.split_data(X, y)
        pd.testing.assert_frame_equal(first.X_test, second.X_test)


class ModelingTests(unittest.TestCase):
    def test_vectorized_fbeta_matches_scikit_learn(self):
        rng = np.random.default_rng(1)
        y = rng.random(500) < 0.3
        proba = np.clip(rng.normal(0.3 + 0.3 * y, 0.2), 0, 1)
        grid = config.THRESHOLD_GRID
        expected = [fbeta_score(y, proba >= t, beta=2.0, zero_division=0) for t in grid]
        np.testing.assert_allclose(modeling.fbeta_for_thresholds(y, proba, grid, 2.0), expected)

    def test_fbeta_is_zero_when_nothing_is_predicted_positive(self):
        scores = modeling.fbeta_for_thresholds(np.array([1, 0, 1]), np.array([0.1, 0.2, 0.3]),
                                               np.array([0.99]), 2.0)
        self.assertEqual(scores[0], 0.0)

    def test_preprocessor_scales_numeric_and_encodes_categorical(self):
        X = make_toy_frame().drop(columns=config.COLUMNS_TO_DROP + [config.TARGET_COLUMN])
        transformed = modeling.make_preprocessor(X).fit_transform(X)
        self.assertEqual(transformed.shape[0], len(X))
        np.testing.assert_allclose(transformed[:, :2].mean(axis=0), 0, atol=1e-9)   # scaled columns
        self.assertEqual(transformed.shape[1], 2 + 3)    # 2 numeric + 3 contract levels


class EvaluationTests(unittest.TestCase):
    def test_compute_metrics_on_known_values(self):
        y = pd.Series([1, 1, 0, 0])
        proba = np.array([0.9, 0.4, 0.6, 0.1])
        m = evaluation.compute_metrics(y, proba, 0.5)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["precision"], 0.5)   # 1 true positive of 2 predicted positives
        self.assertEqual(m["recall"], 0.5)      # 1 of 2 churners found
        self.assertEqual(m["roc_auc"], 0.75)


class ChecksTests(unittest.TestCase):
    def test_check_features_rejects_target_column(self):
        X = pd.DataFrame({"tenure": [1, 2], config.TARGET_COLUMN: [0, 1]})
        with self.assertRaises(AssertionError):
            checks.check_features(X)

    def test_check_features_rejects_missing_values(self):
        with self.assertRaises(AssertionError):
            checks.check_features(pd.DataFrame({"tenure": [1.0, np.nan]}))

    def test_check_results_rejects_chance_level_model(self):
        with self.assertRaises(AssertionError):
            checks.check_results(pd.DataFrame({"roc_auc": [0.49], "recall": [0.5], "precision": [0.5]}))


class CleaningTest(unittest.TestCase):
    """The cleaning functions must reproduce the cleaned dataset used for modeling."""
    RAW = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn_raw.csv"
    CLEAN = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"

    @unittest.skipUnless(RAW.exists() and CLEAN.exists(), "data files not found")
    def test_clean_raw_reproduces_saved_clean_file(self):
        cleaned = cleaning.clean_raw(pd.read_csv(self.RAW))
        pd.testing.assert_frame_equal(cleaned, pd.read_csv(self.CLEAN), check_dtype=False)

    @unittest.skipUnless(RAW.exists(), "raw data not found")
    def test_quality_report_finds_the_known_issue(self):
        report = cleaning.quality_report(pd.read_csv(self.RAW))
        self.assertEqual(report["rows"], 7043)
        self.assertEqual(report["blank_total_charges"], 11)
        self.assertTrue(report["blank_all_tenure_zero"] and report["blank_all_retained"])
        self.assertEqual(report["duplicate_rows"], 0)


class RegressionTest(unittest.TestCase):
    """The pipeline must reproduce the reference results.

    Logistic Regression and Decision Tree are deterministic across scikit-learn versions, so
    they must match tightly. Random Forest output can shift slightly between library versions
    (reference run: scikit-learn 1.8.0), so it is compared with a tolerance.
    """
    DATA = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"
    REFERENCE = PROJECT_ROOT / "tests" / "reference_model_results.csv"
    RANDOM_FOREST_TOLERANCE = 0.03

    @unittest.skipUnless(DATA.exists(), "telco_clean.csv not found")
    def test_results_match_reference(self):
        from churn_pipeline.pipeline import run
        with tempfile.TemporaryDirectory() as out:
            results, _, _ = run(self.DATA, out)
        expected = pd.read_csv(self.REFERENCE)
        numeric = ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc", "train_auc"]
        for (_, got), (_, want) in zip(results.iterrows(), expected.iterrows()):
            self.assertEqual((got["model"], got["operating_point"]), (want["model"], want["operating_point"]))
            tolerance = self.RANDOM_FOREST_TOLERANCE if got["model"] == "Random Forest" else 1e-9
            for column in numeric:
                self.assertAlmostEqual(got[column], want[column], delta=max(tolerance, 1e-9),
                                       msg=f"{got['model']} {got['operating_point']} {column}")
            if got["model"] != "Random Forest":
                self.assertEqual(str(got["best_params"]), want["best_params"])


if __name__ == "__main__":
    unittest.main()
