"""Project-wide constants.

Every value that a person might want to change (or that would otherwise be a "magic number"
buried in the code) lives here, so the rest of the package contains logic only.
"""
import numpy as np

# --- Reproducibility and evaluation protocol -------------------------------------------------
SEED = 42                 # one seed for the split, the CV folds, and every model
TEST_SIZE = 0.20          # share of customers held out for final testing
N_CV_FOLDS = 5            # stratified folds used for tuning and threshold selection
DEFAULT_THRESHOLD = 0.50  # conventional probability cut-off, always reported for comparison
F_BETA = 2.0              # beta > 1 weights recall more than precision when choosing a threshold
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)   # candidate cut-offs searched in steps of 0.01

# --- Data columns ----------------------------------------------------------------------------
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"    # value of TARGET_COLUMN that means the customer churned
NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]   # scaled; every other feature is one-hot encoded
COLUMNS_TO_DROP = [
    "customerID",         # identifier, not a predictor
    "gender",             # no relationship with churn in the EDA
    "PhoneService",       # no relationship with churn in the EDA
    "TotalCharges",       # ~ tenure x MonthlyCharges (r = 0.9996): redundant
]

# --- Hyperparameter grids (keys use scikit-learn's "<pipeline step>__<parameter>" naming) ---
PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {
        "model__max_depth": [3, 4, 5, 6],
        "model__min_samples_leaf": [20, 50, 100],
    },
    "Random Forest": {
        "model__max_depth": [6, 8, 12],
        "model__min_samples_leaf": [5, 20],
    },
}
RANDOM_FOREST_TREES = 300
TUNING_METRIC = "roc_auc"  # threshold-independent, so tuning does not depend on the cut-off

# --- Figure styling ---------------------------------------------------------------------------
BLUE, ORANGE, GRAY = "#1F4E79", "#C55A11", "#7F7F7F"
MODEL_COLORS = {"Logistic Regression": BLUE, "Decision Tree": ORANGE, "Random Forest": GRAY}
FIGURE_DPI = 200
TOP_N_FEATURES = 10
