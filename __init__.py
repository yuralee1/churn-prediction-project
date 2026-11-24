"""Customer churn modeling pipeline for the IBM Telco Customer Churn dataset.

Modules
-------
config      Constants: seed, split sizes, feature lists, hyperparameter grids, colors.
data        Loading the cleaned CSV and creating the stratified train/test split.
modeling    Preprocessing, model pipelines, hyperparameter search, threshold selection.
evaluation  Test-set metrics, results table, model ranking, feature tables.
plotting    All figures (ROC, confusion matrix, precision-recall, feature drivers).
checks      Sanity checks that stop the run if something is wrong (e.g. data leakage).
pipeline    Orchestrates the full run from CSV file to saved outputs.
"""
