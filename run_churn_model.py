"""Command-line entry point.

Example:
    python run_churn_model.py data/processed/telco_clean.csv outputs/latest_run
"""
import argparse
import logging

from churn_pipeline.pipeline import run


def parse_args() -> argparse.Namespace:
    """Read the optional input path and output folder from the command line."""
    parser = argparse.ArgumentParser(description="Train and evaluate customer churn models.")
    parser.add_argument("csv_path", nargs="?", default="data/processed/telco_clean.csv",
                        help="path to the cleaned Telco CSV (default: data/processed/telco_clean.csv)")
    parser.add_argument("output_dir", nargs="?", default="outputs/latest_run",
                        help="folder for tables and figures (default: outputs/latest_run)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run(args.csv_path, args.output_dir)
