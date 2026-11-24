"""Raw-data checks and cleaning (Stage 2 of the project).

Turns the raw IBM Telco CSV into the cleaned table used for modeling. The steps are the same
ones documented in the data preparation milestone; they are collected here as functions so the
notebook can run them and verify the result against ``data/processed/telco_clean.csv``.
"""
import pandas as pd

ADDON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def quality_report(raw: pd.DataFrame) -> dict:
    """Integrity checks on the raw table: shape, duplicates, and the blank TotalCharges issue."""
    total = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = raw[total.isna()]
    return {
        "rows": len(raw),
        "columns": raw.shape[1],
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_customer_ids": int(raw["customerID"].duplicated().sum()),
        "blank_total_charges": int(total.isna().sum()),
        "blank_all_tenure_zero": bool((blank["tenure"] == 0).all()),
        "blank_all_retained": bool((blank["Churn"] == "No").all()),
    }


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned table (no missing values, consistent categories, correct types)."""
    df = raw.copy()
    df.columns = [col.strip() for col in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    # TotalCharges is text because 11 new customers (tenure 0) have a blank value.
    # They have not been billed yet, so the correct value is $0 (not the mean, not a deleted row).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    # "No internet service" / "No phone service" repeat what InternetService / PhoneService say.
    for column in ADDON_COLUMNS:
        df[column] = df[column].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # SeniorCitizen is 0/1 while every other flag uses Yes/No.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def iqr_outlier_counts(df: pd.DataFrame) -> dict:
    """Number of values outside 1.5 x IQR for each numeric column."""
    counts = {}
    for column in NUMERIC_COLUMNS:
        q1, q3 = df[column].quantile([0.25, 0.75])
        spread = q3 - q1
        outside = (df[column] < q1 - 1.5 * spread) | (df[column] > q3 + 1.5 * spread)
        counts[column] = int(outside.sum())
    return counts
