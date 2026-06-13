"""
src/data_preprocessing.py
--------------------------
Handles all data loading, cleaning, and preprocessing steps for the
Credit Scoring Model project (Give Me Some Credit dataset).

Steps performed:
  1. Load raw CSV
  2. Drop the index artifact column ("Unnamed: 0")
  3. Impute missing values with column medians
  4. Separate features (X) from the target (y)
  5. Train / test split with stratification to preserve class balance
  6. Standard-scale features for linear models

Author  : ML Engineering Team
Version : 1.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_COL      = "SeriousDlqin2yrs"
DROP_COLS       = ["Unnamed: 0"]
IMPUTE_COLS     = ["MonthlyIncome", "NumberOfDependents"]
TEST_SIZE       = 0.20
RANDOM_STATE    = 42


# ── Public API ─────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset from *filepath*.

    Parameters
    ----------
    filepath : str
        Path to cs-training.csv (or any compatible CSV).

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as read from disk.
    """
    logger.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Raw shape: %s", df.shape)
    return df


def drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the 'Unnamed: 0' artifact column left by row-indexing in CSV.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with the column removed (no-op if absent).
    """
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in *IMPUTE_COLS* using each column's median.

    Using median rather than mean is more robust to the right-skewed
    distributions typical in financial data.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    for col in IMPUTE_COLS:
        if col not in df.columns:
            logger.warning("Column '%s' not found – skipping imputation.", col)
            continue

        n_missing = df[col].isna().sum()
        if n_missing == 0:
            logger.info("No missing values in '%s'.", col)
            continue

        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        logger.info(
            "Imputed %d missing values in '%s' with median=%.4f",
            n_missing, col, median_val,
        )

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(
        "Feature matrix shape: %s  |  Target distribution:\n%s",
        X.shape,
        y.value_counts(normalize=True).round(4).to_dict(),
    )
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series):
    """
    Stratified train / test split (80 / 20).

    Stratification is essential here because the dataset is heavily
    imbalanced (~6-7 % positives).

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train size: %d  |  Test size: %d  |  "
        "Train positive rate: %.4f  |  Test positive rate: %.4f",
        len(X_train), len(X_test),
        y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit a StandardScaler on the training set and apply it to both splits.

    Fitting only on X_train prevents data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : StandardScaler  (fitted; reuse for inference)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    logger.info("Feature scaling complete.")
    return X_train_scaled, X_test_scaled, scaler


def preprocess(filepath: str):
    """
    Full preprocessing pipeline: load → clean → impute → split → scale.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    X_train_scaled, X_test_scaled, X_train, X_test,
    y_train, y_test, feature_names, scaler
    """
    df      = load_data(filepath)
    df      = drop_index_column(df)
    df      = impute_missing_values(df)
    X, y    = split_features_target(df)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return (
        X_train_scaled, X_test_scaled,
        X_train, X_test,
        y_train, y_test,
        feature_names, scaler,
    )