"""
main.py
--------
Entry point for the Credit Scoring Model pipeline.

Orchestrates:
  1. Data preprocessing (load, clean, impute, split, scale)
  2. Exploratory Data Analysis (EDA plots saved to plots/)
  3. Model training  (Logistic Regression + Random Forest)
  4. Model evaluation (metrics + plots)
  5. Best-model selection and persistence to models/

Usage
-----
    python main.py

The dataset must be present at: data/cs-training.csv

Author  : ML Engineering Team
Version : 1.0
"""

import os
import sys
import logging
import pandas as pd

# ── Logging (must be configured before any module imports logger) ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("credit_scoring.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── Add src/ to path so sub-modules resolve correctly ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import preprocess, load_data, drop_index_column, impute_missing_values
from eda                import run_eda
from train              import train_all_models, save_model
from evaluate           import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    compare_models,
    select_best_model,
)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH       = os.path.join("data", "cs-training.csv")
BEST_MODEL_PATH = os.path.join("models", "best_credit_model.pkl")


def main() -> None:
    logger.info("=" * 60)
    logger.info("   CREDIT SCORING MODEL – PIPELINE START")
    logger.info("=" * 60)

    # ── 1. Data Preprocessing ──────────────────────────────────────────────────
    logger.info("\n[STEP 1/5]  Data Preprocessing")
    (
        X_train_scaled, X_test_scaled,
        X_train,        X_test,
        y_train,        y_test,
        feature_names,  scaler,
    ) = preprocess(DATA_PATH)

    # ── 2. Exploratory Data Analysis ───────────────────────────────────────────
    logger.info("\n[STEP 2/5]  Exploratory Data Analysis")
    # Re-load cleaned data for EDA (we need the full DataFrame, not just splits)
    df_eda = load_data(DATA_PATH)
    df_eda = drop_index_column(df_eda)
    df_eda = impute_missing_values(df_eda)
    run_eda(df_eda)

    # ── 3. Model Training ──────────────────────────────────────────────────────
    logger.info("\n[STEP 3/5]  Model Training")
    # Pass raw (unscaled) splits: each Pipeline handles its own scaling internally
    trained_models = train_all_models(X_train, y_train)

    # ── 4. Evaluation ──────────────────────────────────────────────────────────
    logger.info("\n[STEP 4/5]  Model Evaluation")
    results = []
    for name, model in trained_models.items():
        metrics = compute_metrics(model, X_test, y_test, model_name=name)
        results.append(metrics)
        plot_confusion_matrix(y_test, metrics["y_pred"], model_name=name)

    # Overlay ROC curves
    plot_roc_curves(results, y_test)

    # Feature importance for the Random Forest
    rf_model = trained_models.get("Random Forest")
    if rf_model is not None:
        plot_feature_importance(rf_model, feature_names)

    # Summary comparison table
    comparison_df = compare_models(results)
    print("\n" + "=" * 55)
    print("  MODEL PERFORMANCE COMPARISON")
    print("=" * 55)
    print(comparison_df.to_string(index=False))
    print("=" * 55)

    # ── 5. Save Best Model ─────────────────────────────────────────────────────
    logger.info("\n[STEP 5/5]  Select & Save Best Model")
    best_name, best_model = select_best_model(results, trained_models)
    saved_path = save_model(best_model, BEST_MODEL_PATH)

    logger.info("=" * 60)
    logger.info("   PIPELINE COMPLETE")
    logger.info("   Best model : %s", best_name)
    logger.info("   Saved to   : %s", saved_path)
    logger.info("   Plots      : plots/")
    logger.info("   Log file   : credit_scoring.log")
    logger.info("=" * 60)

    print(f"\n✓  Best model: {best_name}")
    print(f"✓  Saved to  : {saved_path}")
    print("✓  All evaluation plots saved to plots/")


if __name__ == "__main__":
    main()