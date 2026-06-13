"""
src/evaluate.py
----------------
Model evaluation utilities for the Credit Scoring project.

Computes and reports:
  • Accuracy
  • Precision, Recall, F1-Score
  • ROC-AUC
  • Confusion Matrix (absolute + normalised)
  • Classification Report
  • ROC curve comparison plot
  • Feature Importance plot (Random Forest)

Author  : ML Engineering Team
Version : 1.0
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)

logger   = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOTS_DIR = "plots"


# ── Core metrics ───────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    model      : fitted sklearn estimator / Pipeline
    X_test     : feature matrix
    y_test     : true labels
    model_name : str – label for log output

    Returns
    -------
    dict with keys: model_name, accuracy, precision, recall,
                    f1, roc_auc, y_pred, y_prob
    """
    y_pred = model.predict(X_test)

    # predict_proba always returns [p_class0, p_class1]; take positive class
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    metrics = {
        "model_name": model_name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":    roc_auc_score(y_test, y_prob),
        "y_pred":     y_pred,
        "y_prob":     y_prob,
    }

    logger.info(
        "\n[%s] Accuracy=%.4f  Precision=%.4f  Recall=%.4f  "
        "F1=%.4f  ROC-AUC=%.4f",
        model_name,
        metrics["accuracy"],  metrics["precision"],
        metrics["recall"],    metrics["f1"],
        metrics["roc_auc"],
    )
    logger.info(
        "[%s] Classification Report:\n%s",
        model_name,
        classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]),
    )
    return metrics


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str) -> None:
    """
    Save a confusion matrix plot (absolute counts + normalised percentages).

    Parameters
    ----------
    y_test     : true labels
    y_pred     : predicted labels
    model_name : str
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, matrix, fmt, title_suffix in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["(Counts)", "(Normalised)"],
    ):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Non-Default", "Default"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=fmt)
        ax.set_title(f"{model_name} – Confusion Matrix {title_suffix}",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    _save(fig, f"confusion_matrix_{safe_name}.png")


# ── ROC curve comparison ───────────────────────────────────────────────────────

def plot_roc_curves(results: list[dict], y_test) -> None:
    """
    Overlay ROC curves for all models on a single figure.

    Parameters
    ----------
    results : list of metric dicts (output of compute_metrics)
    y_test  : true labels
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{res['model_name']}  (AUC = {res['roc_auc']:.4f})")

    # random-guess baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _save(fig, "roc_curves_comparison.png")


# ── Feature importance (Random Forest) ────────────────────────────────────────

def plot_feature_importance(rf_pipeline, feature_names: list[str]) -> None:
    """
    Horizontal bar chart of mean impurity-decrease importance from the
    Random Forest.

    Parameters
    ----------
    rf_pipeline   : fitted Pipeline containing a RandomForestClassifier
    feature_names : list[str]
    """
    # Navigate to the classifier step inside the Pipeline
    clf = rf_pipeline.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        logger.warning("No feature importances available.")
        return

    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in indices]
    imp_sorted   = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(names_sorted[::-1], imp_sorted[::-1],
                   color="#4C72B0", edgecolor="white")

    for bar, val in zip(bars, imp_sorted[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean Impurity Decrease (Gini Importance)")
    ax.set_title("Random Forest – Feature Importances", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlim(0, imp_sorted.max() * 1.18)
    plt.tight_layout()
    _save(fig, "feature_importance_rf.png")


# ── Metrics comparison table ───────────────────────────────────────────────────

def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a tidy DataFrame comparing all model metrics.

    Parameters
    ----------
    results : list of metric dicts

    Returns
    -------
    pd.DataFrame  sorted by ROC-AUC descending.
    """
    rows = []
    for res in results:
        rows.append({
            "Model":     res["model_name"],
            "Accuracy":  res["accuracy"],
            "Precision": res["precision"],
            "Recall":    res["recall"],
            "F1-Score":  res["f1"],
            "ROC-AUC":   res["roc_auc"],
        })

    comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    comparison = comparison.reset_index(drop=True)

    logger.info("\n=== Model Comparison ===\n%s", comparison.to_string(index=False))
    return comparison


def select_best_model(results: list[dict], trained_models: dict) -> tuple:
    """
    Select the best-performing model based on ROC-AUC score.

    Parameters
    ----------
    results        : list of metric dicts
    trained_models : dict mapping model_name → fitted Pipeline

    Returns
    -------
    (best_model_name: str, best_model: Pipeline)
    """
    best = max(results, key=lambda r: r["roc_auc"])
    best_name  = best["model_name"]
    best_model = trained_models[best_name]

    logger.info(
        "Best model: %s  (ROC-AUC = %.4f)",
        best_name, best["roc_auc"],
    )
    return best_name, best_model


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    """Save figure to PLOTS_DIR."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved plot -> %s", path)