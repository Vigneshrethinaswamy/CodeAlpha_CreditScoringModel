"""
src/eda.py
-----------
Exploratory Data Analysis (EDA) for the Credit Scoring project.

Generates and saves the following plots to the *plots/* directory:
  • Class distribution bar chart
  • Missing-value heatmap
  • Feature correlation heatmap
  • Distribution plots for key numeric features
  • Box plots comparing defaulters vs. non-defaulters

Author  : ML Engineering Team
Version : 1.0
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers / CI
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# ── Style defaults ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOTS_DIR = "plots"


def _save(fig, name: str) -> None:
    """Save *fig* as a tight-layout PNG inside *PLOTS_DIR*."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved plot -> %s", path)


# ── Individual plot functions ──────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs") -> None:
    """Bar chart of target class counts and percentages."""
    counts = df[target_col].value_counts().sort_index()
    labels = ["Non-Default (0)", "Default (1)"]
    pcts   = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.8)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save(fig, "class_distribution.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Horizontal bar chart of missing-value percentages per feature."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        logger.info("No missing values found – skipping missing-value plot.")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(missing_pct) * 0.5)))
    bars = ax.barh(missing_pct.index, missing_pct.values, color="#4C72B0", edgecolor="white")

    for bar, val in zip(bars, missing_pct.values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=10)

    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Value Percentages by Feature", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, missing_pct.max() * 1.2)
    _save(fig, "missing_values.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of the Pearson correlation matrix (numeric columns only)."""
    corr = df.select_dtypes(include=[np.number]).corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))          # show lower triangle only
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.4,
        annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _save(fig, "correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame,
                                target_col: str = "SeriousDlqin2yrs") -> None:
    """
    KDE / histogram overlay comparing distribution of key numeric features
    split by target class.
    """
    features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfDependents",
    ]
    features = [f for f in features if f in df.columns]

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    palette = {
    0: "#4C72B0",
    1: "#DD8452"}
    labels  = {0: "Non-Default", 1: "Default"}

    for ax, feat in zip(axes, features):
        for cls in [0, 1]:
            data = df.loc[df[target_col] == cls, feat].dropna()
            # clip extreme outliers for readability (99th percentile)
            cap = data.quantile(0.99)
            data = data.clip(upper=cap)
            sns.histplot(
                data, ax=ax, kde=False, stat="density",
                color=palette[cls], label=labels[cls],
                alpha=0.45, linewidth=0,
            )
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # hide unused subplot axes
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions by Target Class", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "feature_distributions.png")


def plot_boxplots(df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs") -> None:
    """Box plots of selected features grouped by target class."""
    features = [
        "age",
        "MonthlyIncome",
        "DebtRatio",
        "NumberOfOpenCreditLinesAndLoans",
    ]
    features = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
    axes = np.atleast_1d(axes)

    for ax, feat in zip(axes, features):
        # cap at 99th percentile to suppress extreme outliers in the plot
        cap = df[feat].quantile(0.99)
        plot_df = df[[feat, target_col]].copy()
        plot_df[feat] = plot_df[feat].clip(upper=cap)

        sns.boxplot(
            data=plot_df,
            x=target_col,
            y=feat,
            hue=target_col,
            palette="Set2",
            legend=False,
            ax=ax
        )
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("SeriousDlqin2yrs")

    fig.suptitle("Box Plots by Default Status (capped at 99th pct)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "boxplots_by_target.png")


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, target_col: str = "SeriousDlqin2yrs") -> None:
    """
    Execute the full EDA pipeline on *df*.

    Parameters
    ----------
    df         : pd.DataFrame  (after cleaning / imputation)
    target_col : str
    """
    logger.info("=== Starting EDA ===")
    logger.info("Shape: %s", df.shape)
    logger.info("Data types:\n%s", df.dtypes.to_string())
    logger.info("Descriptive statistics:\n%s", df.describe().to_string())

    plot_class_distribution(df, target_col)
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df, target_col)
    plot_boxplots(df, target_col)

    logger.info("=== EDA complete – all plots saved to '%s/' ===", PLOTS_DIR)