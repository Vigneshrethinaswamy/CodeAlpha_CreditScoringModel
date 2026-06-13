"""
src/train.py
-------------
Model training for the Credit Scoring project.

Trains two classifiers:
  1. Logistic Regression  – fast, interpretable baseline
  2. Random Forest        – ensemble tree model, captures non-linearities

Both are wrapped in scikit-learn Pipeline objects so that any future
preprocessing steps can be inserted cleanly.

Class imbalance is handled via *class_weight='balanced'*, which re-weights
the loss function to give the minority class (defaulters) more influence.

Author  : ML Engineering Team
Version : 1.0
"""

import numpy as np
import joblib
import os
import logging
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler

logger = logging.getLogger(__name__)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
LR_PARAMS = dict(
    max_iter      = 1000,
    class_weight  = "balanced",   # compensate for ~6 % minority class
    solver        = "lbfgs",
    C             = 1.0,
    random_state  = 42,
)

RF_PARAMS = dict(
    n_estimators  = 200,
    max_depth     = 10,
    min_samples_leaf = 50,
    class_weight  = "balanced",
    random_state  = 42,
    n_jobs        = -1,           # use all available CPU cores
)

MODELS_DIR = "models"


# ── Builders ───────────────────────────────────────────────────────────────────

def build_logistic_regression() -> Pipeline:
    """
    Return a Pipeline: StandardScaler → LogisticRegression.

    The scaler is included inside the pipeline so that the model object
    is self-contained and can be applied to raw feature matrices directly.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**LR_PARAMS)),
    ])
    logger.info("Built Logistic Regression pipeline with params: %s", LR_PARAMS)
    return pipeline


def build_random_forest() -> Pipeline:
    """
    Return a Pipeline: RandomForestClassifier.

    Tree-based models do not need feature scaling, but wrapping in a
    Pipeline keeps the interface consistent.
    """
    pipeline = Pipeline([
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])
    logger.info("Built Random Forest pipeline with params: %s", RF_PARAMS)
    return pipeline


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(model: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                model_name: str = "Model") -> Pipeline:
    """
    Fit *model* on the training data.

    Parameters
    ----------
    model      : sklearn Pipeline
    X_train    : np.ndarray or pd.DataFrame  – feature matrix
    y_train    : np.ndarray or pd.Series     – target vector
    model_name : str  – label used in log messages

    Returns
    -------
    Fitted pipeline.
    """
    logger.info("Training %s on %d samples …", model_name, len(y_train))
    model.fit(X_train, y_train)
    logger.info("%s training complete.", model_name)
    return model


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model: Pipeline, filename: str) -> str:
    """
    Persist *model* to disk using joblib.

    Parameters
    ----------
    model    : fitted sklearn Pipeline
    filename : target path (e.g. 'models/best_credit_model.pkl')

    Returns
    -------
    str  – absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    joblib.dump(model, filename)
    logger.info("Model saved -> %s", os.path.abspath(filename))
    return os.path.abspath(filename)


def load_model(filepath: str) -> Pipeline:
    """
    Load a previously saved model from *filepath*.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Fitted sklearn Pipeline.
    """
    model = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return model


# ── Full training pipeline ─────────────────────────────────────────────────────

def train_all_models(X_train, y_train) -> dict:
    """
    Build and train all models.  Returns a dict keyed by model name.

    Parameters
    ----------
    X_train : feature matrix (already scaled for LR if raw arrays are passed)
    y_train : target vector

    Returns
    -------
    dict[str, Pipeline]  – {'Logistic Regression': <fitted>, 'Random Forest': <fitted>}
    """
    models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest":       build_random_forest(),
    }

    trained = {}
    for name, model in models.items():
        trained[name] = train_model(model, X_train, y_train, model_name=name)

    return trained