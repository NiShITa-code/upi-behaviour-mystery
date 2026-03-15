"""
Feature engineering pipeline.

Built as sklearn transformers so it can be composed into a Pipeline,
cross-validated correctly, and deployed without train/test leakage.

Key design decisions:
  - All transforms are fit on training data only (no leakage)
  - Pipeline is picklable for model serving
  - Feature names are preserved through the pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyWindowFeatures(BaseEstimator, TransformerMixin):
    """
    Extract and validate early-window engagement features.

    These are the core predictors — computed from first
    `window_days` of a user's lifecycle.

    Stateless transformer (fit() is a no-op) because all
    features are computed directly from raw columns.
    """

    def __init__(self, window_days: int = CFG.features.early_window_days):
        self.window_days = window_days

    def fit(self, X: pd.DataFrame, y=None) -> "EarlyWindowFeatures":
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        # Derived features
        out["txn_d14_per_day"] = (out["txn_d14"] / self.window_days).round(4)
        out["value_d14_per_txn"] = np.where(
            out["txn_d14"] > 0,
            out["value_d14"] / out["txn_d14"],
            0.0,
        ).round(2)
        out["txn_acceleration"] = np.where(
            out["txn_d7"] > 0,
            out["txn_d14"] / out["txn_d7"],
            1.0,
        ).round(3)  # >1 = accelerating, <1 = decelerating

        # Log-transform skewed monetary features (common in payment data)
        for col in ["value_d7", "value_d14"]:
            out[f"log_{col}"] = np.log1p(out[col])

        logger.debug("EarlyWindowFeatures: added %d derived columns",
                     out.shape[1] - X.shape[1])
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features or [])
        return base + [
            "txn_d14_per_day", "value_d14_per_txn", "txn_acceleration",
            "log_value_d7", "log_value_d14",
        ]


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select exactly the features the model expects."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_names].fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


# Extended feature set (base + derived)
EXTENDED_FEATURES = CFG.features.all_features + [
    "txn_d14_per_day",
    "value_d14_per_txn",
    "txn_acceleration",
    "log_value_d7",
    "log_value_d14",
]


def build_feature_pipeline() -> Pipeline:
    """
    Build the full feature preprocessing pipeline.

    Steps:
      1. EarlyWindowFeatures — add derived engagement features
      2. FeatureSelector     — select final feature set
      3. StandardScaler      — z-score normalise (for logistic regression)

    Note: LightGBM doesn't need scaling, but including it lets us
    run both models through the same pipeline without modification.
    LightGBM is scale-invariant so it makes no difference.
    """
    return Pipeline([
        ("early_window",    EarlyWindowFeatures()),
        ("feature_select",  FeatureSelector(EXTENDED_FEATURES)),
        ("scaler",          StandardScaler()),
    ])
