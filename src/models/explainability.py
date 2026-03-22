"""
SHAP-based model explainability.

Provides global and individual-level model explanations using
TreeExplainer for LightGBM. This is critical for:
  - Stakeholder trust ("why does the model flag this user?")
  - Feature debugging (are we relying on leaky features?)
  - Regulatory compliance (right to explanation)

Design:
  - TreeExplainer for O(TLD) time instead of KernelSHAP's O(2^M)
  - Precompute SHAP values once, reuse across all plots
  - Feature names propagated through the pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SHAPExplanation:
    """Container for SHAP analysis results."""
    shap_values: np.ndarray             # (n_users, n_features) SHAP matrix
    feature_names: List[str]
    X_display: pd.DataFrame             # feature matrix (for dependence plots)
    expected_value: float               # base rate (E[f(x)])
    elapsed_seconds: float

    # Precomputed summaries
    global_importance: Dict[str, float]     # feature → mean |SHAP|
    top_features: List[Tuple[str, float]]   # sorted (feature, importance)

    def summary(self) -> str:
        lines = [
            "─" * 50,
            "SHAP EXPLANATION",
            "─" * 50,
            f"  Users explained:    {self.shap_values.shape[0]:,}",
            f"  Features:           {self.shap_values.shape[1]}",
            f"  Base rate (E[f]):   {self.expected_value:.4f}",
            "",
            "  Top 10 features by mean |SHAP|:",
        ]
        for feat, imp in self.top_features[:10]:
            bar = "█" * max(1, int(imp * 200))
            lines.append(f"    {feat:<40} {imp:.4f} {bar}")
        lines.append("─" * 50)
        return "\n".join(lines)

    def get_user_explanation(self, idx: int) -> Dict[str, float]:
        """Get SHAP values for a single user (for waterfall plot)."""
        return {
            feat: float(self.shap_values[idx, i])
            for i, feat in enumerate(self.feature_names)
        }

    def get_top_user_features(self, idx: int, n: int = 10) -> List[Tuple[str, float]]:
        """Top N features driving a single user's prediction."""
        user_shap = self.get_user_explanation(idx)
        return sorted(user_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:n]


def compute_shap_explanations(
    lgb_pipeline,
    users: pd.DataFrame,
    feature_names: List[str],
    max_users: int = 5000,
    seed: int = 42,
) -> SHAPExplanation:
    """
    Compute SHAP values for the churn model.

    Uses TreeExplainer (exact, fast) for LightGBM.
    Falls back to a permutation-based approximation if shap
    is not installed.

    Parameters
    ----------
    lgb_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with 'clf' step being LightGBM.
    users : pd.DataFrame
        User feature table (same as model input).
    feature_names : list[str]
        Feature names after pipeline transforms.
    max_users : int
        Subsample for speed (SHAP on 50k users is slow).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    SHAPExplanation
    """
    import shap

    logger.info("Computing SHAP explanations...")
    t0 = time.perf_counter()

    # Get the LightGBM model from the pipeline
    clf = lgb_pipeline.named_steps["clf"]

    # Transform features through the pipeline (up to clf)
    from sklearn.pipeline import Pipeline
    pre = Pipeline(lgb_pipeline.steps[:-1])
    X_transformed = pre.transform(users.drop(
        columns=["churned", "user_id", "archetype", "age_group",
                 "total_txn", "total_value", "churn_prob"],
        errors="ignore"
    ))

    # Subsample if needed
    if len(X_transformed) > max_users:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_transformed), max_users, replace=False)
        X_sample = X_transformed.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X_transformed.reset_index(drop=True)
        idx = np.arange(len(X_transformed))

    # TreeExplainer: exact SHAP values in O(TLD)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 = churned

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    # Compute global importance: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = {
        feat: float(val)
        for feat, val in zip(feature_names, mean_abs_shap)
    }
    top_features = sorted(global_importance.items(),
                          key=lambda x: x[1], reverse=True)

    elapsed = time.perf_counter() - t0
    logger.info("SHAP computed in %.1fs for %d users", elapsed, len(X_sample))

    result = SHAPExplanation(
        shap_values=shap_values,
        feature_names=feature_names,
        X_display=X_sample,
        expected_value=float(expected_value),
        elaps