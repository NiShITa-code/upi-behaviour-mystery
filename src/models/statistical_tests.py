"""
Statistical tests for model comparison.

Implements the DeLong test for comparing AUC-ROC curves,
ensuring that observed model differences are statistically
significant and not due to sampling variability.

Reference:
    DeLong, DeLong & Clarke-Pearson (1988).
    "Comparing the Areas under Two or More Correlated Receiver
    Operating Characteristic Curves: A Nonparametric Approach."
    Biometrics, 44(3), 837-845.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeLongResult:
    """Result of a DeLong test comparing two AUC-ROC values."""
    auc_1: float
    auc_2: float
    auc_diff: float
    z_statistic: float
    p_value: float
    ci_lower: float          # 95% CI on AUC difference
    ci_upper: float
    significant: bool        # at alpha = 0.05

    def summary(self) -> str:
        sig_str = "YES" if self.significant else "NO"
        return (
            f"DeLong Test: AUC1={self.auc_1:.4f} vs AUC2={self.auc_2:.4f}\n"
            f"  Difference:   {self.auc_diff:+.4f}\n"
            f"  95% CI:       [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]\n"
            f"  Z-statistic:  {self.z_statistic:.4f}\n"
            f"  P-value:      {self.p_value:.6f}\n"
            f"  Significant (alpha=0.05): {sig_str}"
        )


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for the DeLong statistic."""
    n = len(x)
    idx = np.argsort(x)
    sorted_x = x[idx]

    midranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # Assign average rank to all tied values
        avg_rank = 0.5 * (i + j - 1)
        for k in range(i, j):
            midranks[idx[k]] = avg_rank
        i = j

    return midranks


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong computation.

    Based on the algorithm from:
        Sun & Xu (2014). "Fast Implementation of DeLong's Algorithm
        for Comparing the Areas Under Correlated Receiver Operating
        Characteristic Curves." IEEE Signal Processing Letters.

    Parameters
    ----------
    predictions_sorted_transposed : ndarray, shape (n_models, n_samples)
        Predictions sorted by label (positive first).
    label_1_count : int
        Number of positive samples.

    Returns
    -------
    aucs : ndarray, shape (n_models,)
    delongcov : ndarray, shape (n_models, n_models)
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])

    for r in range(k):
        tz[r] = _compute_midrank(predictions_sorted_transposed[r, :])
        tx[r] = _compute_midrank(positive_examples[r, :])
        ty[r] = _compute_midrank(negative_examples[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delongcov = sx / m + sy / n

    return aucs, delongcov


def delong_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    alpha: float = 0.05,
) -> DeLongResult:
    """
    Compare two AUC-ROC values using the DeLong test.

    Tests the null hypothesis that two models have equal AUC,
    accounting for the correlation between predictions made
    on the same test set.

    Parameters
    ----------
    y_true : ndarray
        True binary labels (0/1).
    y_pred_1 : ndarray
        Predicted probabilities from model 1.
    y_pred_2 : ndarray
        Predicted probabilities from model 2.
    alpha : float
        Significance level for the test.

    Returns
    -------
    DeLongResult
        Test results including z-statistic, p-value, and CI.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_1 = np.asarray(y_pred_1, dtype=np.float64)
    y_pred_2 = np.asarray(y_pred_2, dtype=np.float64)

    # Sort by label (positive first)
    order = (-y_true).argsort()
    y_true_sorted = y_true[order]
    label_1_count = int(y_true_sorted.sum())

    predictions_sorted = np.vstack([y_pred_1[order], y_pred_2[order]])

    aucs, delongcov = _fast_delong(predictions_sorted, label_1_count)

    auc_1, auc_2 = float(aucs[0]), float(aucs[1])
    auc_diff = auc_1 - auc_2

    # Variance of the difference
    # Var(AUC1 - AUC2) = Var(AUC1) + Var(AUC2) - 2*Cov(AUC1, AUC2)
    if np.isscalar(delongcov):
        # Only one model pair
        var_diff = 2 * float(delongcov) - 2 * float(delongcov)
        # Fallback: use individual variances
        var_diff = max(float(delongcov) * 2, 1e-10)
    else:
        var_diff = float(
            delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
        )

    se = np.sqrt(max(var_diff, 1e-10))

    z = auc_diff / se
    p_value = 2 * stats.norm.sf(abs(z))  # two-sided

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = auc_diff - z_crit * se
    ci_upper = auc_diff + z_crit * se

    result = DeLongResult(
        auc_1=round(auc_1, 4),
        auc_2=round(auc_2, 4),
        auc_diff=round(auc_diff, 4),
        z_statistic=round(z, 4),
        p_value=round(p_value, 6),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        significant=p_value < alpha,
    )

    logger.info(result.summary())
    return result
