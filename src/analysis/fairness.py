"""
Model fairness audit.

Checks whether the churn model performs equitably across
protected and demographic groups. This matters because:
  - City tier is a proxy for economic status
  - Age group correlates with digital literacy
  - A model that's accurate overall but biased against
    Tier-3 users could lead to discriminatory interventions

Metrics:
  - Demographic parity: P(predicted churn | group) should be similar
  - Equalised opportunity: TPR should be similar across groups
  - Predictive parity: precision should be similar across groups
  - Calibration: predicted probabilities should match actuals per group
  - Disparate impact ratio: min(rate_group / rate_overall)

Reference: Fairness Definitions Explained (Verma & Rubin, 2018)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Thresholds for flagging potential fairness issues
DISPARATE_IMPACT_THRESHOLD = 0.80   # 80% rule (EEOC guideline)
TPR_DISPARITY_THRESHOLD = 0.10      # max acceptable TPR gap


@dataclass
class GroupMetrics:
    """Fairness metrics for a single group."""
    group_name: str
    group_size: int
    churn_rate_actual: float        # actual churn rate in group
    churn_rate_predicted: float     # predicted churn rate (P(pred=1))
    true_positive_rate: float       # TPR = TP / (TP + FN)
    false_positive_rate: float      # FPR = FP / (FP + TN)
    precision: float                # TP / (TP + FP)
    calibration_error: float        # |mean(pred_prob) - actual_rate|
    auc_roc: float                  # group-specific AUC


@dataclass
class FairnessAttribute:
    """Fairness analysis for one attribute (e.g., city_tier)."""
    attribute_name: str
    groups: Dict[str, GroupMetrics]
    disparate_impact_ratio: float   # min(group_rate / overall_rate)
    max_tpr_gap: float              # max TPR difference between groups
    max_fpr_gap: float
    max_calibration_gap: float
    flagged: bool                   # True if any threshold violated
    flags: List[str]                # specific issues found


@dataclass
class FairnessAudit:
    """Complete fairness audit across all attributes."""
    attributes: Dict[str, FairnessAttribute]
    overall_churn_rate: float
    overall_auc: float
    threshold: float                # churn probability threshold used
    total_flags: int
    passed: bool                    # True if no major issues
    summary_text: str

    def summary(self) -> str:
        return self.summary_text


def _group_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    y_pred: np.ndarray,
    group_name: str,
) -> GroupMetrics:
    """Compute fairness metrics for a single group."""
    from sklearn.metrics import roc_auc_score

    n = len(y_true)
    if n == 0:
        return GroupMetrics(group_name, 0, 0, 0, 0, 0, 0, 0, 0)

    actual_rate = y_true.mean()
    pred_rate = y_pred.mean()

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    calibration_error = abs(float(y_pred_prob.mean()) - float(actual_rate))

    # AUC (needs both classes present)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5  # only one class present

    return GroupMetrics(
        group_name=group_name,
        group_size=n,
        churn_rate_actual=float(actual_rate),
        churn_rate_predicted=float(pred_rate),
        true_positive_rate=float(tpr),
        false_positive_rate=float(fpr),
        precision=float(precision),
        calibration_error=float(calibration_error),
        auc_roc=float(auc),
    )


def _audit_attribute(
    df: pd.DataFrame,
    attribute: str,
    overall_pred_rate: float,
    threshold: float = 0.5,
) -> FairnessAttribute:
    """Run fairness audit on one attribute."""
    y_true = df["churned"].values
    y_prob = df["churn_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    groups = {}
    for group_val in sorted(df[attribute].unique()):
        mask = df[attribute].values == group_val
        gm = _group_metrics(
            y_true[mask], y_prob[mask], y_pred[mask],
            group_name=str(group_val)
        )
        groups[str(group_val)] = gm

    # Disparate impact: min(group_positive_rate / overall_rate)
    group_rates = [g.churn_rate_predicted for g in groups.values() if g.group_size > 0]
    if overall_pred_rate > 0 and group_rates:
        di_ratio = min(r / overall_pred_rate for r in group_rates)
    else:
        di_ratio = 1.0

    # TPR / FPR gaps
    tprs = [g.true_positive_rate for g in groups.values() if g.group_size >= 10]
    fprs = [g.false_positive_rate for g in groups.values() if g.group_size >= 10]
    cals = [g.calibration_error for g in groups.values() if g.group_size >= 10]

    max_tpr_gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    max_fpr_gap = (max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
    max_cal_gap = max(cals) if cals else 0.0

    # Flag issues
    flags = []
    if di_ratio < DISPARATE_IMPACT_THRESHOLD:
        flags.append(
            f"Disparate impact ratio {di_ratio:.2f} < {DISPARATE_IMPACT_THRESHOLD} "
            f"(80% rule violated for {attribute})"
        )
    if max_tpr_gap > TPR_DISPARITY_THRESHOLD:
        flags.append(
            f"TPR gap of {max_tpr_gap:.2f} across {attribute} groups "
            f"(>{TPR_DISPARITY_THRESHOLD} threshold)"
        )
    if max_cal_gap > 0.10:
        flags.append(
            f"Calibration gap of {max_cal_gap:.3f} — model may be miscalibrated "
            f"for some {attribute} groups"
        )

    return FairnessAttribute(
        attribute_name=attribute,
        groups=groups,
        disparate_impact_ratio=round(di_ratio, 4),
        max_tpr_gap=round(max_tpr_gap, 4),
        max_fpr_gap=round(max_fpr_gap, 4),
        max_calibration_gap=round(max_cal_gap, 4),
        flagged=len(flags) > 0,
        flags=flags,
    )


def run_fairness_audit(
    users_scored: pd.DataFrame,
    threshold: float = 0.5,
    attributes: Optional[List[str]] = None,
) -> FairnessAudit:
    """
    Run a comprehensive fairness audit on the churn model.

    Checks model performance parity across demographic groups.

    Parameters
    ----------
    users_scored : pd.DataFrame
        Users with 'churned' (actual), 'churn_prob' (predicted),
        and demographic columns.
    threshold : float
        Probability threshold for binary prediction.
    attributes : list[str], optional
        Columns to audit. Defaults to ['city_tier', 'age_group'].

    Returns
    -------
    FairnessAudit
    """
    logger.info("Running fairness audit...")

    if attributes is None:
        # Auto-detect available attributes
        attributes = []
        for col in ["city_tier", "age_group", "archetype"]:
            if col in users_scored.columns:
                attributes.append(col)

    if not attributes:
        logger.warning("No demographic attributes found for fairness audit")

    from sklearn.metrics import roc_auc_score
    overall_rate = float((users_scored["churn_prob"] >= threshold).mean())
    overall_auc = roc_auc_score(users_scored["churned"], users_scored["churn_prob"])

    audit_results = {}
    total_flags = 0
    for attr in attributes:
        if attr not in users_scored.columns:
            logger.warning("Attribute %s not found, skipping", attr)
            continue
        result = _audit_attribute(users_scored, attr, overall_rate, threshold)
        audit_results[attr] = result
        total_flags += len(result.flags)

    # Build summary
    lines = [
        "═" * 60,
        "MODEL FAIRNESS AUDIT",
        "═" * 60,
        f"  Overall churn rate:  {users_scored['churned'].mean():.1%}",
        f"  Overall AUC:         {overall_auc:.4f}",
        f"  Threshold:           {threshold}",
        "",
    ]

    for attr, result in audit_results.items():
        status = "⚠ FLAGGED" if result.flagged else "✓ PASSED"
        lines.append(f"  ── {attr.upper()} {status} ──")
        lines.append(f"  Disparate impact ratio:  {result.disparate_impact_ratio:.4f}")
        lines.append(f"  Max TPR gap:             {result.max_tpr_gap:.4f}")
        lines.append(f"  Max FPR gap:             {result.max_fpr_gap:.4f}")
        lines.append(f"  Max calibration gap:     {result.max_calibration_gap:.4f}")
        lines.append("")
        lines.append(f"  {'Group':<15} {'N':>6} {'Actual':>8} {'Pred':>8} {'TPR':>6} {'AUC':>6}")
        lines.append(f"  {'─'*15} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
        for name, gm in result.groups.items():
            lines.append(
                f"  {name:<15} {gm.group_size:>6} {gm.churn_rate_actual:>8.1%} "
                f"{gm.churn_rate_predicted:>8.1%} {gm.true_positive_rate:>6.2f} "
                f"{gm.auc_roc:>6.3f}"
            )
        lines.append("")
        if result.flags:
            for flag in result.flags:
                lines.append(f"  ⚠ {flag}")
            lines.append("")

    passed = total_flags == 0
    verdict = "PASSED — no major fairness concerns detected" if passed else \
              f"FLAGGED — {total_flags} potential fairness issue(s) found"
    lines.append("─" * 60)
    lines.append(f"  VERDICT: {verdict}")
    lines.append("─" * 60)

    summary_text = "\n".join(lines)
    logger.info(summary_text)

    retu