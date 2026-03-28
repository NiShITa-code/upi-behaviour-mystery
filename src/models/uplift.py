"""
Causal Uplift Model — T-Learner approach.

Why T-Learner?
  A churn model tells you P(churn). An uplift model tells you
  P(retain | offer) - P(retain | no offer) for each individual.
  These are different — and acting on churn scores alone wastes
  budget on users who'd stay anyway.

T-Learner:
  Fit M1 on treatment group → predicts P(retain | offer)
  Fit M0 on control group   → predicts P(retain | no offer)
  ITE = M1(x) - M0(x)       → individual treatment effect

Four segments from the 2×2 matrix:
  P0 high, P1 high  → Sure Thing    (already retained, offer unnecessary)
  P0 low,  P1 high  → Persuadable   (target these — intervention works)
  P0 low,  P1 low   → Lost Cause    (won't retain regardless)
  P0 high, P1 low   → Sleeping Dog  (rare, intervention may hurt)

Production note:
  In a real deployment, treatment/control assignment comes from
  an actual A/B test. Here we simulate a past experiment to
  estimate the uplift model. The inference pipeline (scoring new users)
  uses only the fitted M0 and M1 — no simulation required at serving time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import CFG
from src.features.engineer import EXTENDED_FEATURES, EarlyWindowFeatures, FeatureSelector
from src.models.churn import ChurnModelResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Threshold for high/low P0 and P1 classification
SEGMENT_THRESHOLD = CFG.model.uplift.classification_threshold


@dataclass
class ROIScenario:
    """Budget allocation comparison: random vs targeted."""
    budget: int
    cashback_per_offer: int
    n_offers: int
    # Random targeting
    n_random_offers_to_persuadables: int
    users_retained_random: int
    # Targeted (Persuadables only)
    n_targeted_offers: int
    users_retained_targeted: int
    efficiency_gain: float  # targeted / random

    def summary(self) -> str:
        return (
            f"Budget ₹{self.budget:,} → {self.n_offers:,} offers at ₹{self.cashback_per_offer}\n"
            f"  Random targeting:  {self.users_retained_random:,} retained\n"
            f"  Targeted (P only): {self.users_retained_targeted:,} retained\n"
            f"  Efficiency gain:   {self.efficiency_gain:.1f}×"
        )


@dataclass
class UpliftResult:
    """Full output of run_uplift_model()."""
    users_segmented: pd.DataFrame   # users + p0, p1, uplift, segment columns
    segment_counts: Dict[str, int]
    segment_pcts: Dict[str, float]
    roi: ROIScenario
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            "─" * 50,
            "UPLIFT MODEL RESULTS",
            "─" * 50,
            "  Segment breakdown:",
        ]
        for seg in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]:
            n   = self.segment_counts.get(seg, 0)
            pct = self.segment_pcts.get(seg, 0.0)
            lines.append(f"    {seg:<15} {n:>6,}  ({pct:.1f}%)")
        lines.append("")
        lines.append(self.roi.summary())
        lines.append("─" * 50)
        return "\n".join(lines)


def run_uplift_model(
    churn_result: ChurnModelResult,
    cashback_amount: Optional[int] = None,
    total_budget: Optional[int] = None,
    response_boost: Optional[float] = None,
    seed: int = 99,
) -> UpliftResult:
    """
    Fit T-Learner uplift model and compute ROI scenarios.

    Parameters
    ----------
    churn_result : ChurnModelResult
        Output of train_churn_model(). Uses scored users DataFrame.
    cashback_amount : int, optional
        Cost per intervention (₹). Defaults to config.
    total_budget : int, optional
        Total budget available (₹). Defaults to config.
    response_boost : float, optional
        Fraction lift in retention probability for Persuadables.
        Calibrated from historical A/B test data (or config default).
    seed : int
        Random seed for simulating the experiment assignment.

    Returns
    -------
    UpliftResult
    """
    cashback = cashback_amount or CFG.intervention.default_cashback
    budget   = total_budget    or CFG.intervention.default_budget
    boost    = response_boost  or CFG.intervention.response_rates

    logger.info("Fitting T-Learner uplift model...")
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)
    df  = churn_result.users_scored.copy()

    # ── Simulate randomised experiment assignment ─────────────────
    # In production: this is replaced by actual A/B test log data
    treatment_frac = CFG.model.uplift.treatment_fraction
    df["treatment"] = rng.binomial(1, treatment_frac, len(df))

    # ── Simulate observed outcomes ────────────────────────────────
    # Each user's outcome = f(churn_prob, treatment, archetype)
    df["retained"] = df.apply(
        lambda row: _simulate_outcome(row, boost, rng), axis=1
    )

    # ── Prepare features ─────────────────────────────────────────
    feature_cols = [c for c in df.columns if c in EXTENDED_FEATURES
                    or c in CFG.features.all_features]
    # Fall back to numeric columns that exist
    numeric_cols = [c for c in CFG.features.numeric if c in df.columns]

    treatment_df = df[df["treatment"] == 1]
    control_df   = df[df["treatment"] == 0]

    logger.info(
        "Treatment: %d users | Control: %d users",
        len(treatment_df), len(control_df)
    )

    # ── Fit M1 (treatment model) and M0 (control model) ──────────
    m1 = _fit_base_model(treatment_df[numeric_cols], treatment_df["retained"])
    m0 = _fit_base_model(control_df[numeric_cols],   control_df["retained"])

    # ── Score all users with both models ─────────────────────────
    X_all = df[numeric_cols].fillna(0)
    df["p1"]     = m1.predict_proba(X_all)[:, 1]   # P(retain | offer)
    df["p0"]     = m0.predict_proba(X_all)[:, 1]   # P(retain | no offer)
    df["uplift"] = (df["p1"] - df["p0"]).round(4)  # ITE

    # ── Classify into segments ────────────────────────────────────
    df["segment"] = df.apply(_classify_segment, axis=1)

    seg_counts = df["segment"].value_counts().to_dict()
    total      = len(df)
    seg_pcts   = {k: round(v / total * 100, 1) for k, v in seg_counts.items()}

    # ── ROI analysis ──────────────────────────────────────────────
    roi = _compute_roi(df, seg_counts, cashback, budget)

    elapsed = time.perf_counter() - t0
    result = UpliftResult(
        users_segmented=df,
        segment_counts=seg_counts,
        segment_pcts=seg_pcts,
        roi=roi,
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result


def _fit_base_model(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Fit one arm of the T-Learner."""
    clf = lgb.LGBMClassifier(
        n_estimators=CFG.model.uplift.n_estimators,
        learning_rate=0.05,
        num_leaves=20,
        random_state=CFG.simulation.seed,
        verbose=-1,
        class_weight="balanced",
    )
    clf.fit(X.fillna(0), y)
    return clf


def _simulate_outcome(
    row: pd.Series,
    boost: dict | float,
    rng: np.random.Generator,
) -> int:
    """
    Simulate whether a user was retained given treatment assignment.
    In production this comes from actual observed outcomes.
    """
    base_retention = 1.0 - row["churn_prob"]

    if row["treatment"] == 1:
        if isinstance(boost, dict):
            lift = boost.get(row.get("archetype", "Regular"), 0.10)
        else:
            lift = float(boost)
        p = min(1.0, base_retention + lift)
    else:
        p = base_retention

    # Add small individual-level noise
    p = float(np.clip(p + rng.normal(0, 0.04), 0, 1))
    return int(p > 0.5)


def _classify_segment(row: pd.Series) -> str:
    """
    Classify user into the four uplift quadrants.

        P0 ↓ high | P1 ↑ high  →  Sure Thing
        P0 ↓ low  | P1 ↑ high  →  Persuadable  ← target
        P0 ↓ low  | P1 ↑ low   →  Lost Cause
        P0 ↓ high | P1 ↑ low   →  Sleeping Dog
    """
    thr = SEGMENT_THRESHOLD
    high_p1 = row["p1"] > thr
    high_p0 = row["p0"] > thr

    if high_p1 and not high_p0:  return "Persuadable"
    if high_p1 and high_p0:      return "Sure Thing"
    if not high_p1 and high_p0:  return "Sleeping Dog"
    return "Lost Cause"


def _compute_roi(
    df: pd.DataFrame,
    seg_counts: Dict[str, int],
    cashback: int,
    budget: int,
) -> ROIScenario:
    """
    Compare retention outcomes: random targeting vs targeted (Persuadables only).
    """
    n_offers       = budget // cashback
    n_persuadable  = seg_counts.get("Persuadable", 0)
    total          = len(df)
    persuadable_rt = 0.67  # estimated response rate for Persuadables

    # Random: offers distributed uniformly across at-risk users
    frac_persuadable   = n_persuadable / max(total, 1)
    n_random_to_p      = int(n_offers * frac_persuadable)
    retained_random    = int(n_random_to_p * persuadable_rt)

    # Targeted: all offers go to identified Persuadables
    n_targeted        = min(n_offers, n_persuadable)
    retained_targeted = int(n_targeted * persuadable_rt)

    efficiency = retained_targeted / max(retained_random, 1)

    return ROIScenario(
        budget=budget,
        cashback_per_offer=cashback,
        n_offers=n_offers,
        n_random_offers_to_persuadables=n_random_to_p,
        users_retained_random=retained_random,
        n_targeted_offers=n_targeted,
        users_retained_targeted=retained_targeted,
        efficiency_gain=round(efficiency, 1),
    )
