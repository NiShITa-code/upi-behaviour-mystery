"""
Experiment design and statistical power analysis.

Answers the question: "If we run an A/B test on this intervention,
how many users do we need per arm, and how long will it take?"

This is the bridge between exploratory analysis and production
experimentation — demonstrates the full DS lifecycle:
  Explore → Model → Design experiment → Run → Evaluate

Design choices:
  - Two-proportion z-test for binary outcomes (retained / churned)
  - Supports minimum detectable effect (MDE) framing
  - Duration estimate from daily traffic
  - Multiple testing correction (Bonferroni) for subgroup analyses
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PowerAnalysisResult:
    """Output of a power analysis calculation."""
    baseline_rate: float            # P(retain | control)
    minimum_detectable_effect: float  # absolute lift to detect
    target_rate: float              # baseline + MDE
    alpha: float                    # significance level
    power: float                    # 1 - beta
    sample_per_arm: int             # required N per group
    total_sample: int               # 2 * sample_per_arm
    daily_eligible_users: int       # users/day entering experiment
    estimated_days: int             # calendar days to reach sample
    estimated_weeks: float

    def summary(self) -> str:
        lines = [
            "─" * 55,
            "EXPERIMENT DESIGN — POWER ANALYSIS",
            "─" * 55,
            f"  Baseline retention rate:    {self.baseline_rate:.1%}",
            f"  Minimum detectable effect:  {self.minimum_detectable_effect:+.1%}pp",
            f"  Target rate (if effective): {self.target_rate:.1%}",
            "",
            f"  Significance level (α):     {self.alpha}",
            f"  Statistical power (1-β):    {self.power}",
            "",
            f"  Sample size per arm:        {self.sample_per_arm:,}",
            f"  Total sample needed:        {self.total_sample:,}",
            "",
            f"  Daily eligible users:       {self.daily_eligible_users:,}",
            f"  Estimated duration:          {self.estimated_days} days ({self.estimated_weeks:.1f} weeks)",
            "─" * 55,
        ]
        return "\n".join(lines)


@dataclass
class ExperimentDesign:
    """Complete experiment design document."""
    primary: PowerAnalysisResult
    subgroup_analyses: Dict[str, PowerAnalysisResult]   # segment → result
    bonferroni_alpha: float         # corrected alpha for subgroups
    guardrail_metrics: List[str]
    recommendations: List[str]
    risks: List[str]

    def summary(self) -> str:
        lines = [self.primary.summary()]
        lines.append("")
        lines.append("SUBGROUP POWER (Bonferroni-corrected α={:.4f}):".format(
            self.bonferroni_alpha))
        for name, result in self.subgroup_analyses.items():
            lines.append(f"  {name:<25} n={result.sample_per_arm:,}/arm  "
                        f"({result.estimated_days} days)")
        lines.append("")
        lines.append("GUARDRAIL METRICS:")
        for m in self.guardrail_metrics:
            lines.append(f"  - {m}")
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        for r in self.recommendations:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def compute_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Required sample size per arm for a two-proportion z-test.

    Uses the normal approximation:
      n = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)^2

    Parameters
    ----------
    baseline_rate : float
        Expected retention rate in control (P(retained | control)).
    mde : float
        Minimum detectable effect (absolute, e.g. 0.05 = 5pp lift).
    alpha : float
        Significance level (two-sided).
    power : float
        Statistical power (1 - Type II error rate).

    Returns
    -------
    int
        Required sample size per arm (rounded up).
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde

    # Clamp to valid probability range
    p2 = max(0.001, min(0.999, p2))

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p2 - p1) ** 2

    if denominator == 0:
        return 999999  # degenerate case

    n = math.ceil(numerator / denominator)
    return max(n, 10)  # floor at 10


def compute_mde_from_sample(
    baseline_rate: float,
    sample_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Given a fixed sample size, compute the minimum detectable effect.

    Useful when you have a fixed user pool and want to know
    what effect size you can reliably detect.

    Uses binary search on the sample size formula.
    """
    lo, hi = 0.001, 0.5
    for _ in range(50):  # binary search iterations
        mid = (lo + hi) / 2
        n = compute_sample_size(baseline_rate, mid, alpha, power)
        if n <= sample_per_arm:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 4)


def design_experiment(
    users_scored,
    uplift_result,
    daily_new_users: int = 500,
    alpha: float = 0.05,
    power: float = 0.80,
    mde: float = 0.05,
) -> ExperimentDesign:
    """
    Generate a complete experiment design for the cashback intervention.

    Parameters
    ----------
    users_scored : pd.DataFrame
        User table with churn_prob and segment columns.
    uplift_result
        UpliftResult with segment counts.
    daily_new_users : int
        Expected daily user registrations (for duration estimate).
    alpha : float
        Significance level.
    power : float
        Target statistical power.
    mde : float
        Minimum detectable effect (absolute retention lift).

    Returns
    -------
    ExperimentDesign
    """
    logger.info("Designing experiment (α=%.2f, power=%.2f, MDE=%.1f%%)",
                alpha, power, mde * 100)

    # ── Primary analysis ─────────────────────────────────────
    baseline_retention = 1 - users_scored["churned"].mean()
    primary_n = compute_sample_size(baseline_retention, mde, alpha, power)

    primary = PowerAnalysisResult(
        baseline_rate=baseline_retention,
        minimum_detectable_effect=mde,
        target_rate=baseline_retention + mde,
        alpha=alpha,
        power=power,
        sample_per_arm=primary_n,
        total_sample=primary_n * 2,
        daily_eligible_users=daily_new_users,
        estimated_days=math.ceil((primary_n * 2) / daily_new_users),
        estimated_weeks=round((primary_n * 2) / daily_new_users / 7, 1),
    )

    # ── Subgroup analyses ─────────────────────────────────────
    # Bonferroni correction for multiple comparisons
    subgroups = {}
    segments_to_test = ["Persuadable", "Sure Thing", "Lost Cause"]
    n_tests = len(segments_to_test)
    bonf_alpha = alpha / n_tests

    for segment_name in segments_to_test:
        if "segment" in users_scored.columns:
            seg_mask = users_scored["segment"] == segment_name
            seg_users = users_scored[seg_mask]
        else:
            seg_users = users_scored

        if len(seg_users) < 10:
            continue

        seg_retention = 1 - seg_users["churned"].mean()
        seg_n = compute_sample_size(seg_retention, mde, bonf_alpha, power)

        # Daily eligible = proportion of all users in this segment
        seg_fraction = len(seg_users) / len(users_scored)
        seg_daily = max(1, int(daily_new_users * seg_fraction))

        subgroups[segment_name] = PowerAnalysisResult(
            baseline_rate=seg_retention,
            minimum_detectable_effect=mde,
            target_rate=seg_retention + mde,
            alpha=bonf_alpha,
            power=power,
            sample_per_arm=seg_n,
            total_sample=seg_n * 2,
            daily_eligible_users=seg_daily,
            estimated_days=math.ceil((seg_n * 2) / seg_daily),
            estimated_weeks=round((seg_n * 2) / seg_daily / 7, 1),
        )

    # ── Guardrail metrics ─────────────────────────────────────
    guardrails = [
        "Average transaction value (must not drop — ensure cashback doesn't devalue organic txns)",
        "P2M ratio (merchant payments should not decrease)",
        "Category diversity (intervention shouldn't narrow usage patterns)",
        "Customer support ticket rate (watch for confusion/complaints)",
        "Cost per retained user (total cashback spend / incremental retentions)",
    ]

    # ── Recommendations ───────────────────────────────────────
    recs = []
    recs.append(
        f"Run a 50/50 randomised experiment with {primary_n:,} users per arm "
        f"(~{primary.estimated_weeks:.0f} weeks at {daily_new_users:,} users/day)."
    )

    if "Persuadable" in subgroups:
        p = subgroups["Persuadable"]
        recs.append(
            f"Pre-register subgroup analysis on Persuadables (need {p.sample_per_arm:,}/arm, "
            f"~{p.estimated_weeks:.0f} weeks). This is where ROI is highest."
        )

    recs.append(
        "Use Bonferroni correction (α={:.4f}) for {} pre-registered subgroup tests.".format(
            bonf_alpha, n_tests)
    )
    recs.append(
        "Monitor guardrail metrics weekly. Stop early if any guardrail "
        "shows >2σ degradation."
    )
    recs.append(
        "Consider sequential testing (group sequential design) to enable "
        "valid early stopping for both efficacy and futility."
    )

    # ── Risks ─────────────────────────────────────────────────
    risks = [
        "Network effects: if cashback users recruit friends, SUTVA may be violated",
        "Novelty effect: short-term lift that fades as users habituate to cashback",
        "Self-selection: users who opt-in to cashback offers differ from random population",
        "Seasonality: festival periods (Diwali, Holi) can inflate baseline retention",
    ]

    design = ExperimentDesign(
        primary=primary,
        subgroup_analyses=subgroups,
        bonferroni_alpha=bonf_alpha,
        guardrail_metrics=guardrails,
        recommendations=recs,
        risks=risks,
    )
    logger.info(design.summary())
    return design
