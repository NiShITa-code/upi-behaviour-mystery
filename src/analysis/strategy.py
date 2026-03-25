"""
Strategy Recommender — plain-English, actionable recommendations.

Takes the output of the full pipeline and generates recommendations
that a product manager can act on immediately. No jargon, no AUC
numbers — just "do this, expect this result."

This is the deliverable a DS hands to a PM after an analysis sprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.models.churn import ChurnModelResult
from src.models.uplift import UpliftResult
from src.analysis.cohorts import CohortResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """A single actionable recommendation."""
    priority: int               # 1 = highest
    title: str
    what: str                   # what to do
    why: str                    # data-driven reason
    expected_impact: str        # quantified expected outcome
    confidence: str             # High / Medium / Low
    effort: str                 # Low / Medium / High


@dataclass
class StrategyReport:
    """Complete strategy output from the recommender."""
    headline: str
    summary: str
    recommendations: List[Recommendation]
    risk_factors: List[str]
    key_metrics: dict

    def to_plain_text(self) -> str:
        """Export as a plain-text memo."""
        lines = [
            "=" * 60,
            "UPI RETENTION STRATEGY — RECOMMENDATIONS",
            "=" * 60,
            "",
            self.headline,
            "",
            self.summary,
            "",
            "-" * 60,
            "RECOMMENDATIONS (by priority)",
            "-" * 60,
        ]
        for rec in self.recommendations:
            lines.extend([
                "",
                f"  [{rec.priority}] {rec.title}",
                f"      What:     {rec.what}",
                f"      Why:      {rec.why}",
                f"      Impact:   {rec.expected_impact}",
                f"      Confidence: {rec.confidence} | Effort: {rec.effort}",
            ])
        lines.extend([
            "",
            "-" * 60,
            "RISK FACTORS",
            "-" * 60,
        ])
        for risk in self.risk_factors:
            lines.append(f"  - {risk}")

        lines.extend([
            "",
            "-" * 60,
            "KEY METRICS TO TRACK",
            "-" * 60,
        ])
        for k, v in self.key_metrics.items():
            lines.append(f"  {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)


def generate_strategy(
    churn_result: ChurnModelResult,
    uplift_result: UpliftResult,
    cohort_result: CohortResult,
    cashback_amount: int,
    total_budget: int,
) -> StrategyReport:
    """
    Generate a complete strategy report from pipeline outputs.

    Returns plain-English recommendations with quantified impact
    estimates. Designed to be handed directly to a PM.
    """
    users = uplift_result.users_segmented
    n_users = len(users)
    churn_rate = users["churned"].mean()

    # ── Extract key signals ──────────────────────────────────────
    d14 = cohort_result.day14_summary
    ret_txn = d14[d14["status"] == "Retained"]["median_txn_d14"].values[0]
    chu_txn = d14[d14["status"] == "Churned"]["median_txn_d14"].values[0]
    ret_p2m = d14[d14["status"] == "Retained"]["pct_with_p2m_d14"].values[0]

    # Feature importance
    fi = churn_result.lgb_metrics.feature_importance
    top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]

    # Uplift segments
    sc = uplift_result.segment_counts
    sp = uplift_result.segment_pcts
    n_persuadable = sc.get("Persuadable", 0)
    pct_persuadable = sp.get("Persuadable", 0.0)
    roi = uplift_result.roi

    # ── Compute segment profiles ─────────────────────────────────
    persuadables = users[users["segment"] == "Persuadable"]
    lost_causes = users[users["segment"] == "Lost Cause"]

    # What makes persuadables different?
    p_txn_median = persuadables["txn_d14"].median() if len(persuadables) > 0 else 0
    p_p2m = persuadables["has_first_p2m_d14"].mean() * 100 if len(persuadables) > 0 else 0

    # City tier concentration
    if len(persuadables) > 0:
        tier_dist = persuadables["city_tier"].value_counts(normalize=True)
        top_tier = tier_dist.idxmax()
        top_tier_pct = tier_dist.max() * 100
        tier_labels = {1: "metro", 2: "tier-2", 3: "tier-3"}
        tier_insight = f"{top_tier_pct:.0f}% of Persuadables are in {tier_labels.get(top_tier, f'tier-{top_tier}')} cities"
    else:
        tier_insight = "No Persuadables identified"

    # ── Build recommendations ────────────────────────────────────
    recommendations = []

    # Rec 1: Targeted cashback
    recommendations.append(Recommendation(
        priority=1,
        title="Deploy targeted cashback to Persuadables only",
        what=(
            f"Send ₹{cashback_amount} cashback offers exclusively to the "
            f"{n_persuadable:,} identified Persuadable users ({pct_persuadable:.1f}% of base). "
            f"Do NOT spray offers to the entire user base."
        ),
        why=(
            f"Random targeting retains {roi.users_retained_random:,} users. "
            f"Targeted retains {roi.users_retained_targeted:,} users with the same "
            f"₹{total_budget:,} budget — {roi.efficiency_gain:.1f}x more efficient. "
            f"Sure Things ({sp.get('Sure Thing', 0):.1f}%) stay anyway. "
            f"Lost Causes ({sp.get('Lost Cause', 0):.1f}%) won't respond regardless."
        ),
        expected_impact=(
            f"Retain ~{roi.users_retained_targeted:,} additional users at "
            f"₹{cashback_amount * roi.n_targeted_offers:,} spend "
            f"(₹{cashback_amount * roi.n_targeted_offers // max(roi.users_retained_targeted, 1):,} per retained user)"
        ),
        confidence="High" if churn_result.cv_auc_mean > 0.80 else "Medium",
        effort="Low — model already identifies targets",
    ))

    # Rec 2: Day-14 onboarding intervention
    activation_threshold = max(int(ret_txn * 0.6), 3)
    recommendations.append(Recommendation(
        priority=2,
        title="Add a merchant payment prompt at Day 7",
        what=(
            f"In the onboarding flow, at Day 7, show users who haven't made "
            f"a merchant payment a nudge: 'Pay at any store with UPI — get ₹10 back.' "
            f"Target users with fewer than {activation_threshold} transactions at Day 7."
        ),
        why=(
            f"Retained users had {ret_txn:.0f} median transactions in first 14 days vs "
            f"{chu_txn:.0f} for churned. {ret_p2m:.0f}% of retained users made their "
            f"first merchant payment within 14 days. This is the highest-signal "
            f"behavioural gate we found ('{top_features[0][0]}' = {top_features[0][1]:.1f}% "
            f"feature importance)."
        ),
        expected_impact=(
            f"If this moves 20% of at-risk Day-7 users to make a merchant payment, "
            f"expected churn reduction: 5-12 percentage points in the next cohort"
        ),
        confidence="Medium — causal effect estimated from observational data",
        effort="Medium — requires onboarding flow change",
    ))

    # Rec 3: Tier-specific strategy
    city_summary = cohort_result.city_tier_summary
    if len(city_summary) > 0:
        worst_tier = city_summary.loc[city_summary["churn_rate"].idxmax()]
        best_tier = city_summary.loc[city_summary["churn_rate"].idxmin()]
        recommendations.append(Recommendation(
            priority=3,
            title=f"Focus acquisition quality in {worst_tier['city_tier']} cities",
            what=(
                f"Review acquisition channels for {worst_tier['city_tier']} users. "
                f"They churn at {worst_tier['churn_rate']*100:.1f}% vs "
                f"{best_tier['churn_rate']*100:.1f}% for {best_tier['city_tier']}. "
                f"Consider gating cashback-only signups or adding a lightweight "
                f"activation requirement before counting a user as 'acquired'."
            ),
            why=(
                f"High churn in specific tiers suggests users are signing up for "
                f"promotional offers and never developing a payment habit. The cost "
                f"of acquiring + churning a user is wasted."
            ),
            expected_impact=(
                f"Reducing {worst_tier['city_tier']} churn by 10pp would retain "
                f"~{int(worst_tier['n_users'] * 0.10):,} additional users from this "
                f"cohort alone"
            ),
            confidence="Medium",
            effort="High — involves acquisition team coordination",
        ))

    # Rec 4: Monthly model refresh
    recommendations.append(Recommendation(
        priority=4,
        title="Rerun the uplift model monthly on fresh data",
        what=(
            f"The user mix shifts as acquisition campaigns change. Retrain "
            f"the churn + uplift models monthly and re-segment. Automate "
            f"the segment export to the CRM."
        ),
        why=(
            f"Model was trained on current cohort distribution. If acquisition "
            f"shifts (e.g., festive campaign brings more Dormant-profile users), "
            f"the Persuadable segment will shift too."
        ),
        expected_impact="Maintains targeting accuracy over time",
        confidence="High",
        effort="Low — pipeline is already automated (Click CLI + artifacts)",
    ))

    # ── Risk factors ─────────────────────────────────────────────
    risk_factors = [
        f"Model AUC ({churn_result.lgb_metrics.auc_roc:.3f}) may degrade as user mix shifts — monitor monthly.",
        f"Persuadable segment ({pct_persuadable:.1f}%) is small — if targeting is imprecise, budget leaks to Sure Things.",
        "Treatment effect estimated from observational data, not a randomised experiment. Run an A/B test to validate before full rollout.",
        f"Sleeping Dogs ({sp.get('Sleeping Dog', 0):.1f}%) — sending offers to these users may actually increase churn. Ensure they are excluded.",
    ]

    # ── Key metrics to track ────────────────────────────────────
    key_metrics = {
        "Primary": f"30-day retention rate among targeted Persuadables (target: >60%)",
        "Secondary": f"Cost per retained user (current: ₹{cashback_amount * roi.n_targeted_offers // max(roi.users_retained_targeted, 1):,})",
        "Guardrail": "Churn rate among Sleeping Dogs (should not increase after campaign)",
        "Leading indicator": f"Day-7 merchant payment rate in new cohorts (current: {ret_p2m:.0f}%)",
    }

    # ── Build headline ───────────────────────────────────────────
    headline = (
        f"Target {n_persuadable:,} Persuadable users with ₹{cashback_amount} cashback. "
        f"Expected: {roi.users_retained_targeted:,} retained at "
        f"{roi.efficiency_gain:.1f}x the efficiency of random targeting."
    )

    summary = (
        f"Analysis of {n_users:,} users over 12 months. Churn rate: {churn_rate*100:.1f}%. "
        f"Day 14 is the critical retention window — users who complete "
        f"{ret_txn:.0f}+ transactions and make their first merchant payment within "
        f"14 days retain at dramatically higher rates. The uplift model identifies "
        f"a {pct_persuadable:.1f}% minority of users where cashback intervention "
        f"actually changes behaviour. Spend budget only on them."
    )

    report = StrategyReport(
        headline=headline,
        summary=summary,
        recommendations=recommendations,
        risk_factors=risk_factors,
        key_metrics=key_metrics,
    )

    logger.info("Strategy report generated: %d recommendations", len(recommendations))
    return report
