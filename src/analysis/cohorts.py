"""
Cohort analysis module.

All operations are SQL-equivalent — the docstrings show the
SQL query each function replaces. In a production BigQuery
environment, these would be actual SQL queries. Here they're
pandas for portability.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.simulator import SimulationResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

RETENTION_CUTOFFS = [14, 30, 60, 90, 180, 365]


@dataclass
class CohortResult:
    """All cohort analysis outputs."""
    retention_by_archetype: pd.DataFrame
    day14_summary: pd.DataFrame
    category_summary: pd.DataFrame
    city_tier_summary: pd.DataFrame
    age_group_summary: pd.DataFrame


def compute_cohorts(sim: SimulationResult) -> CohortResult:
    """Run all cohort analyses."""
    logger.info("Computing cohorts on %d users / %d txns",
                sim.n_users, sim.n_transactions)

    return CohortResult(
        retention_by_archetype=_retention_curves(sim),
        day14_summary=_day14_gap(sim),
        category_summary=_category_breakdown(sim),
        city_tier_summary=_city_tier_breakdown(sim),
        age_group_summary=_age_group_breakdown(sim),
    )


def _retention_curves(sim: SimulationResult) -> pd.DataFrame:
    """
    SQL equivalent:
        SELECT
            archetype,
            day_cutoff,
            COUNT(DISTINCT CASE WHEN day < day_cutoff THEN user_id END)
                * 100.0 / COUNT(DISTINCT user_id) AS retention_pct
        FROM transactions
        JOIN users USING (user_id)
        CROSS JOIN UNNEST([14,30,60,90,180,365]) AS day_cutoff
        GROUP BY archetype, day_cutoff
    """
    rows = []
    for arch, arch_users in sim.users.groupby("archetype"):
        user_ids = set(arch_users["user_id"])
        arch_txns = sim.transactions[sim.transactions["user_id"].isin(user_ids)]
        n_arch = len(user_ids)

        for cutoff in RETENTION_CUTOFFS:
            active = len(set(arch_txns[arch_txns["day"] < cutoff]["user_id"]))
            rows.append({
                "archetype": arch,
                "day_cutoff": cutoff,
                "retention_pct": round(active / max(n_arch, 1) * 100, 2),
                "n_users": n_arch,
            })

    return pd.DataFrame(rows)


def _day14_gap(sim: SimulationResult) -> pd.DataFrame:
    """
    SQL equivalent:
        SELECT
            CASE WHEN churned = 1 THEN 'Churned' ELSE 'Retained' END AS status,
            COUNT(*)                           AS n_users,
            PERCENTILE_CONT(txn_d14, 0.5)     AS median_txn_d14,
            PERCENTILE_CONT(value_d14, 0.5)   AS median_value_d14,
            AVG(has_first_p2m_d14)             AS pct_with_p2m_d14
        FROM users
        GROUP BY status
    """
    df = sim.users.copy()
    df["status"] = df["churned"].map({0: "Retained", 1: "Churned"})

    result = (
        df.groupby("status")
        .agg(
            n_users=("user_id", "count"),
            median_txn_d14=("txn_d14", "median"),
            median_value_d14=("value_d14", "median"),
            pct_with_p2m_d14=("has_first_p2m_d14", "mean"),
            median_cat_diversity=("cat_diversity", "median"),
        )
        .reset_index()
    )
    result["pct_with_p2m_d14"] = (result["pct_with_p2m_d14"] * 100).round(1)
    return result


def _category_breakdown(sim: SimulationResult) -> pd.DataFrame:
    """
    SQL equivalent:
        SELECT
            category,
            COUNT(*)                            AS transaction_count,
            SUM(value) / 1e7                    AS total_value_crore,
            AVG(value)                          AS avg_ticket,
            AVG(is_p2m)                         AS p2m_share
        FROM transactions
        GROUP BY category
        ORDER BY transaction_count DESC
    """
    return (
        sim.transactions.groupby("category")
        .agg(
            transaction_count=("value", "count"),
            total_value_crore=("value", lambda x: round(x.sum() / 1e7, 2)),
            avg_ticket=("value", lambda x: round(x.mean(), 0)),
            p2m_share=("is_p2m", "mean"),
        )
        .reset_index()
        .sort_values("transaction_count", ascending=False)
    )


def _city_tier_breakdown(sim: SimulationResult) -> pd.DataFrame:
    """
    SQL equivalent:
        SELECT
            city_tier,
            COUNT(*)          AS n_users,
            AVG(churned)      AS churn_rate,
            AVG(txn_d14)      AS avg_txn_d14,
            AVG(p2m_ratio)    AS avg_p2m_ratio,
            AVG(total_value)  AS avg_ltv
        FROM users
        GROUP BY city_tier
    """
    result = (
        sim.users.groupby("city_tier")
        .agg(
            n_users=("user_id", "count"),
            churn_rate=("churned", "mean"),
            avg_txn_d14=("txn_d14", "mean"),
            avg_p2m_ratio=("p2m_ratio", "mean"),
            avg_ltv=("total_value", "mean"),
        )
        .reset_index()
    )
    result["city_tier"] = result["city_tier"].map(
        {1: "Tier 1 (Metro)", 2: "Tier 2", 3: "Tier 3"}
    )
    return result


def _age_group_breakdown(sim: SimulationResult) -> pd.DataFrame:
    """Churn and behaviour by age cohort."""
    return (
        sim.users.groupby("age_group")
        .agg(
            n_users=("user_id", "count"),
            churn_rate=("churned", "mean"),
            avg_txn_d14=("txn_d14", "mean"),
            avg_p2m_ratio=("p2m_ratio", "mean"),
        )
        .reset_index()
    )
