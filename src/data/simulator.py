"""
UPI transaction data simulator.

Calibrated to NPCI published aggregate statistics:
  - Monthly volume: ~20B transactions (2025)
  - Average ticket: ₹1,293 (Dec 2025)
  - P2M share: ~63% of volume (H1 2025)
  - City distribution: ~30% metro, 35% tier-2, 35% tier-3

In production this module would be replaced by a connector
to the actual data warehouse. The interface (simulate → DataFrame)
stays identical.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import CFG
from src.data.schema import validate_user_dataframe
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationResult:
    """Typed output of simulate_users()."""
    users: pd.DataFrame
    transactions: pd.DataFrame
    n_users: int
    n_transactions: int
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            f"Simulation complete in {self.elapsed_seconds:.1f}s",
            f"  Users:        {self.n_users:,}",
            f"  Transactions: {self.n_transactions:,}",
            f"  Churn rate:   {self.users['churned'].mean():.1%}",
            f"  Archetype mix:",
        ]
        for arch, grp in self.users.groupby("archetype"):
            lines.append(f"    {arch:<15} {len(grp):>6,}  ({len(grp)/self.n_users:.1%})")
        return "\n".join(lines)


def simulate_users(
    n_users: int | None = None,
    seed: int | None = None,
) -> SimulationResult:
    """
    Simulate UPI user transaction data.

    Parameters
    ----------
    n_users : int, optional
        Override config default.
    seed : int, optional
        Override config default. Pass different seeds to test
        model stability across dataset realisations.

    Returns
    -------
    SimulationResult
        Validated users and transactions DataFrames.

    Raises
    ------
    ValueError
        If generated data fails schema validation.
    """
    n_users = n_users or CFG.simulation.n_users
    seed    = seed    or CFG.simulation.seed

    logger.info("Starting simulation: n_users=%d, seed=%d", n_users, seed)
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)  # modern numpy RNG — reproducible, no global state

    # Sample archetypes according to configured distribution
    arch_names  = list(CFG.simulation.archetypes.keys())
    arch_shares = [CFG.simulation.archetypes[a].share for a in arch_names]
    archetypes  = rng.choice(arch_names, size=n_users, p=arch_shares)

    user_rows: List[dict] = []
    txn_rows:  List[dict] = []

    for uid in range(n_users):
        arch     = archetypes[uid]
        arch_cfg = CFG.simulation.archetypes[arch]

        city_tier = int(rng.choice([1, 2, 3], p=CFG.simulation.city_tier_dist))
        age_group = rng.choice(CFG.simulation.age_groups, p=CFG.simulation.age_dist)

        # --- Per-user counters ---
        d7_cnt = d14_cnt = total_cnt = 0
        d7_val = d14_val = total_val = 0.0
        first_p2m_day: int | None = None
        cats_seen: set[str] = set()

        for day in range(CFG.simulation.n_days):
            # Poisson arrival process — realistic for payment events
            n_today = int(rng.poisson(arch_cfg.daily_txn_rate))

            for _ in range(n_today):
                cat    = rng.choice(CFG.simulation.categories)
                is_p2m = int(cat != "P2P Transfer")

                # Log-normal value distribution calibrated to NPCI avg ticket ₹1,293
                val = float(max(10.0, rng.lognormal(
                    mean  = np.log(arch_cfg.avg_value * (1.2 if is_p2m else 0.7)),
                    sigma = 0.6,
                )))

                txn_rows.append({
                    "user_id":   uid,
                    "archetype": arch,
                    "day":       day,
                    "category":  cat,
                    "is_p2m":    is_p2m,
                    "value":     round(val, 2),
                })

                total_cnt += 1
                total_val += val
                cats_seen.add(cat)

                if day < 7:  d7_cnt  += 1; d7_val  += val
                if day < 14: d14_cnt += 1; d14_val += val
                if is_p2m and first_p2m_day is None:
                    first_p2m_day = day

        # --- Churn label ---
        # Calibrated: high-activity users churn less
        churn_adj = _adjust_churn_prob(arch_cfg.churn_prob, d14_cnt)
        churned   = int(rng.random() < churn_adj)

        # --- P2M ratio (with individual noise) ---
        p2m_ratio = float(np.clip(
            arch_cfg.p2m_ratio + rng.normal(0, 0.05), 0.05, 0.95
        ))

        user_rows.append({
            "user_id":           uid,
            "archetype":         arch,
            "city_tier":         city_tier,
            "age_group":         age_group,
            "txn_d7":            d7_cnt,
            "value_d7":          round(d7_val, 2),
            "txn_d14":           d14_cnt,
            "value_d14":         round(d14_val, 2),
            "has_first_p2m_d14": int(first_p2m_day is not None and first_p2m_day < 14),
            "first_p2m_day":     first_p2m_day if first_p2m_day is not None else 999,
            "cat_diversity":     len(cats_seen),
            "p2m_ratio":         p2m_ratio,
            "total_txn":         total_cnt,
            "total_value":       round(total_val, 2),
            "churned":           churned,
        })

    df_users = pd.DataFrame(user_rows)
    df_txns  = pd.DataFrame(txn_rows)

    # Schema validation at the boundary — fail fast with clear message
    validate_user_dataframe(df_users)

    elapsed = time.perf_counter() - t0
    result  = SimulationResult(
        users=df_users,
        transactions=df_txns,
        n_users=n_users,
        n_transactions=len(df_txns),
        elapsed_seconds=elapsed,
    )

    logger.info(result.summary())
    return result


def _adjust_churn_prob(base: float, txn_d14: int) -> float:
    """
    Adjust base churn probability based on early engagement.
    Users who are very active in first 14 days are less likely to churn.
    Users who barely transact are more likely.
    """
    if txn_d14 == 0:
        multiplier = 1.8   # much more likely to churn if no early engagement
    elif txn_d14 < 3:
        multiplier 