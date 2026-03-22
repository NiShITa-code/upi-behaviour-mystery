"""
Pipeline orchestration.

This is the single entry point for the full analysis.
Run programmatically or via CLI:

    python -m src.pipeline                          # defaults from config
    python -m src.pipeline --n-users 5000 --seed 7 # override params
    python -m src.pipeline --no-save                # skip artifact save
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from src.analysis.cohorts import CohortResult, compute_cohorts
from src.config import CFG
from src.data.simulator import SimulationResult, simulate_users
from src.models.churn import ChurnModelResult, train_churn_model
from src.models.uplift import UpliftResult, run_uplift_model
from src.utils.logging import get_logger

logger = get_logger(__name__, log_file=Path("artifacts/pipeline.log"))


@dataclass
class PipelineResult:
    """Typed output of the full pipeline run."""
    simulation: SimulationResult
    cohorts: CohortResult
    churn_model: ChurnModelResult
    uplift_model: UpliftResult
    total_elapsed: float

    def print_summary(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("UPI BEHAVIOUR MYSTERY — PIPELINE COMPLETE")
        print(sep)
        print(self.simulation.summary())
        print()
        print(self.churn_model.summary())
        print()
        print(self.uplift_model.summary())
        print(f"\nTotal runtime: {self.total_elapsed:.1f}s")
        print(sep)

    def export_segments(self, path: Path) -> None:
        """Export user segments CSV for downstream use."""
        cols = [
            "user_id", "archetype", "city_tier", "age_group",
            "txn_d14", "has_first_p2m_d14", "cat_diversity",
            "churn_prob", "p0", "p1", "uplift", "segment",
        ]
        out = self.uplift_model.users_segmented[
            [c for c in cols if c in self.uplift_model.users_segmented.columns]
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        logger.info("Segments exported → %s (%d rows)", path, len(out))


def run_pipeline(
    n_users: Optional[int] = None,
    seed: Optional[int] = None,
    cashback: Optional[int] = None,
    budget: Optional[int] = None,
    save_artifacts: bool = True,
) -> PipelineResult:
    """
    Run the complete analysis pipeline end-to-end.

    Parameters
    ----------
    n_users : int, optional
        Number of users to simulate. Defaults to config.
    seed : int, optional
        Random seed. Change to test stability across realisations.
    cashback : int, optional
        Cashback offer amount per user (₹).
    budget : int, optional
        Total intervention budget (₹).
    save_artifacts : bool
        Whether to save model artifacts to disk.

    Returns
    -------
    PipelineResult
    """
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("UPI BEHAVIOUR MYSTERY — PIPELINE START")
    logger.info("=" * 60)

    # ── Step 1: Simulate data ─────────────────────────────────────
    logger.info("Step 1/4: Data simulation")
    sim = simulate_users(n_users=n_users, seed=seed)

    # ── Step 2: Cohort analysis ────────────────────────────────────
    logger.info("Step 2/4: Cohort analysis")
    cohorts = compute_cohorts(sim)

    d14 = cohorts.day14_summary
    ret = d14[d14["status"] == "Retained"]["median_txn_d14"].values[0]
    chu = d14[d14["status"] == "Churned"]["median_txn_d14"].values[0]
    logger.info(
        "Day-14 gap: retained=%.0f median txns vs churned=%.0f", ret, chu
    )

    # ── Step 3: Churn model ───────────────────────────────────────
    logger.info("Step 3/4: Churn prediction model")
    churn_result = train_churn_model(sim.users, save_artifact=save_artifacts)

    # ── Step 4: Uplift model ──────────────────────────────────────
    logger.info("Step 4/4: Causal uplift model (T-Learner)")
    uplift_result = run_uplift_model(
        churn_result,
        cashback_amount=cashback,
        total_budget=budget,
    )

    total_elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1fs", total_elapsed)

    result = PipelineResult(
        simulation=sim,
        cohorts=cohorts,
        churn_model=churn_result,
        uplift_model=uplift_result,
        total_elapsed=total_elapsed,
    )

    if save_artifacts:
        result.export_segments(Path("artifacts/user_segments.csv"))

    return result


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
@click.command()
@click.option("--n-users",  default=None, type=int,  help="Number of users to simulate")
@click.option("--seed",     default=None, type=int,  help="Random seed")
@click.option("--cashback", default=None, type=int,  help="Cashback offer amount (₹)")
@click.option("--budget",   default=None, type=int,  help="Total intervention budget (₹)")
@click.option("--no-save",  is_flag=True,            help="Skip saving artifacts")
def cli(n_users, seed, cashback, budget, no_save):
    """
    UPI Behaviour Mystery — Full Analysis Pipeline

    Runs data simulation → cohort analysis → churn model → uplift model.
    Results are printed to stdout and saved to artifacts/.
    """
    result = run_pipeline(
        n_users=n_users,
        seed=