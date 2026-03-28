"""
Tests for churn and uplift models.

These are integration-level tests — they run the real models
on small simulated datasets and assert on outcomes, not
implementation details.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.simulator import simulate_users
from src.models.churn import train_churn_model
from src.models.uplift import run_uplift_model, _classify_segment


@pytest.fixture(scope="module")
def small_pipeline():
    """Run the full pipeline once for the module — reused across tests."""
    sim    = simulate_users(n_users=1000, seed=99)
    churn  = train_churn_model(sim.users, save_artifact=False)
    uplift = run_uplift_model(churn, cashback_amount=20, total_budget=10000)
    return sim, churn, uplift


class TestChurnModel:
    def test_auc_above_random(self, small_pipeline):
        _, churn, _ = small_pipeline
        assert churn.lgb_metrics.auc_roc > 0.55, (
            f"Model barely better than random: AUC={churn.lgb_metrics.auc_roc:.3f}"
        )

    def test_lgb_beats_logistic(self, small_pipeline):
        _, churn, _ = small_pipeline
        # LightGBM should match or beat logistic (allow tiny margin)
        assert churn.lgb_metrics.auc_roc >= churn.lr_metrics.auc_roc - 0.02

    def test_churn_prob_in_range(self, small_pipeline):
        _, churn, _ = small_pipeline
        probs = churn.users_scored["churn_prob"]
        assert probs.between(0, 1).all(), "churn_prob out of [0,1]"

    def test_churn_prob_not_constant(self, small_pipeline):
        _, churn, _ = small_pipeline
        std = churn.users_scored["churn_prob"].std()
        assert std > 0.05, f"churn_prob suspiciously constant (std={std:.4f})"

    def test_cv_auc_reasonable(self, small_pipeline):
        _, churn, _ = small_pipeline
        assert len(churn.cv_auc_scores) == 5
        assert all(s > 0.5 for s in churn.cv_auc_scores)

    def test_cv_std_not_too_high(self, small_pipeline):
        """Model should be stable across folds."""
        _, churn, _ = small_pipeline
        assert churn.cv_auc_std < 0.10, (
            f"High CV variance: std={churn.cv_auc_std:.3f} — possible overfitting"
        )

    def test_feature_importance_sums_to_100(self, small_pipeline):
        _, churn, _ = small_pipeline
        total = sum(churn.lgb_metrics.feature_importance.values())
        assert abs(total - 100.0) < 1.0, f"Importance sum: {total}"

    def test_brier_score_reasonable(self, small_pipeline):
        """Brier score < 0.25 means better than always predicting base rate."""
        _, churn, _ = small_pipeline
        assert churn.lgb_metrics.brier_score < 0.25

    def test_users_scored_same_length(self, small_pipeline):
        sim, churn, _ = small_pipeline
        assert len(churn.users_scored) == len(sim.users)


class TestUpliftModel:
    def test_segments_cover_all_users(self, small_pipeline):
        _, _, uplift = small_pipeline
        total = sum(uplift.segment_counts.values())
        assert total == len(uplift.users_segmented)

    def test_valid_segments_only(self, small_pipeline):
        _, _, uplift = small_pipeline
        valid = {"Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"}
        actual = set(uplift.users_segmented["segment"].unique())
        assert actual.issubset(valid), f"Invalid segments: {actual - valid}"

    def test_p0_p1_in_range(self, small_pipeline):
        _, _, uplift = small_pipeline
        df = uplift.users_segmented
        assert df["p0"].between(0, 1).all()
        assert df["p1"].between(0, 1).all()

    def test_uplift_equals_p1_minus_p0(self, small_pipeline):
        _, _, uplift = small_pipeline
        df = uplift.users_segmented
        computed = (df["p1"] - df["p0"]).round(4)
        assert (computed == df["uplift"]).all()

    def test_persuadables_have_positive_uplift(self, small_pipeline):
        _, _, uplift = small_pipeline
        persuadables = uplift.users_segmented[
            uplift.users_segmented["segment"] == "Persuadable"
        ]
        if len(persuadables) > 0:
            assert (persuadables["uplift"] > 0).all()

    def test_sure_things_have_high_p0(self, small_pipeline):
        _, _, uplift = small_pipeline
        sure = uplift.users_segmented[
            uplift.users_segmented["segment"] == "Sure Thing"
        ]
        if len(sure) > 0:
            threshold = 0.60
            assert (sure["p0"] > threshold).all()

    def test_targeted_geq_random(self, small_pipeline):
        """Targeting Persuadables should always be at least as good as random."""
        _, _, uplift = small_pipeline
        assert uplift.roi.users_retained_targeted >= uplift.roi.users_retained_random

    def test_efficiency_gain_positive(self, small_pipeline):
        _, _, uplift = small_pipeline
        assert uplift.roi.efficiency_gain > 1.0


class TestSegmentClassification:
    """Unit tests for the segment classification logic."""

    def _make_row(self, p0: float, p1: float) -> pd.Series:
        return pd.Series({"p0": p0, "p1": p1})

    def test_persuadable(self):
        row = self._make_row(p0=0.3, p1=0.8)
        assert _classify_segment(row) == "Persuadable"

    def test_sure_thing(self):
        row = self._make_row(p0=0.8, p1=0.9)
        assert _classify_segment(row) == "Sure Thing"

    def test_lost_cause(self):
        row = self._make_row(p0=0.2, p1=0.3)
        assert _classify_segment(row) == "Lost Cause"

    def test_sleeping_dog(self):
        row = self._make_row(p0=0.9, p1=0.2)
        assert _classify_segment(row) == "Sleeping Dog"
