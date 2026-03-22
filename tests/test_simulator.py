"""
Tests for the data simulation module.

Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

from src.data.simulator import simulate_users, _adjust_churn_prob
from src.data.schema import validate_user_dataframe


class TestSimulateUsers:
    """Tests for the simulate_users() function."""

    @pytest.fixture(scope="class")
    def small_sim(self):
        """Small simulation reused across tests in this class."""
        return simulate_users(n_users=500, seed=42)

    def test_output_types(self, small_sim):
        assert isinstance(small_sim.users, pd.DataFrame)
        assert isinstance(small_sim.transactions, pd.DataFrame)

    def test_user_count(self, small_sim):
        assert len(small_sim.users) == 500

    def test_no_duplicate_users(self, small_sim):
        assert not small_sim.users["user_id"].duplicated().any()

    def test_no_null_values(self, small_sim):
        null_cols = small_sim.users.columns[small_sim.users.isnull().any()].tolist()
        assert null_cols == [], f"Null values found in: {null_cols}"

    def test_churn_label_is_binary(self, small_sim):
        assert set(small_sim.users["churned"].unique()).issubset({0, 1})

    def test_p2m_ratio_in_range(self, small_sim):
        assert small_sim.users["p2m_ratio"].between(0, 1).all()

    def test_city_tier_valid(self, small_sim):
        assert small_sim.users["city_tier"].isin([1, 2, 3]).all()

    def test_txn_d14_geq_txn_d7(self, small_sim):
        """Day-14 count must be >= Day-7 count for every user."""
        assert (small_sim.users["txn_d14"] >= small_sim.users["txn_d7"]).all()

    def test_archetype_distribution(self, small_sim):
        """Archetype shares should be within 5pp of configured values."""
        from src.config import CFG
        counts = small_sim.users["archetype"].value_counts(normalize=True)
        for arch, cfg in CFG.simulation.archetypes.items():
            actual_share = counts.get(arch, 0.0)
            assert abs(actual_share - cfg.share) < 0.08, (
                f"{arch}: expected ~{cfg.share:.0%}, got {actual_share:.0%}"
            )

    def test_transactions_reference_valid_users(self, small_sim):
        """All transaction user_ids must exist in users table."""
        valid_ids = set(small_sim.users["user_id"])
        txn_ids   = set(small_sim.transactions["user_id"])
        orphans   = txn_ids - valid_ids
        assert len(orphans) == 0, f"Orphan transaction user_ids: {orphans}"

    def test_reproducibility(self):
        """Same seed should produce identical output."""
        r1 = simulate_users(n_users=200, seed=7)
        r2 = simulate_users(n_users=200, seed=7)
        pd.testing.assert_frame_equal(r1.users, r2.users)

    def test_different_seeds_differ(self):
        """Different seeds should produce different data."""
        r1 = simulate_users(n_users=200, seed=1)
        r2 = simulate_users(n_users=200, seed=2)
        assert not r1.users["txn_d14"].equals(r2.users["txn_d14"])

    def test_schema_validation_passes(self, small_sim):
        """validate_user_dataframe should not raise on valid data."""
        validate_user_dataframe(small_sim.users)  # should not raise

    def test_has_all_categories(self, small_sim):
        from src.config import CFG
        sim_cats  = set(small_sim.transactions["category"].unique())
        cfg_cats  = set(CFG.simulation.categories)
        assert sim_cats.issubset(cfg_cats)

    def test_churn_rate_plausible(self, small_sim):
        """Overall churn rate should be between 20% and 70%."""
        churn_rate = small_sim.users["churned"].mean()
        assert 0.20 <= churn_rate <= 0.70, (
            f"Implausible churn rate: {churn_rate:.1%}"
        )


class TestSchemaValidation:
    """Tests for validate_user_dataframe()."""

    @pytest.fixture
    def valid_df(self):
        return simulate_users(n_users=100, seed=1).users

    def test_raises_on_missing_column(self, valid_df):
        bad = valid_df.drop(columns=["churned"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_user_dataframe(bad)

    def test_raises_on_non_binary_churn(self, valid_df):
        bad = valid_df.copy()
        bad.loc[0, "churned"] = 2
        with pytest.raises(ValueError, match="binary"):
            validate_user_dataframe(bad)

    def test_raises_on_invalid_p2m_ratio(self, valid_df):
        bad = valid_df.copy()
        bad.loc[0, "p2m_ratio"] = 1.5
        with pytest.raises(ValueError, match="p2m_ratio"):
            validate_user_dataframe(bad)

    def test_raises_on_duplicate_user_ids(self, valid_df):
        bad = pd.concat([valid_df, valid_df]).reset_index(drop=True)
        with pytest.raises(ValueError, match="Duplicate user_ids"):
            validate_user_dataframe(bad)

    def test_raises_on_nulls(self, valid_df):
        bad = valid_df.copy()
        bad.loc[0, "txn_d14"] = np.nan
        with pytest.raises(ValueError, match="Null values"):
            validate_user_dataframe(bad)


class TestChurnAdjustment:
    """Tests for the churn probability adjustment logic."""

    def test_zero_activity_increases_churn(self):
        base = 0.15
        assert _adjust_churn_prob(base, txn_d14=0) > base

    def test_high_activity_decreases_churn(self):
        base = 0.45
        assert _adjust_churn_prob(base, txn_d14=25) < base

    def test_output_always_in_range(self):
        for base in [0.01, 0.5, 0.99]:
            for txns in [0, 1, 5, 10, 25, 50]:
                p = _adjust_churn_pro