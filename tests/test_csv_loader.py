"""
Tests for CSV loader and BYOD mode.
"""

import io
import pytest
import pandas as pd
import numpy as np

from src.data.csv_loader import (
    load_user_csv,
    generate_sample_csv,
    REQUIRED_USER_COLS,
)


class TestGenerateSampleCSV:
    def test_returns_dataframe(self):
        df = generate_sample_csv(n_users=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_has_required_columns(self):
        df = generate_sample_csv(n_users=5)
        for col in REQUIRED_USER_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_reproducible(self):
        df1 = generate_sample_csv(n_users=10, seed=42)
        df2 = generate_sample_csv(n_users=10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_sample_csv(n_users=10, seed=1)
        df2 = generate_sample_csv(n_users=10, seed=2)
        assert not df1["txn_d14"].equals(df2["txn_d14"])


class TestLoadUserCSV:
    @pytest.fixture
    def valid_csv_bytes(self):
        df = generate_sample_csv(n_users=50, seed=42)
        return df.to_csv(index=False).encode()

    def test_valid_csv_loads(self, valid_csv_bytes):
        result = load_user_csv(io.BytesIO(valid_csv_bytes))
        assert result.valid
        assert result.n_users == 50
        assert result.users is not None

    def test_missing_column_fails(self):
        df = generate_sample_csv(n_users=10)
        df = df.drop(columns=["churned"])
        csv_bytes = df.to_csv(index=False).encode()
        result = load_user_csv(io.BytesIO(csv_bytes))
        assert not result.valid
        assert any("Missing" in e for e in result.errors)

    def test_empty_csv_fails(self):
        csv_bytes = b"user_id,txn_d7\n"  # headers only
        result = load_user_csv(io.BytesIO(csv_bytes))
        assert not result.valid

    def test_optional_columns_filled(self, valid_csv_bytes):
        # Remove optional columns
        df = generate_sample_csv(n_users=10)
        df = df.drop(columns=["archetype"], errors="ignore")
        csv_bytes = df.to_csv(index=False).encode()
        result = load_user_csv(io.BytesIO(csv_bytes))
        assert result.valid
        assert "archetype" in result.users.columns

    def test_roundtrip_sample_csv(self):
        """Sample CSV should always validate successfully."""
        df = generate_sample_csv(n_users=30, seed=7)
        csv_bytes = df.to_csv(index=False).encode()
        result = load_user_csv(io.BytesIO(csv_bytes))
        assert result.valid
        assert result.n_users == 30


class TestLoadStrategyModule:
    """Verify the strategy recommender doesn't crash."""

    def test_strategy_generates(self):
        from src.data.simulator import simulate_users
        from src.analysis.cohorts import compute_cohorts
        from src.models.churn import train_churn_model
        from src.models.uplift import run_uplift_model
        from src.analysis.strategy import generate_strategy

        sim = simulate_users(n_users=500, seed=42)
        coh = compute_cohorts(sim)
        churn = train_churn_model(sim.users, save_artifact=False)
        uplift = run_uplift_model(churn, cashback_amount=20, total_budget=10000)

        strategy = generate_strategy(churn, uplift, coh, 20, 10000)

        assert len(strategy.recommendations) >= 3
        assert len(strategy.headline) > 20
        assert len(strategy.risk_factors) >= 2
        assert "Primary" in strategy.key_metrics

    def test_strategy_to_text(self):
        from src.data.simulator import simulate_users
        from src.analysis.cohorts import compute_cohorts
        from src.models.churn import train_churn_model
        from src.models.uplift import run_uplift_model
        from src.analysis.strategy import generate_strategy

        sim = simulate_users(n_users=500, seed=42)
        coh = compute_cohorts(sim)
        churn = train_churn_model(sim.users, save_artifact=False)
        uplift = run_uplift_model(churn, cashback_amount=20, total_budget=10000)

        strategy = generate_strategy(churn, uplift, coh, 20, 10000)
        text = strategy.to_plain_text()

        assert "RECOMMENDATIONS" in text
        assert "RISK FACTORS" in text
        assert len(text) > 500
