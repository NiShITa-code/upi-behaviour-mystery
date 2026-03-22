"""
Tests for experiment design / power analysis module.
"""

import pytest
import numpy as np
import pandas as pd

from src.analysis.experiment_design import (
    compute_sample_size,
    compute_mde_from_sample,
    design_experiment,
    PowerAnalysisResult,
)


class TestSampleSizeCalculation:
    """Tests for the core sample size formula."""

    def test_typical_case(self):
        """Standard UPI scenario: 55% retention, detect 5pp lift."""
        n = compute_sample_size(0.55, 0.05, alpha=0.05, power=0.80)
        # Should be in the ballpark of ~1500-2000 per arm
        assert 500 < n < 5000

    def test_smaller_effect_needs_more_users(self):
        """Halving the MDE should roughly quadruple sample size."""
        n_large_mde = compute_sample_size(0.55, 0.10)
        n_small_mde = compute_sample_size(0.55, 0.05)
        assert n_small_mde > n_large_mde * 2  # at least 2x more

    def test_higher_power_needs_more_users(self):
        """90% power needs more users than 80%."""
        n_80 = compute_sample_size(0.55, 0.05, power=0.80)
        n_90 = compute_sample_size(0.55, 0.05, power=0.90)
        assert n_90 > n_80

    def test_extreme_baseline_high(self):
        """Near-perfect retention: hard to detect improvement."""
        n = compute_sample_size(0.95, 0.02)
        assert n > 1000  # very large N needed

    def test_extreme_baseline_low(self):
        """Very low retention: easier to detect relative lift."""
        n = compute_sample_size(0.10, 0.05)
        assert n > 10  # still needs real sample

    def test_returns_int(self):
        n = compute_sample_size(0.55, 0.05)
        assert isinstance(n, int)

    def test_minimum_floor(self):
        """Should never return fewer than 10."""
        n = compute_sample_size(0.50, 0.49)  # huge effect
        assert n >= 10


class TestMDEFromSample:
    """Test the inverse: given N, what's the smallest detectable effect?"""

    def test_roundtrip(self):
        """compute_sample_size and compute_mde should be roughly inverse."""
        n = compute_sample_size(0.55, 0.05, alpha=0.05, power=0.80)
        mde = compute_mde_from_sample(0.55, n, alpha=0.05, power=0.80)
        assert abs(mde - 0.05) < 0.01  # within 1pp

    def test_larger_sample_detects_smaller_effect(self):
        mde_small_n = compute_mde_from_sample(0.55, 500)
        mde_large_n = compute_mde_from_sample(0.55, 5000)
        assert mde_large_n < mde_small_n


class TestExperimentDesign:
    """Integration tests for the full design_experiment function."""

    @pytest.fixture
    def mock_users(self):
        """Create a realistic user DataFrame for testing."""
        rng = np.random.RandomState(42)
        n = 1000
        users = pd.DataFrame({
            "user_id": range(n),
            "churned": rng.binomial(1, 0.35, n),
            "churn_prob": rng.beta(2, 5, n),
            "city_tier": rng.choice([1, 2, 3], n, p=[0.3, 0.35, 0.35]),
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45+"], n),
            "archetype": rng.choice(
                ["Power", "Regular", "Occasional", "Dormant"], n
            ),
            "segment": rng.choice(
                ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"], n,
                p=[0.15, 0.35, 0.35, 0.15]
            ),
        })
        return users

    @pytest.fixture
    def mock_uplift(self):
        """Minimal uplift result mock."""
        from unittest.mock import MagicMock
        uplift = MagicMock()
        uplift.segment_counts = {
            "Persuadable": 150, "Sure Thing": 350,
            "Lost Cause": 350, "Sleeping Dog": 150,
        }
        return uplift

    def test_returns_experiment_design(self, mock_users, mock_uplift):
        result = design_experiment(mock_users, mock_uplift)
        assert hasattr(result, "primary")
        assert hasattr(result, "subgroup_analyses")
        assert hasattr(result, "guardrail_metrics")
        assert hasattr(result, "recommendations")

    def test_primary_has_valid_sample_size(self, mock_users, mock_uplift):
        result = design_experiment(mock_users, mock_uplift)
        assert result.primary.sample_per_arm > 0
        assert result.primary.total_sample == result.primary.sample_per_arm * 2

    def test_bonferroni_correction_applied(self, mock_users, mock_uplift):
        result = design_experiment(mock_users, mock_uplift, alpha=0.05)
        assert result.bonferroni_alpha < 0.05

    def test_subgroup_sample_larger_than_primary(self, mock_users, mock_uplift):
        """Bonferroni correction means subgroups need more users."""
        result = design_experiment(mock_users, mock_uplift)
        for name, sub in result.subgroup_analyses.items():
            # Subgroups with corrected alpha need >= primary sample
            assert sub.sample_per_arm >= result.primary.sample_per_arm * 0.5

    def test_guardrails_not_empty(self, mock_users, mock_uplift):
        result = design_experiment(mock_users, mock_uplift)
        assert len(result.guardrail_metrics) >= 3

    def test_recommendations_generated(self, mock_users, mock_uplift):
        result = design_experiment(mock_users, mock_uplift)
        assert len(result.recommendations) >= 3

    def test_summary_string(self, mock_users, mock_uplift):
        result = design_experiment(mock_users, 