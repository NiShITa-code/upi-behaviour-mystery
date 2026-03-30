"""
Tests for statistical comparison methods.
"""

import pytest
import numpy as np

from src.models.statistical_tests import delong_test


class TestDeLongTest:
    """Tests for the DeLong AUC comparison test."""

    def test_identical_models_not_significant(self):
        """Two identical predictions should not be significantly different."""
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, size=500)
        y_pred = rng.uniform(0, 1, size=500)
        result = delong_test(y_true, y_pred, y_pred)
        assert not result.significant
        assert abs(result.auc_diff) < 0.01

    def test_very_different_models_significant(self):
        """A perfect model vs random should be significantly different."""
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, size=1000)
        # Perfect model
        y_pred_good = y_true.astype(float) + rng.normal(0, 0.1, size=1000)
        y_pred_good = np.clip(y_pred_good, 0, 1)
        # Random model
        y_pred_bad = rng.uniform(0, 1, size=1000)
        result = delong_test(y_true, y_pred_good, y_pred_bad)
        assert result.significant
        assert result.auc_diff > 0

    def test_output_fields_valid(self):
        """All output fields should be in expected ranges."""
        rng = np.random.default_rng(7)
        y_true = rng.binomial(1, 0.4, size=300)
        y_pred_1 = rng.uniform(0, 1, size=300)
        y_pred_2 = rng.uniform(0, 1, size=300)
        result = delong_test(y_true, y_pred_1, y_pred_2)

        assert 0 <= result.auc_1 <= 1
        assert 0 <= result.auc_2 <= 1
        assert -1 <= result.auc_diff <= 1
        assert 0 <= result.p_value <= 1
        assert result.ci_lower <= result.ci_upper

    def test_summary_string(self):
        """Summary should be a non-empty string."""
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, size=200)
        y_pred_1 = rng.uniform(0, 1, size=200)
        y_pred_2 = rng.uniform(0, 1, size=200)
        result = delong_test(y_true, y_pred_1, y_pred_2)
        assert len(result.summary()) > 50
        assert "DeLong" in result.summary()
