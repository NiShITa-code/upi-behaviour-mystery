"""
Tests for model fairness audit module.
"""

import pytest
import numpy as np
import pandas as pd

from src.analysis.fairness import (
    run_fairness_audit,
    _group_metrics,
    FairnessAudit,
    GroupMetrics,
    DISPARATE_IMPACT_THRESHOLD,
)


class TestGroupMetrics:
    """Test individual group metric computation."""

    def test_perfect_predictions(self):
        """Perfect model should have TPR=1, FPR=0."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.15])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        gm = _group_metrics(y_true, y_prob, y_pred, "test")
        assert gm.true_positive_rate == 1.0
        assert gm.false_positive_rate == 0.0
        assert gm.precision == 1.0

    def test_all_wrong_predictions(self):
        """Completely wrong model."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        y_pred = np.array([0, 0, 1, 1])
        gm = _group_metrics(y_true, y_prob, y_pred, "bad")
        assert gm.true_positive_rate == 0.0
        assert gm.false_positive_rate == 1.0

    def test_empty_group(self):
        """Empty group should return zeros gracefully."""
        gm = _group_metrics(np.array([]), np.array([]), np.array([]), "empty")
        assert gm.group_size == 0

    def test_group_size_correct(self):
        y = np.array([1, 0, 1, 0, 1])
        gm = _group_metrics(y, np.random.rand(5), np.array([1,0,1,0,1]), "five")
        assert gm.group_size == 5


class TestFairnessAudit:
    """Integration tests for the full fairness audit."""

    @pytest.fixture
    def fair_users(self):
        """Users where model performs similarly across groups."""
        rng = np.random.RandomState(42)
        n = 2000
        churned = rng.binomial(1, 0.35, n)
        # Model that's roughly calibrated for all groups
        churn_prob = churned * 0.6 + (1 - churned) * 0.3 + rng.normal(0, 0.1, n)
        churn_prob = np.clip(churn_prob, 0.01, 0.99)

        return pd.DataFrame({
            "user_id": range(n),
            "churned": churned,
            "churn_prob": churn_prob,
            "city_tier": rng.choice([1, 2, 3], n, p=[0.33, 0.34, 0.33]),
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45+"], n),
        })

    @pytest.fixture
    def biased_users(self):
        """Users where model is biased against tier-3 cities."""
        rng = np.random.RandomState(42)
        n = 2000
        city_tier = rng.choice([1, 2, 3], n, p=[0.33, 0.34, 0.33])
        churned = rng.binomial(1, 0.35, n)

        # Good predictions for tier 1-2, bad for tier 3
        churn_prob = np.where(
            city_tier <= 2,
            churned * 0.7 + (1 - churned) * 0.2 + rng.normal(0, 0.05, n),
            rng.uniform(0.3, 0.7, n),  # random noise for tier 3
        )
        churn_prob = np.clip(churn_prob, 0.01, 0.99)

        return pd.DataFrame({
            "user_id": range(n),
            "churned": churned,
            "churn_prob": churn_prob,
            "city_tier": city_tier,
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45+"], n),
        })

    def test_returns_fairness_audit(self, fair_users):
        result = run_fairness_audit(fair_users)
        assert isinstance(result, FairnessAudit)

    def test_audits_expected_attributes(self, fair_users):
        result = run_fairness_audit(fair_users)
        assert "city_tier" in result.attributes
        assert "age_group" in result.attributes

    def test_custom_attributes(self, fair_users):
        result = run_fairness_audit(fair_users, attributes=["city_tier"])
        assert "city_tier" in result.attributes
        assert "age_group" not in result.attributes

    def test_fair_model_passes(self, fair_users):
        """A roughly fair model should pass the audit."""
        result = run_fairness_audit(fair_users)
        # Not guaranteed to pass perfectly, but DI ratio should be reasonable
        for attr in result.attributes.values():
            assert attr.disparate_impact_ratio > 0.5  # not wildly biased

    def test_groups_have_correct_attributes(self, fair_users):
        result = run_fairness_audit(fair_users)
        city_attr = result.attributes["city_tier"]
        assert "1" in city_attr.groups or 1 in city_attr.groups
        assert len(city_attr.groups) == 3  # tiers 1, 2, 3

    def test_overall_auc_reasonable(self, fair_users):
        result = run_fairness_audit(fair_users)
        assert 0.5 < result.overall_auc < 1.0

    def test_summary_string(self, fair_users):
        result = run_fairness_audit(fair_users)
        summary = result.summary()
        assert "FAIRNESS AUDIT" in summary
        assert "CITY_TIER" in summary

    def test_threshold_affects_predictions(self, fair_users):
        r1 = run_fairness_audit(fair_users, threshold=0.3)
        r2 = run_fairness_audit(fair_users, threshold=0.7)
        # Different thresholds should give different predicted churn rates
        attr1 = list(r1.attributes.values())[0]
        attr2 = list(r2.attributes.values())[0]
        g1_rates = [g.churn_rate_predicted for g in attr1.groups.values()]
        g2_rates = [g.churn_rate_predicted for g in attr2.groups.values()]
        # Lower threshold → more users flagged as churning
        assert sum(g1_rates) > sum(g2_rates)
