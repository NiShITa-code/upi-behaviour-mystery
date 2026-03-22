"""
Tests for SHAP explainability module.
"""

import pytest
import numpy as np
import pandas as pd

from src.models.explainability import SHAPExplanation


class TestSHAPExplanation:
    """Test the SHAPExplanation dataclass and its methods."""

    @pytest.fixture
    def mock_shap(self):
        """Create a mock SHAPExplanation."""
        n_users = 100
        n_features = 5
        feature_names = ["txn_d14", "value_d14", "p2m_ratio", "city_tier", "cat_diversity"]

        rng = np.random.RandomState(42)
        shap_values = rng.randn(n_users, n_features) * 0.1

        mean_abs = np.abs(shap_values).mean(axis=0)
        global_importance = {
            feat: float(val) for feat, val in zip(feature_names, mean_abs)
        }
        top_features = sorted(global_importance.items(),
                              key=lambda x: x[1], reverse=True)

        return SHAPExplanation(
            shap_values=shap_values,
            feature_names=feature_names,
            X_display=pd.DataFrame(rng.randn(n_users, n_features),
                                   columns=feature_names),
            expected_value=0.35,
            elapsed_seconds=1.5,
            global_importance=global_importance,
            top_features=top_features,
        )

    def test_shape_matches(self, mock_shap):
        assert mock_shap.shap_values.shape == (100, 5)

    def test_feature_names_length(self, mock_shap):
        assert len(mock_shap.feature_names) == 5

    def test_global_importance_keys(self, mock_shap):
        assert set(mock_shap.global_importance.keys()) == set(mock_shap.feature_names)

    def test_global_importance_positive(self, mock_shap):
        for val in mock_shap.global_importance.values():
            assert val >= 0

    def test_top_features_sorted(self, mock_shap):
        values = [v for _, v in mock_shap.top_features]
        assert values == sorted(values, reverse=True)

    def test_get_user_explanation(self, mock_shap):
        explanation = mock_shap.get_user_explanation(0)
        assert isinstance(explanation, dict)
        assert len(explanation) == 5
        assert all(isinstance(v, float) for v in explanation.values())

    def test_get_top_user_features(self, mock_shap):
        top = mock_shap.get_top_user_features(0, n=3)
        assert len(top) == 3
        # Should be sorted by absolute value
        abs_values = [abs(v) for _, v in top]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_summary_string(self, mock_shap):
        summary = mock_shap.summary()
        assert "SHAP EXPLANATION" in summary
 