"""Tests for categorical (multinomial) distribution support.

Verifies that the statistical pipeline correctly handles both:
- Binary (Bernoulli) distributions: shape (d,)
- Categorical (multinomial) distributions: shape (d, K)
"""

from __future__ import annotations

import numpy as np
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import (
    compute_pooled_proportion,
    compute_pooled_variance,
    standardize_proportion_difference,
    _is_categorical,
    _flatten_categorical,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)


class TestCategoricalDetection:
    """Tests for categorical vs binary detection."""

    def test_is_categorical_1d_returns_false(self):
        """1D array is binary, not categorical."""
        arr = np.array([0.3, 0.5, 0.7])
        assert not _is_categorical(arr)

    def test_is_categorical_2d_single_column_returns_false(self):
        """2D array with single column is not categorical."""
        arr = np.array([[0.3], [0.5], [0.7]])
        assert not _is_categorical(arr)

    def test_is_categorical_2d_multiple_columns_returns_true(self):
        """2D array with multiple columns is categorical."""
        arr = np.array([[0.3, 0.7], [0.5, 0.5], [0.2, 0.8]])
        assert _is_categorical(arr)


class TestFlattenCategorical:
    """Tests for flattening categorical distributions."""

    def test_flatten_1d_unchanged(self):
        """1D array should remain unchanged."""
        arr = np.array([0.3, 0.5, 0.7])
        result = _flatten_categorical(arr)
        np.testing.assert_array_equal(result, arr)

    def test_flatten_2d_to_1d(self):
        """2D array should be flattened to 1D."""
        arr = np.array([[0.3, 0.7], [0.5, 0.5]])  # shape (2, 2)
        result = _flatten_categorical(arr)
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, [0.3, 0.7, 0.5, 0.5])


class TestPooledVarianceBinary:
    """Tests for pooled variance with binary distributions."""

    def test_pooled_proportion_binary(self):
        """Pooled proportion for binary distributions."""
        theta_1 = np.array([0.3, 0.6])
        theta_2 = np.array([0.5, 0.4])
        n_1, n_2 = 100.0, 100.0

        pooled = compute_pooled_proportion(theta_1, theta_2, n_1, n_2)
        expected = (n_1 * theta_1 + n_2 * theta_2) / (n_1 + n_2)
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_pooled_variance_binary(self):
        """Pooled variance for binary distributions."""
        theta_1 = np.array([0.3, 0.6])
        theta_2 = np.array([0.5, 0.4])
        n_1, n_2 = 100.0, 100.0

        variance = compute_pooled_variance(theta_1, theta_2, n_1, n_2)
        pooled = compute_pooled_proportion(theta_1, theta_2, n_1, n_2)
        expected = pooled * (1 - pooled) * (1/n_1 + 1/n_2)
        np.testing.assert_array_almost_equal(variance, expected)

    def test_standardize_binary_returns_1d(self):
        """Standardized difference for binary should return 1D."""
        theta_1 = np.array([0.3, 0.6])
        theta_2 = np.array([0.5, 0.4])
        n_1, n_2 = 100.0, 100.0

        z_scores, variance = standardize_proportion_difference(theta_1, theta_2, n_1, n_2)
        assert z_scores.ndim == 1
        assert variance.ndim == 1
        assert len(z_scores) == 2


class TestPooledVarianceCategorical:
    """Tests for pooled variance with categorical distributions."""

    def test_pooled_proportion_categorical(self):
        """Pooled proportion for categorical distributions."""
        # 3 features, 4 categories each
        theta_1 = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
        ])
        theta_2 = np.array([
            [0.2, 0.3, 0.3, 0.2],
            [0.1, 0.4, 0.4, 0.1],
            [0.3, 0.3, 0.2, 0.2],
        ])
        n_1, n_2 = 50.0, 150.0

        pooled = compute_pooled_proportion(theta_1, theta_2, n_1, n_2)
        assert pooled.shape == (3, 4)
        expected = (n_1 * theta_1 + n_2 * theta_2) / (n_1 + n_2)
        np.testing.assert_array_almost_equal(pooled, expected)

    def test_pooled_variance_categorical(self):
        """Pooled variance for categorical distributions."""
        theta_1 = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
        ])
        theta_2 = np.array([
            [0.2, 0.3, 0.3, 0.2],
            [0.1, 0.4, 0.4, 0.1],
        ])
        n_1, n_2 = 100.0, 100.0

        variance = compute_pooled_variance(theta_1, theta_2, n_1, n_2)
        assert variance.shape == (2, 4)
        # Variance should be positive
        assert np.all(variance > 0)

    def test_standardize_categorical_flattens(self):
        """Standardized difference for categorical should flatten to 1D."""
        theta_1 = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
        ])
        theta_2 = np.array([
            [0.2, 0.3, 0.3, 0.2],
            [0.1, 0.4, 0.4, 0.1],
        ])
        n_1, n_2 = 100.0, 100.0

        z_scores, variance = standardize_proportion_difference(theta_1, theta_2, n_1, n_2)
        # Should be flattened: 2 features * 4 categories = 8
        assert z_scores.shape == (8,)
        assert variance.shape == (8,)


class TestSiblingDivergenceTestBinary:
    """Tests for sibling divergence test with binary distributions."""

    def test_sibling_test_binary_identical(self):
        """Identical binary distributions should have high p-value."""
        theta = np.array([0.3, 0.5, 0.7])
        stat, df, p_value = sibling_divergence_test(theta, theta, 100.0, 100.0)
        assert p_value > 0.05

    def test_sibling_test_binary_different(self):
        """Very different binary distributions should have low p-value."""
        theta_left = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        theta_right = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        stat, df, p_value = sibling_divergence_test(
            theta_left, theta_right, 500.0, 500.0
        )
        assert p_value < 0.05


class TestSiblingDivergenceTestCategorical:
    """Tests for sibling divergence test with categorical distributions."""

    def test_sibling_test_categorical_identical(self):
        """Identical categorical distributions should have high p-value."""
        theta = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
        ])
        stat, df, p_value = sibling_divergence_test(theta, theta, 100.0, 100.0)
        assert p_value > 0.05

    def test_sibling_test_categorical_different(self):
        """Very different categorical distributions should have low p-value."""
        theta_left = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.9, 0.05, 0.025, 0.025],
            [0.9, 0.05, 0.025, 0.025],
        ])
        theta_right = np.array([
            [0.025, 0.025, 0.05, 0.9],
            [0.025, 0.025, 0.05, 0.9],
            [0.025, 0.025, 0.05, 0.9],
        ])
        stat, df, p_value = sibling_divergence_test(
            theta_left, theta_right, 500.0, 500.0
        )
        assert p_value < 0.05

    def test_sibling_test_categorical_returns_valid_stats(self):
        """Sibling test should return valid statistics for categorical."""
        theta_left = np.array([
            [0.3, 0.3, 0.2, 0.2],
            [0.1, 0.4, 0.4, 0.1],
        ])
        theta_right = np.array([
            [0.2, 0.2, 0.3, 0.3],
            [0.4, 0.1, 0.1, 0.4],
        ])
        stat, df, p_value = sibling_divergence_test(
            theta_left, theta_right, 100.0, 100.0
        )
        assert stat >= 0
        assert df > 0
        assert 0 <= p_value <= 1


class TestShapeConsistency:
    """Tests to ensure shape consistency across the pipeline."""

    def test_binary_shape_preserved_through_pipeline(self):
        """Binary distributions maintain correct shapes through pipeline."""
        d = 10
        theta_left = np.random.uniform(0.1, 0.9, size=d)
        theta_right = np.random.uniform(0.1, 0.9, size=d)

        # Pooled variance
        pooled = compute_pooled_proportion(theta_left, theta_right, 100.0, 100.0)
        assert pooled.shape == (d,)

        variance = compute_pooled_variance(theta_left, theta_right, 100.0, 100.0)
        assert variance.shape == (d,)

        # Standardized difference (flattened)
        z, var = standardize_proportion_difference(theta_left, theta_right, 100.0, 100.0)
        assert z.shape == (d,)

    def test_categorical_shape_preserved_through_pipeline(self):
        """Categorical distributions maintain correct shapes through pipeline."""
        d, K = 10, 5
        # Generate valid probability simplices
        theta_left = np.random.dirichlet(np.ones(K), size=d)
        theta_right = np.random.dirichlet(np.ones(K), size=d)

        # Pooled variance (preserves 2D)
        pooled = compute_pooled_proportion(theta_left, theta_right, 100.0, 100.0)
        assert pooled.shape == (d, K)

        variance = compute_pooled_variance(theta_left, theta_right, 100.0, 100.0)
        assert variance.shape == (d, K)

        # Standardized difference (flattened to 1D)
        z, var = standardize_proportion_difference(theta_left, theta_right, 100.0, 100.0)
        assert z.shape == (d * K,)
