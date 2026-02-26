"""Tests for data-adaptive PROJECTION_MIN_K estimation.

Verifies that the effective-rank-based floor estimation in
``random_projection.py`` returns sensible values for different data
characteristics, and that the resolution/caching pipeline works correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.projection import random_projection
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    compute_projection_dimension,
    estimate_min_projection_dimension,
    resolve_min_k,
)

# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_resolved_cache():
    """Reset module-level _RESOLVED_MIN_K before and after each test."""
    random_projection._RESOLVED_MIN_K = None
    yield
    random_projection._RESOLVED_MIN_K = None


def _make_low_rank_data(n: int = 50, d: int = 200, true_rank: int = 3, seed: int = 42):
    """Create binary data with controlled effective rank.

    Generates data from ``true_rank`` latent factors, then binarises.
    """
    rng = np.random.default_rng(seed)
    # Low-rank continuous data
    W = rng.standard_normal((n, true_rank))
    V = rng.standard_normal((true_rank, d))
    X_cont = W @ V
    # Binarise via median threshold per column
    X_bin = (X_cont > np.median(X_cont, axis=0)).astype(int)
    return pd.DataFrame(X_bin, columns=[f"F{j}" for j in range(d)])


def _make_high_rank_data(n: int = 100, d: int = 50, seed: int = 42):
    """Create binary data with high effective rank (near full-rank)."""
    rng = np.random.default_rng(seed)
    # Each feature independent Bernoulli with varied p
    probs = rng.uniform(0.2, 0.8, size=d)
    X = (rng.random((n, d)) < probs).astype(int)
    return pd.DataFrame(X, columns=[f"F{j}" for j in range(d)])


def _make_degenerate_data(n: int = 5, d: int = 10):
    """Create all-zeros data (rank 0)."""
    return pd.DataFrame(np.zeros((n, d), dtype=int), columns=[f"F{j}" for j in range(d)])


# -----------------------------------------------------------------------
# Tests for estimate_min_projection_dimension
# -----------------------------------------------------------------------


class TestEstimateMinProjectionDimension:
    """Tests for the effective-rank-based floor estimation."""

    def test_low_rank_gives_small_floor(self):
        data = _make_low_rank_data(true_rank=3)
        min_k = estimate_min_projection_dimension(data)
        # Binarisation inflates effective rank above the true latent rank,
        # but it should still be modest (well below the hard_cap=20).
        assert 2 <= min_k <= 12, f"Expected small min_k for rank-3 data, got {min_k}"

    def test_high_rank_gives_larger_floor(self):
        data = _make_high_rank_data(n=100, d=50)
        min_k = estimate_min_projection_dimension(data)
        # High-rank data should give a larger floor
        assert min_k >= 5, f"Expected moderate min_k for high-rank data, got {min_k}"

    def test_hard_floor_respected(self):
        data = _make_degenerate_data()
        min_k = estimate_min_projection_dimension(data, hard_floor=3)
        assert min_k >= 3

    def test_hard_cap_respected(self):
        data = _make_high_rank_data(n=200, d=100)
        min_k = estimate_min_projection_dimension(data, hard_cap=8)
        assert min_k <= 8

    def test_tiny_data_returns_hard_floor(self):
        """Single-row data should return the hard floor."""
        data = pd.DataFrame(np.ones((1, 5), dtype=int), columns=[f"F{j}" for j in range(5)])
        min_k = estimate_min_projection_dimension(data)
        assert min_k == 2  # default hard_floor

    def test_single_feature_returns_hard_floor(self):
        data = pd.DataFrame(np.array([[0], [1], [0], [1]]), columns=["F0"])
        min_k = estimate_min_projection_dimension(data)
        assert min_k == 2

    def test_constant_columns_ignored(self):
        """Columns with zero variance shouldn't inflate the effective rank."""
        rng = np.random.default_rng(42)
        n, d_active, d_const = 50, 5, 95
        X_active = rng.integers(0, 2, size=(n, d_active))
        X_const = np.ones((n, d_const), dtype=int)
        X = np.hstack([X_active, X_const])
        data = pd.DataFrame(X, columns=[f"F{j}" for j in range(d_active + d_const)])
        min_k = estimate_min_projection_dimension(data)
        # Should be based on the 5 active features, not 100
        assert min_k <= 10


# -----------------------------------------------------------------------
# Tests for resolve_min_k
# -----------------------------------------------------------------------


class TestResolveMinK:
    """Tests for the config resolution function."""

    def test_int_passthrough(self):
        assert resolve_min_k(10) == 10
        assert resolve_min_k(5) == 5

    def test_auto_with_data(self):
        data = _make_low_rank_data()
        result = resolve_min_k("auto", leaf_data=data)
        assert isinstance(result, int)
        assert result >= 2

    def test_auto_without_data_returns_fallback(self):
        """Should fall back to 2 when leaf_data is None."""
        result = resolve_min_k("auto", leaf_data=None)
        assert result == 2

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="must be an int or 'auto'"):
            resolve_min_k("invalid_value")

    def test_resolved_value_cached(self):
        """resolve_min_k should populate _RESOLVED_MIN_K."""
        assert random_projection._RESOLVED_MIN_K is None
        data = _make_low_rank_data()
        result = resolve_min_k("auto", leaf_data=data)
        assert random_projection._RESOLVED_MIN_K == result

    def test_int_also_caches(self):
        resolve_min_k(7)
        assert random_projection._RESOLVED_MIN_K == 7


# -----------------------------------------------------------------------
# Tests for compute_projection_dimension with resolved cache
# -----------------------------------------------------------------------


class TestComputeProjectionDimensionAdaptive:
    """Test that compute_projection_dimension reads the resolved cache."""

    def test_uses_cached_value(self):
        """When _RESOLVED_MIN_K is set, compute_projection_dimension should use it."""
        random_projection._RESOLVED_MIN_K = 5

        # With n_samples=3, d=100, info cap kicks in (d >= 4n):
        # k_JL = big, capped to n=3, then floored to min_k=5
        k = compute_projection_dimension(3, 100)
        assert k == 5

    def test_without_cache_reads_config(self):
        """When _RESOLVED_MIN_K is None, should fall back to config or 2."""
        # _RESOLVED_MIN_K is None (reset by fixture)
        # config.PROJECTION_MIN_K is "auto" → falls back to 2
        k = compute_projection_dimension(3, 100)
        # With info cap: min(JL, 3) → 3, then max(3, 2) → 3
        assert k >= 2

    def test_explicit_min_k_overrides_cache(self):
        """Explicit min_k parameter should override the cache."""
        random_projection._RESOLVED_MIN_K = 5
        k = compute_projection_dimension(3, 100, min_k=8)
        assert k == 8  # explicit 8 overrides cached 5
