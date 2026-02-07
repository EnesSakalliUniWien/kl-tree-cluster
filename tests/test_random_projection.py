from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
    generate_projection_matrix,
    _PROJECTOR_CACHE,
    _PROJECTION_CACHE,
)
from kl_clustering_analysis import config


def _to_dense(matrix):
    """Convert matrix to dense numpy array if sparse."""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


@pytest.fixture(autouse=True)
def clear_projector_cache():
    """Fixture to clear the projector cache before each test."""
    _PROJECTOR_CACHE.clear()
    _PROJECTION_CACHE.clear()
    yield
    _PROJECTOR_CACHE.clear()
    _PROJECTION_CACHE.clear()


def test_compute_projection_dimension_basic():
    """Test basic computation of k with typical values."""
    n_samples = 1000
    n_features = 10000
    eps = 0.3
    # k = 8 * ln(n) / eps^2 = 8 * ln(1000) / (0.3^2) approx 8 * 6.907 / 0.09 = 55.256 / 0.09 approx 613
    # sklearn's johnson_lindenstrauss_min_dim formula is k >= (4 * log(n_samples)) / (eps ** 2 / 2 - eps ** 3 / 3)
    # For n_samples=1000, eps=0.3, this gives k >= 383
    # Our config.PROJECTION_MIN_K is 10
    k = compute_projection_dimension(n_samples, n_features, eps=eps, min_k=10)
    assert k >= 10
    assert k <= n_features
    # Verify it's in the expected range for these parameters
    assert 760 <= k <= 770


def test_compute_projection_dimension_min_k():
    """Test min_k constraint."""
    n_samples = 5
    n_features = 10
    eps = 0.5
    # For n_samples=5, eps=0.5, k is very small, capped by n_features=10, not min_k=50.
    k = compute_projection_dimension(n_samples, n_features, eps=eps, min_k=50)
    assert k == n_features


def test_compute_projection_dimension_max_k_n_features():
    """Test n_features constraint (k should not exceed n_features)."""
    n_samples = 1000
    n_features = 50
    eps = 0.1  # Very low eps, would usually result in k > 50
    k = compute_projection_dimension(n_samples, n_features, eps=eps, min_k=10)
    assert k == n_features


def test_generate_projection_matrix_shape():
    """Test the shape of the generated projection matrix."""
    n_features = 500
    k = 50
    matrix = generate_projection_matrix(n_features, k)
    assert matrix.shape == (k, n_features)


def test_generate_projection_matrix_determinism():
    """Test that the projection matrix is deterministic with a fixed random_state."""
    n_features = 1000
    k = 100
    random_state = 42

    matrix1 = _to_dense(
        generate_projection_matrix(n_features, k, random_state=random_state)
    )
    matrix2 = _to_dense(
        generate_projection_matrix(n_features, k, random_state=random_state)
    )
    assert np.array_equal(matrix1, matrix2)


def test_generate_projection_matrix_different_random_state_different_matrices():
    """Test that different random_states produce different matrices."""
    n_features = 1000
    k = 100
    random_state1 = 42
    random_state2 = 43

    matrix1 = _to_dense(
        generate_projection_matrix(n_features, k, random_state=random_state1)
    )
    matrix2 = _to_dense(
        generate_projection_matrix(n_features, k, random_state=random_state2)
    )
    assert not np.array_equal(matrix1, matrix2)


def test_generate_projection_matrix_caching():
    """Test that the projection matrix is cached and reused."""
    n_features = 200
    k = 20
    random_state = 123

    # First call, should populate cache
    matrix1 = _to_dense(
        generate_projection_matrix(n_features, k, random_state=random_state)
    )

    # Check cache based on current projection method
    method = getattr(config, "PROJECTION_METHOD", "sparse")
    cache_key = (method, n_features, k, random_state)
    assert cache_key in _PROJECTION_CACHE

    # Second call with same parameters, should use cached matrix
    matrix2 = _to_dense(
        generate_projection_matrix(n_features, k, random_state=random_state)
    )

    assert np.array_equal(matrix1, matrix2)

    # Call with different k, should create new entry in cache
    k_new = 25
    matrix3 = _to_dense(
        generate_projection_matrix(n_features, k_new, random_state=random_state)
    )
    cache_key_new = (method, n_features, k_new, random_state)
    assert cache_key_new in _PROJECTION_CACHE

    assert not np.array_equal(matrix1, matrix3)


def test_sparse_random_projection_effect_on_output():
    """
    Test that sparse projection output is sparse and usable.
    """
    n_features = 1000
    k = 50
    random_state = 42
    # Force sparse method for this test
    matrix = _to_dense(
        generate_projection_matrix(
            n_features, k, random_state=random_state, method="sparse"
        )
    )

    # The matrix should contain non-zero elements
    assert np.any(matrix != 0)
    # Check that it's not all zeros, and that it contains more than 1% non-zeros
    assert np.sum(matrix != 0) > (k * n_features) * 0.01  # Expect some non-zeros
    # Sparse projection should have many zeros (not all values filled)
    assert np.sum(matrix != 0) < (k * n_features) * 0.5

    # Check that it returns a numpy array
    assert isinstance(matrix, np.ndarray)


def test_orthonormal_projection_is_orthonormal():
    """Test that orthonormal projection produces orthonormal rows."""
    n_features = 500
    k = 50
    random_state = 42

    matrix = generate_projection_matrix(
        n_features, k, random_state=random_state, method="orthonormal"
    )

    # R @ R.T should be identity (rows are orthonormal)
    RRT = matrix @ matrix.T
    expected = np.eye(k)

    assert np.allclose(RRT, expected, atol=1e-10), (
        "Orthonormal projection should satisfy R @ R.T = I"
    )


def test_orthonormal_projection_chi_square_distribution():
    """Test that ||R @ z||^2 follows chi-square(k) when z ~ N(0, I)."""
    n_features = 500
    k = 50
    random_state = 42
    n_samples = 1000

    R = generate_projection_matrix(
        n_features, k, random_state=random_state, method="orthonormal"
    )

    # Generate standard normal samples
    rng = np.random.default_rng(123)
    z_samples = rng.standard_normal((n_samples, n_features))

    # Compute test statistics
    projected = z_samples @ R.T  # Shape: (n_samples, k)
    test_stats = np.sum(projected**2, axis=1)  # Sum of squared projected components

    # Test statistics should have mean ≈ k and variance ≈ 2k (chi-square properties)
    assert np.abs(np.mean(test_stats) - k) < 3, (
        f"Mean should be ~{k}, got {np.mean(test_stats):.2f}"
    )
    assert np.abs(np.var(test_stats) - 2 * k) < 20, (
        f"Variance should be ~{2 * k}, got {np.var(test_stats):.2f}"
    )
