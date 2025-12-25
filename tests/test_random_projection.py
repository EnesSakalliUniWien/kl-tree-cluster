from __future__ import annotations

import numpy as np
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
    generate_projection_matrix,
    _PROJECTOR_CACHE,
)
from kl_clustering_analysis import config


@pytest.fixture(autouse=True)
def clear_projector_cache():
    """Fixture to clear the projector cache before each test."""
    _PROJECTOR_CACHE.clear()
    yield
    _PROJECTOR_CACHE.clear()


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

    matrix1 = generate_projection_matrix(n_features, k, random_state=random_state).toarray()
    matrix2 = generate_projection_matrix(n_features, k, random_state=random_state).toarray()
    assert np.array_equal(matrix1, matrix2)


def test_generate_projection_matrix_different_random_state_different_matrices():
    """Test that different random_states produce different matrices."""
    n_features = 1000
    k = 100
    random_state1 = 42
    random_state2 = 43

    matrix1 = generate_projection_matrix(n_features, k, random_state=random_state1).toarray()
    matrix2 = generate_projection_matrix(n_features, k, random_state=random_state2).toarray()
    assert not np.array_equal(matrix1, matrix2)


def test_generate_projection_matrix_caching():
    """Test that the projector is cached and reused."""
    n_features = 200
    k = 20
    random_state = 123

    # First call, should populate cache
    matrix1 = generate_projection_matrix(n_features, k, random_state=random_state).toarray()
    assert (n_features, k, random_state) in _PROJECTOR_CACHE
    projector1_id = id(_PROJECTOR_CACHE[(n_features, k, random_state)])

    # Second call with same parameters, should use cached projector
    matrix2 = generate_projection_matrix(n_features, k, random_state=random_state).toarray()
    projector2_id = id(_PROJECTOR_CACHE[(n_features, k, random_state)])

    assert np.array_equal(matrix1, matrix2)
    assert projector1_id == projector2_id  # Ensure the same object is returned

    # Call with different k, should create new projector and cache it
    k_new = 25
    matrix3 = generate_projection_matrix(n_features, k_new, random_state=random_state).toarray()
    assert (n_features, k_new, random_state) in _PROJECTOR_CACHE
    projector3_id = id(_PROJECTOR_CACHE[(n_features, k_new, random_state)])

    assert projector1_id != projector3_id
    assert not np.array_equal(matrix1, matrix3)


def test_sparse_random_projection_effect_on_output():
    """
    Test that the output matrix is not entirely zero and is usable,
    implying sparse projection works as expected.
    """
    n_features = 1000
    k = 50
    random_state = 42
    matrix = generate_projection_matrix(n_features, k, random_state=random_state).toarray()

    # The matrix should contain non-zero elements
    assert np.any(matrix != 0)
    # The matrix should not be full of zeros, and not full of identical values (sparse)
    # SparseRandomProjection uses 1/sqrt(n_components) as the default density, resulting in many zeros
    # Check that it's not all zeros, and that it contains more than 1% non-zeros, but not too many (still sparse)
    assert np.sum(matrix != 0) > (k * n_features) * 0.01  # Expect some non-zeros
    assert np.sum(matrix != 0) < (k * n_features) * 0.5  # Expect it to be sparse (not all values)

    # Check that it returns a numpy array, as expected by the type hint (after .toarray())
    assert isinstance(matrix, np.ndarray)
