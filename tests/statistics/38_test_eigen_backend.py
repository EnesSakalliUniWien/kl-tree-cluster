from __future__ import annotations

import numpy as np
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.decomposition import (
    eigendecompose_covariance,
)
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.operators import (
    build_primal_covariance_matrix,
)
from tree_break_selection.hierarchy_analysis.decomposition.backends.eigen.projection import (
    build_pca_projection,
)


def _feature_covariance_eigensystem(data_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance_matrix = build_primal_covariance_matrix(data_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]


def test_dual_decomposition_exposes_feature_covariance_eigenvalue_scale() -> None:
    rng = np.random.default_rng(20260516)
    data_matrix = rng.normal(size=(5, 9))

    eig = eigendecompose_covariance(data_matrix, compute_eigenvectors=False)
    expected_eigenvalues, _ = _feature_covariance_eigensystem(data_matrix)

    assert eig is not None
    assert eig.use_dual is True
    np.testing.assert_allclose(
        eig.eigenvalues,
        expected_eigenvalues[: eig.eigenvalues.shape[0]],
        rtol=1e-10,
        atol=1e-10,
    )


def test_dual_projection_uses_the_same_scale_as_primal_covariance() -> None:
    rng = np.random.default_rng(20260517)
    data_matrix = rng.normal(size=(5, 9))

    eig = eigendecompose_covariance(data_matrix, compute_eigenvectors=True)
    expected_eigenvalues, expected_eigenvectors = _feature_covariance_eigensystem(data_matrix)
    projection, projection_eigenvalues = build_pca_projection(
        eig,
        projection_dimension=3,
        n_features_total=data_matrix.shape[1],
    )

    assert eig is not None
    assert projection is not None
    assert projection_eigenvalues is not None
    np.testing.assert_allclose(
        projection_eigenvalues,
        expected_eigenvalues[:3],
        rtol=1e-10,
        atol=1e-10,
    )

    alignment = np.abs(projection @ expected_eigenvectors[:, :3])
    np.testing.assert_allclose(alignment, np.eye(3), rtol=1e-10, atol=1e-10)


def test_dual_projection_exposes_only_positive_rank_directions() -> None:
    for n_samples in (4, 5):
        rng = np.random.default_rng(20260518 + n_samples)
        data_matrix = rng.normal(size=(n_samples, 9))

        eig = eigendecompose_covariance(data_matrix, compute_eigenvectors=True)

        assert eig is not None
        assert eig.use_dual is True
        positive_rank = int(np.count_nonzero(eig.eigenvalues > 1e-10))
        assert positive_rank == data_matrix.shape[0] - 1

        projection, projection_eigenvalues = build_pca_projection(
            eig,
            projection_dimension=data_matrix.shape[0],
            n_features_total=data_matrix.shape[1],
        )

        assert projection is not None
        assert projection_eigenvalues is not None
        assert projection.shape == (positive_rank, data_matrix.shape[1])
        assert projection_eigenvalues.shape == (positive_rank,)
        assert np.all(projection_eigenvalues > 1e-10)


def test_single_active_feature_returns_explicit_one_dimensional_basis() -> None:
    data_matrix = np.array(
        [
            [0.0, 4.0, 9.0],
            [1.0, 4.0, 9.0],
            [2.0, 4.0, 9.0],
            [3.0, 4.0, 9.0],
        ],
        dtype=np.float64,
    )

    eig = eigendecompose_covariance(data_matrix, compute_eigenvectors=True)
    projection, projection_eigenvalues = build_pca_projection(
        eig,
        projection_dimension=2,
        n_features_total=data_matrix.shape[1],
    )

    assert eig is not None
    assert eig.active_feature_count == 1
    np.testing.assert_allclose(eig.eigenvalues, np.array([1.25]))
    assert projection is not None
    assert projection_eigenvalues is not None
    np.testing.assert_allclose(np.abs(projection), np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(projection_eigenvalues, np.array([1.25]))
