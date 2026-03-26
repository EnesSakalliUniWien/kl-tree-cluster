"""Numerical eigen backend for spectral decomposition internals."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg

from ..core.eigen_result import EigenResult


def _decompose_and_sort(
    matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run symmetric eigendecomposition and return descending, floored outputs."""
    if compute_eigenvectors:
        eigenvalues, eigenvectors = linalg.eigh(matrix, check_finite=False)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
    else:
        eigenvalues = np.sort(linalg.eigvalsh(matrix, check_finite=False))[::-1]
        eigenvectors = None

    return np.maximum(eigenvalues, 0.0), eigenvectors


def eigendecompose_correlation_backend(
    data_matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> Optional[EigenResult]:
    """Eigendecompose a correlation matrix in primal or dual form.

    Uses the dual n×n Gram matrix when ``n_samples < n_active_features``.
    """
    data_matrix = np.asarray(data_matrix, dtype=np.float64)
    feature_variances = np.var(data_matrix, axis=0)
    is_active_feature = feature_variances > 0
    n_active_features = int(np.sum(is_active_feature))

    if n_active_features < 2:
        return None

    active_data = data_matrix[:, is_active_feature]
    n_samples = data_matrix.shape[0]
    use_dual = n_samples < n_active_features

    if use_dual:

        mu_active = active_data.mean(axis=0)
        sigma_active = active_data.std(axis=0, ddof=0)
        sigma_active[sigma_active == 0] = 1.0

        X_std = (active_data - mu_active) / sigma_active
        gram_dual = X_std @ X_std.T / n_active_features
        eigenvalues, gram_eigenvectors = _decompose_and_sort(
            gram_dual,
            compute_eigenvectors=compute_eigenvectors,
        )

        return EigenResult(
            eigenvalues=eigenvalues,
            is_active_feature=is_active_feature,
            active_feature_count=n_active_features,
            use_dual=True,
            dual_sample_eigenvectors=gram_eigenvectors,
            standardized_data_active=X_std,
        )

    correlation_matrix = np.corrcoef(active_data.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    eigenvalues, active_eigenvectors = _decompose_and_sort(
        correlation_matrix,
        compute_eigenvectors=compute_eigenvectors,
    )

    return EigenResult(
        eigenvalues=eigenvalues,
        is_active_feature=is_active_feature,
        active_feature_count=n_active_features,
        use_dual=False,
        eigenvectors_active=active_eigenvectors,
    )


def build_pca_projection_backend(
    eig: EigenResult,
    *,
    projection_dimension: int,
    n_features_total: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a PCA projection matrix and eigenvalue array.

    Returns a ``(effective_dimension × n_features_total)`` projection matrix
    and the corresponding top eigenvalues.
    """
    n_samples = (
        eig.standardized_data_active.shape[0]
        if eig.standardized_data_active is not None
        else eig.eigenvectors_active.shape[0] if eig.eigenvectors_active is not None else 0
    )

    effective_dimension = (
        min(projection_dimension, n_samples) if eig.use_dual else projection_dimension
    )

    if effective_dimension <= 0:
        return None, None

    active_eigenvectors = eig.eigenvectors_active

    # Recover feature-space (d-space) eigenvectors from dual (n×n) Gram eigenvectors:
    # Formula: feature_eigenvector = (Standardized_X^T @ sample_eigenvector) / (sqrt(eigenvalue) * sqrt(d_active))
    if (
        eig.use_dual
        and eig.dual_sample_eigenvectors is not None
        and eig.standardized_data_active is not None
    ):
        dual_sample_vectors = eig.dual_sample_eigenvectors[:, :effective_dimension]
        top_eigenvalues_floored = np.maximum(eig.eigenvalues[:effective_dimension], 1e-12)

        # Scale factor combines the eigenvalue stretch and the feature-count normalization
        recovery_scale = np.sqrt(top_eigenvalues_floored) * np.sqrt(eig.active_feature_count)
        recovered_feature_vectors = (
            eig.standardized_data_active.T @ dual_sample_vectors / recovery_scale
        )

        # Ensure unit length for numerical stability after recovery
        recovered_norms = np.linalg.norm(recovered_feature_vectors, axis=0)
        recovered_norms[recovered_norms == 0] = 1.0
        active_eigenvectors = recovered_feature_vectors / recovered_norms

    if active_eigenvectors is None:
        return None, None

    full_projection = np.zeros((n_features_total, effective_dimension), dtype=np.float64)
    full_projection[eig.is_active_feature, :] = active_eigenvectors[:, :effective_dimension]
    projection_matrix = full_projection.T
    top_eigenvalues = np.maximum(eig.eigenvalues[:effective_dimension], 1e-12).astype(np.float64)

    return projection_matrix, top_eigenvalues


__all__ = [
    "EigenResult",
    "eigendecompose_correlation_backend",
    "build_pca_projection_backend",
]
