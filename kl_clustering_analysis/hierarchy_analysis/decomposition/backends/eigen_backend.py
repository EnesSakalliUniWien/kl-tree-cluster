"""Numerical eigen backend for spectral decomposition internals."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg

from ..core.eigen_result import EigenResult
from ...statistics.projection.k_estimators import marchenko_pastur_signal_count


def eigendecompose_correlation_backend(
    data_matrix: np.ndarray,
    *,
    need_eigh: bool,
) -> Optional[EigenResult]:
    """Eigendecompose a correlation matrix in primal or dual form.

    Uses the dual n×n Gram matrix when ``n_samples < n_active_features``.
    """
    data_matrix = np.asarray(data_matrix, dtype=np.float64)
    column_variances = np.var(data_matrix, axis=0)
    active_mask = column_variances > 0
    n_active_features = int(np.sum(active_mask))

    if n_active_features < 2:
        return None

    active_data = data_matrix[:, active_mask]
    n_samples = data_matrix.shape[0]
    use_dual = n_samples < n_active_features

    if use_dual:

        column_means = active_data.mean(axis=0)
        column_stds = active_data.std(axis=0, ddof=0)
        column_stds[column_stds == 0] = 1.0
        standardized_data = (active_data - column_means) / column_stds
        gram_matrix = standardized_data @ standardized_data.T / n_active_features

        if need_eigh:
            eigenvalues, gram_eigenvectors = linalg.eigh(gram_matrix, check_finite=False)
            eigenvalues = eigenvalues[::-1]
            gram_eigenvectors = gram_eigenvectors[:, ::-1]
        else:
            eigenvalues = np.sort(linalg.eigvalsh(gram_matrix, check_finite=False))[::-1]
            gram_eigenvectors = None

        eigenvalues = np.maximum(eigenvalues, 0.0)

        return EigenResult(
            eigenvalues=eigenvalues,
            active_mask=active_mask,
            d_active=n_active_features,
            use_dual=True,
            gram_vecs=gram_eigenvectors,
            X_std=standardized_data,
        )

    correlation_matrix = np.corrcoef(active_data.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)

    if need_eigh:
        eigenvalues, active_eigenvectors = linalg.eigh(correlation_matrix, check_finite=False)
        eigenvalues = eigenvalues[::-1]
        active_eigenvectors = active_eigenvectors[:, ::-1]
    else:
        eigenvalues = np.sort(linalg.eigvalsh(correlation_matrix, check_finite=False))[::-1]
        active_eigenvectors = None

    eigenvalues = np.maximum(eigenvalues, 0.0)

    return EigenResult(
        eigenvalues=eigenvalues,
        active_mask=active_mask,
        d_active=n_active_features,
        use_dual=False,
        eigenvectors_active=active_eigenvectors,
    )


def estimate_spectral_k_backend(
    eigenvalues: np.ndarray,
    *,
    method: str,
    n_samples: int,
    n_features: int,
    minimum_projection_dimension: int,
) -> int:
    """Estimate projection dimension from eigenvalues."""
    k = int(marchenko_pastur_signal_count(eigenvalues, n_samples, n_features))

    k = max(k, int(minimum_projection_dimension))
    k = min(k, int(n_features))
    return k


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
        eig.X_std.shape[0]
        if eig.X_std is not None
        else eig.eigenvectors_active.shape[0] if eig.eigenvectors_active is not None else 0
    )

    effective_dimension = (
        min(projection_dimension, n_samples) if eig.use_dual else projection_dimension
    )

    if effective_dimension <= 0:
        return None, None

    active_eigenvectors = eig.eigenvectors_active

    # Recover d-space eigenvectors from dual (n×n) Gram eigenvectors:
    #   v_i = X_std^T @ u_i / (sqrt(λ_i) * sqrt(d_active))
    if eig.use_dual and eig.gram_vecs is not None and eig.X_std is not None:
        top_gram_vectors = eig.gram_vecs[:, :effective_dimension]
        floored_eigenvalues = np.maximum(eig.eigenvalues[:effective_dimension], 1e-12)
        normalization_scale = np.sqrt(floored_eigenvalues) * np.sqrt(eig.d_active)
        active_eigenvectors = eig.X_std.T @ top_gram_vectors / normalization_scale
        column_norms = np.linalg.norm(active_eigenvectors, axis=0)
        column_norms[column_norms == 0] = 1.0
        active_eigenvectors = active_eigenvectors / column_norms

    if active_eigenvectors is None:
        return None, None

    full_projection = np.zeros((n_features_total, effective_dimension), dtype=np.float64)
    full_projection[eig.active_mask, :] = active_eigenvectors[:, :effective_dimension]
    projection_matrix = full_projection.T
    top_eigenvalues = np.maximum(eig.eigenvalues[:effective_dimension], 1e-12).astype(np.float64)

    return projection_matrix, top_eigenvalues


__all__ = [
    "EigenResult",
    "eigendecompose_correlation_backend",
    "estimate_spectral_k_backend",
    "build_pca_projection_backend",
]
