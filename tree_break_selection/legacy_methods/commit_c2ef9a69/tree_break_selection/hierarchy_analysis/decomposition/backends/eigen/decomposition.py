"""Spectral decomposition orchestration for local correlation structure."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg

from ...core.eigen_result import EigenResult
from .operators import (
    build_dual_gram_matrix,
    build_primal_correlation_matrix,
    standardize_active_data,
)


def eigendecompose_correlation(
    data_matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> Optional[EigenResult]:
    """Eigendecompose the active correlation structure in primal or dual form.

    The input is a dense matrix of shape ``(n_samples, n_features)``.
    Constant features are removed first. When ``n_samples < n_active_features``,
    the decomposition switches to the smaller dual sample-space Gram matrix;
    otherwise it decomposes the feature-space correlation matrix directly.
    """
    active_data, is_active_feature = _prepare_active_data(data_matrix)
    n_active_features = int(np.sum(is_active_feature))

    if n_active_features < 2:
        return None

    if _should_use_dual(n_samples=active_data.shape[0], n_active_features=n_active_features):
        return _eigendecompose_dual(
            active_data,
            is_active_feature=is_active_feature,
            compute_eigenvectors=compute_eigenvectors,
        )

    return _eigendecompose_primal(
        active_data,
        is_active_feature=is_active_feature,
        compute_eigenvectors=compute_eigenvectors,
    )


def _prepare_active_data(
    data_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert input to float64 and remove constant features."""
    data_matrix = np.asarray(data_matrix, dtype=np.float64)
    feature_variances = np.var(data_matrix, axis=0)
    is_active_feature = feature_variances > 0
    active_data = data_matrix[:, is_active_feature]
    return active_data, is_active_feature


def _should_use_dual(*, n_samples: int, n_active_features: int) -> bool:
    """Return whether the dual Gram representation is smaller."""
    return n_samples < n_active_features


def _eigendecompose_dual(
    active_data: np.ndarray,
    *,
    is_active_feature: np.ndarray,
    compute_eigenvectors: bool,
) -> EigenResult:
    """Run dual eigendecomposition and package the result."""
    standardized_active_data = standardize_active_data(active_data)
    gram_dual = build_dual_gram_matrix(standardized_active_data)
    eigenvalues, dual_sample_eigenvectors = _decompose_and_sort(
        gram_dual,
        compute_eigenvectors=compute_eigenvectors,
    )

    return EigenResult(
        eigenvalues=eigenvalues,
        is_active_feature=is_active_feature,
        active_feature_count=int(np.sum(is_active_feature)),
        use_dual=True,
        dual_sample_eigenvectors=dual_sample_eigenvectors,
        standardized_data_active=standardized_active_data,
    )


def _eigendecompose_primal(
    active_data: np.ndarray,
    *,
    is_active_feature: np.ndarray,
    compute_eigenvectors: bool,
) -> EigenResult:
    """Run primal eigendecomposition and package the result."""
    correlation_matrix = build_primal_correlation_matrix(active_data)
    eigenvalues, eigenvectors_active = _decompose_and_sort(
        correlation_matrix,
        compute_eigenvectors=compute_eigenvectors,
    )

    return EigenResult(
        eigenvalues=eigenvalues,
        is_active_feature=is_active_feature,
        active_feature_count=int(np.sum(is_active_feature)),
        use_dual=False,
        eigenvectors_active=eigenvectors_active,
    )


def _decompose_and_sort(
    matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run symmetric eigendecomposition and return descending, floored outputs."""
    if compute_eigenvectors:
        eigenvalues, eigenvectors = linalg.eigh(matrix, check_finite=False)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
    else:
        eigenvalues = np.sort(linalg.eigvalsh(matrix, check_finite=False))[::-1]
        eigenvectors = None

    return np.maximum(eigenvalues, 0.0), eigenvectors
