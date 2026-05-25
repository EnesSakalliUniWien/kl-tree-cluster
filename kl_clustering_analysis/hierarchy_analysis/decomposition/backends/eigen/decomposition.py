"""Spectral decomposition orchestration for local covariance structure."""

from __future__ import annotations

import numpy as np

from ...core.eigen_result import EigenResult
from .operators import (
    build_dual_covariance_gram_matrix,
    build_primal_covariance_matrix,
    center_active_data,
)
from .preparation import PreparedCovarianceData, prepare_covariance_data
from .representation import CovarianceRepresentation, select_covariance_representation
from .solver import decompose_symmetric_matrix


def eigendecompose_covariance(
    data_matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> EigenResult | None:
    """Eigendecompose the active covariance structure in primal or dual form.

    The input is a dense matrix of shape ``(n_samples, n_features)``.
    Constant features are removed first. When ``n_samples < n_active_features``,
    the decomposition switches to the smaller dual sample-space Gram matrix;
    otherwise it decomposes the feature-space covariance matrix directly.
    Returns ``None`` only for the zero-rank case where no finite, varying
    feature remains after preparation.
    """
    prepared_data = prepare_covariance_data(data_matrix)

    if prepared_data.n_active_features == 0:
        return None

    representation = select_covariance_representation(prepared_data)
    if representation is CovarianceRepresentation.DUAL:
        return _eigendecompose_dual(
            prepared_data,
            compute_eigenvectors=compute_eigenvectors,
        )

    return _eigendecompose_primal(
        prepared_data,
        compute_eigenvectors=compute_eigenvectors,
    )


def _eigendecompose_dual(
    prepared_data: PreparedCovarianceData,
    *,
    compute_eigenvectors: bool,
) -> EigenResult:
    """Run dual eigendecomposition and package the result."""
    centered_active_data = center_active_data(prepared_data.active_data)
    gram_dual = build_dual_covariance_gram_matrix(centered_active_data)
    eigenvalues, dual_sample_eigenvectors = decompose_symmetric_matrix(
        gram_dual,
        compute_eigenvectors=compute_eigenvectors,
    )

    return EigenResult(
        eigenvalues=eigenvalues,
        is_active_feature=prepared_data.is_active_feature,
        active_feature_count=prepared_data.n_active_features,
        use_dual=True,
        dual_sample_eigenvectors=dual_sample_eigenvectors,
        centered_data_active=(
            centered_active_data if dual_sample_eigenvectors is not None else None
        ),
    )


def _eigendecompose_primal(
    prepared_data: PreparedCovarianceData,
    *,
    compute_eigenvectors: bool,
) -> EigenResult:
    """Run primal eigendecomposition and package the result."""
    covariance_matrix = build_primal_covariance_matrix(prepared_data.active_data)
    eigenvalues, eigenvectors_active = decompose_symmetric_matrix(
        covariance_matrix,
        compute_eigenvectors=compute_eigenvectors,
    )

    return EigenResult(
        eigenvalues=eigenvalues,
        is_active_feature=prepared_data.is_active_feature,
        active_feature_count=prepared_data.n_active_features,
        use_dual=False,
        eigenvectors_active=eigenvectors_active,
    )


__all__ = ["eigendecompose_covariance"]
