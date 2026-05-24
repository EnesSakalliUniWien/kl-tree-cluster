"""Spectral decomposition orchestration for local correlation structure."""

from __future__ import annotations

import numpy as np

from ...core.eigen_result import EigenResult
from .operators import (
    build_dual_correlation_gram_matrix,
    build_primal_correlation_matrix,
    standardize_active_data,
)
from .preparation import PreparedCorrelationData, prepare_correlation_data
from .representation import CorrelationRepresentation, select_correlation_representation
from .solver import decompose_symmetric_matrix


def eigendecompose_correlation(
    data_matrix: np.ndarray,
    *,
    compute_eigenvectors: bool,
) -> EigenResult | None:
    """Eigendecompose the active correlation structure in primal or dual form.

    The input is a dense matrix of shape ``(n_samples, n_features)``.
    Constant features are removed first. When ``n_samples < n_active_features``,
    the decomposition switches to the smaller dual sample-space Gram matrix;
    otherwise it decomposes the feature-space correlation matrix directly.
    Returns ``None`` only for the zero-rank case where no finite, varying
    feature remains after preparation.
    """
    prepared_data = prepare_correlation_data(data_matrix)

    if prepared_data.n_active_features == 0:
        return None

    representation = select_correlation_representation(prepared_data)
    if representation is CorrelationRepresentation.DUAL:
        return _eigendecompose_dual(
            prepared_data,
            compute_eigenvectors=compute_eigenvectors,
        )

    return _eigendecompose_primal(
        prepared_data,
        compute_eigenvectors=compute_eigenvectors,
    )


def _eigendecompose_dual(
    prepared_data: PreparedCorrelationData,
    *,
    compute_eigenvectors: bool,
) -> EigenResult:
    """Run dual eigendecomposition and package the result."""
    standardized_active_data = standardize_active_data(prepared_data.active_data)
    gram_dual = build_dual_correlation_gram_matrix(standardized_active_data)
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
        standardized_data_active=(
            standardized_active_data if dual_sample_eigenvectors is not None else None
        ),
    )


def _eigendecompose_primal(
    prepared_data: PreparedCorrelationData,
    *,
    compute_eigenvectors: bool,
) -> EigenResult:
    """Run primal eigendecomposition and package the result."""
    correlation_matrix = build_primal_correlation_matrix(prepared_data.active_data)
    eigenvalues, eigenvectors_active = decompose_symmetric_matrix(
        correlation_matrix,
        compute_eigenvectors=compute_eigenvectors,
    )

    return EigenResult(
        eigenvalues=eigenvalues,
        is_active_feature=prepared_data.is_active_feature,
        active_feature_count=prepared_data.n_active_features,
        use_dual=False,
        eigenvectors_active=eigenvectors_active,
    )
