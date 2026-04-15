"""Global JL floor estimation from leaf data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ....statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    effective_rank,
)


@dataclass(frozen=True)
class _PreparedActiveFeatureMatrix:
    """Prepared leaf data for global spectrum estimation.

    The original input is a leaf-by-feature matrix of shape
    ``(n_samples, n_features)``. Constant features are removed before
    spectrum estimation, yielding ``active_feature_matrix`` with shape
    ``(n_samples, n_active_features)``.

    Example
    -------
    Original matrix::

        [[1, 0, 1, 0],
         [1, 0, 1, 1],
         [0, 1, 1, 0]]

    If the third column is constant, the active matrix becomes::

        [[1, 0, 0],
         [1, 0, 1],
         [0, 1, 0]]
    """

    active_feature_matrix: np.ndarray
    n_samples: int
    n_features: int
    n_active_features: int


def estimate_projection_dimension_floor(
    leaf_feature_matrix: pd.DataFrame,
    *,
    minimum_dimension_floor: int = 2,
    maximum_dimension_cap: int = 20,
) -> int:
    """Estimate a global lower bound for JL projection dimension.

    Parameters
    ----------
    leaf_feature_matrix
        Leaf-by-feature matrix of shape ``(n_samples, n_features)``.

        Example
        -------
        ``leafA``, ``leafB``, and ``leafC`` measured over four features::

            index   f1  f2  f3  f4
            leafA    1   0   1   0
            leafB    1   0   1   1
            leafC    0   1   1   0

    Returns
    -------
    int
        Dataset-level minimum projection dimension used as the floor for the
        JL fallback path.

    Notes
    -----
    This function does not build a projection matrix. It only returns a
    single global floor derived from the effective rank of the active feature
    spectrum. The conceptual steps are:

    1. Drop constant features.
    2. Compute an eigenvalue spectrum from the active data.
    3. Convert that spectrum to an effective-rank floor.
    4. Clamp the result to the configured minimum and maximum bounds.
    """
    prepared_matrix = _prepare_active_feature_matrix(leaf_feature_matrix)

    if prepared_matrix.n_samples < 2 or prepared_matrix.n_features < 2:
        return minimum_dimension_floor

    if prepared_matrix.n_active_features < 2:
        return minimum_dimension_floor

    spectrum_eigenvalues = _compute_spectrum_eigenvalues(prepared_matrix)
    estimated_projection_dimension_floor = _estimate_effective_rank_floor(spectrum_eigenvalues)
    return _clamp_projection_dimension_floor(
        estimated_projection_dimension_floor,
        minimum_dimension_floor=minimum_dimension_floor,
        maximum_dimension_cap=maximum_dimension_cap,
    )


def _prepare_active_feature_matrix(
    leaf_feature_matrix: pd.DataFrame,
) -> _PreparedActiveFeatureMatrix:
    """Prepare the active leaf-by-feature matrix for spectrum estimation.

    Parameters
    ----------
    leaf_feature_matrix
        Dense leaf-by-feature matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    _PreparedActiveFeatureMatrix
        Float64 matrix with constant columns removed, plus shape metadata for
        the original and active representations.
    """
    leaf_feature_values = leaf_feature_matrix.values.astype(np.float64)
    n_samples, n_features = leaf_feature_values.shape
    feature_variances = np.var(leaf_feature_values, axis=0)
    nonconstant_feature_mask = feature_variances > 0
    active_feature_matrix = leaf_feature_values[:, nonconstant_feature_mask]
    return _PreparedActiveFeatureMatrix(
        active_feature_matrix=active_feature_matrix,
        n_samples=n_samples,
        n_features=n_features,
        n_active_features=int(np.sum(nonconstant_feature_mask)),
    )


def _compute_spectrum_eigenvalues(
    prepared_matrix: _PreparedActiveFeatureMatrix,
) -> np.ndarray:
    """Compute the spectrum used for effective-rank floor estimation.

    Given an active matrix ``X`` of shape ``(n_samples, n_active_features)``,
    this function chooses the numerically smaller of two equivalent
    second-order representations:

    - sample-space Gram matrix with shape ``(n_samples, n_samples)``
    - feature-space correlation matrix with shape
      ``(n_active_features, n_active_features)``
    """
    if prepared_matrix.n_samples < prepared_matrix.n_active_features:
        return _compute_sample_space_spectrum(prepared_matrix.active_feature_matrix)
    return _compute_feature_space_spectrum(prepared_matrix.active_feature_matrix)


def _compute_sample_space_spectrum(
    active_feature_matrix: np.ndarray,
) -> np.ndarray:
    """Compute eigenvalues from the standardized sample-space Gram matrix.

    This path is used when ``n_samples < n_active_features`` so that the
    eigendecomposition happens on a smaller matrix of shape
    ``(n_samples, n_samples)`` instead of ``(n_active_features, n_active_features)``.
    """
    feature_means = active_feature_matrix.mean(axis=0)
    feature_standard_deviations = active_feature_matrix.std(axis=0, ddof=0)
    feature_standard_deviations[feature_standard_deviations == 0] = 1.0
    standardized_feature_matrix = (
        active_feature_matrix - feature_means
    ) / feature_standard_deviations
    gram_matrix = standardized_feature_matrix @ standardized_feature_matrix.T
    gram_matrix /= active_feature_matrix.shape[1]
    return np.sort(np.linalg.eigvalsh(gram_matrix))[::-1]


def _compute_feature_space_spectrum(
    active_feature_matrix: np.ndarray,
) -> np.ndarray:
    """Compute eigenvalues from the feature-space correlation matrix.

    This path is used when ``n_samples >= n_active_features`` so the
    correlation matrix over features is the simpler representation.
    """
    correlation_matrix = np.corrcoef(active_feature_matrix.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    return np.sort(np.linalg.eigvalsh(correlation_matrix))[::-1]


def _estimate_effective_rank_floor(
    spectrum_eigenvalues: np.ndarray,
) -> int:
    """Convert a spectrum into an unclamped integer floor.

    Example
    -------
    For eigenvalues ``[3.9, 1.4, 0.5, 0.2]``, the effective rank might be
    ``2.37`` and the returned floor would be ``ceil(2.37) == 3``.
    """
    return int(np.ceil(effective_rank(np.maximum(spectrum_eigenvalues, 0.0))))


def _clamp_projection_dimension_floor(
    estimated_floor: int,
    *,
    minimum_dimension_floor: int,
    maximum_dimension_cap: int,
) -> int:
    """Clamp a raw projection floor to the configured minimum and cap."""
    return min(max(int(estimated_floor), minimum_dimension_floor), maximum_dimension_cap)
