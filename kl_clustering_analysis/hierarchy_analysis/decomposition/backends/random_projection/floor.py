"""Global JL floor estimation from leaf data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ....statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    effective_rank,
)
from ..eigen.decomposition import eigendecompose_correlation


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
        JL projection path.

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
    leaf_feature_values = leaf_feature_matrix.to_numpy(dtype=np.float64, copy=False)
    n_samples, n_features = leaf_feature_values.shape

    if n_samples < 2 or n_features < 2:
        return minimum_dimension_floor

    eigendecomposition_result = eigendecompose_correlation(
        leaf_feature_values,
        compute_eigenvectors=False,
    )
    if (
        eigendecomposition_result is None
        or eigendecomposition_result.active_feature_count < 2
    ):
        return minimum_dimension_floor

    spectrum_eigenvalues = eigendecomposition_result.eigenvalues
    estimated_projection_dimension_floor = _estimate_effective_rank_floor(spectrum_eigenvalues)
    return _clamp_projection_dimension_floor(
        estimated_projection_dimension_floor,
        minimum_dimension_floor=minimum_dimension_floor,
        maximum_dimension_cap=maximum_dimension_cap,
    )


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
