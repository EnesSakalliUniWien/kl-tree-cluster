"""Input preparation for local feature-correlation eigendecomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedCorrelationData:
    """Dense local data with constant features removed."""

    active_data: np.ndarray
    is_active_feature: np.ndarray
    n_samples: int
    n_features: int
    n_active_features: int


def prepare_correlation_data(data_matrix: np.ndarray) -> PreparedCorrelationData:
    """Convert input to float64 and expose the active-feature submatrix."""
    feature_matrix = np.asarray(data_matrix, dtype=np.float64)
    if feature_matrix.ndim != 2:
        raise ValueError("Correlation eigendecomposition requires a 2D data matrix.")

    n_samples, n_features = feature_matrix.shape
    if n_samples == 0 or n_features == 0:
        is_active_feature = np.zeros(n_features, dtype=bool)
    else:
        feature_variances = np.var(feature_matrix, axis=0)
        is_active_feature = np.isfinite(feature_variances) & (feature_variances > 0)

    active_data = feature_matrix[:, is_active_feature]
    return PreparedCorrelationData(
        active_data=active_data,
        is_active_feature=is_active_feature,
        n_samples=n_samples,
        n_features=n_features,
        n_active_features=int(np.count_nonzero(is_active_feature)),
    )


__all__ = ["PreparedCorrelationData", "prepare_correlation_data"]
