"""Matrix-building helpers for eigen decomposition."""

from __future__ import annotations

import numpy as np


def standardize_active_data(active_data: np.ndarray) -> np.ndarray:
    """Center and scale active features for dual Gram construction."""
    mu_active = active_data.mean(axis=0)
    sigma_active = active_data.std(axis=0, ddof=0)
    sigma_active[sigma_active == 0] = 1.0
    return (active_data - mu_active) / sigma_active


def build_dual_correlation_gram_matrix(standardized_active_data: np.ndarray) -> np.ndarray:
    """Build the sample-space matrix with feature-correlation eigenvalue scale."""
    n_samples = standardized_active_data.shape[0]
    return standardized_active_data @ standardized_active_data.T / n_samples


def build_primal_correlation_matrix(active_data: np.ndarray) -> np.ndarray:
    """Build the sanitized feature-space correlation matrix."""
    if active_data.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if active_data.shape[1] == 1:
        return np.ones((1, 1), dtype=np.float64)

    correlation_matrix = np.corrcoef(active_data.T)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix
