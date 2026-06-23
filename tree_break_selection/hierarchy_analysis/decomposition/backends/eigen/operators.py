"""Matrix-building helpers for eigen decomposition."""

from __future__ import annotations

import numpy as np


def center_active_data(active_data: np.ndarray) -> np.ndarray:
    """Center active features while preserving their null-whitened scale."""
    return active_data - active_data.mean(axis=0)


def build_dual_covariance_gram_matrix(centered_active_data: np.ndarray) -> np.ndarray:
    """Build the sample-space matrix with feature-covariance eigenvalue scale."""
    n_samples = centered_active_data.shape[0]
    return centered_active_data @ centered_active_data.T / n_samples


def build_primal_covariance_matrix(active_data: np.ndarray) -> np.ndarray:
    """Build the feature-space covariance matrix in the original coordinate scale."""
    if active_data.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64)
    centered_active_data = center_active_data(active_data)
    return centered_active_data.T @ centered_active_data / active_data.shape[0]


__all__ = [
    "build_dual_covariance_gram_matrix",
    "build_primal_covariance_matrix",
    "center_active_data",
]
