"""Kernel weights for branch-length-aware selected neighborhoods."""

from __future__ import annotations

import numpy as np


def structural_log_dimension_kernel(
    source_log_k: np.ndarray,
    target_log_k: float,
    h_k: float,
) -> np.ndarray:
    """Gaussian kernel over log projection dimension with exact-match zero width."""

    source = np.asarray(source_log_k, dtype=np.float64)
    bandwidth = float(h_k)
    if bandwidth <= 0.0:
        return np.where(np.isclose(source, float(target_log_k), atol=1e-12), 1.0, 0.0)
    return np.exp(-0.5 * ((source - float(target_log_k)) / bandwidth) ** 2)


def tree_exponential_kernel(distances: np.ndarray, tau: float) -> np.ndarray:
    """Exponential decay kernel over branch-length tree distances."""

    distance_array = np.asarray(distances, dtype=np.float64)
    tau_value = float(tau)
    if tau_value <= 0.0:
        return np.where(np.isclose(distance_array, 0.0, atol=1e-12), 1.0, 0.0)
    return np.exp(-distance_array / tau_value)


def selected_neighborhood_kernel_weights(
    *,
    distances: np.ndarray,
    source_log_k: np.ndarray,
    target_log_k: float,
    tau: float,
    h_k: float,
) -> np.ndarray:
    """Return the guarded legacy neighborhood kernel weights."""

    return tree_exponential_kernel(distances, tau) * structural_log_dimension_kernel(
        source_log_k,
        target_log_k,
        h_k,
    )


def effective_support(weights: np.ndarray) -> float:
    """Return Kish effective sample size for non-negative kernel weights."""

    weight_array = np.asarray(weights, dtype=np.float64)
    positive = weight_array[np.isfinite(weight_array) & (weight_array > 0.0)]
    if positive.size == 0:
        return 0.0
    squared_sum = float(np.sum(positive) ** 2)
    sum_squares = float(np.sum(positive**2))
    if sum_squares <= 0.0:
        return 0.0
    return squared_sum / sum_squares


__all__ = [
    "effective_support",
    "selected_neighborhood_kernel_weights",
    "structural_log_dimension_kernel",
    "tree_exponential_kernel",
]
