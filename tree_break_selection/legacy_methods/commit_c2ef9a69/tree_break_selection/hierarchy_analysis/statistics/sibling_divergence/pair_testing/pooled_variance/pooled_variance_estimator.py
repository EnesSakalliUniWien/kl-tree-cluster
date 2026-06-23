"""Diagonal pooled-variance estimates for sibling proportion tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .pooled_proportion import compute_pooled_proportion


def compute_pooled_variance(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute the variance of the difference between two proportions."""
    pooled = compute_pooled_proportion(theta_1, theta_2, n_1, n_2, eps)
    variance = pooled * (1.0 - pooled) * (1.0 / n_1 + 1.0 / n_2)
    return np.maximum(variance, eps)


__all__ = ["compute_pooled_variance"]
