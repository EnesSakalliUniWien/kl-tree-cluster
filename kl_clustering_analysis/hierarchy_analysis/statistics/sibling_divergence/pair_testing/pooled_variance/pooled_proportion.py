"""Pooled null-mean estimates for sibling proportion tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_pooled_proportion(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute the pooled proportion estimate under H0: theta_1 = theta_2."""
    n_total = n_1 + n_2
    pooled = (n_1 * theta_1 + n_2 * theta_2) / n_total
    return np.clip(pooled, eps, 1.0 - eps)


__all__ = ["compute_pooled_proportion"]
