"""Standardized sibling contrasts derived from pooled variance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    felsenstein_sibling_multiplier,
)

from .categorical_shape import _flatten_categorical
from .pooled_variance_estimator import compute_pooled_variance


def standardize_proportion_difference(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
    branch_length_sum: float | None = None,
    mean_branch_length: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute standardized difference (z-scores) between two proportions."""
    variance = compute_pooled_variance(theta_1, theta_2, n_1, n_2, eps)

    if branch_length_sum is not None and branch_length_sum > 0:
        variance = variance * felsenstein_sibling_multiplier(branch_length_sum, mean_branch_length)

    difference = theta_1 - theta_2
    variance_flat = _flatten_categorical(variance)
    difference_flat = _flatten_categorical(difference)

    z_scores = difference_flat / np.sqrt(variance_flat)
    return z_scores, variance_flat


__all__ = ["standardize_proportion_difference"]
