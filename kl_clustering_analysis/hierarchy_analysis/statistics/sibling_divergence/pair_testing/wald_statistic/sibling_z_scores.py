"""Sibling contrast standardization for Wald testing."""

from __future__ import annotations

from importlib import import_module

import numpy as np


def _compute_sibling_z_scores(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
) -> np.ndarray:
    """Return the standardized sibling contrast vector."""
    wald_statistic_module = import_module(__package__)
    left_array = np.asarray(left_distribution)

    if wald_statistic_module._is_categorical(left_array):
        return wald_statistic_module.categorical_whitened_vector(
            np.asarray(left_distribution, dtype=np.float64),
            np.asarray(right_distribution, dtype=np.float64),
            float(left_sample_size),
            float(right_sample_size),
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )

    z_scores, _ = wald_statistic_module.standardize_proportion_difference(
        left_distribution,
        right_distribution,
        left_sample_size,
        right_sample_size,
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
    )
    return z_scores


__all__ = ["_compute_sibling_z_scores"]
