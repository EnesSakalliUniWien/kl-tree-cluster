"""Standardization for child-parent projected Wald edge tests."""

from __future__ import annotations

import numpy as np


def compute_child_parent_standardized_z_scores(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
) -> np.ndarray:
    """Compute standardized z-scores for child vs parent."""
    nested_factor = 1.0 / n_child - 1.0 / n_parent
    if nested_factor <= 0:
        raise ValueError(
            f"Invalid tree structure: child sample size ({n_child}) must be strictly "
            f"less than parent sample size ({n_parent}). Got nested_factor={nested_factor:.6f}. "
            f"This indicates a degenerate or incorrectly constructed tree."
        )

    variance = parent_dist * (1 - parent_dist) * nested_factor

    if (
        branch_length is not None
        and np.isfinite(branch_length)
        and branch_length > 0
        and mean_branch_length is not None
        and np.isfinite(mean_branch_length)
        and mean_branch_length > 0
    ):
        normalized_branch_length_multiplier = 1.0 + branch_length / mean_branch_length
        variance = variance * normalized_branch_length_multiplier

    variance = np.maximum(variance, 1e-10)
    z_scores = (child_dist - parent_dist) / np.sqrt(variance)
    return z_scores.ravel()


__all__ = ["compute_child_parent_standardized_z_scores"]
