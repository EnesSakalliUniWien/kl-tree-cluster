"""Projected Wald math for child-parent divergence tests."""

from __future__ import annotations

import logging

import numpy as np

from ...decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_projection_dimension,
)
from ...decomposition.methods.projected_wald import run_projected_wald_kernel

logger = logging.getLogger(__name__)


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


def run_child_parent_projected_wald_test(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    seed: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    minimum_projection_dimension: int | None = None,
) -> tuple[float, float, float, bool]:
    """Compute projected Wald test for one child-parent edge."""
    standardized_z_scores = compute_child_parent_standardized_z_scores(
        child_dist,
        parent_dist,
        n_child,
        n_parent,
        branch_length,
        mean_branch_length,
    )

    standardized_z_scores = standardized_z_scores.astype(np.float64, copy=False)

    test_statistic, projection_dim, _effective_degrees_of_freedom, p_value = (
        run_projected_wald_kernel(
            standardized_z_scores,
            seed=seed,
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=pca_eigenvalues,
            k_fallback=lambda dim: compute_projection_dimension(
                n_child,
                dim,
                minimum_projection_dimension=minimum_projection_dimension,
            ),
        )
    )

    return test_statistic, float(projection_dim), p_value, False


__all__ = [
    "compute_child_parent_standardized_z_scores",
    "run_child_parent_projected_wald_test",
]
