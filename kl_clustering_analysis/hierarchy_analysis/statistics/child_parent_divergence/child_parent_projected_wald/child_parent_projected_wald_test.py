"""Edge-level projected Wald test wrapper for child-parent divergence."""

from __future__ import annotations

import numpy as np

from ...projection.projected_wald.projected_wald_kernel import run_projected_wald_kernel
from .child_parent_standardized_z_scores import (
    compute_child_parent_standardized_z_scores,
)


def run_child_parent_projected_wald_test(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    branch_length: float | None = None,
    mean_branch_length: float | None = None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
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

    test_statistic, _projection_dim, effective_degrees_of_freedom, p_value = (
        run_projected_wald_kernel(
            standardized_z_scores,
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=pca_eigenvalues,
        )
    )

    return test_statistic, float(effective_degrees_of_freedom), p_value, False


__all__ = [
    "run_child_parent_projected_wald_test",
]
