"""Edge-level projected Wald test wrapper for child-parent divergence."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from kl_clustering_analysis.tree.feature_space import FeatureSpace

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
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
) -> tuple[float, float, float, bool]:
    """Compute projected Wald test for one child-parent edge."""
    standardized_z_scores = compute_child_parent_standardized_z_scores(
        child_dist,
        parent_dist,
        n_child,
        n_parent,
        branch_length,
        mean_branch_length,
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
    )

    standardized_z_scores = standardized_z_scores.astype(np.float64, copy=False)

    result = run_projected_wald_kernel(
        standardized_z_scores,
        spectral_k=spectral_k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
    )

    return result.statistic, float(result.degrees_of_freedom), result.p_value, False


__all__ = [
    "run_child_parent_projected_wald_test",
]
