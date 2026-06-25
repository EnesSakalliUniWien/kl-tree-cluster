"""Top-level sibling Wald test orchestration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from tree_break_selection.tree.feature_space import FeatureSpace

from ....projection.projected_wald.projected_wald_kernel import run_projected_wald_kernel
from ...projection.pair_testing.projection_dimension import (
    resolve_sibling_projection_dimension,
)
from .sibling_z_scores import _compute_sibling_z_scores


def sibling_divergence_test(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    *,
    branch_length_sum: float | None = None,
    mean_branch_length: float | None = None,
    projection_dimension_from_edge_comparisons: int | None = None,
    parent_principal_component_projection: np.ndarray | None = None,
    parent_principal_component_eigenvalues: np.ndarray | None = None,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
    adaptive_projection_dimension_energy_fraction: float | None = None,
) -> tuple[float, float, float, float]:
    """Two-sample Wald test for sibling divergence."""
    z_scores = _compute_sibling_z_scores(
        left_distribution,
        right_distribution,
        left_sample_size,
        right_sample_size,
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
    )

    n_features = int(z_scores.shape[0])
    if not np.isfinite(z_scores).all():
        raise ValueError(
            "Sibling Wald z-scores must be finite; "
            f"found {int(np.sum(~np.isfinite(z_scores)))} non-finite component(s)."
        )

    sibling_projection_dimension, _ = resolve_sibling_projection_dimension(
        projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
        left_sample_size=left_sample_size,
        right_sample_size=right_sample_size,
        n_features=n_features,
    )

    z_scores = z_scores.astype(np.float64, copy=False)

    result = run_projected_wald_kernel(
        z_scores,
        spectral_k=sibling_projection_dimension,
        pca_projection=parent_principal_component_projection,
        pca_eigenvalues=parent_principal_component_eigenvalues,
        adaptive_dimension_energy_fraction=(adaptive_projection_dimension_energy_fraction),
    )

    return (
        result.statistic,
        result.reference_scale,
        result.degrees_of_freedom,
        result.p_value,
    )


__all__ = ["sibling_divergence_test"]
