"""Top-level sibling Wald test orchestration."""

from __future__ import annotations

import logging

import numpy as np

from kl_clustering_analysis import config

from .....decomposition.backends.random_projection.seed import derive_projection_seed
from ....projection.projected_wald.projected_wald_kernel import run_projected_wald_kernel
from ...projection.pair_testing.projection_dimension import (
    resolve_sibling_projection_dimension,
)
from .branch_length import _resolve_sibling_branch_length_sum
from .sibling_z_scores import _compute_sibling_z_scores

logger = logging.getLogger(__name__)


def sibling_divergence_test(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    branch_length_left: float | None = None,
    branch_length_right: float | None = None,
    mean_branch_length: float | None = None,
    *,
    test_id: str | None = None,
    projection_dimension_from_edge_comparisons: int | None = None,
    parent_principal_component_projection: np.ndarray | None = None,
    parent_principal_component_eigenvalues: np.ndarray | None = None,
    projection_diagnostics: dict[str, object] | None = None,
) -> tuple[float, float, float]:
    """Two-sample Wald test for sibling divergence."""
    branch_length_sum = _resolve_sibling_branch_length_sum(
        branch_length_left,
        branch_length_right,
        mean_branch_length,
    )

    z_scores = _compute_sibling_z_scores(
        left_distribution,
        right_distribution,
        left_sample_size,
        right_sample_size,
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
    )

    n_features = int(z_scores.shape[0])
    try:
        (
            resolved_projection_dimension,
            projection_dimension_source,
        ) = resolve_sibling_projection_dimension(
            projection_dimension_from_edge_comparisons=projection_dimension_from_edge_comparisons,
            left_sample_size=left_sample_size,
            right_sample_size=right_sample_size,
            n_features=n_features,
        )
    except ValueError:
        if not np.isfinite(z_scores).all():
            logger.warning(
                "Found %d non-finite z-scores in sibling test; marking test invalid "
                "(raw outputs NaN, conservative p=1.0 for correction).",
                int(np.sum(~np.isfinite(z_scores))),
            )
            return np.nan, np.nan, np.nan
        raise

    if projection_diagnostics is not None:
        projection_diagnostics["source"] = projection_dimension_source
        projection_diagnostics["resolved_projection_dimension"] = int(
            resolved_projection_dimension
        )

    if not np.isfinite(z_scores).all():
        logger.warning(
            "Found %d non-finite z-scores in sibling test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(~np.isfinite(z_scores))),
        )
        return np.nan, np.nan, np.nan

    z_scores = z_scores.astype(np.float64, copy=False)

    if test_id is None:
        test_id = (
            f"sibling:shapeL={tuple(np.shape(left_distribution))}:"
            f"shapeR={tuple(np.shape(right_distribution))}:"
            f"leftN={float(left_sample_size):.6g}:rightN={float(right_sample_size):.6g}"
        )

    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)

    test_statistic, _k_nominal, effective_df, p_value = run_projected_wald_kernel(
        z_scores,
        seed=test_seed,
        spectral_k=resolved_projection_dimension,
        pca_projection=parent_principal_component_projection,
        pca_eigenvalues=parent_principal_component_eigenvalues,
    )

    return test_statistic, effective_df, p_value


__all__ = ["sibling_divergence_test"]
