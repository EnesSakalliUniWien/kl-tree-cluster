"""Edge-level projected Wald test wrapper for child-parent divergence."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from time import perf_counter

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
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
    stage_timings: MutableMapping[str, float] | None = None,
) -> tuple[float, float, float, bool]:
    """Compute projected Wald test for one child-parent edge."""
    contrast_start_sec = perf_counter()
    standardized_z_scores = compute_child_parent_standardized_z_scores(
        child_dist,
        parent_dist,
        n_child,
        n_parent,
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
    )
    if stage_timings is not None:
        stage_timings["edge_gate_contrast_covariance_sec"] = float(
            stage_timings.get("edge_gate_contrast_covariance_sec", 0.0)
        ) + float(perf_counter() - contrast_start_sec)

    standardized_z_scores = standardized_z_scores.astype(np.float64, copy=False)

    result = run_projected_wald_kernel(
        standardized_z_scores,
        spectral_k=spectral_k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        stage_timings=stage_timings,
        timing_prefix="edge_gate",
    )

    return result.statistic, float(result.degrees_of_freedom), result.p_value, False


__all__ = [
    "run_child_parent_projected_wald_test",
]
