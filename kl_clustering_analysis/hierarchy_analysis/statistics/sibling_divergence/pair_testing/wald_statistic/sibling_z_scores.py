"""Sibling contrast standardization for Wald testing."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from kl_clustering_analysis.tree.feature_space import FeatureSpace

from ....contrast_covariance import compute_whitened_wald_contrast


def _compute_sibling_z_scores(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
) -> np.ndarray:
    """Return the standardized sibling contrast vector."""
    return compute_whitened_wald_contrast(
        left_distribution,
        right_distribution,
        float(left_sample_size),
        float(right_sample_size),
        comparison="sibling",
        feature_space=feature_space,
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
        continuous_covariance_by_block=continuous_covariance_by_block,
    )


__all__ = ["_compute_sibling_z_scores"]
