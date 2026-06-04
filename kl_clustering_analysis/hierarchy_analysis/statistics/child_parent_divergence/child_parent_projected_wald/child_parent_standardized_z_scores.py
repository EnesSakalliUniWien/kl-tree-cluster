"""Standardization for child-parent projected Wald edge tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from kl_clustering_analysis.tree.feature_space import FeatureSpace

from ...contrast_covariance import compute_whitened_wald_contrast


def compute_child_parent_standardized_z_scores(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    feature_space: FeatureSpace | None = None,
    continuous_covariance_by_block: Mapping[str, NDArray[np.floating]] | None = None,
) -> np.ndarray:
    """Compute standardized z-scores for child vs parent."""
    return compute_whitened_wald_contrast(
        child_dist,
        parent_dist,
        float(n_child),
        float(n_parent),
        comparison="child_parent",
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
    )


__all__ = ["compute_child_parent_standardized_z_scores"]
