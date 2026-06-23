"""Continuous-feature tree distances.

These helpers build continuous tree-construction distances from the explicit
continuous ``FeatureSpace`` contract. ``mahalanobis_time`` is a Brownian-time
diagnostic distance for validated covariance regimes. ``standardized_euclidean``
is a location-geometry distance for mean-shift continuous benchmarks where
pooled full-covariance whitening can remove the signal used to build the tree.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy import linalg
from scipy.spatial.distance import pdist

from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    validate_feature_matrix,
)

CONTINUOUS_TREE_DISTANCE_METRIC = "mahalanobis_time"
CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC = "standardized_euclidean"


def estimate_continuous_covariance_by_block(
    data_matrix: npt.NDArray[np.floating],
    feature_space: FeatureSpace,
    *,
    shrinkage: float = 0.05,
    ridge: float = 1e-8,
) -> dict[str, npt.NDArray[np.float64]]:
    """Estimate shrinkage covariance blocks for continuous feature data.

    Each block uses

        Sigma_hat = (1 - lambda) S + lambda * tau I + epsilon * tau I

    where ``S`` is the unbiased empirical covariance and
    ``tau = trace(S) / d``. The small ridge is scale-relative and only ensures
    Cholesky stability for low-rank or nearly constant blocks.
    """
    matrix = validate_feature_matrix(
        np.asarray(data_matrix, dtype=np.float64),
        feature_space,
        value_name="continuous_tree_distance_data",
    )
    if not feature_space.has_continuous_blocks:
        raise ValueError("Continuous covariance estimation requires continuous blocks.")
    if matrix.shape[0] < 2:
        raise ValueError("At least two rows are required to estimate continuous covariance.")
    shrinkage_value = float(shrinkage)
    if not np.isfinite(shrinkage_value) or not 0.0 <= shrinkage_value <= 1.0:
        raise ValueError(f"shrinkage must lie in [0, 1]; got {shrinkage!r}.")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError(f"ridge must be finite and non-negative; got {ridge!r}.")

    covariance_by_block: dict[str, npt.NDArray[np.float64]] = {}
    for block in feature_space.continuous_blocks:
        block_values = matrix[:, list(block.column_indices)]
        centered = block_values - np.mean(block_values, axis=0, keepdims=True)
        empirical = centered.T @ centered / float(matrix.shape[0] - 1)
        empirical = np.atleast_2d(np.asarray(empirical, dtype=np.float64))
        dimension = int(block.raw_dimension)
        target_scale = float(np.trace(empirical) / max(dimension, 1))
        if target_scale <= 0.0 or not np.isfinite(target_scale):
            target_scale = 1.0
        identity = np.eye(dimension, dtype=np.float64)
        covariance = (
            (1.0 - shrinkage_value) * empirical
            + shrinkage_value * target_scale * identity
            + ridge_value * target_scale * identity
        )
        covariance_by_block[block.name] = covariance
    return covariance_by_block


def continuous_time_distance_condensed(
    data_matrix: npt.NDArray[np.floating],
    feature_space: FeatureSpace,
    *,
    covariance_by_block: Mapping[str, npt.NDArray[np.floating]] | None = None,
    shrinkage: float = 0.05,
    ridge: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """Return pairwise Brownian-time distances for continuous data.

    The output distance between rows ``a`` and ``b`` is

        (x_a - x_b)^T Sigma^{-1} (x_a - x_b) / d,

    accumulated across continuous blocks. This is a time-like dissimilarity:
    under a Gaussian diffusion with covariance ``Sigma`` per unit time, its
    expectation is proportional to elapsed divergence time.
    """
    matrix = validate_feature_matrix(
        np.asarray(data_matrix, dtype=np.float64),
        feature_space,
        value_name="continuous_tree_distance_data",
    )
    if feature_space.family_label != "continuous":
        raise ValueError(
            "continuous_time_distance_condensed currently requires a pure continuous "
            f"feature space; got {feature_space.family_label!r}."
        )
    covariances = (
        estimate_continuous_covariance_by_block(
            matrix,
            feature_space,
            shrinkage=shrinkage,
            ridge=ridge,
        )
        if covariance_by_block is None
        else {
            name: np.asarray(covariance, dtype=np.float64)
            for name, covariance in covariance_by_block.items()
        }
    )
    expected_names = {block.name for block in feature_space.continuous_blocks}
    if set(covariances) != expected_names:
        raise ValueError(
            "covariance_by_block keys must match continuous feature blocks. "
            f"expected={sorted(expected_names)!r}, actual={sorted(covariances)!r}."
        )

    whitened_blocks: list[npt.NDArray[np.float64]] = []
    for block in feature_space.continuous_blocks:
        block_values = matrix[:, list(block.column_indices)]
        centered = block_values - np.mean(block_values, axis=0, keepdims=True)
        covariance = np.asarray(covariances[block.name], dtype=np.float64)
        expected_shape = (block.raw_dimension, block.raw_dimension)
        if covariance.shape != expected_shape:
            raise ValueError(
                f"Covariance block {block.name!r} has shape {covariance.shape}; "
                f"expected {expected_shape}."
            )
        cho, lower = linalg.cho_factor(covariance, lower=True, check_finite=False)
        whitened = linalg.solve_triangular(
            cho,
            centered.T,
            lower=lower,
            check_finite=False,
        ).T
        whitened_blocks.append(whitened)

    whitened_matrix = np.column_stack(whitened_blocks)
    squared_mahalanobis = pdist(whitened_matrix, metric="sqeuclidean")
    return squared_mahalanobis / float(feature_space.contrast_dimension)


def standardized_euclidean_distance_condensed(
    data_matrix: npt.NDArray[np.floating],
    feature_space: FeatureSpace,
) -> npt.NDArray[np.float64]:
    """Return pairwise Euclidean distances after coordinate standardization.

    The output distance between rows ``a`` and ``b`` is

        ||D^{-1/2}(x_a - x_b)||_2,

    where ``D`` is the diagonal matrix of empirical coordinate variances.
    Constant coordinates receive unit scale so they contribute zero distance
    without making the transformation singular.
    """
    matrix = validate_feature_matrix(
        np.asarray(data_matrix, dtype=np.float64),
        feature_space,
        value_name="continuous_tree_distance_data",
    )
    if feature_space.family_label != "continuous":
        raise ValueError(
            "standardized_euclidean_distance_condensed currently requires a pure "
            f"continuous feature space; got {feature_space.family_label!r}."
        )
    if matrix.shape[0] < 2:
        raise ValueError("At least two rows are required to standardize continuous data.")

    standardized_blocks: list[npt.NDArray[np.float64]] = []
    for block in feature_space.continuous_blocks:
        block_values = matrix[:, list(block.column_indices)]
        centered = block_values - np.mean(block_values, axis=0, keepdims=True)
        scale = np.std(block_values, axis=0, ddof=1, keepdims=True)
        scale = np.asarray(scale, dtype=np.float64)
        scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
        standardized_blocks.append(centered / scale)

    standardized_matrix = np.column_stack(standardized_blocks)
    return pdist(standardized_matrix, metric="euclidean")


__all__ = [
    "CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC",
    "CONTINUOUS_TREE_DISTANCE_METRIC",
    "continuous_time_distance_condensed",
    "estimate_continuous_covariance_by_block",
    "standardized_euclidean_distance_condensed",
]
