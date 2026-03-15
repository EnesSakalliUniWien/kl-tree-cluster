"""Wald chi-square kernel for sibling divergence testing.

Tests whether two sibling distributions differ significantly using a
projected Wald chi-square statistic calibrated against a χ²(k) reference law.

For binary data, z is the standardised proportion difference.
For categorical data, z is first covariance-whitened via a multinomial
Mahalanobis construction (drop-last basis).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from kl_clustering_analysis import config

from ....decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_projection_dimension,
)
from ....decomposition.backends.random_projection_backend import (
    derive_projection_seed_backend as derive_projection_seed,
)
from ....decomposition.methods.projected_wald import run_projected_wald_kernel
from ...branch_length_utils import sanitize_positive_branch_length
from ...categorical_mahalanobis import categorical_whitened_vector
from .pooled_variance import _is_categorical, standardize_proportion_difference

logger = logging.getLogger(__name__)


def sibling_divergence_test(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    n_left: float,
    n_right: float,
    branch_length_left: float | None = None,
    branch_length_right: float | None = None,
    mean_branch_length: float | None = None,
    *,
    test_id: str | None = None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    minimum_projection_dimension: int | None = None,
) -> Tuple[float, float, float]:
    """Two-sample Wald test for sibling divergence.

    Parameters
    ----------
    left_distribution, right_distribution : np.ndarray
        Distributions of left and right siblings.
    n_left, n_right : float
        Sample sizes.
    branch_length_left, branch_length_right : float, optional
        Branch lengths (distance to parent) for each sibling.
    mean_branch_length : float, optional
        Mean branch length across the tree, used only by the optional
        branch-length variance hook in the standardization step.

    Returns
    -------
    Tuple[float, float, float]
        (test_statistic, degrees_of_freedom, p_value).
    """
    # Compute branch-length sum for the optional variance adjustment hook.
    # When mean_branch_length is None (disabled via config),
    # skip branch-length computation entirely to avoid triggering the
    # ValueError in standardize_proportion_difference().
    branch_length_sum = None
    if mean_branch_length is not None:
        sanitized_left = sanitize_positive_branch_length(branch_length_left)
        sanitized_right = sanitize_positive_branch_length(branch_length_right)
        if sanitized_left is not None and sanitized_right is not None:
            branch_length_sum = sanitized_left + sanitized_right
            if branch_length_sum <= 0:
                logger.warning(
                    "Non-positive sibling branch length sum encountered "
                    "(left=%s, right=%s). Disabling branch-length variance adjustment "
                    "for this test.",
                    sanitized_left,
                    sanitized_right,
                )
                branch_length_sum = None

    if _is_categorical(np.asarray(left_distribution)):
        z_scores = categorical_whitened_vector(
            np.asarray(left_distribution, dtype=np.float64),
            np.asarray(right_distribution, dtype=np.float64),
            float(n_left),
            float(n_right),
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )
    else:
        z_scores, _ = standardize_proportion_difference(
            left_distribution,
            right_distribution,
            n_left,
            n_right,
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )

    # Explicit invalid-test path: never coerce non-finite z-scores.
    if not np.isfinite(z_scores).all():
        logger.warning(
            "Found %d non-finite z-scores in sibling test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(~np.isfinite(z_scores))),
        )
        return np.nan, np.nan, np.nan
    z_scores = z_scores.astype(np.float64, copy=False)

    # Use n_left + n_right (total observations) for the fallback JL cap.
    total_sample_size = int(n_left + n_right)

    # Project and compute test statistic
    if test_id is None:
        test_id = (
            f"sibling:shapeL={tuple(np.shape(left_distribution))}:"
            f"shapeR={tuple(np.shape(right_distribution))}:"
            f"nL={float(n_left):.6g}:nR={float(n_right):.6g}"
        )
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)

    test_statistic, _k_nominal, effective_df, p_value = run_projected_wald_kernel(
        z_scores,
        seed=test_seed,
        spectral_k=spectral_k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        k_fallback=lambda dim: compute_projection_dimension(
            total_sample_size, dim, minimum_projection_dimension=minimum_projection_dimension
        ),
    )

    return test_statistic, effective_df, p_value


__all__ = ["sibling_divergence_test"]
