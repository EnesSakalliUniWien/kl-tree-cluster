"""Wald chi-square kernel for sibling divergence testing.

Tests whether two sibling distributions differ significantly using a
projected Wald chi-square statistic calibrated against a χ²(k) reference law.

For binary data, z is the standardised proportion difference.
For categorical data, z is first covariance-whitened via a multinomial
Mahalanobis construction (drop-last basis).
"""

from __future__ import annotations

import logging

import numpy as np

from kl_clustering_analysis import config

from ....decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_projection_dimension,
)
from ....decomposition.backends.random_projection_backend import (
    derive_projection_seed_backend as derive_projection_seed,
)
from ...branch_length_utils import sanitize_positive_branch_length
from ...categorical_mahalanobis import categorical_whitened_vector
from ...projection.chi2_pvalue import WhiteningMode
from ...projection.projected_wald import run_projected_wald_kernel
from .pooled_variance import _is_categorical, standardize_proportion_difference

logger = logging.getLogger(__name__)


def _resolve_sibling_branch_length_sum(
    branch_length_left: float | None,
    branch_length_right: float | None,
    mean_branch_length: float | None,
) -> float | None:
    """Return the sibling branch-length sum when variance adjustment is enabled."""
    if mean_branch_length is None:
        return None

    sanitized_left = sanitize_positive_branch_length(branch_length_left)
    sanitized_right = sanitize_positive_branch_length(branch_length_right)
    if sanitized_left is None or sanitized_right is None:
        return None

    return sanitized_left + sanitized_right


def _compute_sibling_z_scores(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: float,
    right_sample_size: float,
    *,
    branch_length_sum: float | None,
    mean_branch_length: float | None,
) -> np.ndarray:
    left_array = np.asarray(left_distribution)
    right_array = np.asarray(right_distribution)

    if _is_categorical(left_array):
        return categorical_whitened_vector(
            np.asarray(left_array, dtype=np.float64),
            np.asarray(right_array, dtype=np.float64),
            float(left_sample_size),
            float(right_sample_size),
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )

    z_scores, _ = standardize_proportion_difference(
        left_distribution,
        right_distribution,
        left_sample_size,
        right_sample_size,
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
    )
    return z_scores


def _resolve_sibling_projection_k(
    *,
    spectral_k: int | None,
    left_sample_size: float,
    right_sample_size: float,
    n_features: int,
) -> tuple[int, str]:
    """Resolve the projection dimension and record which path supplied it.

    Returns
    -------
    tuple[int, str]
        ``(resolved_k, source)`` where ``source`` is either ``"spectral"``
        for a caller-supplied positive dimension or ``"jl_fallback"`` when
        the sibling test has to synthesize a Johnson-Lindenstrauss dimension.
        ``spectral_k=None`` means no sibling-specific spectral dimension is
        available for this pair. That is the legitimate fallback case used for
        leaf-leaf sibling pairs and explicit no-spectral configurations.
        Non-positive explicit values are invalid caller input.
    """
    if spectral_k is None:
        total_sample_size = int(left_sample_size + right_sample_size)
        return (
            compute_projection_dimension(total_sample_size, n_features),
            "jl_fallback",
        )

    if spectral_k <= 0:
        raise ValueError(
            f"Invalid spectral_k={spectral_k}; expected None or a positive integer."
        )

    return int(spectral_k), "spectral"


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
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    whitening: WhiteningMode = "per_component",
) -> tuple[float, float, float]:
    """Two-sample Wald test for sibling divergence.

    Parameters
    ----------
    left_distribution, right_distribution : np.ndarray
        Distributions of left and right siblings.
    left_sample_size, right_sample_size : float
        Sample sizes of the left and right sibling nodes.
    branch_length_left, branch_length_right : float, optional
        Branch lengths (distance to parent) for each sibling.
    mean_branch_length : float, optional
        Mean branch length across the tree, used only by the optional
        branch-length variance hook in the standardization step.
    spectral_k : int | None, optional
        Pair-specific projection dimension. ``None`` means no sibling spectral
        override is available, so the test falls back to a JL dimension.
        Positive integers are used as-is. Non-positive explicit values are
        rejected as invalid input.

    Returns
    -------
    tuple[float, float, float]
        (test_statistic, degrees_of_freedom, p_value).
    """
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

    # Explicit invalid-test path: never coerce non-finite z-scores.
    if not np.isfinite(z_scores).all():
        logger.warning(
            "Found %d non-finite z-scores in sibling test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(~np.isfinite(z_scores))),
        )
        return np.nan, np.nan, np.nan

    z_scores = z_scores.astype(np.float64, copy=False)

    n_features = int(z_scores.shape[0])
    spectral_k, _projection_k_source = _resolve_sibling_projection_k(
        spectral_k=spectral_k,
        left_sample_size=left_sample_size,
        right_sample_size=right_sample_size,
        n_features=n_features,
    )

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
        spectral_k=spectral_k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        whitening=whitening,
    )

    return test_statistic, effective_df, p_value


__all__ = ["sibling_divergence_test"]
