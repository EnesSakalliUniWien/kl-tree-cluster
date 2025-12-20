"""Mutual Information-based feature selection for sibling divergence test.

Computes per-feature MI between feature values and sibling assignment,
then filters out low-information features before statistical testing.

This reduces noise and focuses the test on discriminative features.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


def binary_entropy(theta: np.ndarray) -> np.ndarray:
    """Compute entropy of Bernoulli distribution.

    H(θ) = -θ log(θ) - (1-θ) log(1-θ)

    Parameters
    ----------
    theta : np.ndarray
        Bernoulli parameters (probabilities), shape (d,).

    Returns
    -------
    np.ndarray
        Entropy values, shape (d,).
    """
    # Clip to avoid log(0)
    theta = np.clip(theta, 1e-10, 1 - 1e-10)
    return -theta * np.log(theta) - (1 - theta) * np.log(1 - theta)


def compute_feature_mi(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    n_left: float,
    n_right: float,
) -> np.ndarray:
    """Compute MI between each feature and sibling assignment.

    MI(X_j; S) = H(X_j) - H(X_j | S)

    where S ∈ {left, right} is the sibling indicator.

    Parameters
    ----------
    theta_left : np.ndarray
        Feature means for left sibling, shape (d,).
    theta_right : np.ndarray
        Feature means for right sibling, shape (d,).
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.

    Returns
    -------
    np.ndarray
        Per-feature MI values, shape (d,). Always non-negative.
    """
    # Sibling proportions
    n_total = n_left + n_right
    p_left = n_left / n_total
    p_right = n_right / n_total

    # Pooled distribution (marginal over S)
    theta_pooled = p_left * theta_left + p_right * theta_right

    # Marginal entropy H(X_j)
    H_marginal = binary_entropy(theta_pooled)

    # Conditional entropies H(X_j | S)
    H_left = binary_entropy(theta_left)
    H_right = binary_entropy(theta_right)
    H_conditional = p_left * H_left + p_right * H_right

    # MI = H(X) - H(X|S)
    mi = H_marginal - H_conditional

    # Ensure non-negative (numerical precision)
    return np.maximum(mi, 0.0)


def select_informative_features(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    n_left: float,
    n_right: float,
    min_fraction: float = 0.1,
    quantile_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Select features with high MI for sibling discrimination.

    Uses two criteria:
    1. Keep features with MI > median(MI) of non-zero MI features
    2. Always keep at least min_fraction of features

    Parameters
    ----------
    theta_left : np.ndarray
        Feature means for left sibling, shape (d,).
    theta_right : np.ndarray
        Feature means for right sibling, shape (d,).
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.
    min_fraction : float
        Minimum fraction of features to keep (default 0.1 = 10%).
    quantile_threshold : float
        Keep features above this quantile of MI (default 0.5 = median).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (informative_mask, mi_values, n_informative)
        - informative_mask: boolean mask of selected features
        - mi_values: MI for all features
        - n_informative: number of selected features
    """
    mi = compute_feature_mi(theta_left, theta_right, n_left, n_right)
    d = len(mi)

    # Compute threshold from non-zero MI features
    nonzero_mi = mi[mi > 1e-10]
    if len(nonzero_mi) > 0:
        threshold = np.quantile(nonzero_mi, quantile_threshold)
    else:
        threshold = 0.0

    # Select features above threshold
    mask = mi > threshold
    n_selected = mask.sum()

    # Ensure minimum number of features
    min_features = max(1, int(min_fraction * d))
    if n_selected < min_features:
        # Take top min_features by MI
        top_indices = np.argsort(mi)[-min_features:]
        mask = np.zeros(d, dtype=bool)
        mask[top_indices] = True
        n_selected = min_features

    return mask, mi, n_selected


def mi_weighted_projection(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    n_left: float,
    n_right: float,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Apply MI-weighted random projection.

    Weights each feature by sqrt(MI) before projection,
    so informative features contribute more.

    Parameters
    ----------
    theta_left : np.ndarray
        Feature means for left sibling.
    theta_right : np.ndarray
        Feature means for right sibling.
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.
    projection_matrix : np.ndarray
        Random projection matrix, shape (k, d).

    Returns
    -------
    np.ndarray
        Projected difference vector, shape (k,).
    """
    mi = compute_feature_mi(theta_left, theta_right, n_left, n_right)

    # Normalize MI to [0, 1] and use sqrt for variance weighting
    mi_normalized = mi / (mi.max() + 1e-10)
    weights = np.sqrt(mi_normalized + 0.01)  # Small floor to keep some contribution

    # Weight the difference
    diff = theta_left - theta_right
    weighted_diff = diff * weights

    # Project
    return projection_matrix @ weighted_diff


def filter_and_project(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    n_left: float,
    n_right: float,
    projection_matrix: np.ndarray,
    mi_threshold_quantile: float = 0.5,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Filter features by MI, then project the informative ones.

    This is the recommended hybrid approach:
    1. Remove low-MI features (noise reduction)
    2. Project remaining features (dimensionality reduction)

    Parameters
    ----------
    theta_left : np.ndarray
        Feature means for left sibling.
    theta_right : np.ndarray
        Feature means for right sibling.
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.
    projection_matrix : np.ndarray
        Random projection matrix for FILTERED features, shape (k, d_filtered).
    mi_threshold_quantile : float
        Keep features above this quantile of MI.

    Returns
    -------
    Tuple[np.ndarray, int, np.ndarray]
        (projected_diff, n_informative, informative_mask)
    """
    # Select informative features
    mask, mi, n_informative = select_informative_features(
        theta_left,
        theta_right,
        n_left,
        n_right,
        quantile_threshold=mi_threshold_quantile,
    )

    # Filter to informative features only
    diff = theta_left - theta_right
    diff_filtered = diff[mask]

    # Project filtered difference
    projected = projection_matrix @ diff_filtered

    return projected, n_informative, mask


__all__ = [
    "binary_entropy",
    "compute_feature_mi",
    "select_informative_features",
    "mi_weighted_projection",
    "filter_and_project",
]
