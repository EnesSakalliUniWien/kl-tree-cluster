"""Mutual Information-based feature selection for sibling divergence test.

Computes per-feature MI between feature values and sibling assignment,
then filters out low-information features before statistical testing.

Supports both binary (Bernoulli) and categorical (multinomial) distributions.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.stats import entropy as scipy_entropy


def _feature_entropy(p: np.ndarray) -> np.ndarray:
    """Compute entropy for each feature.

    Automatically handles both binary and categorical distributions:
    - Binary (1D): p is shape (d,), each element is P(X=1)
    - Categorical (2D): p is shape (d, K), each row is a probability simplex

    Parameters
    ----------
    p : np.ndarray
        Distribution parameters. Shape (d,) for binary or (d, K) for categorical.

    Returns
    -------
    np.ndarray
        Entropy values in nats, shape (d,).
    """
    p = np.asarray(p)

    if p.ndim == 1:
        # Binary: expand to [p, 1-p] and compute entropy
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        p_binary = np.stack([p_clipped, 1 - p_clipped], axis=-1)
        return scipy_entropy(p_binary, axis=-1)
    else:
        # Categorical: compute entropy along class dimension
        p_clipped = np.clip(p, 1e-10, 1.0)
        return scipy_entropy(p_clipped, axis=-1)


def _compute_feature_mi(
    p_left: np.ndarray,
    p_right: np.ndarray,
    n_left: float,
    n_right: float,
) -> np.ndarray:
    """Compute MI between each feature and sibling assignment.

    MI(X_j; S) = H(X_j) - H(X_j | S)

    where S ∈ {left, right} is the sibling indicator.

    Supports both binary and categorical distributions.

    Parameters
    ----------
    p_left : np.ndarray
        Distribution for left sibling. Shape (d,) for binary or (d, K) for categorical.
    p_right : np.ndarray
        Distribution for right sibling. Same shape as p_left.
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.

    Returns
    -------
    np.ndarray
        Per-feature MI values in nats, shape (d,). Always non-negative.
    """
    n_total = n_left + n_right
    w_left = n_left / n_total
    w_right = n_right / n_total

    # Pooled distribution (marginal over S)
    p_pooled = w_left * p_left + w_right * p_right

    # MI = H(X) - H(X|S)
    H_marginal = _feature_entropy(p_pooled)
    H_conditional = w_left * _feature_entropy(p_left) + w_right * _feature_entropy(p_right)

    return np.maximum(H_marginal - H_conditional, 0.0)


def select_informative_features(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    n_left: float,
    n_right: float,
    min_fraction: float = 0.1,
    quantile_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Select features with high MI for sibling discrimination.

    Supports both binary (Bernoulli) and categorical (multinomial) distributions.

    Uses two criteria:
    1. Keep features with MI > quantile threshold of non-zero MI features
    2. Always keep at least min_fraction of features

    Parameters
    ----------
    theta_left : np.ndarray
        Distribution for left sibling. Shape (d,) for binary or (d, K) for categorical.
    theta_right : np.ndarray
        Distribution for right sibling. Same shape as theta_left.
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
    """
    mi = _compute_feature_mi(theta_left, theta_right, n_left, n_right)
    d = len(mi)

    # Compute threshold from non-zero MI features
    nonzero_mi = mi[mi > 1e-10]
    threshold = np.quantile(nonzero_mi, quantile_threshold) if len(nonzero_mi) > 0 else 0.0

    # Select features above threshold
    mask = mi > threshold
    n_selected = mask.sum()

    # Ensure minimum number of features
    min_features = max(1, int(min_fraction * d))
    if n_selected < min_features:
        top_indices = np.argsort(mi)[-min_features:]
        mask = np.zeros(d, dtype=bool)
        mask[top_indices] = True
        n_selected = min_features

    return mask, mi, n_selected


__all__ = ["select_informative_features"]
