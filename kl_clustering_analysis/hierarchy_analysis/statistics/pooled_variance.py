"""Pooled variance estimation for two-sample proportion tests.

Supports both binary (Bernoulli) and categorical (multinomial) distributions.

Statistical Background
----------------------
For two independent samples with observations from the same distribution family:

Binary (Bernoulli):
    Var[θ̂] = θ(1-θ)/n

Categorical (Multinomial):
    Var[p̂_k] = p_k(1-p_k)/n  (diagonal elements)
    Cov[p̂_j, p̂_k] = -p_j·p_k/n  (off-diagonal, but ignored for Wald test)

For the Wald test, we use the diagonal variance approximation, which is
conservative and works well with random projection.

References
----------
Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _is_categorical(arr: np.ndarray) -> bool:
    """Check if array represents categorical distributions (2D)."""
    return arr.ndim == 2 and arr.shape[1] > 1


def _flatten_categorical(arr: np.ndarray) -> np.ndarray:
    """Flatten categorical distribution to 1D for Wald test.

    For shape (d, K), returns shape (d * K,).
    For shape (d,), returns unchanged.
    """
    if _is_categorical(arr):
        return arr.ravel()
    return arr


def compute_pooled_proportion(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute the pooled proportion estimate under H₀: θ₁ = θ₂.

    Supports both binary (1D) and categorical (2D) distributions.

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Estimated proportions for sample 1.
        Shape (d,) for binary or (d, K) for categorical.
    theta_2 : NDArray[np.floating]
        Estimated proportions for sample 2. Same shape as theta_1.
    n_1 : float
        Sample size of group 1.
    n_2 : float
        Sample size of group 2.
    eps : float, optional
        Small constant to clip proportions (default: 1e-10).

    Returns
    -------
    NDArray[np.floating]
        Pooled proportion estimates, same shape as input.
    """
    n_total = n_1 + n_2
    pooled = (n_1 * theta_1 + n_2 * theta_2) / n_total
    return np.clip(pooled, eps, 1.0 - eps)


def compute_pooled_variance(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute the variance of the difference between two proportions.

    Supports both binary (1D) and categorical (2D) distributions.

    For categorical, uses diagonal variance approximation:
        Var[p̂_k] = p_k(1-p_k)/n

    This ignores off-diagonal covariance terms, which is conservative
    and works well with random projection.

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Estimated proportions for sample 1.
        Shape (d,) for binary or (d, K) for categorical.
    theta_2 : NDArray[np.floating]
        Estimated proportions for sample 2. Same shape as theta_1.
    n_1 : float
        Sample size of group 1.
    n_2 : float
        Sample size of group 2.
    eps : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns
    -------
    NDArray[np.floating]
        Variance of the difference, same shape as input.
    """
    pooled = compute_pooled_proportion(theta_1, theta_2, n_1, n_2, eps)
    # Var = p(1-p) * (1/n1 + 1/n2) works for both binary and categorical
    variance = pooled * (1.0 - pooled) * (1.0 / n_1 + 1.0 / n_2)
    return np.maximum(variance, eps)


def standardize_proportion_difference(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
    branch_length_sum: float | None = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute standardized difference (z-scores) between two proportions.

    Supports both binary (1D) and categorical (2D) distributions.
    For categorical, flattens to 1D for the Wald test.

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Estimated proportions for sample 1.
        Shape (d,) for binary or (d, K) for categorical.
    theta_2 : NDArray[np.floating]
        Estimated proportions for sample 2. Same shape as theta_1.
    n_1 : float
        Sample size of group 1.
    n_2 : float
        Sample size of group 2.
    eps : float, optional
        Small constant for numerical stability (default: 1e-10).
    branch_length_sum : float, optional
        Sum of branch lengths (b_L + b_R) for Felsenstein's phylogenetic
        independent contrasts adjustment. If provided, variance is scaled
        by this factor to account for expected divergence over topological
        distance. Longer branches -> more expected variance -> smaller Z.

    Returns
    -------
    Tuple[NDArray[np.floating], NDArray[np.floating]]
        (z_scores, variance) where both are 1D arrays.
        For categorical input, these are flattened.
    """
    variance = compute_pooled_variance(theta_1, theta_2, n_1, n_2, eps)

    # Felsenstein (1985) branch-length adjustment with normalization:
    # For sibling contrasts, variance scales with sum of branch lengths.
    # We normalize: BL_norm = 1 + BL_sum/(2*mean_BL)
    # The factor of 2 accounts for summing two branches.
    # This ensures BL_norm ≥ 1, preventing variance shrinkage.
    #
    # Longer total branch length → more expected divergence → larger variance
    # → smaller z-scores → harder to declare siblings as different.
    if branch_length_sum is not None and branch_length_sum > 0:
        # Default mean_bl estimate: assume typical BL ~ branch_length_sum/2
        # This gives BL_norm ~ 2 on average
        # TODO: Pass actual mean_bl from tree for more accurate normalization
        mean_bl_estimate = branch_length_sum / 2  # rough estimate
        bl_normalized = 1.0 + branch_length_sum / (2 * mean_bl_estimate)
        variance = variance * bl_normalized

    difference = theta_1 - theta_2

    # Flatten if categorical
    variance_flat = _flatten_categorical(variance)
    difference_flat = _flatten_categorical(difference)

    z_scores = difference_flat / np.sqrt(variance_flat)
    return z_scores, variance_flat


__all__ = [
    "compute_pooled_proportion",
    "compute_pooled_variance",
    "standardize_proportion_difference",
]
