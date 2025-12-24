"""Pooled variance estimation for two-sample proportion tests.

This module provides utilities for computing the pooled variance of the
difference between two Bernoulli proportion estimates, as used in the
standard Wald z-test for comparing proportions.

Statistical Background
----------------------
For two independent samples with Bernoulli observations:
    - Sample 1: n₁ observations with estimated proportion θ̂₁
    - Sample 2: n₂ observations with estimated proportion θ̂₂

Under the null hypothesis H₀: θ₁ = θ₂ = θ, the pooled estimator is:
    θ̂_pooled = (n₁·θ̂₁ + n₂·θ̂₂) / (n₁ + n₂)

This is the maximum likelihood estimator (MLE) under H₀.

The variance of the difference Δθ̂ = θ̂₁ - θ̂₂ is:
    Var[Δθ̂] = Var[θ̂₁] + Var[θ̂₂]
             = θ(1-θ)/n₁ + θ(1-θ)/n₂
             = θ(1-θ) × (1/n₁ + 1/n₂)

Using the pooled estimate:
    Var[Δθ̂] ≈ θ̂_pooled(1 - θ̂_pooled) × (1/n₁ + 1/n₂)

The standardized difference (z-score) is:
    z = Δθ̂ / √Var[Δθ̂]

Under H₀, z ~ N(0,1) asymptotically, and z² ~ χ²(1).

For multivariate case (p features), T = Σⱼ zⱼ² ~ χ²(p) under H₀.

References
----------
Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley.
Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical Methods
    for Rates and Proportions (3rd ed.). Wiley.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def compute_pooled_proportion(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute the pooled proportion estimate under H₀: θ₁ = θ₂.

    Uses sample-size weighted averaging, which is the MLE under H₀:
        θ̂_pooled = (n₁·θ̂₁ + n₂·θ̂₂) / (n₁ + n₂)

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Estimated proportions for sample 1, shape (p,) for p features.
    theta_2 : NDArray[np.floating]
        Estimated proportions for sample 2, shape (p,).
    n_1 : float
        Sample size of group 1.
    n_2 : float
        Sample size of group 2.
    eps : float, optional
        Small constant to clip proportions away from 0 and 1,
        preventing degenerate variance estimates (default: 1e-10).

    Returns
    -------
    NDArray[np.floating]
        Pooled proportion estimates, shape (p,), clipped to [eps, 1-eps].

    Examples
    --------
    >>> theta_1 = np.array([0.3, 0.5, 0.7])
    >>> theta_2 = np.array([0.4, 0.5, 0.6])
    >>> pooled = compute_pooled_proportion(theta_1, theta_2, n_1=100, n_2=200)
    >>> # With n_2 = 2×n_1, pooled is weighted 2:1 toward theta_2
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

    Under H₀: θ₁ = θ₂, the variance of Δθ̂ = θ̂₁ - θ̂₂ is:
        Var[Δθ̂] = θ_pooled(1 - θ_pooled) × (1/n₁ + 1/n₂)

    This uses the pooled proportion estimate (MLE under H₀) rather than
    separate variance estimates, which is standard for hypothesis testing.

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Estimated proportions for sample 1, shape (p,).
    theta_2 : NDArray[np.floating]
        Estimated proportions for sample 2, shape (p,).
    n_1 : float
        Sample size of group 1.
    n_2 : float
        Sample size of group 2.
    eps : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns
    -------
    NDArray[np.floating]
        Variance of the difference for each feature, shape (p,).
        Minimum value is eps to prevent division by zero.

    Notes
    -----
    The formula Var = θ(1-θ)(1/n₁ + 1/n₂) comes from:
    - Var[θ̂₁] = θ(1-θ)/n₁  (variance of sample proportion)
    - Var[θ̂₂] = θ(1-θ)/n₂
    - Var[θ̂₁ - θ̂₂] = Var[θ̂₁] + Var[θ̂₂]  (independence)
    """
    pooled = compute_pooled_proportion(theta_1, theta_2, n_1, n_2, eps)
    variance = pooled * (1.0 - pooled) * (1.0 / n_1 + 1.0 / n_2)
    return np.maximum(variance, eps)


def standardize_proportion_difference(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    eps: float = 1e-10,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the standardized difference (z-scores) between two proportions.

    For each feature j, computes:
        z_j = (θ̂₁_j - θ̂₂_j) / √Var[Δθ̂_j]

    Under H₀: θ₁ = θ₂, each z_j ~ N(0,1) asymptotically.

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Estimated proportions for sample 1, shape (p,).
    theta_2 : NDArray[np.floating]
        Estimated proportions for sample 2, shape (p,).
    n_1 : float
        Sample size of group 1.
    n_2 : float
        Sample size of group 2.
    eps : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns
    -------
    Tuple[NDArray[np.floating], NDArray[np.floating]]
        (z_scores, variance) where:
        - z_scores: Standardized differences, shape (p,)
        - variance: Pooled variance estimates, shape (p,)

    Examples
    --------
    >>> theta_1 = np.array([0.3, 0.5])
    >>> theta_2 = np.array([0.4, 0.5])
    >>> z, var = standardize_proportion_difference(theta_1, theta_2, 100, 100)
    >>> # z[0] < 0 (theta_1 < theta_2), z[1] ≈ 0 (equal proportions)
    """
    variance = compute_pooled_variance(theta_1, theta_2, n_1, n_2, eps)
    difference = theta_1 - theta_2
    z_scores = difference / np.sqrt(variance)
    return z_scores, variance


__all__ = [
    "compute_pooled_proportion",
    "compute_pooled_variance",
    "standardize_proportion_difference",
]
