"""Production categorical Wald statistic (multinomial Mahalanobis).

This module intentionally keeps a single implementation path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def categorical_whitened_vector(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    *,
    branch_length_sum: float | None = None,
    mean_branch_length: float | None = None,
    ridge: float = 1e-12,
) -> NDArray[np.floating]:
    """Build a covariance-whitened categorical difference vector.

    The returned vector z satisfies approximately:
        sum(z**2) == multinomial Wald statistic (drop-last parametrization).
    """
    theta_1 = np.asarray(theta_1, dtype=np.float64)
    theta_2 = np.asarray(theta_2, dtype=np.float64)

    if theta_1.shape != theta_2.shape:
        raise ValueError(
            f"theta_1 and theta_2 must have identical shape. "
            f"Got {theta_1.shape} vs {theta_2.shape}."
        )
    if theta_1.ndim != 2:
        raise ValueError(f"Categorical input must be 2D (d, K). Got ndim={theta_1.ndim}.")
    d, K = theta_1.shape
    if K < 2:
        raise ValueError(f"Categorical input must have K>=2. Got K={K}.")
    if not np.isfinite(n_1) or not np.isfinite(n_2) or n_1 <= 0 or n_2 <= 0:
        raise ValueError(f"n_1 and n_2 must be finite positive values. Got n_1={n_1}, n_2={n_2}.")

    pooled = (n_1 * theta_1 + n_2 * theta_2) / (n_1 + n_2)
    n_eff = (n_1 * n_2) / (n_1 + n_2)

    # Drop the final category to remove simplex singularity.
    diff = theta_1[:, :-1] - theta_2[:, :-1]  # shape (d, K-1)
    pooled_reduced = pooled[:, :-1]  # keep raw probabilities (no renormalization)

    dim = K - 1
    eye = np.eye(dim)
    blocks: list[np.ndarray] = []
    for i in range(d):
        p = pooled_reduced[i]
        sigma = (np.diag(p) - np.outer(p, p)) / n_eff
        sigma = sigma + ridge * eye
        cho, lower = linalg.cho_factor(sigma, lower=True, check_finite=False)
        # y = L^{-1} diff_i, where sigma = L L^T
        y = linalg.solve_triangular(cho, diff[i], lower=lower, check_finite=False)
        blocks.append(y)

    z = np.concatenate(blocks, axis=0)

    # Apply sibling branch-length scaling as variance inflation.
    # Since T = ||z||^2, scaling variance by c implies T/c, i.e. z/sqrt(c).
    if branch_length_sum is not None and branch_length_sum > 0:
        if mean_branch_length is None or mean_branch_length <= 0:
            raise ValueError(
                "mean_branch_length must be finite positive when branch_length_sum is provided. "
                f"Got mean_branch_length={mean_branch_length!r}, branch_length_sum={branch_length_sum!r}."
            )
        bl_normalized = 1.0 + branch_length_sum / (2.0 * mean_branch_length)
        z = z / np.sqrt(bl_normalized)

    return z


def mahalanobis_wald_categorical(
    theta_1: NDArray[np.floating],
    theta_2: NDArray[np.floating],
    n_1: float,
    n_2: float,
    *,
    branch_length_sum: float | None = None,
    mean_branch_length: float | None = None,
    ridge: float = 1e-12,
) -> tuple[float, int]:
    """Compute multinomial Wald statistic with drop-last parametrization.

    Parameters
    ----------
    theta_1 : NDArray[np.floating]
        Distribution for group 1, shape (d, K).
    theta_2 : NDArray[np.floating]
        Distribution for group 2, shape (d, K).
    n_1 : float
        Sample size for group 1.
    n_2 : float
        Sample size for group 2.
    branch_length_sum : float, optional
        Sibling branch-length sum for Felsenstein variance scaling.
    mean_branch_length : float, optional
        Mean branch length used to normalize branch_length_sum.
    ridge : float, optional
        Small diagonal regularization for numeric stability.

    Returns
    -------
    tuple[float, int]
        (wald_statistic, degrees_of_freedom).
    """
    theta_1 = np.asarray(theta_1, dtype=np.float64)
    if theta_1.ndim != 2:
        raise ValueError(f"Categorical input must be 2D (d, K). Got ndim={theta_1.ndim}.")
    d, K = theta_1.shape
    z = categorical_whitened_vector(
        theta_1=theta_1,
        theta_2=theta_2,
        n_1=n_1,
        n_2=n_2,
        branch_length_sum=branch_length_sum,
        mean_branch_length=mean_branch_length,
        ridge=ridge,
    )
    T = float(np.sum(z**2))
    df = d * (K - 1)
    return float(T), int(df)


__all__ = ["categorical_whitened_vector", "mahalanobis_wald_categorical"]
