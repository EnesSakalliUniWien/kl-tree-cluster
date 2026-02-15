"""Central Limit Theorem validity checks using Berry-Esseen bounds.

This module provides data-adaptive sample size validation for Wald tests
using the Berry-Esseen theorem, which quantifies how quickly the sampling
distribution of the sample mean converges to a normal distribution.

The key insight is that for the Wald chi-square test to be valid, the
normal approximation must hold for each feature being tested. Rather than
using arbitrary thresholds like n >= 10, we check whether the Berry-Esseen
bound guarantees the approximation error is below the test's significance
level.

References
----------
- Berry (1941): "The Accuracy of the Gaussian Approximation to the Sum of
  Independent Variates"
- Esseen (1942): "On the Liapounoff limit of error in the theory of
  probability distributions"
- Shevtsova (2011): "On the absolute constants in the Berry-Esseen type
  inequalities for identically distributed summands"
- Agresti (2002): "Categorical Data Analysis" (2nd ed.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Shevtsova (2011) constant - tightest known upper bound for Berry-Esseen
# For i.i.d. case with finite third moment
SHEVTSOVA_CONSTANT: float = 0.4748

# Van Beek (1972) constant - more general, slightly looser
VAN_BEEK_CONSTANT: float = 0.7975


@dataclass(frozen=True)
class CLTValidityResult:
    """Result of CLT validity check for a node.

    Attributes
    ----------
    is_valid : bool
        True if the normal approximation is valid for all features.
    min_required_n : int
        Minimum sample size required for validity (per feature).
    actual_n : int
        Actual sample size at this node.
    max_approximation_error : float
        Upper bound on the approximation error from Berry-Esseen.
    features_valid : NDArray[np.bool_]
        Boolean array indicating which features pass validity check.
    fraction_valid : float
        Fraction of features that pass the validity check.
    """

    is_valid: bool
    min_required_n: int
    actual_n: int
    max_approximation_error: float
    features_valid: NDArray[np.bool_]
    fraction_valid: float


def compute_third_absolute_moment(
    probs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute third absolute central moment for Bernoulli distribution.

    For Bernoulli(p): E[|X - μ|³] = p(1-p)[p² + (1-p)²]

    Parameters
    ----------
    probs : NDArray[np.float64]
        Probability vector (shape: n_features,).

    Returns
    -------
    NDArray[np.float64]
        Third absolute central moment for each feature.
    """
    p = probs
    mu = p  # mean of Bernoulli
    # E[|X - μ|³] = p(1-p)[p² + (1-p)²]
    return p * (1 - p) * (p**2 + (1 - p) ** 2)


def compute_third_absolute_moment_categorical(
    probs: NDArray[np.float64],
) -> float:
    """Compute third absolute central moment for Categorical distribution.

    For Categorical with probabilities (p₁, ..., pₖ):
    E[|X - μ|³] = Σᵢ pᵢ ||eᵢ - p||³

    where μ = E[X] = p (probability vector), and eᵢ are one-hot vectors.

    This is computed as: Σᵢ pᵢ (Σⱼ (δᵢⱼ - pⱼ)²)^(3/2)

    Parameters
    ----------
    probs : NDArray[np.float64]
        Probability vector for categorical distribution (shape: n_categories,).

    Returns
    -------
    float
        Third absolute central moment.
    """
    p = probs / probs.sum()  # normalize
    # For one-hot encoding, ||eᵢ - p||² = (1-pᵢ)² + Σ_{j≠i} pⱼ²
    # = 1 - 2pᵢ + pᵢ² + Σ_{j≠i} pⱼ² = 1 - 2pᵢ + ||p||²
    norm_sq = np.sum(p**2)
    sq_norms = 1 - 2 * p + norm_sq  # squared L2 distance from eᵢ to p
    # E[|X - μ|³] = Σᵢ pᵢ ||eᵢ - p||³
    return float(np.sum(p * sq_norms**1.5))


def compute_variance_bernoulli(
    probs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute variance for Bernoulli distribution.

    Var(X) = p(1-p) for Bernoulli(p).

    Parameters
    ----------
    probs : NDArray[np.float64]
        Probability vector (shape: n_features,).

    Returns
    -------
    NDArray[np.float64]
        Variance for each feature.
    """
    return probs * (1 - probs)


def berry_esseen_bound(
    n: int,
    rho: float | NDArray[np.float64],
    sigma_sq: float | NDArray[np.float64],
    constant: float = SHEVTSOVA_CONSTANT,
) -> float | NDArray[np.float64]:
    """Compute Berry-Esseen upper bound on normal approximation error.

    The bound is: sup_x |P(Z_n <= x) - Φ(x)| <= C * ρ / (σ³ * sqrt(n))

    where:
    - C is the Berry-Esseen constant (default: Shevtsova 2011)
    - ρ is the third absolute central moment
    - σ² is the variance
    - n is the sample size

    Parameters
    ----------
    n : int
        Sample size.
    rho : float or NDArray[np.float64]
        Third absolute central moment(s).
    sigma_sq : float or NDArray[np.float64]
        Variance(s). Must be positive.
    constant : float, default=SHEVTSOVA_CONSTANT
        Berry-Esseen constant to use.

    Returns
    -------
    float or NDArray[np.float64]
        Upper bound on approximation error.

    Raises
    ------
    ValueError
        If variance is not positive.
    """
    if np.any(sigma_sq <= 0):
        raise ValueError("Variance must be positive for Berry-Esseen bound")

    sigma_cubed = sigma_sq**1.5
    return constant * rho / (sigma_cubed * np.sqrt(n))


def check_clt_validity_bernoulli(
    probs: NDArray[np.float64],
    n: int,
    alpha: float = 0.05,
    min_fraction_valid: float = 0.5,
    constant: float = SHEVTSOVA_CONSTANT,
) -> CLTValidityResult:
    """Check if CLT approximation is valid for Bernoulli features.

    Uses the Berry-Esseen bound to check whether the normal approximation
    error is below the significance level alpha. The approximation is
    considered valid if the bound guarantees the error is small enough.

    Parameters
    ----------
    probs : NDArray[np.float64]
        Probability vector for Bernoulli features (shape: n_features,).
    n : int
        Sample size.
    alpha : float, default=0.05
        Significance level. The approximation error should be less than alpha.
    min_fraction_valid : float, default=0.5
        Minimum fraction of features that must pass the validity check
        for the overall result to be valid.
    constant : float, default=SHEVTSOVA_CONSTANT
        Berry-Esseen constant to use.

    Returns
    -------
    CLTValidityResult
        Result of the validity check.

    Notes
    -----
    For degenerate features (p = 0 or p = 1), the variance is zero and
    the Berry-Esseen bound is undefined. These features are excluded from
    the validity check but counted in the fraction_valid calculation.
    """
    if n < 1:
        return CLTValidityResult(
            is_valid=False,
            min_required_n=1,
            actual_n=n,
            max_approximation_error=np.inf,
            features_valid=np.array([], dtype=bool),
            fraction_valid=0.0,
        )

    # Compute moments
    rho = compute_third_absolute_moment(probs)
    sigma_sq = compute_variance_bernoulli(probs)

    # Identify non-degenerate features (0 < p < 1)
    # For degenerate features, variance = 0, so CLT is trivially "valid"
    # (the distribution is a point mass at the mean)
    non_degenerate = (probs > 0) & (probs < 1)
    n_features = len(probs)

    if not np.any(non_degenerate):
        # All features are degenerate
        return CLTValidityResult(
            is_valid=True,
            min_required_n=1,
            actual_n=n,
            max_approximation_error=0.0,
            features_valid=np.ones(n_features, dtype=bool),
            fraction_valid=1.0,
        )

    # Compute Berry-Esseen bound for non-degenerate features
    rho_nd = rho[non_degenerate]
    sigma_sq_nd = sigma_sq[non_degenerate]

    bounds = berry_esseen_bound(n, rho_nd, sigma_sq_nd, constant)

    # The approximation is valid if the bound is below alpha
    # (guarantees the error is less than alpha)
    valid_nd = bounds < alpha

    # Construct full features_valid array
    features_valid = np.ones(n_features, dtype=bool)
    features_valid[non_degenerate] = valid_nd

    # Count degenerate features as valid
    n_degenerate = n_features - np.sum(non_degenerate)
    fraction_valid = (np.sum(valid_nd) + n_degenerate) / n_features

    # Compute minimum required sample size for validity
    # Solve: C * rho / (sigma^3 * sqrt(n)) = alpha
    # => n = (C * rho / (sigma^3 * alpha))^2
    min_n_per_feature = np.ceil((constant * rho_nd / (sigma_sq_nd**1.5 * alpha)) ** 2).astype(int)
    min_required_n = int(np.max(min_n_per_feature))

    max_bound = float(np.max(bounds)) if len(bounds) > 0 else 0.0

    is_valid = fraction_valid >= min_fraction_valid

    return CLTValidityResult(
        is_valid=is_valid,
        min_required_n=min_required_n,
        actual_n=n,
        max_approximation_error=max_bound,
        features_valid=features_valid,
        fraction_valid=fraction_valid,
    )


def compute_minimum_n_berry_esseen(
    probs: NDArray[np.float64],
    alpha: float = 0.05,
    constant: float = SHEVTSOVA_CONSTANT,
) -> int:
    """Compute minimum sample size required for CLT validity.

    This computes the minimum n such that the Berry-Esseen bound
    guarantees the normal approximation error is below alpha.

    Parameters
    ----------
    probs : NDArray[np.float64]
        Probability vector.
    alpha : float, default=0.05
        Target approximation error (significance level).
    constant : float, default=SHEVTSOVA_CONSTANT
        Berry-Esseen constant.

    Returns
    -------
    int
        Minimum required sample size.
    """
    rho = compute_third_absolute_moment(probs)
    sigma_sq = compute_variance_bernoulli(probs)

    # Only consider non-degenerate features
    non_degenerate = (probs > 0) & (probs < 1)

    if not np.any(non_degenerate):
        return 1

    rho_nd = rho[non_degenerate]
    sigma_sq_nd = sigma_sq[non_degenerate]

    # n = (C * rho / (sigma^3 * alpha))^2
    min_n = np.ceil((constant * rho_nd / (sigma_sq_nd**1.5 * alpha)) ** 2).astype(int)

    return int(np.max(min_n))


def check_split_clt_validity(
    dist_left: NDArray[np.float64],
    dist_right: NDArray[np.float64],
    n_left: int,
    n_right: int,
    alpha: float = 0.05,
    min_fraction_valid: float = 0.5,
) -> tuple[CLTValidityResult, CLTValidityResult]:
    """Check CLT validity for both children of a split.

    Parameters
    ----------
    dist_left, dist_right : NDArray[np.float64]
        Probability distributions for left and right children.
    n_left, n_right : int
        Sample sizes for left and right children.
    alpha : float, default=0.05
        Significance level.
    min_fraction_valid : float, default=0.5
        Minimum fraction of features that must be valid.

    Returns
    -------
    tuple[CLTValidityResult, CLTValidityResult]
        Validity results for left and right children.
    """
    left_result = check_clt_validity_bernoulli(dist_left, n_left, alpha, min_fraction_valid)
    right_result = check_clt_validity_bernoulli(dist_right, n_right, alpha, min_fraction_valid)
    return left_result, right_result
