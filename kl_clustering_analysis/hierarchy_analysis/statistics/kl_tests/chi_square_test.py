"""Chi-square test for KL divergence significance testing.

Provides goodness-of-fit testing for KL divergence values using the
chi-square approximation: 2·n·KL ~ χ²(F) with variance-weighted effective df.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2


def kl_divergence_chi_square_test(
    kl_divergence: float,
    sample_size: int,
    num_features: int,
    parent_distribution: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[float, float, float]:
    """Test KL divergence significance using chi-square approximation.

    Uses the approximation 2·n·KL ~ χ²(F) where n is sample size
    and F is effective degrees of freedom weighted by feature informativeness.

    By default, uses variance-weighted df where features are weighted by
    their Bernoulli variance: w_i = 4·θ_i·(1-θ_i). This down-weights
    uninformative features (θ near 0 or 1) and provides more accurate
    statistical tests.

    Parameters
    ----------
    kl_divergence : float
        KL divergence value to test
    sample_size : int
        Number of observations (e.g., leaf nodes in a tree)
    num_features : int
        Number of features (used if parent_distribution is None)
    parent_distribution : Optional[np.ndarray]
        Parent node's probability distribution (theta vector).
        If provided, calculates variance-weighted effective df.
        If None, uses theoretical df = num_features.

    Returns
    -------
    test_statistic : float
        Chi-square test statistic (2·n·KL)
    degrees_of_freedom : float
        Effective degrees of freedom (variance-weighted)
    p_value : float
        Right-tail p-value from chi-square distribution

    Notes
    -----
    This test assumes the KL divergence is computed from independent samples
    and that the sample size is sufficiently large for the chi-square
    approximation to be valid.

    The variance-weighted df accounts for varying feature informativeness:
    - Features with θ ≈ 0.5 get full weight (variance = 0.25)
    - Features with θ ≈ 0 or 1 get low weight (near-constant)

    References
    ----------
    The 2·n·KL ~ χ²(F) approximation is based on the asymptotic properties
    of the likelihood ratio test statistic (Wilks' theorem).
    """
    # Convert inputs to appropriate types for numerical stability
    n = float(sample_size)
    kl = float(kl_divergence)

    # Calculate chi-square test statistic: χ² = 2·n·KL
    test_statistic = 2.0 * n * kl

    # Calculate degrees of freedom
    if parent_distribution is not None:
        # Variance-weighted effective df (DEFAULT)
        # Weight by feature informativeness: Bernoulli variance = θ(1-θ)
        # Normalized to [0,1] by multiplying by 4 (max variance = 0.25)
        parent_theta = np.asarray(parent_distribution, dtype=np.float64)
        variance_weights = 4.0 * parent_theta * (1.0 - parent_theta)
        degrees_of_freedom = float(np.sum(variance_weights))
    else:
        # Theoretical df (fallback when parent distribution unavailable)
        degrees_of_freedom = float(num_features)

    # Calculate right-tail p-value using survival function (1 - CDF)
    p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))

    return test_statistic, degrees_of_freedom, p_value


__all__ = [
    "kl_divergence_chi_square_test",
]
