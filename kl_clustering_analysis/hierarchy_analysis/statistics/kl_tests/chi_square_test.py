"""Chi-square test for KL divergence significance testing.

Provides goodness-of-fit testing for KL divergence values using the
chi-square approximation: 2·n·KL ~ χ²(F).
"""

from __future__ import annotations

from typing import Tuple

from scipy.stats import chi2


def kl_divergence_chi_square_test(
    kl_divergence: float, sample_size: int, num_features: int
) -> Tuple[float, int, float]:
    """Test KL divergence significance using chi-square approximation.

    Uses the approximation 2·n·KL ~ χ²(F) where n is sample size
    and F is degrees of freedom (number of features).

    Parameters
    ----------
    kl_divergence : float
        KL divergence value to test
    sample_size : int
        Number of observations (e.g., leaf nodes in a tree)
    num_features : int
        Number of features (degrees of freedom)

    Returns
    -------
    test_statistic : float
        Chi-square test statistic (2·n·KL)
    degrees_of_freedom : int
        Degrees of freedom for the test (equals num_features)
    p_value : float
        Right-tail p-value from chi-square distribution

    Notes
    -----
    This test assumes the KL divergence is computed from independent samples
    and that the sample size is sufficiently large for the chi-square
    approximation to be valid.

    References
    ----------
    The 2·n·KL ~ χ²(F) approximation is based on the asymptotic properties
    of the likelihood ratio test statistic.
    """
    # Convert inputs to appropriate types for numerical stability
    n = float(sample_size)
    kl = float(kl_divergence)

    # Calculate chi-square test statistic: χ² = 2·n·KL
    test_statistic = 2.0 * n * kl

    # Degrees of freedom equals number of features
    degrees_of_freedom = int(num_features)

    # Calculate right-tail p-value using survival function (1 - CDF)
    p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))

    return test_statistic, degrees_of_freedom, p_value


__all__ = [
    "kl_divergence_chi_square_test",
]
