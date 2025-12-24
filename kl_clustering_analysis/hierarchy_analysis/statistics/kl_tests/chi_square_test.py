"""Chi-square test for KL divergence significance testing.

Provides goodness-of-fit testing for KL divergence values using the
chi-square approximation: 2·n·KL ~ χ²(df).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import chi2


def kl_divergence_chi_square_test(
    kl_divergence: float,
    sample_size: int,
    degrees_of_freedom: Union[int, float],
    weight: float = 1.0,
) -> Tuple[float, float, float]:
    """Test KL divergence significance using chi-square approximation.

    Uses the approximation χ² = 2·n·KL/w ~ χ²(df) where n is sample size,
    df is degrees of freedom, and w is an optional weight.

    Parameters
    ----------
    kl_divergence : float
        KL divergence value to test
    sample_size : int
        Number of observations (e.g., leaf nodes in a tree)
    degrees_of_freedom : int or float
        Degrees of freedom for the chi-square distribution.
        Can be variance-weighted (float) or fixed (int).
    weight : float, default=1.0
        Weight divisor for the test statistic. Values > 1 penalize
        (harder to reject), values < 1 give bonus (easier to reject).

    Returns
    -------
    test_statistic : float
        Chi-square test statistic (2·n·KL/w)
    degrees_of_freedom : float
        Degrees of freedom used for the test
    p_value : float
        Right-tail p-value from chi-square distribution

    Notes
    -----
    This test assumes the KL divergence is computed from independent samples
    and that the sample size is sufficiently large for the chi-square
    approximation to be valid.

    References
    ----------
    The 2·n·KL ~ χ²(df) approximation is based on the asymptotic properties
    of the likelihood ratio test statistic.
    """
    n = float(sample_size)
    kl = float(kl_divergence)
    df = float(degrees_of_freedom)
    w = float(weight)

    # Calculate chi-square test statistic: χ² = 2·n·KL/w
    test_statistic = 2.0 * n * kl / w

    # Calculate right-tail p-value using survival function (1 - CDF)
    p_value = float(chi2.sf(test_statistic, df=df))

    return test_statistic, df, p_value


def kl_divergence_chi_square_test_batch(
    kl_divergences: np.ndarray,
    sample_sizes: np.ndarray,
    degrees_of_freedom: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized chi-square test for multiple KL divergence values.

    Tests χ² = 2·n·KL/w ~ χ²(df) for each edge simultaneously.

    Parameters
    ----------
    kl_divergences : np.ndarray
        KL divergence values for each edge
    sample_sizes : np.ndarray
        Sample sizes (e.g., leaf counts) for each edge
    degrees_of_freedom : np.ndarray
        Degrees of freedom for each edge (can be variance-weighted)
    weights : np.ndarray, optional
        Weight divisors for each edge. If None, uses 1.0 for all.

    Returns
    -------
    test_statistics : np.ndarray
        Chi-square test statistics (2·n·KL/w)
    p_values : np.ndarray
        Right-tail p-values from chi-square distribution

    Notes
    -----
    Invalid edges (NaN KL or sample_size <= 0) will have NaN p-values.
    """
    n_edges = len(kl_divergences)

    if weights is None:
        weights = np.ones(n_edges, dtype=float)

    # Compute test statistics: χ² = 2·n·KL/w
    test_statistics = 2.0 * sample_sizes * kl_divergences / weights

    # Compute p-values using vectorized survival function
    p_values = chi2.sf(test_statistics, df=degrees_of_freedom)

    return test_statistics, p_values


__all__ = [
    "kl_divergence_chi_square_test",
    "kl_divergence_chi_square_test_batch",
]
