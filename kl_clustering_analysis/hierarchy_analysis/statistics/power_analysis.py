"""Power analysis for hierarchical clustering statistical tests.

This module provides power calculations for:
1. Child-parent divergence tests (nested variance)
2. Sibling divergence tests (two-sample Wald)

The key insight is that with small sample sizes, even large true effects
may not be detectable at reasonable significance levels. When power is
too low, tests should be skipped or interpreted with caution.

References
----------
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PowerResult:
    """Result of a power analysis calculation.

    Attributes
    ----------
    power : float
        Statistical power (1 - β) to detect the effect
    n_required : float
        Sample size required to achieve target power
    effect_size : float
        Standardized effect size (Cohen's h for proportions)
    is_sufficient : bool
        Whether current power meets the threshold
    """

    power: float
    n_required: float
    effect_size: float
    is_sufficient: bool


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions.

    Cohen's h is the difference between arcsine-transformed proportions:
        h = 2 * (arcsin(√p1) - arcsin(√p2))

    Interpretation (Cohen, 1988):
        - h = 0.20: small effect
        - h = 0.50: medium effect
        - h = 0.80: large effect

    Parameters
    ----------
    p1, p2 : float
        Proportions in [0, 1]

    Returns
    -------
    float
        Cohen's h effect size
    """
    # Clip to valid range to avoid domain errors
    p1 = np.clip(p1, 1e-10, 1 - 1e-10)
    p2 = np.clip(p2, 1e-10, 1 - 1e-10)

    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def power_wald_two_sample(
    n1: float,
    n2: float,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Compute power for a two-sample Wald test of proportions.

    Under the alternative hypothesis H₁: p1 ≠ p2, the test statistic
    follows a non-central χ² distribution with non-centrality parameter:
        λ = (p1 - p2)² / Var(θ̂₁ - θ̂₂)

    Parameters
    ----------
    n1, n2 : float
        Sample sizes for the two groups
    p1, p2 : float
        True proportions under H₁
    alpha : float
        Significance level (Type I error rate)
    alternative : str
        'two-sided' or 'one-sided'

    Returns
    -------
    float
        Statistical power in [0, 1]

    References
    ----------
    Agresti, A. (2013). Categorical Data Analysis, Section 3.3.5
    """
    # Effect size
    delta = p1 - p2

    # Pooled variance under H₀ (for critical value)
    p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
    var_pool = p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)

    # Variance under H₁ (for non-centrality)
    var_alt = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2

    if var_pool <= 0 or var_alt <= 0:
        return 0.0

    # Non-centrality parameter for chi-square
    # λ = (E[T] under H₁) = (δ)² / Var(θ̂₁ - θ̂₂ | H₁)
    ncp = delta**2 / var_alt

    # Critical value from central chi-square
    df = 1  # Single comparison
    if alternative == "two-sided":
        alpha_adj = alpha
    else:
        alpha_adj = 2 * alpha  # One-sided uses α/2 in each direction

    critical = stats.chi2.ppf(1 - alpha_adj, df=df)

    # Power = P(χ²(df, ncp) > critical)
    power = 1 - stats.ncx2.cdf(critical, df=df, nc=ncp)

    return float(power)


def power_wald_nested(
    n_child: float,
    n_parent: float,
    p_child: float,
    p_parent: float,
    alpha: float = 0.05,
) -> float:
    """Compute power for nested child-parent divergence test.

    The nested test has variance:
        Var(θ̂_c - θ̂_p) = θ(1-θ) × (1/n_c - 1/n_p)

    This is more powerful than independent samples (variance is smaller),
    but requires n_c < n_parent.

    Parameters
    ----------
    n_child, n_parent : float
        Sample sizes (child must be proper subset of parent)
    p_child, p_parent : float
        True proportions under H₁
    alpha : float
        Significance level

    Returns
    -------
    float
        Statistical power
    """
    if n_child >= n_parent:
        return 0.0  # Invalid nested structure

    delta = p_child - p_parent

    # Nested variance (smaller than independent)
    nested_factor = 1.0 / n_child - 1.0 / n_parent
    var_nested = p_parent * (1 - p_parent) * nested_factor

    if var_nested <= 0:
        return 0.0

    # Non-centrality parameter
    ncp = delta**2 / var_nested

    # Critical value and power
    df = 1
    critical = stats.chi2.ppf(1 - alpha, df=df)
    power = 1 - stats.ncx2.cdf(critical, df=df, nc=ncp)

    return float(power)


def compute_child_parent_power(
    n_child: int,
    n_parent: int,
    min_effect_size: float = 0.2,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> PowerResult:
    """Compute power for child-parent divergence test.

    Uses a conservative baseline proportion of p = 0.5 (maximum variance).
    The effect size specifies the minimum detectable difference.

    Parameters
    ----------
    n_child, n_parent : int
        Sample sizes
    min_effect_size : float
        Minimum Cohen's h to detect (default: 0.2 = small effect)
    alpha : float
        Significance level
    target_power : float
        Target power for sample size calculation

    Returns
    -------
    PowerResult
        Power analysis results
    """
    if n_child >= n_parent or n_child < 2:
        return PowerResult(
            power=0.0,
            n_required=float("inf"),
            effect_size=min_effect_size,
            is_sufficient=False,
        )

    # Conservative: assume p_parent = 0.5 (max variance)
    # Compute p_child that gives the target effect size
    p_parent = 0.5

    # For small effects: h ≈ 2 * (p1 - p2) / √(p(1-p)) = 4 * (p1 - p2)
    # So delta ≈ h / 4
    delta = min_effect_size / 4
    p_child = np.clip(p_parent + delta, 0.01, 0.99)

    # Compute actual power
    power = power_wald_nested(
        n_child=n_child,
        n_parent=n_parent,
        p_child=p_child,
        p_parent=p_parent,
        alpha=alpha,
    )

    # Compute required sample size (approximate)
    # For nested test: n_c ≈ 4 * z² * p(1-p) / δ²
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)
    z_sum = z_alpha + z_beta

    # Conservative: assume n_c << n_p, so nested_factor ≈ 1/n_c
    n_required = 4 * z_sum**2 * p_parent * (1 - p_parent) / delta**2

    return PowerResult(
        power=power,
        n_required=n_required,
        effect_size=min_effect_size,
        is_sufficient=power >= target_power,
    )


def compute_sibling_power(
    n_left: int,
    n_right: int,
    min_effect_size: float = 0.2,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> PowerResult:
    """Compute power for sibling divergence test.

    Parameters
    ----------
    n_left, n_right : int
        Sample sizes for the two siblings
    min_effect_size : float
        Minimum Cohen's h to detect (default: 0.2 = small effect)
    alpha : float
        Significance level
    target_power : float
        Target power for sample size calculation

    Returns
    -------
    PowerResult
        Power analysis results
    """
    if n_left < 2 or n_right < 2:
        return PowerResult(
            power=0.0,
            n_required=float("inf"),
            effect_size=min_effect_size,
            is_sufficient=False,
        )

    # Conservative baseline
    p_left = 0.5
    delta = min_effect_size / 4
    p_right = np.clip(p_left + delta, 0.01, 0.99)

    # Compute power
    power = power_wald_two_sample(n1=n_left, n2=n_right, p1=p_left, p2=p_right, alpha=alpha)

    # Required sample size (balanced design)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(target_power)
    z_sum = z_alpha + z_beta

    # For two-sample: n ≈ 2 * (z_α/2 + z_β)² * p(1-p) / δ²
    p_pool = (p_left + p_right) / 2
    n_per_group = 2 * z_sum**2 * p_pool * (1 - p_pool) / delta**2

    return PowerResult(
        power=power,
        n_required=n_per_group,
        effect_size=min_effect_size,
        is_sufficient=power >= target_power,
    )


def check_power_sufficient(
    n_samples: int,
    test_type: str = "sibling",
    min_effect_size: float = 0.2,
    alpha: float = 0.05,
    min_power: float = 0.5,
    **kwargs,
) -> bool:
    """Quick check if power is sufficient for a test.

    Parameters
    ----------
    n_samples : int
        Sample size (or min of two for siblings)
    test_type : str
        'child_parent' or 'sibling'
    min_effect_size : float
        Minimum effect size to detect
    alpha : float
        Significance level
    min_power : float
        Minimum acceptable power (default: 0.5)
    **kwargs
        Additional args for specific test types

    Returns
    -------
    bool
        True if power is sufficient, False otherwise
    """
    if test_type == "child_parent":
        n_parent = kwargs.get("n_parent", n_samples * 2)
        result = compute_child_parent_power(
            n_child=n_samples,
            n_parent=n_parent,
            min_effect_size=min_effect_size,
            alpha=alpha,
            target_power=min_power,
        )
    else:  # sibling
        n_other = kwargs.get("n_other", n_samples)
        min_n = min(n_samples, n_other)
        result = compute_sibling_power(
            n_left=n_samples,
            n_right=n_other,
            min_effect_size=min_effect_size,
            alpha=alpha,
            target_power=min_power,
        )

    return result.is_sufficient


__all__ = [
    "PowerResult",
    "cohens_h",
    "power_wald_two_sample",
    "power_wald_nested",
    "compute_child_parent_power",
    "compute_sibling_power",
    "check_power_sufficient",
]
