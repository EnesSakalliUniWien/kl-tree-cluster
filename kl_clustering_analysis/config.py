"""
Central configuration for the KL-TE clustering analysis library.
"""

import numpy as np

# --- Statistical Parameters ---

# Default significance level (alpha) for hypothesis tests.
SIGNIFICANCE_ALPHA: float = 0.05

# Default significance level (alpha) for sibling-independence *gating* in clustering.
# This is intentionally more conservative than SIGNIFICANCE_ALPHA because the sibling
# CMI permutation test can be noisy at low permutation counts; using a smaller alpha
# reduces over-merging at high levels of the tree.
SIBLING_ALPHA: float = 0.01

# Default number of permutations for permutation tests.
N_PERMUTATIONS: int = 100

# Default standard deviation threshold for z-score-based deviation tests.
STD_DEVIATION_THRESHOLD: float = 2.0

# Epsilon value for numerical stability in KL-divergence and probability calculations.
EPSILON: float = 1e-9

# --- Attention Mechanism Parameters ---
ATTENTION_TAU: float = 1.0
ATTENTION_GAMMA: float = 1.0
ATTENTION_N_ITERATIONS: int = 10

# --- Decomposition Parameters ---

# Default significance level for local (child-vs-parent) tests in decomposition.
ALPHA_LOCAL: float = 0.05

# --- Adaptive Significance Level ---

# Enable adaptive α based on sample-to-dimension ratio
# Set to False by default - only enable after validation
USE_ADAPTIVE_ALPHA: bool = False

# Exponent for adaptive α scaling: α_eff = α * (n/df)^exponent
# - 0.0: No scaling (always use base α)
# - 0.5: Square root scaling (moderate adjustment)
# - 1.0: Linear scaling (aggressive adjustment)
ADAPTIVE_ALPHA_EXPONENT: float = 0.5


def compute_adaptive_alpha(
    base_alpha: float,
    sample_size: float,
    degrees_of_freedom: float,
    exponent: float = ADAPTIVE_ALPHA_EXPONENT,
) -> float:
    """Compute sample-size-adjusted significance level.

    When sample size (n) is smaller than degrees of freedom (df), the chi-square
    approximation becomes unreliable and p-values are inflated. This function
    scales α upward to compensate for reduced power.

    The scaling is: α_eff = α * min(1.0, (n/df)^exponent)

    Parameters
    ----------
    base_alpha : float
        Original significance level (e.g., 0.05)
    sample_size : float
        Effective sample size (n or harmonic mean for two-sample tests)
    degrees_of_freedom : float
        Effective degrees of freedom from the test
    exponent : float
        Scaling exponent (0.5 = square root, 1.0 = linear)

    Returns
    -------
    float
        Adjusted significance level, always in [base_alpha, 1.0]

    Examples
    --------
    >>> compute_adaptive_alpha(0.05, 100, 50)  # n > df: well-powered
    0.05
    >>> compute_adaptive_alpha(0.05, 50, 200)  # n < df: underpowered
    0.1  # α scaled up to compensate
    """
    if degrees_of_freedom <= 0:
        return base_alpha

    ratio = sample_size / degrees_of_freedom

    if ratio >= 1.0:
        # Well-powered test: use standard α
        return base_alpha

    # Underpowered: scale α upward (more liberal to compensate)
    # α_eff = α / (ratio^exponent) but capped at reasonable level
    scaling_factor = 1.0 / (ratio ** exponent) if ratio > 0 else 1.0

    # Cap the scaling to avoid absurdly high α
    max_scaling = 4.0  # α can be scaled up by at most 4x
    scaling_factor = min(scaling_factor, max_scaling)

    return min(base_alpha * scaling_factor, 0.20)  # Never exceed α = 0.20
