"""
Central configuration for the KL-TE clustering analysis library.
"""

# --- Statistical Parameters ---

# Default significance level (alpha) for hypothesis tests.
SIGNIFICANCE_ALPHA: float = 0.05

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
