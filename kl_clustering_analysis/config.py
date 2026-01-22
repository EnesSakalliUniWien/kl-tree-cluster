"""
Central configuration for the KL-TE clustering analysis library.
"""


# --- Statistical Parameters ---

# Default significance level (alpha) for hypothesis tests.
SIGNIFICANCE_ALPHA: float = 0.05

# Default significance level (alpha) for sibling-divergence gating in clustering.
# This is intentionally more conservative than SIGNIFICANCE_ALPHA to reduce
# over-merging at high levels of the tree.
SIBLING_ALPHA: float = 0.05

# --- Decomposition Parameters ---

# Default significance level for local (child-vs-parent) tests in decomposition.
ALPHA_LOCAL: float = 0.05


# Epsilon value for numerical stability in KL-divergence and probability calculations.
EPSILON: float = 1e-9


# --- Post-Hoc Merge Parameters ---

# Enable tree-respecting post-hoc merge by default.
# This iteratively merges clusters whose underlying distributions are NOT
# significantly different, working bottom-up through the tree to reduce over-splitting.
POSTHOC_MERGE: bool = True

# Significance level for post-hoc merge tests.
# If None, defaults to SIBLING_ALPHA at runtime.
POSTHOC_MERGE_ALPHA: float | None = None

# --- Tree Inference Parameters ---

# Distance metric for hierarchical clustering
# Options: 'hamming', 'rogerstanimoto', 'jaccard', 'dice', 'euclidean'
# Rogers-Tanimoto double-weights mismatches, making it more sensitive to cluster boundaries
TREE_DISTANCE_METRIC: str = "rogerstanimoto"

# Linkage method for hierarchical clustering
# Options: 'average', 'complete', 'single', 'ward'
# Average (UPGMA) produces more balanced trees than complete linkage
TREE_LINKAGE_METHOD: str = "average"

# --- Random Projection Parameters ---

# Enable random projection for high-dimensional sibling tests
USE_RANDOM_PROJECTION: bool = True

# Projection decision (JL-based):
# Apply projection when n_features > n_samples AND the JL-computed target
# dimension k = compute_projection_dimension(n_samples, n_features) is less than n_features.
# Random projection is applied in the hypothesis tests via
# `statistics.random_projection.compute_projection_dimension`.

# Distortion tolerance for Johnson-Lindenstrauss projection.
# Controls the trade-off between dimension reduction and distance preservation:
#   eps=0.1 -> ±10% distortion, many dimensions (conservative)
#   eps=0.3 -> ±30% distortion, moderate dimensions (good for hypothesis testing)
#   eps=0.5 -> ±50% distortion, few dimensions (aggressive)
# Uses sklearn's johnson_lindenstrauss_min_dim for theoretically-grounded dimension.
PROJECTION_EPS: float = 0.3

# Minimum projected dimension
PROJECTION_MIN_K: int = 10


# Random seed for projection reproducibility (None for random)
PROJECTION_RANDOM_SEED: int | None = 42

# --- MI Feature Filter Parameters ---

# Enable MI-based feature filtering before projection
USE_MI_FEATURE_FILTER: bool = True

# Quantile threshold for MI filtering (keep features above this quantile)
MI_FILTER_QUANTILE: float = 0.5

# Minimum fraction of features to retain
MI_FILTER_MIN_FRACTION: float = 0.1


