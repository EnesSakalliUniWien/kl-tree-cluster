"""Central configuration for the KL-TE clustering analysis library."""

# --- Statistical Parameters ---

# Default significance level (alpha) for sibling-divergence gating in clustering.
# This is intentionally conservative to reduce over-merging at high levels
# of the tree.
SIBLING_ALPHA: float = 0.01

# --- Decomposition Parameters ---

# Default significance level for edge-divergence (child-vs-parent) tests.
EDGE_ALPHA: float = 0.001


# --- Legacy Branch-Length Variance Scaling ---

# Scale Wald test variance by normalized branch length.
# Disabled: empirical comparison shows branch-length scaling systematically
# under-splits on data-dependent trees (mean ARI 0.694 → 1.000 without it).
FELSENSTEIN_SCALING: bool = False


# --- Tree Inference Parameters ---

# Distance metric for hierarchical clustering
# Options: 'hamming', 'rogerstanimoto', 'jaccard', 'dice', 'euclidean'
# Hamming is the simplest, most standard binary distance with no arbitrary weighting
TREE_DISTANCE_METRIC: str = "hamming"

# Linkage method for hierarchical clustering
# Options: 'average', 'complete', 'single', 'ward'
# Average (UPGMA) produces more balanced trees than complete linkage
TREE_LINKAGE_METHOD: str = "average"

# Include internal-node distribution vectors in the spectral data matrix.
# Internal distributions are convex combinations of leaf data — they do NOT
# increase rank but inflate n_desc, which tightens the MP noise bounds
# (smaller √(d/n)) and makes signal detection slightly more sensitive.
# WARNING: Setting to False catastrophically inflates edge-gate false positives
# on null data (edge_T1 goes from 2% to 89%).
# Keeping True is recommended for consistency.
INCLUDE_INTERNAL_IN_SPECTRAL: bool = True

# --- Pass-Through Traversal ---

# When True, the DFS traversal continues past nodes where the sibling gate fails
# (siblings declared "same") IF any descendant has a significant sibling
# split (Sibling_BH_Different == True).  This prevents deep structure from
# being masked by a non-significant split at a higher level.  The descendant-
# signal flag is precomputed bottom-up in O(n) so it adds no cost to the
# hot path.  When False, the original greedy behaviour is used: sibling-gate
# failure immediately merges all descendants into one cluster.
PASSTHROUGH: bool = True
