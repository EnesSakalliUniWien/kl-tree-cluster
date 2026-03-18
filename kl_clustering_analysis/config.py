"""
Central configuration for the KL-TE clustering analysis library.
"""

from typing import Literal

# --- Statistical Parameters ---

# Default significance level (alpha) for sibling-divergence gating in clustering.
# This is intentionally conservative to reduce over-merging at high levels
# of the tree.
SIBLING_ALPHA: float = 0.01

# --- Decomposition Parameters ---

# Default significance level for Gate 2 edge (child-vs-parent) tests.
EDGE_ALPHA: float = 0.001


# --- Legacy Branch-Length Variance Scaling ---

# Scale Wald test variance by normalized branch length.
# Disabled: empirical comparison shows branch-length scaling systematically
# under-splits on data-dependent trees (mean ARI 0.694 → 1.000 without it).
FELSENSTEIN_SCALING: bool = False

# Explicit branch-length semantics for pairwise path distances used by
# branch-length scaling in localization and post-hoc merge:
# - "phylogeny": use edge ``branch_length`` values as true path lengths.
# - "topology": ignore edge values and use unit hop-length per edge.
FELSENSTEIN_BRANCH_LENGTH_MODE: Literal["phylogeny", "topology"] = "topology"

# Behavior when branch-length scaling is enabled in phylogeny mode but some edges have
# missing/invalid branch lengths:
# - "warn_disable": emit a warning and disable branch-length scaling for that run.
# - "error": raise ValueError and fail fast.
FELSENSTEIN_INCOMPLETE_BRANCH_POLICY: Literal["warn_disable", "error"] = "warn_disable"

# --- Tree Inference Parameters ---

# Distance metric for hierarchical clustering
# Options: 'hamming', 'rogerstanimoto', 'jaccard', 'dice', 'euclidean'
# Hamming is the simplest, most standard binary distance with no arbitrary weighting
TREE_DISTANCE_METRIC: str = "hamming"

# Linkage method for hierarchical clustering
# Options: 'average', 'complete', 'single', 'ward'
# Average (UPGMA) produces more balanced trees than complete linkage
TREE_LINKAGE_METHOD: str = "average"

# --- Spectral Dimension Estimation ---

# Per-node projection dimension method.  Uses eigendecomposition of the local
# correlation matrix at each node to determine signal rank.
# Options:
#   "marchenko_pastur"    - Count eigenvalues above MP upper bound (default)
# Marchenko-Pastur uses random matrix theory to separate signal eigenvalues
# from the noise bulk.  For the correlation matrix σ²=1 exactly, so the
# MP bounds are (1±√(d/n))².  When MP yields k=0 (no signal eigenvalues),
# the test is skipped and the node is treated as a merge.
SPECTRAL_METHOD: str | None = "marchenko_pastur"

# Minimum projection dimension for the SPECTRAL (Gate 2) path only.
# With Marchenko-Pastur, noise nodes get k=1 (no signal eigenvalues above
# the MP bound).  The floor of 2 prevents χ²(1) tests, which have a
# singularity at 0 that can cause numerical instability.  At pure-noise
# subtrees, the χ²(2) test correctly fails to reject.
SPECTRAL_MINIMUM_DIMENSION: int = 2

# Include internal-node distribution vectors in the spectral data matrix.
# Internal distributions are convex combinations of leaf data — they do NOT
# increase rank but inflate n_desc, which tightens the MP noise bounds
# (smaller √(d/n)) and makes signal detection slightly more sensitive.
# WARNING: Setting to False catastrophically inflates Gate 2 false positives
# on null data (edge_T1 goes from 2% to 89%).
# Keeping True is recommended for consistency.
INCLUDE_INTERNAL_IN_SPECTRAL: bool = True

# --- Sibling Test Method ---

# Method for sibling divergence testing.
# Options:
#   "wald"                  - Standard projected Wald χ² test (original, known anti-conservative)
#   "cousin_adjusted_wald"  - Cousin-adjusted Wald: estimates post-selection inflation c
#                             from null-like pairs (neither child edge-significant) via
#                             one log-linear regression on (BL_sum, n_parent), then deflates
#                             focal pair stats: T_adj = T / ĉ ~ χ²(k).
#                             Production default.
SIBLING_TEST_METHOD: str = "cousin_adjusted_wald"

# --- Sibling Whitening Mode ---
# Controls how projected statistics are converted to p-values in Gate 3.
#   "per_component"   - T = Σ (vᵢᵀz)²/λᵢ ~ χ²(k).  Exact under H₀ when
#                        eigenvalues come from null covariance.  Default.
#   "satterthwaite"   - T = Σ (vᵢᵀz)² (unwhitened), referenced against a
#                        moment-matched c × χ²(ν).  Use when eigenvalues
#                        are contaminated by between-group signal (e.g.
#                        parent PCA for sibling test).
SIBLING_WHITENING: Literal["per_component", "satterthwaite"] = "satterthwaite"

# --- Pass-Through Traversal ---

# When True, the DFS traversal continues past nodes where Gate 3 fails
# (siblings declared "same") IF any descendant has a significant sibling
# split (Sibling_BH_Different == True).  This prevents deep structure from
# being masked by a non-significant split at a higher level.  The descendant-
# signal flag is precomputed bottom-up in O(n) so it adds no cost to the
# hot path.  When False, the original greedy behaviour is used: Gate 3
# failure immediately merges all descendants into one cluster.
PASSTHROUGH: bool = True
