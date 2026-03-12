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
EDGE_ALPHA: float = 0.01


# Epsilon value for numerical stability in KL-divergence and probability calculations.
EPSILON: float = 1e-9


# --- Felsenstein Branch-Length Scaling ---

# Scale Wald test variance by normalised branch length (Felsenstein PIC, 1985).
# Disabled: empirical comparison shows Felsenstein scaling systematically
# under-splits on data-dependent trees (mean ARI 0.694 → 1.000 without it).
FELSENSTEIN_SCALING: bool = False

# Explicit branch-length semantics for pairwise path distances used by
# Felsenstein scaling in localization and post-hoc merge:
# - "phylogeny": use edge ``branch_length`` values as true path lengths.
# - "topology": ignore edge values and use unit hop-length per edge.
FELSENSTEIN_BRANCH_LENGTH_MODE: Literal["phylogeny", "topology"] = "topology"

# Behavior when scaling is enabled in phylogeny mode but some edges have
# missing/invalid branch lengths:
# - "warn_disable": emit a warning and disable Felsenstein scaling for that run.
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

# --- Random Projection Parameters ---

# Random projection is integral to the Wald chi-square test and always applied.
# The projection dimension k is computed adaptively via the JL lemma:
# k = compute_projection_dimension(n_samples, n_features).
# When n_features <= k, no effective dimensionality reduction occurs.

# Distortion tolerance for Johnson-Lindenstrauss projection.
# Controls the trade-off between dimension reduction and distance preservation:
#   eps=0.1 -> ±10% distortion, many dimensions (conservative)
#   eps=0.3 -> ±30% distortion, moderate dimensions (good for hypothesis testing)
#   eps=0.5 -> ±50% distortion, few dimensions (aggressive)
# Uses sklearn's johnson_lindenstrauss_min_dim for theoretically-grounded dimension.
PROJECTION_EPS: float = 0.3

# Minimum projected dimension.
# Set to an integer for a fixed floor, or "auto" to estimate from the data's
# effective rank (Shannon entropy of eigenvalue spectrum).  When "auto", the
# floor is computed once per pipeline run as:
#   minimum_projection_dimension = max(2, min(ceil(effective_rank(full_data)), 20))
# This prevents adding pure-noise χ² components when the data has low
# intrinsic dimensionality, and avoids under-projecting high-rank data.
PROJECTION_MINIMUM_DIMENSION: int | str = "auto"


# Random seed for projection reproducibility (None for random)
PROJECTION_RANDOM_SEED: int | None = 42

# --- Spectral Dimension Estimation ---

# Per-node projection dimension method.  When set, replaces JL-based dimension
# selection with eigendecomposition of the local correlation matrix at each node.
# Options:
#   None                  - Legacy JL-based dimension
#   "effective_rank"      - Shannon entropy of eigenvalue spectrum (Roy & Vetterli 2007)
#   "marchenko_pastur"    - Count eigenvalues above MP upper bound (default)
#   "active_features"     - Count features with non-zero variance (no eigendecomp)
# Marchenko-Pastur is the default: it uses random matrix theory to separate
# signal eigenvalues from the noise bulk.  For the correlation matrix σ²=1
# exactly, so the MP bounds are (1±√(d/n))².  Empirically, MP has MAE 1.3
# vs effective_rank's MAE 16+ when compared to true signal rank.
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
# on null data with effective_rank (edge_T1 goes from 2% to 89%).
# With Marchenko-Pastur the effect is milder but keeping True is still
# recommended for consistency.
INCLUDE_INTERNAL_IN_SPECTRAL: bool = True

# --- Edge (Gate 2) Calibration ---

# Calibrate edge test statistics via eigenvalue-weighted Gamma GLM.
# When True, estimates post-selection inflation factor ĉ_edge from
# null-like edges (high leaf-count ratio × flat eigenvalue spectrum)
# and deflates all edge statistics: T_adj = T / ĉ_edge.
# Analogous to the sibling test's cousin-weighted Wald calibration.
EDGE_CALIBRATION: bool = False

# Adaptive sibling-aware filtering for edge calibration.
# When edge calibration has low effective sample size (effective_n),
# calibration can be contaminated by a few high-signal edges and massively
# over-deflate (ĉ >> 1). In that regime we exclude edges whose parent
# sibling test is already significant (Sibling_BH_Different=True) from the
# calibration fit. For well-powered calibration (effective_n >= threshold),
# the filter is not applied.
EDGE_CAL_MIN_EFFECTIVE_N_FOR_SIB_FILTER: float = 10.0

# --- Eigenvalue Whitening ---

# How to handle eigenvalues when PCA eigenvectors are used for projection.
# Options:
#   True   - Whitened: T = Σ (vᵢᵀz)²/λᵢ ~ χ²(k)  (exact under H₀,
#            but dividing by large λᵢ dampens signal → lower power)
#   False  - Unwhitened with Satterthwaite correction:
#            T = Σ (vᵢᵀz)² ~ Σ λᵢ·χ²(1), approximate as c·χ²(ν)
#            where c = Σλᵢ²/Σλᵢ, ν = (Σλᵢ)²/Σλᵢ² (preserves power)
# Satterthwaite preserves power because signal in high-eigenvalue
# directions is not dampened by dividing by λᵢ.
# WARNING: Satterthwaite (False) is anti-conservative on null Gaussian data —
# the approximate χ²(ν) calibration breaks down when the eigenspectrum is flat,
# causing massive Gate 2 false positive inflation (e.g. gauss_null K=1 → K=18).
EIGENVALUE_WHITENING: bool = True

# --- Sibling Test Method ---

# Method for sibling divergence testing.
# Options:
#   "wald"                  - Standard projected Wald χ² test (original, known anti-conservative)
#   "cousin_ftest"          - Cousin-calibrated F-test: uses uncle's sibling split as reference
#                             to cancel post-selection bias. Falls back to Wald when cousin
#                             reference is unavailable (uncle is leaf or root's children).
#                             Empirically calibrated: 5.0% Type I under null, KS p=0.98.
#   "cousin_adjusted_wald"  - Cousin-adjusted Wald: estimates post-selection inflation c
#                             from null-like pairs (neither child edge-significant) via
#                             log-linear regression on (BL_sum, n_parent), then deflates
#                             focal pair stats: T_adj = T / ĉ ~ χ²(k).
#                             Preserves power better than F-test for multi-cluster cases.
#   "cousin_tree_guided"    - Tree-guided cousin: walks up the tree from each focal pair
#                             to find topologically nearest null-like relatives, uses their
#                             median T/k as local ĉ. No global regression — adapts to local
#                             tree structure. Falls back to global median when no local
#                             null-like pairs are found.
#   "cousin_weighted_wald"  - Cousin-weighted Wald: like cousin_adjusted_wald but
#                             uses ALL sibling pairs in calibration regression, weighted by
#                             edge p-values: w_i = min(p_edge_L, p_edge_R). Continuous
#                             weighting avoids hard binary null/non-null split; more stable.
#                             NOTE: empirical null FPR ≈ 68–100% — does NOT correct
#                             post-selection inflation. Kept for comparison only.
SIBLING_TEST_METHOD: str = "cousin_adjusted_wald"

# --- Pass-Through Traversal ---

# When True, the DFS traversal continues past nodes where Gate 3 fails
# (siblings declared "same") IF any descendant has a significant sibling
# split (Sibling_BH_Different == True).  This prevents deep structure from
# being masked by a non-significant split at a higher level.  The descendant-
# signal flag is precomputed bottom-up in O(n) so it adds no cost to the
# hot path.  When False, the original greedy behaviour is used: Gate 3
# failure immediately merges all descendants into one cluster.
PASSTHROUGH: bool = True
