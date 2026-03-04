"""
Central configuration for the KL-TE clustering analysis library.
"""

from typing import Literal

# --- Statistical Parameters ---

# Default significance level (alpha) for hypothesis tests.
SIGNIFICANCE_ALPHA: float = 0.01

# Default significance level (alpha) for sibling-divergence gating in clustering.
# This is intentionally more conservative than SIGNIFICANCE_ALPHA to reduce
# over-merging at high levels of the tree.
SIBLING_ALPHA: float = 0.01

# --- Decomposition Parameters ---

# Default significance level for local (child-vs-parent) tests in decomposition.
ALPHA_LOCAL: float = 0.01


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
#   min_k = max(2, min(ceil(effective_rank(full_data)), 20))
# This prevents adding pure-noise χ² components when the data has low
# intrinsic dimensionality, and avoids under-projecting high-rank data.
PROJECTION_MIN_K: int | str = "auto"


# Random seed for projection reproducibility (None for random)
PROJECTION_RANDOM_SEED: int | None = 42

# --- Spectral Dimension Estimation ---

# Per-node projection dimension method.  When set, replaces JL-based dimension
# selection with eigendecomposition of the local covariance at each node.
# Options:
#   None                  - Legacy JL-based dimension (default)
#   "effective_rank"      - Shannon entropy of eigenvalue spectrum (Roy & Vetterli 2007)
#   "marchenko_pastur"    - Count eigenvalues above MP upper bound
#   "active_features"     - Count features with non-zero variance (no eigendecomp)
SPECTRAL_METHOD: str | None = "effective_rank"

# Include internal-node distribution vectors in the spectral data matrix.
# Internal distributions are convex combinations of leaf data — they do NOT
# increase rank but shift the mean toward the global average, concentrating
# variance in the top PCs and REDUCING effective rank (typically ~30%).
# Recommended: False (leaves-only gives more accurate rank estimates).
INCLUDE_INTERNAL_IN_SPECTRAL: bool = False

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
EIGENVALUE_WHITENING: bool = False

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
