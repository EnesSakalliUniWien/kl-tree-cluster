"""
Numerical walkthrough: INCLUDE_INTERNAL_IN_SPECTRAL effect on a real case.

Uses the 'binary_low_noise_4c' benchmark case (n=80, d=100, K=4).
Diagnostic showed: I=True → K=4 (correct), I=False → K=40 (catastrophic).

This script traces eigenvalues at the ROOT node for both configurations,
showing exactly where and why k diverges.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projection_dimension_estimation.projection_dimension_estimators import (
    marchenko_pastur_signal_count,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ── Generate the benchmark case ──────────────────────────────────────
leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
    n_rows=80,
    n_cols=100,
    entropy_param=0.05,
    n_clusters=4,
    random_seed=2001,
    balanced_clusters=True,
    feature_sparsity=0.05,
)
# Convert dict → DataFrame (sorted by leaf name)
leaf_names = sorted(leaf_matrix_dict.keys())
X = np.array([leaf_matrix_dict[n] for n in leaf_names], dtype=float)
true_labels = np.array([cluster_assignments[n] for n in leaf_names])
data = pd.DataFrame(X, index=leaf_names, columns=[f"F{j}" for j in range(X.shape[1])])

print("=" * 70)
print("CASE: binary_low_noise_4c  (n=80, d=100, K_true=4)")
print("=" * 70)
print(f"  Data shape: {X.shape}")
print(f"  Cluster sizes: {np.bincount(true_labels)}")
print(f"  Mean row sum: {X.sum(axis=1).mean():.1f}")

# ── Build tree ────────────────────────────────────────────────────────
Z = linkage(pdist(X, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
root = tree.root


# ── Collect leaf data for root subtree ────────────────────────────────
leaf_nodes = sorted(tree.get_leaves(), key=str)
leaf_indices = list(range(len(leaf_nodes)))  # all leaves are under root
leaf_data = X[leaf_indices, :]


# ── Collect internal node distributions ───────────────────────────────
internal_dists = []
for node in tree.nodes():
    if node not in set(leaf_nodes):
        dist = tree.nodes[node].get("distribution")
        if dist is not None:
            internal_dists.append(np.asarray(dist, dtype=np.float64))

# Populate distributions if needed
if not internal_dists:
    from kl_clustering_analysis.tree.distributions import populate_distributions

    populate_distributions(tree, data)
    for node in tree.nodes():
        if node not in set(leaf_nodes):
            dist = tree.nodes[node].get("distribution")
            if dist is not None:
                internal_dists.append(np.asarray(dist, dtype=np.float64))


# ── Helper: trace eigenvalue analysis ─────────────────────────────────
def trace_eigenvalues(label: str, feature_matrix: np.ndarray) -> int:
    """Run eigendecomposition + MP counting like the production code."""
    n, d_total = feature_matrix.shape

    result = eigendecompose_correlation_backend(
        feature_matrix,
        compute_eigenvectors=False,
    )
    if result is None:
        print(f"  {label}: eigendecomposition returned None")
        return 1

    eigvals = result.eigenvalues
    d_active = result.active_feature_count
    mode = "DUAL (n x n Gram)" if result.use_dual else "PRIMAL (d x d corr)"

    # Run MP signal count
    k = marchenko_pastur_signal_count(eigvals, n, d_active)

    # Reconstruct internals for display
    positive = eigvals[eigvals > 0]
    sigma2 = float(np.median(positive)) if positive.size > 0 else 0.0
    gamma = float(d_active) / float(n)
    lp = sigma2 * (1.0 + np.sqrt(gamma)) ** 2

    print(f"\n  {label}")
    print(f"  {'─' * 60}")
    print(f"  Matrix:  {n} rows x {d_total} cols  ({d_active} active features)")
    print(f"  Mode:    {mode}  =>  {len(eigvals)} eigenvalues")
    print(f"  Positive eigenvalues: {len(positive)} out of {len(eigvals)}")
    print()

    # Show eigenvalues: top signal, around the median, and bottom
    top_n = min(10, len(positive))
    print(f"  Top {top_n} positive eigenvalues:")
    for i in range(top_n):
        ev = eigvals[i]
        tag = " << SIGNAL" if ev > lp else ""
        print(f"    lambda_{i+1:3d} = {ev:.6f}{tag}")
    if len(positive) > top_n:
        print(f"    ... ({len(positive) - top_n} more positive eigenvalues)")

    # Mark the median
    median_idx = len(positive) // 2
    if len(positive) > 1:
        print(f"\n  Median eigenvalue (index {median_idx+1} of {len(positive)} positive):")
        lo = max(0, median_idx - 1)
        hi = min(len(positive), median_idx + 2)
        for j in range(lo, hi):
            marker = " <-- MEDIAN" if j == median_idx else ""
            if j == median_idx - 1 and len(positive) % 2 == 0:
                marker = " <-- MEDIAN (avg with next)"
            print(f"    lambda_{j+1:3d} = {positive[j]:.6f}{marker}")

    zero_count = len(eigvals) - len(positive)
    if zero_count > 0:
        print(f"\n  Zero eigenvalues: {zero_count} (rank deficit = {zero_count})")

    print("\n  Marchenko-Pastur calculation:")
    print(f"    sigma^2 = median(positive eigvals) = {sigma2:.6f}")
    print(f"    gamma   = d_active/n = {d_active}/{n} = {gamma:.4f}")
    print(f"    lambda+ = {sigma2:.6f} * (1 + sqrt({gamma:.4f}))^2")
    print(f"            = {sigma2:.6f} * {(1 + np.sqrt(gamma))**2:.4f}")
    print(f"            = {lp:.6f}")
    print(f"    k       = {k}  signal dimension(s)")
    return k


# ── Case 1: Leaves only ─────────────────────────────────────────────
print()
print("=" * 70)
print("ROOT NODE — LEAVES ONLY  (include_internal = False)")
print("=" * 70)
k1 = trace_eigenvalues("Leaves only", leaf_data)

# ── Case 2: Leaves + internal nodes ──────────────────────────────────
if internal_dists:
    augmented = np.vstack([leaf_data] + [v[np.newaxis, :] for v in internal_dists])
else:
    augmented = leaf_data

print()
print("=" * 70)
print("ROOT NODE — LEAVES + INTERNAL NODES  (include_internal = True)")
print("=" * 70)
k2 = trace_eigenvalues("Leaves + internals", augmented)


# ── Side-by-side summary ─────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY:  Root node spectral dimension (k)")
print("=" * 70)
print()
print(f"  include_internal = False:  k = {k1}")
print(f"  include_internal = True:   k = {k2}")
print()
if k1 != k2:
    print(f"  DIFFERENCE: k changes by {abs(k2 - k1)} dimensions.")
    print(f"  With k={k1} (I=False), the chi-square test at Gate 2 has df={k1}.")
    print(f"  With k={k2} (I=True),  the chi-square test at Gate 2 has df={k2}.")
    print()
    if k1 < k2:
        print(f"  Lower df={k1} means even small per-dimension differences accumulate")
        print("  enough to cross the chi-square threshold → Gate 2 fires on noise")
        print("  → over-splitting (diagnostic showed K=40 instead of K=4).")
    else:
        print(f"  Higher df={k1} without real signal in those directions is wasteful.")
else:
    print("  Same k at the root. The effect may appear at deeper subtree nodes")
    print("  where n is smaller and signal/noise eigenvalue mixing is worse.")

# ── Run full decomposition for both configs ───────────────────────────
print()
print("=" * 70)
print("FULL PIPELINE RESULT")
print("=" * 70)

orig_val = config.INCLUDE_INTERNAL_IN_SPECTRAL

config.INCLUDE_INTERNAL_IN_SPECTRAL = False
tree_false = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
res_false = tree_false.decompose(leaf_data=data)

config.INCLUDE_INTERNAL_IN_SPECTRAL = True
tree_true = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
res_true = tree_true.decompose(leaf_data=data)

config.INCLUDE_INTERNAL_IN_SPECTRAL = orig_val

print(f"\n  include_internal = False:  K = {res_false['num_clusters']}  (true K = 4)")
print(f"  include_internal = True:   K = {res_true['num_clusters']}  (true K = 4)")


# ======================================================================
# PART 2: SMALL SUBTREE DRILL-DOWN
# ======================================================================
# The root had n=80.  The real damage happens at SUBTREES where a single
# cluster's worth of leaves (n≈20) are isolated.  Here we simulate what
# the code sees at such a node.
# ======================================================================

from scipy.stats import chi2 as chi2_dist

print()
print()
print("#" * 70)
print("#  PART 2 — SMALL SUBTREE ANALYSIS  (where the cascade happens)")
print("#" * 70)

# Pick one cluster's leaves (cluster 0, n=20)
cluster_id = 0
cluster_mask = true_labels == cluster_id
X_cluster = X[cluster_mask]
n_cluster = X_cluster.shape[0]

print(f"\n  Subtree: cluster {cluster_id} leaves only")
print(f"  n = {n_cluster} leaves,  d = {X_cluster.shape[1]} features")

# Collect internal distributions for nodes whose ALL leaves are in this cluster
# For simplicity, we build fake internal nodes as progressive averages
# (this matches what the tree would compute for a balanced sub-tree)
cluster_internal = []
n_c = n_cluster
# Build a binary-tree worth of internal nodes from bottom up
level_rows = [X_cluster[i : i + 1] for i in range(n_c)]
while len(level_rows) > 1:
    next_level = []
    for i in range(0, len(level_rows), 2):
        if i + 1 < len(level_rows):
            parent = np.vstack([level_rows[i], level_rows[i + 1]]).mean(axis=0, keepdims=True)
            cluster_internal.append(parent[0])
            next_level.append(np.vstack([level_rows[i], level_rows[i + 1]]))
        else:
            next_level.append(level_rows[i])
    level_rows = next_level

n_internal = len(cluster_internal)
X_cluster_aug = np.vstack([X_cluster] + [v[np.newaxis, :] for v in cluster_internal])
n_cluster_aug = X_cluster_aug.shape[0]

print(f"  Internal nodes synthesized: {n_internal}")
print(f"  n_aug = {n_cluster} + {n_internal} = {n_cluster_aug}")
print()

# ── Eigenvalues: leaves only ─────────────────────────────────────────
print("=" * 70)
print(f"SUBTREE — LEAVES ONLY  (n={n_cluster}, d=100)")
print("=" * 70)
k_sub1 = trace_eigenvalues(f"Cluster {cluster_id}: leaves only", X_cluster)

# ── Eigenvalues: augmented ────────────────────────────────────────────
print()
print("=" * 70)
print(f"SUBTREE — LEAVES + INTERNAL NODES  (n_aug={n_cluster_aug}, d=100)")
print("=" * 70)
k_sub2 = trace_eigenvalues(f"Cluster {cluster_id}: augmented", X_cluster_aug)


# ── Gate 2 simulation ────────────────────────────────────────────────
# Show what happens when Gate 2 tests a child vs parent at this subtree.
# We simulate: parent = full cluster distribution, child = half the cluster.
print()
print("=" * 70)
print("GATE 2 SIMULATION — child (n=10) vs parent (n=20) within one cluster")
print("=" * 70)

half = n_cluster // 2
child_data = X_cluster[:half]
parent_data = X_cluster
child_dist = child_data.mean(axis=0)
parent_dist = parent_data.mean(axis=0)

# Compute the Wald statistic for different k values
# T = sum_j (child_j - parent_j)^2 / Var_j
# where Var_j = parent_j*(1-parent_j) * (1/n_child - 1/n_parent)
eps = 1e-9
theta = np.clip(parent_dist, eps, 1 - eps)
var_per_feature = theta * (1 - theta) * (1.0 / half - 1.0 / n_cluster)

diff = child_dist - parent_dist
wald_per_feature = diff**2 / np.where(var_per_feature > 0, var_per_feature, 1.0)

# With projection: T = sum of top-k wald contributions (simplified)
# In practice, projection rotates into eigenvector space. For illustration,
# sort wald contributions and accumulate.
sorted_wald = np.sort(wald_per_feature)[::-1]

print(f"\n  child:  mean of L0-L{half-1}  (n_child = {half})")
print(f"  parent: mean of L0-L{n_cluster-1}  (n_parent = {n_cluster})")
print("  These are SAME cluster — no real divergence. Gate 2 should NOT fire.")
print()

for k_test in [k_sub1, k_sub2, 3, 1]:
    # Approximate projected T as sum of top-k Wald contributions
    T_proj = float(np.sum(sorted_wald[:k_test]))
    p_value = 1.0 - chi2_dist.cdf(T_proj, df=k_test)
    fires = p_value < 0.001  # EDGE_ALPHA
    verdict = "FIRES (false positive!)" if fires else "does not fire (correct)"
    print(f"  k={k_test:2d}:  T={T_proj:8.3f},  df={k_test},  p={p_value:.6f}  →  Gate 2 {verdict}")

print()
print("  INTERPRETATION:")
print(f"  With k={k_sub1} (I=False), the test accumulates {k_sub1} dimensions of noise.")
print(f"  With k={k_sub2} (I=True),  the test uses {k_sub2} dimensions — typically closer")
print("  to the actual signal rank, so the chi-square df matches the projection.")
print()
print("  When k is WRONG (too high or too low for the actual signal structure):")
print("  • k too high → noise dimensions inflate T → false positives → over-splitting")
print("  • k too low  → misses signal directions → false negatives at higher levels")
print("  Both errors cascade: each wrong Gate 2 decision creates a new subtree")
print("  where the SAME k-estimation problem recurs at even smaller n.")

# ── Summary table ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUBTREE SUMMARY")
print("=" * 70)
print()
print(f"  {'':30s} {'I=False':>12s}   {'I=True':>12s}")
print(f"  {'-' * 58}")
print(f"  {'n (rows)':30s} {n_cluster:>12d}   {n_cluster_aug:>12d}")
print(
    f"  {'n vs d':30s} {'n < d (DUAL)':>12s}   {('n > d (PRIMAL)' if n_cluster_aug > 100 else 'n < d (DUAL)'):>12s}"
)
print(f"  {'# eigenvalues':30s} {n_cluster:>12d}   {min(n_cluster_aug, 100):>12d}")

result_sub1 = eigendecompose_correlation_backend(X_cluster, compute_eigenvectors=False)
result_sub2 = eigendecompose_correlation_backend(X_cluster_aug, compute_eigenvectors=False)
pos1 = int(np.sum(result_sub1.eigenvalues > 0)) if result_sub1 else 0
pos2 = int(np.sum(result_sub2.eigenvalues > 0)) if result_sub2 else 0
print(f"  {'# positive eigenvalues':30s} {pos1:>12d}   {pos2:>12d}")
print(f"  {'k (signal dimensions)':30s} {k_sub1:>12d}   {k_sub2:>12d}")
print()
print(f"  With n={n_cluster} leaves and d=100:")
print(f"  - DUAL mode gives {n_cluster} eigenvalues, ALL positive (no zeros)")
print("  - median(positive) is contaminated by signal eigenvalues")
print("  - sigma^2 too high → lambda+ too high → k may be wrong")
print()
print(f"  With n_aug={n_cluster_aug} rows:")
if n_cluster_aug > 100:
    print(f"  - PRIMAL mode: 100 eigenvalues, {100 - pos2} zeros separate signal from noise")
else:
    print(f"  - Still DUAL but {n_cluster_aug} eigenvalues (vs {n_cluster})")
    print("  - More eigenvalues → better median → cleaner sigma^2")
print("  - The cascade: wrong k at this level → wrong Gate 2 decisions")
print("    → creates child subtrees with even smaller n → even worse k estimation")
