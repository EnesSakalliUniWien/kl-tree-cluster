"""Diagnose projection dimension behaviour for high-d / low-n cases.

Tests: gaussian_extreme_noise_2 (n=300, d=2000, K=30)
Shows per-node k values from JL, rank-cap, and spectral method.
Profiles overall pipeline timing.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs

# ── Setup ────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    compute_projection_dimension,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral_dimension import (
    compute_spectral_decomposition,
)
from kl_clustering_analysis.tree.io import tree_from_linkage

# ── 1.  Show projection dimension table ──────────────────────────────
print("=" * 72)
print("1.  Projection dimension: JL vs rank-cap vs spectral")
print("=" * 72)

d = 2000
header = f"{'n':>6}  {'d':>6}  {'k (current)':>12}  {'rank_cap':>10}  {'n<d?':>5}"
print(header)
print("-" * len(header))

for n in [2, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300]:
    k = compute_projection_dimension(n, d)
    rank_cap = n if n < d else d
    print(f"{n:6d}  {d:6d}  {k:12d}  {rank_cap:10d}  {'yes' if n < d else 'no':>5}")

# Also test n >= d case (should NOT be capped)
print()
for n, d2 in [(30, 30), (100, 50), (200, 100)]:
    k = compute_projection_dimension(n, d2)
    print(f"{n:6d}  {d2:6d}  {k:12d}  {'n/a (n>=d)':>10}  {'yes' if n < d2 else 'no':>5}")


# ── 2.  Generate the extreme-noise test case ─────────────────────────
print()
print("=" * 72)
print("2.  gaussian_extreme_noise_2: n=300, d=2000, K=30")
print("=" * 72)

n_samples = 300
n_features = 2000
n_clusters = 30
cluster_std = 2.0
seed = 44

X, y_true = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    cluster_std=cluster_std,
    random_state=seed,
)

# Median-binarise
X_binary = (X > np.median(X, axis=0)).astype(int)
data_df = pd.DataFrame(
    X_binary,
    index=[f"S{i}" for i in range(n_samples)],
    columns=[f"F{j}" for j in range(n_features)],
)
print(f"Data shape: {data_df.shape}")
print(f"True K: {n_clusters}")
print(f"Samples per cluster: {np.bincount(y_true).tolist()[:5]}... (showing first 5)")


# ── 3.  Build tree and measure timing ────────────────────────────────
print()
print("=" * 72)
print("3.  Build tree + spectral decomposition timing")
print("=" * 72)

t0 = time.perf_counter()
dist_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
t_dist = time.perf_counter() - t0
print(f"  pdist:    {t_dist:.2f}s")

t0 = time.perf_counter()
Z = linkage(dist_condensed, method=config.TREE_LINKAGE_METHOD)
t_link = time.perf_counter() - t0
print(f"  linkage:  {t_link:.2f}s")

t0 = time.perf_counter()
tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())
t_tree = time.perf_counter() - t0
print(f"  tree:     {t_tree:.2f}s")

# Populate distributions
t0 = time.perf_counter()
tree.populate_node_divergences(data_df)
t_pop = time.perf_counter() - t0
print(f"  populate: {t_pop:.2f}s")

n_internal = sum(1 for n in tree.nodes if not tree.nodes[n].get("is_leaf", False))
n_edges = tree.number_of_edges()
print(f"  Internal nodes: {n_internal}, Edges: {n_edges}")


# ── 4.  Spectral decomposition ───────────────────────────────────────
print()
print("=" * 72)
print("4.  Spectral decomposition (effective_rank)")
print("=" * 72)

t0 = time.perf_counter()
spectral_dims, pca_projs, pca_evals = compute_spectral_decomposition(
    tree,
    data_df,
    method="effective_rank",
    min_k=config.PROJECTION_MIN_K,
    compute_projections=False,
)
t_spectral = time.perf_counter() - t0
print(f"  Time: {t_spectral:.2f}s")

# Show distribution of spectral k values for internal nodes
internal_spectral_ks = {
    nid: k for nid, k in spectral_dims.items() if not tree.nodes[nid].get("is_leaf", False)
}
ks_sorted = sorted(internal_spectral_ks.values())
print(f"  Internal nodes: {len(ks_sorted)}")
print(
    f"  Spectral k: min={min(ks_sorted)}, median={np.median(ks_sorted):.0f}, "
    f"mean={np.mean(ks_sorted):.1f}, max={max(ks_sorted)}"
)
print(f"  Histogram: {np.histogram(ks_sorted, bins=[1, 5, 10, 20, 50, 100, 200, 500, 2001])[0]}")


# ── 5.  Compare JL k vs spectral k vs rank-cap at sample nodes ──────
print()
print("=" * 72)
print("5.  Per-node comparison: JL k vs rank-cap k vs spectral k")
print("=" * 72)

# Gather leaf counts per internal node
internal_nodes = [nid for nid in tree.nodes if not tree.nodes[nid].get("is_leaf", False)]

rows = []
for nid in internal_nodes:
    n_desc = tree.nodes[nid].get("leaf_count", 0)
    if n_desc < 1:
        continue
    k_jl_raw = compute_projection_dimension(n_desc, n_features)
    k_spectral = spectral_dims.get(nid, -1)
    rows.append(
        {
            "node": nid,
            "n_desc": n_desc,
            "k_jl_rankcap": k_jl_raw,
            "k_spectral": k_spectral,
        }
    )

df_cmp = pd.DataFrame(rows).sort_values("n_desc")

# Show a selection: smallest, some mid, largest
sample_idx = (
    list(range(min(5, len(df_cmp))))
    + [len(df_cmp) // 4, len(df_cmp) // 2, 3 * len(df_cmp) // 4]
    + list(range(max(0, len(df_cmp) - 5), len(df_cmp)))
)
sample_idx = sorted(set(i for i in sample_idx if 0 <= i < len(df_cmp)))

print(f"{'node':>10}  {'n_desc':>7}  {'k_jl+cap':>8}  {'k_spectral':>10}")
print("-" * 40)
prev = -1
for idx in sample_idx:
    if idx == prev:
        continue
    prev = idx
    r = df_cmp.iloc[idx]
    print(
        f"{r['node']:>10}  {int(r['n_desc']):7d}  {int(r['k_jl_rankcap']):8d}  "
        f"{int(r['k_spectral']):10d}"
    )


# ── 6. Full pipeline timing: decompose() ─────────────────────────────
print()
print("=" * 72)
print("6.  Full decompose() timing")
print("=" * 72)

# Re-build tree (clean state)
tree2 = tree_from_linkage(Z, leaf_names=data_df.index.tolist())

t0 = time.perf_counter()
result = tree2.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
t_decompose = time.perf_counter() - t0

K_found = result.get("num_clusters", "?")
print(f"  Time: {t_decompose:.2f}s")
print(f"  Found K={K_found} (true K={n_clusters})")

# ARI
from sklearn.metrics import adjusted_rand_score

if "cluster_assignments" in result:
    labels_pred = np.full(n_samples, -1, dtype=int)
    for cluster_id, (_, leaf_list) in enumerate(result["cluster_assignments"].items()):
        for leaf in leaf_list:
            if leaf.startswith("S") and leaf[1:].isdigit():
                idx = int(leaf[1:])
                labels_pred[idx] = cluster_id
    ari = adjusted_rand_score(y_true, labels_pred)
    print(f"  ARI: {ari:.3f}")

print()
print("Done.")
