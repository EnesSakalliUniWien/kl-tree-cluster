#!/usr/bin/env python3
"""Full numerical pipeline trace on a tiny 10×12 binary dataset."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ================================================================
# STAGE 0: Generate a 4-cluster binary dataset (block-diagonal)
# ================================================================
print("=" * 70)
print("STAGE 0: Input data  X ∈ {0,1}^{40×20}  (4 clusters × 10)")
print("=" * 70)
rng = np.random.default_rng(42)
n_per = 10  # samples per cluster
n_clusters = 4
d = 20  # 5 features per cluster
n = n_per * n_clusters  # 40 total

X = np.zeros((n, d), dtype=int)
# Block-diagonal: cluster k "owns" features [5k : 5k+5] with prob_one=0.9
# Background prob_one = 0.1
bg = 0.1
fg = 0.9

for k in range(n_clusters):
    rows = slice(k * n_per, (k + 1) * n_per)
    for j in range(d):
        p = fg if (k * 5 <= j < (k + 1) * 5) else bg
        X[rows, j] = rng.binomial(1, p, size=n_per)

labels = [f"S{i}" for i in range(n)]
features = [f"f{j}" for j in range(d)]
data = pd.DataFrame(X, index=labels, columns=features)
print(f"Shape: {n}×{d}, 4 clusters of {n_per}, block-diagonal signal")
print(f"First 6 rows:\n{data.head(6).to_string()}")
print(f"...\nLast 6 rows:\n{data.tail(6).to_string()}")
print("\nColumn sums per cluster:")
for k in range(n_clusters):
    rows = slice(k * n_per, (k + 1) * n_per)
    sums = X[rows].sum(axis=0)
    print(f"  Cluster {k} (S{k*n_per}..S{(k+1)*n_per-1}): {sums}")
print("True clusters: C0={S0-9}, C1={S10-19}, C2={S20-29}, C3={S30-39}")

# ================================================================
# STAGE 1: Distance matrix + linkage
# ================================================================
print("\n" + "=" * 70)
print("STAGE 1: pdist(hamming) → linkage(average)")
print("=" * 70)
D = pdist(X, metric="hamming")
D_sq = squareform(D)
print(f"Condensed distance vector: {len(D)} pairs")
print("\nWithin-cluster vs between-cluster hamming distances:")
within, between = [], []
for i in range(n):
    for j in range(i + 1, n):
        ci, cj = i // n_per, j // n_per
        if ci == cj:
            within.append(D_sq[i, j])
        else:
            between.append(D_sq[i, j])
print(f"  Within-cluster:  mean={np.mean(within):.3f}, std={np.std(within):.3f}")
print(f"  Between-cluster: mean={np.mean(between):.3f}, std={np.std(between):.3f}")

Z = linkage(D, method="average")
print(f"\nLinkage matrix ({Z.shape[0]} merges):")
print(f"{'merge':>6} {'left':>6} {'right':>6} {'dist':>10} {'size':>5}")
for i, row in enumerate(Z):
    l = f"L{int(row[0])}" if row[0] < n else f"N{int(row[0]) - n}"
    r = f"L{int(row[1])}" if row[1] < n else f"N{int(row[1]) - n}"
    print(f"  N{i:<4} {l:>5}  {r:>5}  {row[2]:>10.4f} {int(row[3]):>5}")

# ================================================================
# STAGE 2: PosetTree + node distributions
# ================================================================
print("\n" + "=" * 70)
print("STAGE 2: PosetTree construction + node distributions")
print("=" * 70)
from kl_clustering_analysis.tree.poset_tree import PosetTree

tree = PosetTree.from_linkage(Z, leaf_names=labels)
tree.populate_node_divergences(data)

root = next(n_ for n_, deg in tree.in_degree() if deg == 0)
print(f"Root node: {root}")
all_leaves = sorted(n_ for n_ in tree.nodes if tree.out_degree(n_) == 0)
print(f"Leaves: {all_leaves}")
print(f"Total nodes: {tree.number_of_nodes()}, edges: {tree.number_of_edges()}")

# Show distributions for root and its first-level children
print("\nNode distributions θ̂ (internal nodes only — leaves omitted):")
for node in sorted(tree.nodes, key=str):
    nd = tree.nodes[node]
    dist = nd.get("distribution")
    if dist is None:
        continue
    is_leaf = nd.get("is_leaf", False)
    if is_leaf:
        continue
    n_desc = nd.get("n_descendants", "?")
    label = nd.get("label", node)
    arr = np.array(dist)
    tag = f"n={n_desc}"
    # Show first 10 features for compactness
    short = ", ".join(f"{v:.2f}" for v in arr[:10])
    print(f"  {node:>5} ({tag:>6}):  [{short}, ...]")

# Branch lengths — just summary
bls = [
    tree.edges[p, c]["branch_length"]
    for p, c in tree.edges()
    if "branch_length" in tree.edges[p, c]
]
if bls:
    print(
        f"\nBranch lengths: n={len(bls)}, mean={np.mean(bls):.4f}, min={np.min(bls):.4f}, max={np.max(bls):.4f}"
    )
    # Show just the top-level edges from root
    for c in tree.successors(root):
        bl = tree.edges[root, c].get("branch_length")
        print(f"  {root} → {c}: {bl:.4f}")

# ================================================================
# STAGE 3: Fallback floor from effective rank
# ================================================================
print("\n" + "=" * 70)
print("STAGE 3: Fallback floor (k_floor from effective rank)")
print("=" * 70)

X_f = X.astype(float)
var = np.var(X_f, axis=0)
mask = var > 0
d_active = int(mask.sum())
print(f"Feature variances: {var}")
print(f"Non-constant features: {d_active}/{d}")

X_active = X_f[:, mask]
C = np.corrcoef(X_active.T)
C = np.nan_to_num(C, nan=0.0)
np.fill_diagonal(C, 1.0)
eigs = np.sort(np.linalg.eigvalsh(C))[::-1]
eigs = np.maximum(eigs, 0.0)
print(f"\nCorrelation matrix eigenvalues (d_active={d_active}):")
for i, lam in enumerate(eigs):
    bar = "█" * max(1, int(lam * 3))
    print(f"  λ_{i + 1:>2} = {lam:7.4f}  {bar}")

p = eigs / eigs.sum()
H = -np.sum(p[p > 0] * np.log(p[p > 0]))
r_eff = np.exp(H)
floor = min(20, max(2, int(np.ceil(r_eff))))
print(f"\nShannon entropy:  H = −Σ p_i·ln(p_i) = {H:.4f}")
print(f"Effective rank:   r_eff = exp(H) = exp({H:.4f}) = {r_eff:.2f}")
print(f"Floor:            min(20, max(2, ⌈{r_eff:.2f}⌉)) = {floor}")

# ================================================================
# STAGE 4: Gate 2 — edge (child-parent divergence) test
# ================================================================
print("\n" + "=" * 70)
print("STAGE 4: Gate 2 — child-parent divergence (edge test)")
print("=" * 70)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence_with_context,
)

ann = tree.annotations_df.copy()
edge_df, spectral_context = annotate_child_parent_divergence_with_context(
    tree, ann, significance_level_alpha=0.001, leaf_data=data
)

spectral_dims = spectral_context.spectral_projection_dimensions_by_node or {}
# Summarize spectral dims
k_vals = list(spectral_dims.values())
k_nonzero = [v for v in k_vals if v > 0]
print(f"\nSpectral dimensions: {len(k_vals)} nodes, {len(k_nonzero)} with k>0")
print(f"  Leaves: all k=0 ({sum(1 for v in k_vals if v == 0)} nodes)")
if k_nonzero:
    print(f"  Internal: k values = {sorted(set(k_nonzero))}")
# Show dims for root and its immediate children/grandchildren
print("\n  Nodes of interest (root + first two levels):")
for node in [root] + list(tree.successors(root)):
    k = spectral_dims.get(node, "?")
    print(f"    {node}: k_spectral = {k}")
    for gc in tree.successors(node):
        k = spectral_dims.get(gc, "?")
        print(f"      {gc}: k_spectral = {k}")

print("\nEdge test results (showing nodes with p_BH < 0.1 or root's children):")
root_children = set(tree.successors(root))
print(f"{'Node':>6} {'Parent':>8} {'Significant':>12} " f"{'p_raw':>12} {'p_BH':>12} {'k':>4}")
for node in sorted(edge_df.index, key=str):
    sig = edge_df.loc[node, "Child_Parent_Divergence_Significant"]
    p_bh = edge_df.loc[node, "Child_Parent_Divergence_P_Value_BH"]
    p_raw = edge_df.loc[node].get("Child_Parent_Divergence_P_Value", float("nan"))
    k_s = spectral_dims.get(node, "-")
    parent = list(tree.predecessors(node))
    parent_str = parent[0] if parent else "root"
    if pd.notna(p_bh) and (p_bh < 0.1 or node in root_children or sig):
        print(
            f"{node:>6} {parent_str:>8} {str(sig):>12} "
            f"{p_raw:>12.6f} {p_bh:>12.6f} {str(k_s):>4}"
        )
total_sig = int(edge_df["Child_Parent_Divergence_Significant"].sum())
total_tested = int(edge_df["Child_Parent_Divergence_P_Value_BH"].notna().sum())
print(f"\nTotal: {total_sig}/{total_tested} edges significant at α=0.001")

# ================================================================
# STAGE 5: Gate 2 → Gate 3 handoff
# ================================================================
print("\n" + "=" * 70)
print("STAGE 5: Spectral k handoff → sibling spectral dims")
print("=" * 70)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)

sib_dims = derive_sibling_projection_dimensions_from_child_edge_comparisons(
    tree,
    spectral_context=spectral_context,
)
print("Sibling spectral dims (geometric mean of children k):")
for parent, k in sorted((sib_dims or {}).items(), key=str):
    children = list(tree.successors(parent))
    k_L = spectral_dims.get(children[0], 0) if len(children) == 2 else "?"
    k_R = spectral_dims.get(children[1], 0) if len(children) == 2 else "?"
    print(f"  {parent}: √({k_L}×{k_R}) → k_sibling = {k}")

# Which nodes will use JL fallback?
print("\nFallback assignment:")
for parent in sorted(tree.nodes, key=str):
    children = list(tree.successors(parent))
    if len(children) != 2:
        continue
    L, R = children
    if sib_dims and parent in sib_dims:
        print(f"  {parent}: SPECTRAL (k={sib_dims[parent]})")
    else:
        print(f"  {parent}: JL FALLBACK (both children are leaves → k_L=0, k_R=0)")

# ================================================================
# STAGE 6: Gate 3 — sibling divergence test
# ================================================================
print("\n" + "=" * 70)
print("STAGE 6: Gate 3 — sibling divergence (adjusted Wald)")
print("=" * 70)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.adjusted_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)

sib_pca_proj, sib_pca_eig = collect_parent_principal_component_inputs_for_sibling_tests(
    sib_dims,
    spectral_context=spectral_context,
)
final_df = annotate_sibling_divergence(
    tree,
    edge_df.copy(),
    significance_level_alpha=0.01,
    sibling_projection_dimensions_from_edge_comparisons=sib_dims,
    parent_principal_component_projections=sib_pca_proj,
    parent_principal_component_eigenvalues=sib_pca_eig,
    edge_projection_dimensions_by_node=spectral_dims,
)

print("\nSibling test results:")
for parent in sorted(final_df.index, key=str):
    children = list(tree.successors(parent))
    if len(children) != 2:
        continue
    skipped = final_df.loc[parent, "Sibling_Divergence_Skipped"]
    if skipped:
        print(f"  {parent}: SKIPPED (leaf or no signal)")
        continue
    T = final_df.loc[parent, "Sibling_Test_Statistic"]
    df_val = final_df.loc[parent, "Sibling_Degrees_of_Freedom"]
    p_raw = final_df.loc[parent, "Sibling_Divergence_P_Value"]
    p_corr = final_df.loc[parent, "Sibling_Divergence_P_Value_Corrected"]
    diff = final_df.loc[parent, "Sibling_BH_Different"]
    method = final_df.loc[parent, "Sibling_Test_Method"]
    print(
        f"  {parent}: T={T:.3f}, df={df_val:.0f}, "
        f"p_raw={p_raw:.6f}, p_BH={p_corr:.6f}, "
        f"different={diff} [{method}]"
    )

audit = final_df.attrs.get("sibling_divergence_audit", {})
if audit:
    print("\n  Adjusted Wald calibration:")
    print(f"    ĉ (global inflation) = {audit.get('global_c_hat', '?')}")
    print(f"    calibration_n        = {audit.get('calibration_n', '?')}")
    print(f"    method               = {audit.get('calibration_method', '?')}")

# ================================================================
# STAGE 7+8: Gate evaluation DFS → cluster extraction
# ================================================================
print("\n" + "=" * 70)
print("STAGE 7: Gate evaluation (DFS from root)")
print("=" * 70)

# Walk manually to show each gate decision
from collections import deque

worklist = deque([root])
cluster_roots = set()
while worklist:
    node = worklist.popleft()
    children = list(tree.successors(node))

    if len(children) == 0:
        # Leaf — becomes its own cluster root
        cluster_roots.add(node)
        print(f"  {node}: LEAF → cluster root")
        continue

    # Gate 1: binary?
    if len(children) != 2:
        cluster_roots.add(node)
        print(f"  {node}: Gate 1 FAIL ({len(children)} children) → MERGE")
        continue

    L, R = children

    # Gate 2: either child diverges from parent?
    sig_L = final_df.loc[L, "Child_Parent_Divergence_Significant"] if L in final_df.index else False
    sig_R = final_df.loc[R, "Child_Parent_Divergence_Significant"] if R in final_df.index else False
    if not sig_L and not sig_R:
        cluster_roots.add(node)
        print(f"  {node}: Gate 2 FAIL (L_sig={sig_L}, R_sig={sig_R}) → MERGE")
        continue

    # Gate 3: siblings different?
    skipped = final_df.loc[node, "Sibling_Divergence_Skipped"]
    if skipped:
        cluster_roots.add(node)
        print(f"  {node}: Gate 3 SKIPPED → MERGE")
        continue

    diff = final_df.loc[node, "Sibling_BH_Different"]
    if diff:
        print(f"  {node}: G1 ✓  G2 ✓ (L={sig_L},R={sig_R})  G3 ✓ (different) → SPLIT into {L}, {R}")
        worklist.append(L)
        worklist.append(R)
    else:
        cluster_roots.add(node)
        print(f"  {node}: G1 ✓  G2 ✓ (L={sig_L},R={sig_R})  G3 FAIL (same) → MERGE")

# ================================================================
# STAGE 8: Actual decompose result
# ================================================================
print("\n" + "=" * 70)
print("STAGE 8: Final cluster assignments")
print("=" * 70)
result = tree.decompose(leaf_data=data, alpha_local=0.001, sibling_alpha=0.01)
assignments = result["cluster_assignments"]
print(f"Number of clusters: {result['num_clusters']}")
for cid, info in sorted(assignments.items()):
    print(
        f"  Cluster {cid}: root_node={info['root_node']}, "
        f"leaves={sorted(info['leaves'])}, size={info['size']}"
    )

# ARI
y_true = [i // n_per for i in range(n)]
y_pred = [0] * n
for cid, info in assignments.items():
    for leaf in info["leaves"]:
        idx = labels.index(leaf)
        y_pred[idx] = cid

from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, y_pred)
print(f"\nTrue labels:  {y_true}")
print(f"Pred labels:  {y_pred}")
print(f"ARI = {ari:.4f}")
