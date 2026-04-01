#!/usr/bin/env python3
"""Minimal 4-taxa tree: numerical walkthrough of every structure."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(precision=4, suppress=True, linewidth=100)

# ============================================================
# INPUT: 4 taxa × 6 features, 2 clusters of 2
# ============================================================
X = np.array(
    [
        [1, 1, 1, 0, 0, 0],  # S0  cluster A
        [1, 1, 0, 0, 0, 1],  # S1  cluster A
        [0, 0, 0, 1, 1, 1],  # S2  cluster B
        [0, 0, 1, 1, 1, 0],  # S3  cluster B
    ],
    dtype=int,
)
labels = ["S0", "S1", "S2", "S3"]
features = ["f0", "f1", "f2", "f3", "f4", "f5"]
data = pd.DataFrame(X, index=labels, columns=features)

print("=" * 60)
print("INPUT:  X in {0,1}^{4x6}")
print("=" * 60)
print(data.to_string())
print()

# ─────────────────────────────────────────────────────────────
# 1. Pairwise hamming distances
# ─────────────────────────────────────────────────────────────
D = pdist(X, metric="hamming")
D_sq = squareform(D)
print("HAMMING DISTANCES:")
print("       S0    S1    S2    S3")
for i in range(4):
    row = "  ".join(f"{D_sq[i, j]:.4f}" for j in range(4))
    print(f"  S{i}:  {row}")
print()

# ─────────────────────────────────────────────────────────────
# 2. UPGMA linkage
# ─────────────────────────────────────────────────────────────
Z = linkage(D, method="average")
print("LINKAGE (average/UPGMA):")
print("  merge  left  right    dist  size")
for i, row in enumerate(Z):
    l = f"L{int(row[0])}" if row[0] < 4 else f"N{int(row[0]) - 4}"
    r = f"L{int(row[1])}" if row[1] < 4 else f"N{int(row[1]) - 4}"
    print(f"  N{i}     {l:>3}   {r:>3}   {row[2]:.4f}    {int(row[3])}")
print()

# ─────────────────────────────────────────────────────────────
# 3. PosetTree
# ─────────────────────────────────────────────────────────────
from kl_clustering_analysis.tree.poset_tree import PosetTree

tree = PosetTree.from_linkage(Z, leaf_names=labels)
tree.populate_node_divergences(data)

root = next(n for n, deg in tree.in_degree() if deg == 0)

print("=" * 60)
print(f"POSET TREE  (root = {root})")
print("=" * 60)
print(f"Nodes ({tree.number_of_nodes()}): {sorted(tree.nodes, key=str)}")
print(f"Edges ({tree.number_of_edges()}): {sorted(tree.edges, key=str)}")
print()


# ── Visual tree ──
def show_tree(node, prefix="", is_last=True):
    nd = tree.nodes[node]
    dist = nd.get("distribution")
    is_leaf = nd.get("is_leaf", False)
    label = nd.get("label", node)
    children = list(tree.successors(node))

    parents = list(tree.predecessors(node))
    bl = tree.edges[parents[0], node].get("branch_length") if parents else None
    bl_str = f"  bl={bl:.4f}" if bl is not None else ""

    connector = "└── " if is_last else "├── "
    d_arr = np.array(dist) if dist is not None else None

    if is_leaf:
        print(f"{prefix}{connector}{node} (leaf={label}){bl_str}")
        ext = "    " if is_last else "│   "
        print(f"{prefix}{ext}theta = [{', '.join(f'{v:.1f}' for v in d_arr)}]")
    else:
        desc = [n_ for n_ in nx.descendants(tree, node) if tree.out_degree(n_) == 0]
        leaf_labels = sorted(nd_.get("label", n_) for n_, nd_ in ((l, tree.nodes[l]) for l in desc))
        print(f"{prefix}{connector}{node} (contains {leaf_labels}){bl_str}")
        ext = "    " if is_last else "│   "
        if d_arr is not None:
            print(f"{prefix}{ext}theta = [{', '.join(f'{v:.2f}' for v in d_arr)}]")
        for i, c in enumerate(children):
            show_tree(c, prefix + ext, is_last=(i == len(children) - 1))


show_tree(root)
print()

# ── Edge list with branch lengths ──
print("EDGE TABLE:")
print(f"  {'from':>5} {'to':>5} {'branch_len':>10}")
for p, c in sorted(tree.edges, key=str):
    bl = tree.edges[p, c].get("branch_length")
    bl_s = f"{bl:.4f}" if bl is not None else "N/A"
    print(f"  {p:>5} {c:>5} {bl_s:>10}")

bls = [
    tree.edges[p, c]["branch_length"]
    for p, c in tree.edges()
    if "branch_length" in tree.edges[p, c]
]
mean_bl = float(np.mean(bls)) if bls else None
print(f"\n  mean_branch_length = {mean_bl}")
print()

# ── Node contents ──
print("NODE CONTENTS:")
for node in sorted(tree.nodes, key=str):
    desc = [n_ for n_ in nx.descendants(tree, node) if tree.out_degree(n_) == 0]
    if tree.out_degree(node) == 0:
        desc = [node]
    leaf_labels_ = sorted(tree.nodes[l].get("label", l) for l in desc)
    n_desc = tree.nodes[node].get("n_descendants", len(leaf_labels_))
    print(f"  {node:>5}: leaves={leaf_labels_}, n_desc={n_desc}")
print()

# ── Distribution computation ──
print("=" * 60)
print("DISTRIBUTION COMPUTATION")
print("=" * 60)
ch = list(tree.successors(root))
L, R = ch[0], ch[1]
for node in [root, L, R]:
    desc = [n_ for n_ in nx.descendants(tree, node) if tree.out_degree(n_) == 0]
    if tree.out_degree(node) == 0:
        desc = [node]
    leaf_labels_ = sorted(tree.nodes[l].get("label", l) for l in desc)
    leaf_idx = [labels.index(ll) for ll in leaf_labels_]
    subX = X[leaf_idx]
    theta = subX.mean(axis=0)
    print(f"\n  {node} (leaves={leaf_labels_}):")
    for li, idx in zip(leaf_labels_, leaf_idx):
        print(f"    {li}: {X[idx]}")
    print(f"    theta = mean(rows) = [{', '.join(f'{v:.2f}' for v in theta)}]")

# ── Covariance / Correlation / Eigendecomposition ──
print()
print("=" * 60)
print("COVARIANCE & CORRELATION MATRICES + EIGENDECOMPOSITION")
print("=" * 60)

# --- Global data matrix (what the fallback floor is computed from) ---
print("\n--- GLOBAL: Leaf data matrix X (used for fallback floor) ---")
X_f = X.astype(float)
print(f"  X ({X_f.shape[0]} x {X_f.shape[1]}):")
for i, lab in enumerate(labels):
    print(f"    {lab}: {X_f[i]}")

# Covariance matrix (features x features)
cov_global = np.cov(X_f.T, ddof=0)  # population covariance
print(f"\n  Covariance matrix (features x features, {cov_global.shape}):")
print(f"         {' '.join(f'{f:>7s}' for f in features)}")
for j, f in enumerate(features):
    print(f"    {f}:  {' '.join(f'{cov_global[j,k]:+7.4f}' for k in range(len(features)))}")

# Correlation matrix (what is actually eigendecomposed)
var_global = np.var(X_f, axis=0)
mask = var_global > 0
d_active = int(mask.sum())
print(f"\n  Feature variances: {var_global}")
print(f"  Non-constant features: {d_active}/{len(features)}")

X_active = X_f[:, mask]
active_names = [f for f, m in zip(features, mask) if m]
corr_global = np.corrcoef(X_active.T)
corr_global = np.nan_to_num(corr_global, nan=0.0)
np.fill_diagonal(corr_global, 1.0)

print("\n  >>> CORRELATION matrix (this is what gets eigendecomposed) <<<")
print("  Why correlation, not covariance? Standardization ensures each feature")
print("  contributes equally regardless of its variance.")
print(f"\n         {' '.join(f'{f:>7s}' for f in active_names)}")
for j, f in enumerate(active_names):
    print(f"    {f}:  {' '.join(f'{corr_global[j,k]:+7.4f}' for k in range(len(active_names)))}")

eigs_global = np.sort(np.linalg.eigvalsh(corr_global))[::-1]
eigs_global = np.maximum(eigs_global, 0.0)
print("\n  Eigenvalues of the correlation matrix:")
for i, lam in enumerate(eigs_global):
    bar = "#" * max(1, int(lam * 5))
    print(f"    lambda_{i+1} = {lam:7.4f}  {bar}")
print(f"    Sum = {eigs_global.sum():.4f} (= d_active = {d_active})")

p_global = eigs_global / eigs_global.sum()
H_global = -np.sum(p_global[p_global > 0] * np.log(p_global[p_global > 0]))
r_eff = np.exp(H_global)
floor = min(20, max(2, int(np.ceil(r_eff))))
print(f"\n  Effective rank: r_eff = exp(H) = exp({H_global:.4f}) = {r_eff:.2f}")
print(f"  Fallback floor = min(20, max(2, ceil({r_eff:.2f}))) = {floor}")

# --- Per-node: what Gate 2 actually decomposes ---
print()
print("--- PER-NODE: What Gate 2 eigendecomposes (root N6 example) ---")
print()
print("  For each internal node, Gate 2 builds a LOCAL data matrix from:")
print("    1. Leaf feature rows (descendants of that node)")
print("    2. Internal node distribution vectors (if INCLUDE_INTERNAL=True)")
print("  Then computes the CORRELATION matrix of that local matrix.")
print()

for node in [root, L, R]:
    children = list(tree.successors(node))
    if len(children) == 0:
        continue

    # Collect descendant leaves
    desc_leaves = [n_ for n_ in nx.descendants(tree, node) if tree.out_degree(n_) == 0]
    if tree.out_degree(node) == 0:
        desc_leaves = [node]
    leaf_labels_ = sorted(tree.nodes[l].get("label", l) for l in desc_leaves)
    leaf_idx = [labels.index(ll) for ll in leaf_labels_]

    # Collect descendant internal node distributions (INCLUDE_INTERNAL=True)
    desc_internal = [n_ for n_ in nx.descendants(tree, node) if tree.out_degree(n_) > 0]
    internal_dists = []
    internal_names = []
    for n_ in sorted(desc_internal, key=str):
        d = tree.nodes[n_].get("distribution")
        if d is not None:
            internal_dists.append(np.array(d))
            internal_names.append(n_)

    # Build data matrix
    leaf_rows = X_f[leaf_idx]
    if internal_dists:
        local_data = np.vstack([leaf_rows, np.array(internal_dists)])
        row_labels = leaf_labels_ + internal_names
    else:
        local_data = leaf_rows
        row_labels = leaf_labels_

    n_rows, n_cols = local_data.shape
    print(f"  Node {node}: local data matrix ({n_rows} rows x {n_cols} cols)")
    print(f"    Rows: {row_labels}")
    for i, rl in enumerate(row_labels):
        print(f"      {rl}: [{', '.join(f'{v:.2f}' for v in local_data[i])}]")

    # Active features (non-constant in this local matrix)
    local_var = np.var(local_data, axis=0)
    local_mask = local_var > 0
    local_d_active = int(local_mask.sum())
    local_active_names = [features[j] for j in range(n_cols) if local_mask[j]]
    print(f"    Active features (var>0): {local_d_active} → {local_active_names}")

    if local_d_active < 2:
        print("    Too few active features for correlation matrix → k_spectral = 0")
        continue

    local_active = local_data[:, local_mask]

    # Choose primal vs dual
    use_dual = n_rows < local_d_active
    if use_dual:
        mu = local_active.mean(axis=0)
        sigma = local_active.std(axis=0, ddof=0)
        sigma[sigma == 0] = 1.0
        X_std = (local_active - mu) / sigma
        gram = X_std @ X_std.T / local_d_active
        print(f"\n    n={n_rows} < d_active={local_d_active} → DUAL FORM (Gram matrix)")
        print("    X_std = (X_local - mu) / sigma, then Gram = X_std @ X_std.T / d_active")
        print(f"    Standardized X_std ({n_rows}x{local_d_active}):")
        for i, rl in enumerate(row_labels):
            print(f"      {rl}: [{', '.join(f'{v:+.4f}' for v in X_std[i])}]")
        print(f"\n    >>> Gram matrix ({n_rows}x{n_rows}) — this is what gets eigendecomposed <<<")
        print(f"         {' '.join(f'{rl:>7s}' for rl in row_labels)}")
        for i, rl in enumerate(row_labels):
            print(f"    {rl:>4}: {' '.join(f'{gram[i,j]:+7.4f}' for j in range(n_rows))}")
        local_eigs = np.sort(np.linalg.eigvalsh(gram))[::-1]
    else:
        local_corr = np.corrcoef(local_active.T)
        local_corr = np.nan_to_num(local_corr, nan=0.0)
        np.fill_diagonal(local_corr, 1.0)
        print(f"\n    n={n_rows} >= d_active={local_d_active} → PRIMAL FORM (Correlation matrix)")
        print(
            f"    >>> Correlation matrix ({local_d_active}x{local_d_active}) — eigendecomposed <<<"
        )
        print(f"         {' '.join(f'{f:>7s}' for f in local_active_names)}")
        for j, f in enumerate(local_active_names):
            print(
                f"    {f:>4}: {' '.join(f'{local_corr[j,k]:+7.4f}' for k in range(local_d_active))}"
            )
        local_eigs = np.sort(np.linalg.eigvalsh(local_corr))[::-1]

    local_eigs = np.maximum(local_eigs, 0.0)
    gamma = local_d_active / n_rows  # aspect ratio
    mp_upper = (1 + np.sqrt(gamma)) ** 2
    print("\n    Eigenvalues:")
    for i, lam in enumerate(local_eigs):
        above = "  > MP_upper  *** SIGNAL" if lam > mp_upper else ""
        print(f"      lambda_{i+1} = {lam:7.4f}{above}")
    print(f"    Marchenko-Pastur threshold: gamma = d/n = {local_d_active}/{n_rows} = {gamma:.2f}")
    print(f"      MP_upper = (1 + sqrt({gamma:.2f}))^2 = {mp_upper:.4f}")
    k_spectral = int(np.sum(local_eigs > mp_upper))
    print(f"    k_spectral = #{'{'}eigenvalues > {mp_upper:.4f}{'}'} = {k_spectral}")
    print()

# ── KL divergence ──
print()
print("=" * 60)
print("KL DIVERGENCE (child || parent)")
print("=" * 60)
eps = 1e-9
for child_node in [L, R]:
    child_dist = np.array(tree.nodes[child_node]["distribution"])
    parent_dist = np.array(tree.nodes[root]["distribution"])
    kl_per_feat = []
    for j in range(len(child_dist)):
        p = np.clip(child_dist[j], eps, 1 - eps)
        q = np.clip(parent_dist[j], eps, 1 - eps)
        kl_j = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        kl_per_feat.append(kl_j)
    total_kl = sum(kl_per_feat)
    print(f"\n  KL({child_node} || {root}):")
    for j, kl_j in enumerate(kl_per_feat):
        p_ = np.clip(child_dist[j], eps, 1 - eps)
        q_ = np.clip(parent_dist[j], eps, 1 - eps)
        print(f"    f{j}: theta_child={p_:.2f}, theta_parent={q_:.2f} -> KL_j = {kl_j:.4f}")
    print(f"    TOTAL KL = {total_kl:.4f}")

# ── Wald test ingredients ──
print()
print("=" * 60)
print("WALD TEST INGREDIENTS (edge test, child vs parent)")
print("=" * 60)
for child_node in [L, R]:
    child_dist = np.array(tree.nodes[child_node]["distribution"])
    parent_dist = np.array(tree.nodes[root]["distribution"])
    desc_child = [n_ for n_ in nx.descendants(tree, child_node) if tree.out_degree(n_) == 0]
    if tree.out_degree(child_node) == 0:
        desc_child = [child_node]
    n_child = len(desc_child)
    n_parent = 4

    print(f"\n  {child_node} vs {root}:")
    print(f"    n_child={n_child}, n_parent={n_parent}")
    z = child_dist - parent_dist
    print("    z = theta_child - theta_parent:")
    print(f"      [{', '.join(f'{v:+.4f}' for v in z)}]")

    var = parent_dist * (1 - parent_dist) * (1.0 / n_child - 1.0 / n_parent)
    print(f"    Var_j = theta_j(1-theta_j)(1/{n_child} - 1/{n_parent}):")
    for j in range(len(var)):
        theta_j = parent_dist[j]
        fac = 1.0 / n_child - 1.0 / n_parent
        print(f"      f{j}: {theta_j:.2f}*(1-{theta_j:.2f})*{fac:.4f} = {var[j]:.4f}")

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.where(var > 0, z / np.sqrt(var), 0)
    print("    z_scores = z / sqrt(Var):")
    print(f"      [{', '.join(f'{v:+.4f}' for v in z_scores)}]")
    print(f"    sum(z^2) = {np.sum(z_scores**2):.4f}")
    print("    -> project to k dims, then T ~ chi2(k)")

# ── Sibling comparison ──
print()
print("=" * 60)
print(f"SIBLING COMPARISON ({L} vs {R})")
print("=" * 60)
l_dist = np.array(tree.nodes[L]["distribution"])
r_dist = np.array(tree.nodes[R]["distribution"])
print(f"  theta_L = [{', '.join(f'{v:.2f}' for v in l_dist)}]")
print(f"  theta_R = [{', '.join(f'{v:.2f}' for v in r_dist)}]")
delta = l_dist - r_dist
print(f"  delta   = [{', '.join(f'{v:+.2f}' for v in delta)}]")
print(f"  |delta|^2 = {np.sum(delta ** 2):.4f}")
