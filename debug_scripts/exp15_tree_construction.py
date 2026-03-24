#!/usr/bin/env python3
"""
exp15: Compare tree construction strategies on benchmark cases.

Tests three alternative tree construction methods from new_method.md:
  1. Spectral bisection  — recursive Fiedler vector (nx.spectral_bisection)
  2. BisectingKMeans tree — sklearn top-down bisecting k-means
  3. Diffusion + HAC      — diffusion distance from SNN graph → standard linkage

All are compared against the baseline (Hamming + average linkage).

Usage:
    python debug_scripts/exp15_tree_construction.py [case_name ...]

    # Single case
    python debug_scripts/exp15_tree_construction.py cat_overlap_3cat_4c

    # Multiple cases
    python debug_scripts/exp15_tree_construction.py cat_overlap_3cat_4c gauss_clear_small

    # All cases (slow)
    python debug_scripts/exp15_tree_construction.py --all
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ────────────────────────────────────────────────────────────────────
# Strategy 1: Spectral Bisection (recursive Fiedler vector)
# ────────────────────────────────────────────────────────────────────


def _build_spectral_bisection_tree(
    data: pd.DataFrame, k_neighbors: int = 15, min_bisect: int = 8
) -> PosetTree:
    """Build a PosetTree via recursive spectral bisection on an SNN graph.

    Stops recursing when subtree size <= min_bisect and builds a balanced
    sub-tree for the remaining leaves (avoids hundreds of tiny eigendecompositions).
    """
    import networkx as nx

    n = len(data)
    leaf_names = data.index.tolist()

    # Build k-NN similarity graph
    sim_graph = _build_snn_graph(data, k_neighbors=k_neighbors)

    tree = PosetTree()
    node_counter = [n]  # mutable counter for internal node IDs

    def _next_internal() -> str:
        idx = node_counter[0]
        node_counter[0] += 1
        return f"N{idx}"

    def _bisect(sample_indices: list[int], parent_id: str | None) -> str:
        """Recursively bisect and return the node ID for this subtree."""
        if len(sample_indices) == 1:
            leaf_id = f"L{sample_indices[0]}"
            tree.add_node(leaf_id, is_leaf=True, label=leaf_names[sample_indices[0]])
            if parent_id is not None:
                tree.add_edge(parent_id, leaf_id, branch_length=1.0)
            return leaf_id

        # Early stop: small subtrees → balanced sub-tree (no eigendecomp)
        if len(sample_indices) <= min_bisect:
            node_id = _next_internal()
            tree.add_node(node_id, is_leaf=False)
            if parent_id is not None:
                tree.add_edge(parent_id, node_id, branch_length=1.0)
            _build_balanced_subtree(
                tree, node_id, sample_indices, leaf_names, node_counter
            )
            return node_id

        # Create subgraph for these samples
        sg = sim_graph.subgraph(sample_indices).copy()

        # Handle disconnected subgraph: add tiny edges to largest component
        if not nx.is_connected(sg):
            components = list(nx.connected_components(sg))
            largest = max(components, key=len)
            for comp in components:
                if comp is largest:
                    continue
                bridge_from = next(iter(comp))
                bridge_to = next(iter(largest))
                sg.add_edge(bridge_from, bridge_to, weight=1e-6)

        # Spectral bisection via Fiedler vector
        try:
            set_a, set_b = nx.spectral_bisection(sg, weight="weight")
        except Exception:
            # Fallback: split in half
            mid = len(sample_indices) // 2
            set_a = set(sample_indices[:mid])
            set_b = set(sample_indices[mid:])

        list_a = sorted(set_a)
        list_b = sorted(set_b)

        # Ensure non-empty
        if not list_a or not list_b:
            mid = len(sample_indices) // 2
            list_a = sample_indices[:mid]
            list_b = sample_indices[mid:]

        node_id = _next_internal()
        tree.add_node(node_id, is_leaf=False)
        if parent_id is not None:
            tree.add_edge(parent_id, node_id, branch_length=1.0)

        _bisect(list_a, node_id)
        _bisect(list_b, node_id)
        return node_id

    root_id = _bisect(list(range(n)), None)
    tree.graph["root"] = root_id
    return tree


# ────────────────────────────────────────────────────────────────────
# Strategy 2: BisectingKMeans tree
# ────────────────────────────────────────────────────────────────────


def _build_bisecting_kmeans_tree(data: pd.DataFrame) -> PosetTree:
    """Build a PosetTree from sklearn BisectingKMeans internal tree."""
    from sklearn.cluster import BisectingKMeans

    n = len(data)
    leaf_names = data.index.tolist()
    X = data.values.astype(float)

    # Use a reasonable max K so each leaf-cluster is small
    max_k = min(n // 2, 200)

    bkm = BisectingKMeans(
        n_clusters=max_k,
        bisecting_strategy="biggest_inertia",
        random_state=42,
        n_init=3,
    )
    bkm.fit(X)

    # Map each sample to its leaf-cluster label
    labels = bkm.labels_  # shape (n,)

    # Collect leaf nodes and their centers
    bkm_leaves = list(bkm._bisecting_tree.iter_leaves())
    leaf_centers = np.array([l.center for l in bkm_leaves])

    # Map cluster-centers to tree leaves via nearest center
    # bkm.cluster_centers_[label] is the center for cluster `label`
    from scipy.spatial.distance import cdist

    # Match each bkm leaf to the cluster_centers_ index
    center_dists = cdist(leaf_centers, bkm.cluster_centers_)
    leaf_to_cluster = center_dists.argmin(axis=1)
    # Reverse: cluster -> bkm_leaf_index
    cluster_to_leaf_idx = {int(leaf_to_cluster[i]): i for i in range(len(bkm_leaves))}

    # Assign samples to bkm_leaf_index
    leaf_samples: dict[int, list[int]] = {}
    for sample_idx in range(n):
        cl = int(labels[sample_idx])
        li = cluster_to_leaf_idx.get(cl, cl)
        leaf_samples.setdefault(li, []).append(sample_idx)

    # Build PosetTree by traversing bkm tree structure
    tree = PosetTree()
    node_counter = [n]

    def _next_internal() -> str:
        idx = node_counter[0]
        node_counter[0] += 1
        return f"N{idx}"

    # Map bkm node -> list of sample indices it contains
    def _get_node_samples(bkm_node) -> list[int]:
        """Recursively collect all sample indices under this node."""
        if bkm_node.left is None and bkm_node.right is None:
            # Find which leaf index this is
            for li, leaf in enumerate(bkm_leaves):
                if leaf is bkm_node:
                    return leaf_samples.get(li, [])
            return []
        samples = []
        if bkm_node.left is not None:
            samples.extend(_get_node_samples(bkm_node.left))
        if bkm_node.right is not None:
            samples.extend(_get_node_samples(bkm_node.right))
        return samples

    def _convert_node(bkm_node, parent_id: str | None) -> str:
        """Recursively convert BisectingKMeans tree node to PosetTree."""
        is_leaf_node = bkm_node.left is None and bkm_node.right is None
        samples = _get_node_samples(bkm_node)

        if not samples:
            # Empty node — shouldn't happen, but guard
            node_id = _next_internal()
            tree.add_node(node_id, is_leaf=False)
            if parent_id is not None:
                tree.add_edge(parent_id, node_id, branch_length=1.0)
            return node_id

        if is_leaf_node:
            if len(samples) == 1:
                leaf_id = f"L{samples[0]}"
                tree.add_node(leaf_id, is_leaf=True, label=leaf_names[samples[0]])
                if parent_id is not None:
                    tree.add_edge(parent_id, leaf_id, branch_length=1.0)
                return leaf_id

            # Multi-sample leaf cluster: build balanced sub-tree
            node_id = _next_internal()
            tree.add_node(node_id, is_leaf=False)
            if parent_id is not None:
                tree.add_edge(parent_id, node_id, branch_length=1.0)
            _build_balanced_subtree(tree, node_id, samples, leaf_names, node_counter)
            return node_id

        # Internal node — recurse
        node_id = _next_internal()
        tree.add_node(node_id, is_leaf=False)
        if parent_id is not None:
            tree.add_edge(parent_id, node_id, branch_length=1.0)

        if bkm_node.left is not None:
            _convert_node(bkm_node.left, node_id)
        if bkm_node.right is not None:
            _convert_node(bkm_node.right, node_id)

        return node_id

    root_id = _convert_node(bkm._bisecting_tree, None)
    tree.graph["root"] = root_id
    return tree


def _build_balanced_subtree(
    tree: PosetTree,
    parent_id: str,
    indices: list[int],
    leaf_names: list[str],
    node_counter: list[int],
) -> None:
    """Build a balanced binary sub-tree for a set of leaf indices."""
    if len(indices) == 1:
        leaf_id = f"L{indices[0]}"
        tree.add_node(leaf_id, is_leaf=True, label=leaf_names[indices[0]])
        tree.add_edge(parent_id, leaf_id, branch_length=1.0)
        return

    if len(indices) == 2:
        for i in indices:
            leaf_id = f"L{i}"
            tree.add_node(leaf_id, is_leaf=True, label=leaf_names[i])
            tree.add_edge(parent_id, leaf_id, branch_length=1.0)
        return

    mid = len(indices) // 2
    left_indices = indices[:mid]
    right_indices = indices[mid:]

    for sub_indices in (left_indices, right_indices):
        if len(sub_indices) == 1:
            leaf_id = f"L{sub_indices[0]}"
            tree.add_node(leaf_id, is_leaf=True, label=leaf_names[sub_indices[0]])
            tree.add_edge(parent_id, leaf_id, branch_length=1.0)
        else:
            child_id = f"N{node_counter[0]}"
            node_counter[0] += 1
            tree.add_node(child_id, is_leaf=False)
            tree.add_edge(parent_id, child_id, branch_length=1.0)
            _build_balanced_subtree(tree, child_id, sub_indices, leaf_names, node_counter)


# ────────────────────────────────────────────────────────────────────
# Strategy 3: Diffusion distance + HAC
# ────────────────────────────────────────────────────────────────────


def _build_diffusion_hac_tree(
    data: pd.DataFrame,
    k_neighbors: int = 15,
    diffusion_time: int = 3,
    n_components: int = 30,
) -> PosetTree:
    """Build a PosetTree using diffusion distance → average linkage."""
    n = len(data)
    leaf_names = data.index.tolist()

    # Build k-NN similarity matrix
    from sklearn.neighbors import NearestNeighbors

    X = data.values.astype(float)
    k = min(k_neighbors, n - 1)

    nn = NearestNeighbors(n_neighbors=k, metric="hamming")
    nn.fit(X)
    knn_dist, knn_idx = nn.kneighbors(X)

    # Similarity: 1 - hamming distance
    from scipy.sparse import lil_matrix

    W = lil_matrix((n, n), dtype=float)
    for i in range(n):
        for j_pos in range(k):
            j = knn_idx[i, j_pos]
            sim = max(1.0 - knn_dist[i, j_pos], 1e-10)
            W[i, j] = max(W[i, j], sim)
            W[j, i] = max(W[j, i], sim)

    W = W.toarray()
    np.fill_diagonal(W, 0)

    # Normalize to transition matrix
    D = W.sum(axis=1)
    D[D == 0] = 1e-10
    D_inv = 1.0 / D
    T = W * D_inv[:, None]  # row-normalized transition matrix

    # Eigendecomposition of transition matrix
    from scipy.linalg import eigh

    # T is not symmetric — use the symmetric normalization trick:
    # D^{-1/2} W D^{-1/2} has the same eigenvalues as D^{-1} W
    D_sqrt_inv = 1.0 / np.sqrt(D)
    T_sym = W * D_sqrt_inv[:, None] * D_sqrt_inv[None, :]

    n_comps = min(n_components, n - 1)
    eigenvalues, eigenvectors = eigh(T_sym)

    # Take top eigenvalues (excluding the trivial one ≈ 1)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the first (trivial) eigenvector, take next n_comps
    eigenvalues = eigenvalues[1 : n_comps + 1]
    eigenvectors = eigenvectors[:, 1 : n_comps + 1]

    # Clamp negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)

    # Diffusion coordinates at time t
    diffusion_coords = eigenvectors * (eigenvalues[None, :] ** diffusion_time)

    # Compute pairwise distance on diffusion coordinates
    diff_dist = pdist(diffusion_coords, metric="euclidean")

    # Standard HAC with average linkage
    Z = linkage(diff_dist, method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=leaf_names)
    return tree


# ────────────────────────────────────────────────────────────────────
# Shared: SNN graph builder
# ────────────────────────────────────────────────────────────────────


def _build_snn_graph(data: pd.DataFrame, k_neighbors: int = 15) -> "nx.Graph":
    """Build a Shared Nearest Neighbor (SNN) weighted graph."""
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors

    n = len(data)
    X = data.values.astype(float)
    k = min(k_neighbors, n - 1)

    nn = NearestNeighbors(n_neighbors=k, metric="hamming")
    nn.fit(X)
    _, knn_idx = nn.kneighbors(X)

    # SNN similarity: |N(i) ∩ N(j)| / k
    neighbor_sets = [set(knn_idx[i]) for i in range(n)]

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in neighbor_sets[i]:
            if j <= i:
                continue
            shared = len(neighbor_sets[i] & neighbor_sets[j])
            if shared > 0:
                w = shared / k
                G.add_edge(i, j, weight=w)

    # Ensure connected: add tiny edges between components
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        for comp in components:
            if comp is largest:
                continue
            bridge_from = next(iter(comp))
            bridge_to = next(iter(largest))
            G.add_edge(bridge_from, bridge_to, weight=1e-6)

    return G


# ────────────────────────────────────────────────────────────────────
# Baseline: Standard Hamming + average linkage
# ────────────────────────────────────────────────────────────────────


def _build_baseline_tree(data: pd.DataFrame) -> PosetTree:
    """Standard pipeline: Hamming distance + average linkage."""
    dist = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    return PosetTree.from_linkage(Z, leaf_names=data.index.tolist())


# ────────────────────────────────────────────────────────────────────
# Tree quality metrics
# ────────────────────────────────────────────────────────────────────


def _tree_diagnostics(tree: PosetTree, data: pd.DataFrame) -> dict:
    """Compute tree quality metrics."""
    root = tree.graph.get("root") or next(n for n, d in tree.in_degree() if d == 0)
    children = list(tree.successors(root))
    n = len(data)
    p = data.shape[1]

    # Root imbalance
    def _count_leaves(node: str) -> int:
        if tree.out_degree(node) == 0:
            return 1
        return sum(_count_leaves(c) for c in tree.successors(node))

    if len(children) == 2:
        n_l = _count_leaves(children[0])
        n_r = _count_leaves(children[1])
        imbalance = max(n_l, n_r) / (n_l + n_r)
    else:
        imbalance = float("nan")

    # γ distribution
    internal = [nd for nd in tree.nodes() if tree.out_degree(nd) > 0]
    gammas = []
    for nd in internal:
        nl = _count_leaves(nd)
        if nl > 0:
            gammas.append(p / nl)
    gammas = np.array(gammas) if gammas else np.array([0.0])

    return {
        "root_imbalance": imbalance,
        "gamma_median": float(np.median(gammas)),
        "gamma_pct_gt1": float((gammas > 1).mean() * 100),
        "n_internal": len(internal),
    }


# ────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "baseline_hamming_avg": _build_baseline_tree,
    "spectral_bisection": _build_spectral_bisection_tree,
    "bisecting_kmeans": _build_bisecting_kmeans_tree,
    "diffusion_hac": _build_diffusion_hac_tree,
}


def run_case(case_name: str) -> list[dict]:
    """Run all strategies on a single benchmark case."""
    all_cases = get_default_test_cases()
    tc = next((c for c in all_cases if c["name"] == case_name), None)
    if tc is None:
        print(f"ERROR: Case '{case_name}' not found.")
        return []

    data_t, y_t, _, _ = generate_case_data(tc)
    n, p = data_t.shape
    true_k = tc.get("n_clusters", 1)

    print(f"\n{'='*72}")
    print(f"  CASE: {case_name}  (n={n}, p={p}, true_K={true_k})")
    print(f"{'='*72}")

    results = []
    for strategy_name, builder in STRATEGIES.items():
        print(f"\n  --- {strategy_name} ---")
        try:
            tree = builder(data_t)

            # Diagnostics
            diag = _tree_diagnostics(tree, data_t)
            print(
                f"    Root imbalance: {diag['root_imbalance']:.3f}, "
                f"γ median: {diag['gamma_median']:.1f}, "
                f"γ>1: {diag['gamma_pct_gt1']:.0f}%"
            )

            # Decompose
            decomp = tree.decompose(
                leaf_data=data_t,
                alpha_local=config.EDGE_ALPHA,
                sibling_alpha=config.SIBLING_ALPHA,
            )
            K_found = decomp["num_clusters"]

            # ARI
            if y_t is not None:
                y_pred = np.full(n, -1, dtype=int)
                for cid, cinfo in decomp["cluster_assignments"].items():
                    for leaf in cinfo["leaves"]:
                        idx = data_t.index.get_loc(leaf)
                        y_pred[idx] = cid
                ari = adjusted_rand_score(y_t, y_pred)
            else:
                ari = float("nan")

            # Calibration audit
            stats = tree.annotations_df
            audit = stats.attrs.get("sibling_divergence_audit", {})
            c_hat = audit.get("global_inflation_factor", float("nan"))

            print(f"    K_found={K_found} (true={true_k}), ARI={ari:.3f}, " f"ĉ={c_hat:.1f}")

            # Cluster composition
            if y_t is not None:
                for cid, cinfo in sorted(decomp["cluster_assignments"].items()):
                    comp: dict[int, int] = {}
                    for leaf in cinfo["leaves"]:
                        idx = data_t.index.get_loc(leaf)
                        comp[int(y_t[idx])] = comp.get(int(y_t[idx]), 0) + 1
                    dominant = max(comp, key=comp.get)
                    purity = comp[dominant] / cinfo["size"]
                    print(
                        f"      Cluster {cid}: size={cinfo['size']}, "
                        f"purity={purity:.2f}, comp={dict(sorted(comp.items()))}"
                    )

            results.append(
                {
                    "case": case_name,
                    "strategy": strategy_name,
                    "true_k": true_k,
                    "found_k": K_found,
                    "ari": ari,
                    "c_hat": c_hat,
                    "root_imbalance": diag["root_imbalance"],
                    "gamma_median": diag["gamma_median"],
                    "exact_k": K_found == true_k,
                }
            )

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "case": case_name,
                    "strategy": strategy_name,
                    "true_k": true_k,
                    "found_k": -1,
                    "ari": float("nan"),
                    "c_hat": float("nan"),
                    "root_imbalance": float("nan"),
                    "gamma_median": float("nan"),
                    "exact_k": False,
                }
            )

    return results


def print_summary(all_results: list[dict]) -> None:
    """Print a summary table of all results."""
    df = pd.DataFrame(all_results)
    if df.empty:
        return

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")

    # Per-strategy aggregates
    agg = (
        df.groupby("strategy")
        .agg(
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            exact_k_pct=("exact_k", "mean"),
            mean_c_hat=("c_hat", "mean"),
            mean_imbalance=("root_imbalance", "mean"),
            n_cases=("case", "count"),
        )
        .sort_values("mean_ari", ascending=False)
    )
    agg["exact_k_pct"] = (agg["exact_k_pct"] * 100).round(1)

    print("\nStrategy ranking:")
    print(agg.to_string(float_format="%.3f"))

    # Per-case comparison
    if df["case"].nunique() <= 20:
        print("\nPer-case detail:")
        pivot = df.pivot(index="case", columns="strategy", values="ari")
        print(pivot.to_string(float_format="%.3f"))

        pivot_k = df.pivot(index="case", columns="strategy", values="found_k")
        print("\nK found:")
        print(pivot_k.to_string())


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

DEFAULT_CASES = [
    "cat_overlap_3cat_4c",  # Case 61 — the focus case
    "gauss_clear_small",  # Easy baseline
    "trivial_2c",  # Trivial
    "block_4c",  # Clean block-diagonal
    "sparse_72x72",  # Sparse 4-cluster
    "gauss_overlap_3c_small",  # Moderate overlap
    "overlap_heavy_4c_small_feat",  # Hard overlap
]

if __name__ == "__main__":
    if "--all" in sys.argv:
        all_cases = get_default_test_cases()
        case_names = [c["name"] for c in all_cases]
    elif len(sys.argv) > 1:
        case_names = [a for a in sys.argv[1:] if not a.startswith("-")]
    else:
        case_names = DEFAULT_CASES

    all_results: list[dict] = []
    for cn in case_names:
        all_results.extend(run_case(cn))

    print_summary(all_results)
