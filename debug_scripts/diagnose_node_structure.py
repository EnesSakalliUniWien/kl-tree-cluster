#!/usr/bin/env python3
"""
Dump node-level structure for a benchmark case: leaf counts, spectral dims,
gamma ratios, and min-child vs parent vs geomean k comparison.

Usage:
    python debug_scripts/diagnose_node_structure.py <case_name>
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ.setdefault("KL_TE_N_JOBS", "1")

from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    GateEvaluator,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.traversal import (
    iterate_worklist,
    process_node,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import extract_bool_column_dict
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _count_leaves(tree, node: str) -> int:
    children = list(tree.successors(node))
    if not children:
        return 1
    return sum(_count_leaves(tree, c) for c in children)


def _decompose_from_annotations(tree: PosetTree, ann: pd.DataFrame) -> dict:
    """Run the traversal directly from pre-annotated df, no re-annotation."""
    root = next(n for n, d in tree.in_degree() if d == 0)

    children_map = {n: list(tree.successors(n)) for n in tree.nodes()}
    descendant_leaf_sets = tree.compute_descendant_sets(use_labels=True)

    local_sig = extract_bool_column_dict(ann, "Child_Parent_Divergence_Significant")
    sib_diff = extract_bool_column_dict(ann, "Sibling_BH_Different")
    sib_skip = extract_bool_column_dict(ann, "Sibling_Divergence_Skipped")

    gate = GateEvaluator(
        tree=tree,
        local_significant=local_sig,
        sibling_different=sib_diff,
        sibling_skipped=sib_skip,
        children_map=children_map,
        descendant_leaf_sets=descendant_leaf_sets,
        passthrough=config.PASSTHROUGH,
    )

    nodes_to_visit = [root]
    final_leaf_sets: list[set[str]] = []
    processed: set[str] = set()
    for node in iterate_worklist(nodes_to_visit, processed):
        process_node(node, gate, nodes_to_visit, final_leaf_sets)

    cluster_assignments = {}
    for i, leaves in enumerate(final_leaf_sets):
        cluster_assignments[i] = {"leaves": leaves, "size": len(leaves)}

    return {"cluster_assignments": cluster_assignments, "num_clusters": len(cluster_assignments)}


def diagnose(case_name: str) -> None:
    all_cases = get_default_test_cases()
    tc = next((c for c in all_cases if c["name"] == case_name), None)
    if tc is None:
        print(f"ERROR: Case '{case_name}' not found.")
        return

    data_t, y_t, _, _ = generate_case_data(tc)
    n, p = data_t.shape

    dist = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())
    tree.populate_node_divergences(data_t)
    base = tree.stats_df.copy()

    result = run_gate_annotation_pipeline(
        tree,
        base.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_t,
        spectral_method=config.SPECTRAL_METHOD,
        minimum_projection_dimension=config.PROJECTION_MINIMUM_DIMENSION,
        sibling_method=config.SIBLING_TEST_METHOD,
        sibling_whitening=config.SIBLING_WHITENING,
        fdr_method=config.EDGE_FDR_METHOD,
    )
    stats = result.annotated_df
    edge_dims = stats.attrs.get("_spectral_dims", {})

    # Build per-parent table
    rows = []
    for parent in tree.nodes():
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        n_left = _count_leaves(tree, left)
        n_right = _count_leaves(tree, right)
        n_parent = n_left + n_right
        k_left = edge_dims.get(left, 0)
        k_right = edge_dims.get(right, 0)
        k_parent = edge_dims.get(parent, 0)
        k_min = (
            min(k for k in [k_left, k_right] if k > 0)
            if any(k > 0 for k in [k_left, k_right])
            else 0
        )
        k_geo_children = (
            int(np.round(np.sqrt(k_left * k_right))) if k_left > 0 and k_right > 0 else k_min
        )
        # geomean(k_parent, k_min_child)
        k_geo_parent_min = (
            int(np.round(np.sqrt(k_parent * k_min)))
            if k_parent > 0 and k_min > 0
            else max(k_parent, k_min)
        )
        # geomean(k_parent, k_left, k_right)
        k_geo3 = (
            int(np.round((k_parent * k_left * k_right) ** (1 / 3)))
            if k_parent > 0 and k_left > 0 and k_right > 0
            else k_geo_parent_min
        )
        # gamma-aware: scale by sqrt(n_small / p) to penalize high-gamma nodes
        n_small = min(n_left, n_right)
        gamma_small = p / n_small if n_small > 0 else float("inf")
        gamma_parent = p / n_parent if n_parent > 0 else float("inf")
        # effective k: geomean(k_parent, k_min_child) * sqrt(n_small / p)
        # = pull toward parent k but discount by how data-starved the small child is
        k_gamma_adj = (
            int(np.round(k_geo_parent_min * np.sqrt(n_small / p)))
            if k_geo_parent_min > 0 and np.isfinite(gamma_small)
            else 0
        )
        k_gamma_adj = max(k_gamma_adj, 2)  # floor at 2
        # parent-k scaled by parent gamma: k_parent * min(1, sqrt(n_parent / p))
        k_parent_gamma = (
            int(np.round(k_parent * min(1.0, np.sqrt(n_parent / p)))) if k_parent > 0 else 0
        )
        k_parent_gamma = max(k_parent_gamma, 2) if k_parent > 0 else 0
        rows.append(
            {
                "parent": parent,
                "n_parent": n_parent,
                "n_left": n_left,
                "n_right": n_right,
                "n_small": n_small,
                "gamma_small": gamma_small,
                "gamma_parent": gamma_parent,
                "k_left": k_left,
                "k_right": k_right,
                "k_parent": k_parent,
                "k_min_child": k_min,
                "k_geo_children": k_geo_children,
                "k_geo_parent_min": k_geo_parent_min,
                "k_geo3": k_geo3,
                "k_gamma_adj": k_gamma_adj,
                "k_parent_gamma": k_parent_gamma,
            }
        )

    df = pd.DataFrame(rows)

    print(f"Case: {case_name}")
    print(f"p (features) = {p}, n (samples) = {n}")
    print(f"Total binary parents: {len(df)}")

    # Leaf count distribution
    print("\n--- Leaf count distribution (n_parent) ---")
    print(f"  min={df.n_parent.min()}, median={df.n_parent.median():.0f}, max={df.n_parent.max()}")
    bins = [2, 5, 10, 20, 50, 100, 200, 500, n + 1]
    for lo, hi in zip(bins[:-1], bins[1:]):
        cnt = ((df.n_parent >= lo) & (df.n_parent < hi)).sum()
        print(f"  [{lo:>3d}, {hi:>3d}): {cnt} nodes")

    # Smaller child distribution
    small = df[["n_left", "n_right"]].min(axis=1)
    print("\n--- Smaller child leaf count ---")
    print(
        f"  min={small.min()}, median={small.median():.0f}, max={small.max()}, mean={small.mean():.1f}"
    )
    for thresh in [1, 2, 3, 5, 10, 20, 50]:
        pct = (small <= thresh).mean() * 100
        print(f"  smaller child <={thresh}: {(small <= thresh).sum()} ({pct:.0f}%)")

    # Edge spectral dims
    k_vals = np.array([v for v in edge_dims.values() if v > 0])
    print("\n--- Edge spectral dims (k) ---")
    print(f"  Nodes with k>0: {len(k_vals)}")
    print(
        f"  min={k_vals.min()}, median={np.median(k_vals):.0f}, max={k_vals.max()}, mean={k_vals.mean():.1f}"
    )
    for thresh in [1, 2, 3, 5, 10, 20, 50]:
        print(f"  k<={thresh}: {(k_vals <= thresh).sum()} ({(k_vals <= thresh).mean()*100:.0f}%)")

    # Comparison of all k strategies
    print("\n--- k strategy comparison (across binary parents) ---")
    strategies = [
        ("k_min_child", "min(k_L, k_R)"),
        ("k_geo_children", "geomean(k_L, k_R)"),
        ("k_parent", "k_parent"),
        ("k_geo_parent_min", "geomean(k_par, k_min)"),
        ("k_geo3", "geomean(k_par, k_L, k_R)"),
        ("k_gamma_adj", "geo(k_par,k_min)*sqrt(n_s/p)"),
        ("k_parent_gamma", "k_par*min(1,sqrt(n_par/p))"),
    ]
    for col, desc in strategies:
        vals = df[col]
        print(
            f"  {col:>18s}: min={vals.min():>3}, median={vals.median():>5.0f}, "
            f"max={vals.max():>4}, mean={vals.mean():>5.1f}  ({desc})"
        )

    # gamma for smaller child
    gamma_small = p / small
    print("\n--- gamma (d/n) for smaller child ---")
    print(
        f"  min={gamma_small.min():.1f}, median={gamma_small.median():.1f}, max={gamma_small.max():.1f}"
    )
    print(
        f"  gamma > 1: {(gamma_small > 1).mean()*100:.0f}%, gamma > 10: {(gamma_small > 10).mean()*100:.0f}%"
    )

    # Top 15 by parent size
    print("\n--- Top 15 nodes by parent size ---")
    print(
        f"  {'node':>6s} {'n':>4s} {'split':>10s} {'k_L':>4s} {'k_R':>4s} {'k_par':>5s} {'min':>4s} {'geo_c':>5s} {'geo_pm':>6s} {'geo3':>5s} {'γ_adj':>5s} {'p_γ':>5s} {'γ_s':>6s}"
    )
    for _, r in df.nlargest(15, "n_parent").iterrows():
        print(
            f"  {r.parent:>6s} {r.n_parent:>4.0f} {r.n_left:>4.0f}+{r.n_right:<4.0f} "
            f"{r.k_left:>4.0f} {r.k_right:>4.0f} {r.k_parent:>5.0f} {r.k_min_child:>4.0f} "
            f"{r.k_geo_children:>5.0f} {r.k_geo_parent_min:>6.0f} {r.k_geo3:>5.0f} "
            f"{r.k_gamma_adj:>5.0f} {r.k_parent_gamma:>5.0f} {r.gamma_small:>6.1f}"
        )

    # Bottom 15 by parent size
    print("\n--- Bottom 15 nodes by parent size ---")
    print(
        f"  {'node':>6s} {'n':>4s} {'split':>10s} {'k_L':>4s} {'k_R':>4s} {'k_par':>5s} {'min':>4s} {'geo_c':>5s} {'geo_pm':>6s} {'geo3':>5s} {'γ_adj':>5s} {'p_γ':>5s} {'γ_s':>6s}"
    )
    for _, r in df.nsmallest(15, "n_parent").iterrows():
        print(
            f"  {r.parent:>6s} {r.n_parent:>4.0f} {r.n_left:>4.0f}+{r.n_right:<4.0f} "
            f"{r.k_left:>4.0f} {r.k_right:>4.0f} {r.k_parent:>5.0f} {r.k_min_child:>4.0f} "
            f"{r.k_geo_children:>5.0f} {r.k_geo_parent_min:>6.0f} {r.k_geo3:>5.0f} "
            f"{r.k_gamma_adj:>5.0f} {r.k_parent_gamma:>5.0f} {r.gamma_small:>6.1f}"
        )

    # ── Counterfactual decomposition: run pipeline with each k-strategy ──
    print("\n" + "=" * 72)
    print("  COUNTERFACTUAL DECOMPOSITION — K found per strategy")
    print("=" * 72)

    true_k = tc.get("n_clusters")

    # Build sibling_spectral_dims dicts for each strategy
    strategy_dims: dict[str, dict[str, int]] = {}
    strategy_cols = [
        "k_min_child",
        "k_geo_children",
        "k_parent",
        "k_geo_parent_min",
        "k_geo3",
        "k_gamma_adj",
        "k_parent_gamma",
    ]
    for col in strategy_cols:
        dims = {}
        for _, r in df.iterrows():
            k = int(r[col])
            if k > 0:
                dims[r["parent"]] = k
        strategy_dims[col] = dims

    for strategy_name in strategy_cols:
        dims = strategy_dims[strategy_name]
        # Run full pipeline with this strategy's sibling dims
        pipeline_result = run_gate_annotation_pipeline(
            tree,
            base.copy(),
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
            leaf_data=data_t,
            spectral_method=config.SPECTRAL_METHOD,
            minimum_projection_dimension=config.PROJECTION_MINIMUM_DIMENSION,
            sibling_method=config.SIBLING_TEST_METHOD,
            sibling_whitening=config.SIBLING_WHITENING,
            fdr_method=config.EDGE_FDR_METHOD,
            sibling_spectral_dims=dims,
        )
        ann = pipeline_result.annotated_df

        # Run decomposition directly from annotations (no re-annotation)
        decomp = _decompose_from_annotations(tree, ann)
        k_found = decomp["num_clusters"]

        ari_str = "N/A"
        if y_t is not None and true_k is not None:
            y_pred = np.full(n, -1, dtype=int)
            for cid, cinfo in decomp["cluster_assignments"].items():
                for leaf in cinfo["leaves"]:
                    y_pred[data_t.index.get_loc(leaf)] = cid
            ari_str = f"{adjusted_rand_score(y_t, y_pred):.3f}"

        # Gate 3 pass count
        g3_pass = 0
        for node in tree.nodes():
            if tree.out_degree(node) == 0:
                continue
            if not bool(ann.loc[node, "Sibling_Divergence_Skipped"]) and bool(
                ann.loc[node, "Sibling_BH_Different"]
            ):
                g3_pass += 1

        audit = ann.attrs.get("sibling_divergence_audit", {})
        c_hat = audit.get("global_inflation_factor", "?")

        print(
            f"  {strategy_name:>18s}: K={k_found:>3}, ARI={ari_str}, "
            f"G3_pass={g3_pass:>4}, ĉ={c_hat}"
        )

    # Also run raw wald (no deflation) with each strategy
    print("\n  --- Raw Wald (no deflation) ---")
    for strategy_name in strategy_cols:
        dims = strategy_dims[strategy_name]
        pipeline_result = run_gate_annotation_pipeline(
            tree,
            base.copy(),
            alpha_local=config.EDGE_ALPHA,
            sibling_alpha=config.SIBLING_ALPHA,
            leaf_data=data_t,
            spectral_method=config.SPECTRAL_METHOD,
            minimum_projection_dimension=config.PROJECTION_MINIMUM_DIMENSION,
            sibling_method="wald",
            sibling_whitening=config.SIBLING_WHITENING,
            fdr_method=config.EDGE_FDR_METHOD,
            sibling_spectral_dims=dims,
        )
        ann = pipeline_result.annotated_df

        decomp = _decompose_from_annotations(tree, ann)
        k_found = decomp["num_clusters"]

        ari_str = "N/A"
        if y_t is not None and true_k is not None:
            y_pred = np.full(n, -1, dtype=int)
            for cid, cinfo in decomp["cluster_assignments"].items():
                for leaf in cinfo["leaves"]:
                    y_pred[data_t.index.get_loc(leaf)] = cid
            ari_str = f"{adjusted_rand_score(y_t, y_pred):.3f}"

        g3_pass = 0
        for node in tree.nodes():
            if tree.out_degree(node) == 0:
                continue
            if not bool(ann.loc[node, "Sibling_Divergence_Skipped"]) and bool(
                ann.loc[node, "Sibling_BH_Different"]
            ):
                g3_pass += 1

        print(f"  {strategy_name:>18s}: K={k_found:>3}, ARI={ari_str}, G3_pass={g3_pass:>4}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_name>")
        sys.exit(1)
    diagnose(sys.argv[1])
    diagnose(sys.argv[1])
