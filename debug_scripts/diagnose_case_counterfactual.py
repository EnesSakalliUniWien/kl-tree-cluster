#!/usr/bin/env python3
"""
Sibling-gate diagnosis with raw-vs-configured and parent-k counterfactual views.

Usage:
    python debug_scripts/diagnose_case_counterfactual.py <case_name> [case_name2 ...]
    python debug_scripts/diagnose_case_counterfactual.py --no-parent-k <case_name>

Example:
    python debug_scripts/diagnose_case_counterfactual.py overlap_heavy_8c_large_feat
"""

from __future__ import annotations

import argparse
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
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


ROOT_COMPARE_COLUMNS = [
    ("Sibling_Degrees_of_Freedom", "df"),
    ("Sibling_Test_Statistic", "stat"),
    ("Sibling_Divergence_P_Value", "p"),
    ("Sibling_Divergence_P_Value_Corrected", "p_corr"),
    ("Sibling_BH_Different", "different"),
    ("Sibling_Test_Method", "method"),
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", help="Benchmark case names to diagnose.")
    parser.add_argument(
        "--no-parent-k",
        action="store_true",
        help="Skip the parent-k counterfactual rerun.",
    )
    return parser.parse_args(argv)


def _format_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value))
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "nan"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if val == 0.0:
            return "0"
        if abs(val) >= 1e4 or abs(val) < 1e-3:
            return f"{val:.3e}"
        return f"{val:.6g}"
    return str(value)


def _run_pipeline_variant(
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    leaf_data: pd.DataFrame,
    *,
    sibling_method: str,
    sibling_spectral_dims: dict[str, int] | None = None,
) -> pd.DataFrame:
    return run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=leaf_data,
        spectral_method=config.SPECTRAL_METHOD,
        minimum_projection_dimension=config.PROJECTION_MINIMUM_DIMENSION,
        sibling_method=sibling_method,
        sibling_whitening=config.SIBLING_WHITENING,
        fdr_method=config.EDGE_FDR_METHOD,
        sibling_spectral_dims=sibling_spectral_dims,
    ).annotated_df


def _count_gate2_pass(tree: PosetTree, stats: pd.DataFrame) -> int:
    count = 0
    for node in tree.nodes():
        if tree.out_degree(node) == 0:
            continue
        children = list(tree.successors(node))
        if len(children) != 2:
            continue
        left_sig = bool(stats.loc[children[0], "Child_Parent_Divergence_Significant"])
        right_sig = bool(stats.loc[children[1], "Child_Parent_Divergence_Significant"])
        if left_sig or right_sig:
            count += 1
    return count


def _count_gate3(tree: PosetTree, stats: pd.DataFrame) -> tuple[int, int]:
    tested = 0
    passed = 0
    for node in tree.nodes():
        if tree.out_degree(node) == 0:
            continue
        skipped = bool(stats.loc[node, "Sibling_Divergence_Skipped"])
        different = bool(stats.loc[node, "Sibling_BH_Different"])
        if not skipped:
            tested += 1
        if different and not skipped:
            passed += 1
    return tested, passed


def _print_root_comparison(
    root: str,
    raw_stats: pd.DataFrame,
    configured_stats: pd.DataFrame,
    *,
    title: str,
    configured_label: str,
) -> None:
    print(f"\n--- Gate 3 at root ({title}) ---")
    for col, label in ROOT_COMPARE_COLUMNS:
        raw_value = raw_stats.loc[root, col] if col in raw_stats.columns else None
        configured_value = (
            configured_stats.loc[root, col] if col in configured_stats.columns else None
        )
        print(
            f"  {label}: raw_wald={_format_value(raw_value)} | "
            f"{configured_label}={_format_value(configured_value)}"
        )


def _print_gate_summary(
    tree: PosetTree,
    raw_stats: pd.DataFrame,
    configured_stats: pd.DataFrame,
    *,
    title: str,
    configured_label: str,
) -> None:
    internal_count = sum(1 for node in tree.nodes() if tree.out_degree(node) > 0)
    gate2_pass = _count_gate2_pass(tree, configured_stats)
    raw_tested, raw_pass = _count_gate3(tree, raw_stats)
    configured_tested, configured_pass = _count_gate3(tree, configured_stats)

    print(f"\n--- Gate Summary ({title}) ---")
    print(f"  Gate 2 pass: {gate2_pass}/{internal_count}")
    print(f"  Gate 3 tested: raw_wald={raw_tested}/{internal_count}")
    print(f"  Gate 3 pass: raw_wald={raw_pass}/{internal_count}")
    print(f"  Gate 3 tested: {configured_label}={configured_tested}/{internal_count}")
    print(f"  Gate 3 pass: {configured_label}={configured_pass}/{internal_count}")


def _print_top_sibling_nodes(
    tree: PosetTree,
    stats: pd.DataFrame,
    *,
    title: str,
    limit: int = 5,
) -> None:
    if "Sibling_Divergence_P_Value_Corrected" not in stats.columns:
        return
    internal = [node for node in tree.nodes() if tree.out_degree(node) > 0]
    tested_df = stats.loc[internal].copy()
    tested_df = tested_df[~tested_df["Sibling_Divergence_Skipped"].astype(bool)]
    if tested_df.empty:
        return
    tested_df = tested_df.sort_values("Sibling_Divergence_P_Value_Corrected")
    print(f"\n--- Top {limit} nodes by lowest corrected sibling p-value ({title}) ---")
    for node, row in tested_df.head(limit).iterrows():
        children = list(tree.successors(node))
        left_sig = bool(stats.loc[children[0], "Child_Parent_Divergence_Significant"])
        right_sig = bool(stats.loc[children[1], "Child_Parent_Divergence_Significant"])
        gate2_str = "G2✓" if (left_sig or right_sig) else "G2✗"
        print(
            f"  {node}: p_corr={_format_value(row['Sibling_Divergence_P_Value_Corrected'])}, "
            f"stat={_format_value(row['Sibling_Test_Statistic'])}, "
            f"df={_format_value(row['Sibling_Degrees_of_Freedom'])}, "
            f"diff={_format_value(row['Sibling_BH_Different'])}, {gate2_str}"
        )


def _print_sibling_audit(stats: pd.DataFrame, *, title: str) -> None:
    audit = stats.attrs.get("sibling_divergence_audit", {})
    if not audit:
        return
    print(f"\n--- Sibling calibration audit ({title}) ---")
    for key, value in audit.items():
        if key == "diagnostics" and isinstance(value, dict):
            print("  diagnostics:")
            for diag_key, diag_value in value.items():
                print(f"    {diag_key}: {_format_value(diag_value)}")
        else:
            print(f"  {key}: {_format_value(value)}")


def _print_dfs_tree(tree: PosetTree, stats: pd.DataFrame, node: str, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return

    indent = "  " * depth
    children = list(tree.successors(node))
    if not children:
        leaf_count = int(stats.loc[node, "leaf_count"]) if "leaf_count" in stats.columns else 1
        print(f"{indent}leaf {node} (n={leaf_count})")
        return
    if len(children) != 2:
        print(f"{indent}non-binary {node} ({len(children)} children)")
        return

    leaf_count = int(stats.loc[node, "leaf_count"]) if "leaf_count" in stats.columns else -1
    left_sig = bool(stats.loc[children[0], "Child_Parent_Divergence_Significant"])
    right_sig = bool(stats.loc[children[1], "Child_Parent_Divergence_Significant"])
    gate2 = left_sig or right_sig
    skipped = bool(stats.loc[node, "Sibling_Divergence_Skipped"])
    different = bool(stats.loc[node, "Sibling_BH_Different"])
    gate3 = different and not skipped

    if gate2 and gate3:
        decision = "SPLIT"
    elif not gate2:
        decision = "MERGE (Gate 2 fail)"
    else:
        decision = "MERGE (Gate 3 fail)"

    print(
        f"{indent}{node} (n={leaf_count}): {decision} "
        f"[edge L={left_sig}, R={right_sig}; sib={'DIFF' if different else 'SAME'}"
        f"{' SKIP' if skipped else ''}]"
    )
    for child in children:
        _print_dfs_tree(tree, stats, child, depth + 1, max_depth)


def diagnose_case(case_name: str, *, include_parent_k: bool) -> None:
    all_cases = get_default_test_cases()
    case = next((entry for entry in all_cases if entry["name"] == case_name), None)
    if case is None:
        print(f"ERROR: Case '{case_name}' not found.")
        print("Available:", [entry["name"] for entry in all_cases][:20], "...")
        return

    print("=" * 72)
    print(f"  CASE: {case_name}")
    print("=" * 72)

    print("\nCase config:")
    for key, value in case.items():
        if key != "generator":
            print(f"  {key}: {value}")
    print(f"  parent_k_counterfactual: {include_parent_k}")

    data_t, y_t, x_original, meta = generate_case_data(case)
    n_samples, n_features = data_t.shape
    true_k = case.get("n_clusters")

    print(f"\nData: {n_samples} samples x {n_features} features")
    if y_t is not None:
        label_counts = dict(zip(*np.unique(y_t, return_counts=True)))
        print(f"True labels (K={len(label_counts)}): {label_counts}")
    print(f"Sparsity: {1 - data_t.values.mean():.3f}")
    print(
        f"Column variance: min={data_t.values.var(0).min():.4f}, "
        f"max={data_t.values.var(0).max():.4f}, "
        f"mean={data_t.values.var(0).mean():.4f}"
    )
    constant_columns = int((data_t.values.var(0) == 0).sum())
    print(f"Constant columns: {constant_columns}/{n_features}")

    if y_t is not None:
        print("\n--- Cluster means (per-feature average of binary values) ---")
        cluster_means: list[np.ndarray] = []
        for label in sorted(np.unique(y_t)):
            mask = y_t == label
            cluster_mean = data_t.values[mask].mean(axis=0)
            cluster_means.append(cluster_mean)
            print(
                f"  Cluster {int(label)}: n={mask.sum()}, "
                f"col_mean range=[{cluster_mean.min():.3f}, {cluster_mean.max():.3f}], "
                f"overall theta_bar={cluster_mean.mean():.3f}"
            )
        if len(cluster_means) > 1:
            inter_dists = pdist(np.array(cluster_means), metric="hamming")
            print(
                f"  Inter-cluster Hamming: min={inter_dists.min():.3f}, "
                f"max={inter_dists.max():.3f}, mean={inter_dists.mean():.3f}"
            )

    dist = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    linkage_matrix = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_t.index.tolist())
    tree.populate_node_divergences(data_t)
    base_annotations = tree.stats_df.copy()

    print(
        f"\nConfig: SPECTRAL_METHOD={config.SPECTRAL_METHOD}, "
        f"SPECTRAL_MINIMUM_DIMENSION={config.SPECTRAL_MINIMUM_DIMENSION}, "
        f"SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}"
    )

    decomp = tree.decompose(
        annotations_df=base_annotations.copy(),
        leaf_data=data_t,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    configured_stats = tree.stats_df.copy()
    raw_stats = _run_pipeline_variant(
        tree,
        base_annotations,
        data_t,
        sibling_method="wald",
    )

    ari_str = "N/A"
    if y_t is not None and true_k is not None:
        y_pred = np.full(n_samples, -1, dtype=int)
        for cluster_id, cluster_info in decomp["cluster_assignments"].items():
            for leaf in cluster_info["leaves"]:
                y_pred[data_t.index.get_loc(leaf)] = cluster_id
        ari_str = f"{adjusted_rand_score(y_t, y_pred):.3f}"

    print(f"\n>>> K_found={decomp['num_clusters']} (true={true_k}), ARI={ari_str}")

    root = next(node for node, degree in tree.in_degree() if degree == 0)
    children = list(tree.successors(root))
    print(f"\nRoot: {root}, children: {children}")

    print("\n--- Gate 2 (child-parent divergence) at root ---")
    for child in children:
        print(
            f"  {child}: sig={configured_stats.loc[child, 'Child_Parent_Divergence_Significant']}, "
            f"p_BH={configured_stats.loc[child, 'Child_Parent_Divergence_P_Value_BH']:.2e}, "
            f"df={configured_stats.loc[child, 'Child_Parent_Divergence_df']}"
        )

    _print_root_comparison(
        root,
        raw_stats,
        configured_stats,
        title="current sibling dims",
        configured_label=config.SIBLING_TEST_METHOD,
    )
    _print_gate_summary(
        tree,
        raw_stats,
        configured_stats,
        title="current sibling dims",
        configured_label=config.SIBLING_TEST_METHOD,
    )

    spectral_dims = configured_stats.attrs.get("_spectral_dims", {})
    if spectral_dims:
        internal_dims = [value for node, value in spectral_dims.items() if tree.out_degree(node) > 0]
        if internal_dims:
            print("\n--- Spectral dimensions (Gate 2 edge dims) ---")
            print(
                f"  min={min(internal_dims)}, max={max(internal_dims)}, "
                f"median={np.median(internal_dims):.0f}, mean={np.mean(internal_dims):.1f}"
            )

    _print_top_sibling_nodes(
        tree,
        configured_stats,
        title=f"configured {config.SIBLING_TEST_METHOD}",
    )
    _print_sibling_audit(configured_stats, title="configured current dims")

    if include_parent_k and spectral_dims:
        parent_k_configured = _run_pipeline_variant(
            tree,
            base_annotations,
            data_t,
            sibling_method=config.SIBLING_TEST_METHOD,
            sibling_spectral_dims=spectral_dims,
        )
        parent_k_raw = _run_pipeline_variant(
            tree,
            base_annotations,
            data_t,
            sibling_method="wald",
            sibling_spectral_dims=spectral_dims,
        )

        _print_root_comparison(
            root,
            parent_k_raw,
            parent_k_configured,
            title="parent-k counterfactual",
            configured_label=f"{config.SIBLING_TEST_METHOD}/parent_k",
        )
        _print_gate_summary(
            tree,
            parent_k_raw,
            parent_k_configured,
            title="parent-k counterfactual",
            configured_label=f"{config.SIBLING_TEST_METHOD}/parent_k",
        )
        _print_top_sibling_nodes(
            tree,
            parent_k_configured,
            title=f"configured {config.SIBLING_TEST_METHOD}, parent-k",
        )
        _print_sibling_audit(parent_k_configured, title="configured parent-k")

    print("\n--- DFS decision tree (configured current dims, first 3 levels) ---")
    _print_dfs_tree(tree, configured_stats, root, depth=0, max_depth=3)
    print()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    for case_name in args.cases:
        diagnose_case(case_name, include_parent_k=not args.no_parent_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
