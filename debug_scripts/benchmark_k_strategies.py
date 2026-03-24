#!/usr/bin/env python3
"""
Benchmark all k-finding strategies across ALL benchmark test cases.

For each case, builds the tree once, runs Gate 2 (edge annotation) once,
then re-runs Gate 3 (sibling annotation) + decomposition for each k strategy.

Two modes per strategy:
  - Deflated (cousin_adjusted_wald) — the current default
  - Raw Wald (no deflation)

Outputs:
  - Per-case results table (printed)
  - Summary CSV saved to debug_scripts/results/k_strategy_benchmark.csv

Usage:
    python debug_scripts/benchmark_k_strategies.py
    python debug_scripts/benchmark_k_strategies.py --category gaussian
    python debug_scripts/benchmark_k_strategies.py --case gauss_clear_small
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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

from benchmarks.shared.cases import get_default_test_cases, list_categories
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

# ── Strategy names ──────────────────────────────────────────────────────
STRATEGY_NAMES = [
    "k_min_child",
    "k_geo_children",
    "k_parent",
    "k_geo_parent_min",
    "k_geo3",
    "k_gamma_adj",
    "k_parent_gamma",
]

SIBLING_METHODS = {
    "deflated": config.SIBLING_TEST_METHOD,  # cousin_adjusted_wald
    "raw_wald": "wald",
}


# ── Helpers ─────────────────────────────────────────────────────────────


def _count_leaves(tree: PosetTree, node: str) -> int:
    children = list(tree.successors(node))
    if not children:
        return 1
    return sum(_count_leaves(tree, c) for c in children)


def _decompose_from_annotations(tree: PosetTree, ann: pd.DataFrame) -> dict:
    """Run traversal directly from pre-annotated df, bypassing re-annotation."""
    root = tree.root()
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

    return {
        "cluster_assignments": cluster_assignments,
        "num_clusters": len(cluster_assignments),
    }


def _compute_ari(decomp: dict, y_t: np.ndarray, data_index: pd.Index, n: int) -> float:
    """Compute ARI from decomposition result."""
    y_pred = np.full(n, -1, dtype=int)
    for cid, cinfo in decomp["cluster_assignments"].items():
        for leaf in cinfo["leaves"]:
            y_pred[data_index.get_loc(leaf)] = cid
    return adjusted_rand_score(y_t, y_pred)


def _compute_strategy_dims(
    tree: PosetTree,
    edge_dims: dict[str, int],
    p: int,
) -> dict[str, dict[str, int]]:
    """Compute sibling_spectral_dims dicts for all 7 strategies."""
    strategy_dims: dict[str, dict[str, int]] = {name: {} for name in STRATEGY_NAMES}

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

        # k_min_child
        pos_ks = [k for k in [k_left, k_right] if k > 0]
        k_min = min(pos_ks) if pos_ks else 0
        if k_min > 0:
            strategy_dims["k_min_child"][parent] = k_min

        # k_geo_children
        k_geo_c = int(np.round(np.sqrt(k_left * k_right))) if k_left > 0 and k_right > 0 else k_min
        if k_geo_c > 0:
            strategy_dims["k_geo_children"][parent] = k_geo_c

        # k_parent
        if k_parent > 0:
            strategy_dims["k_parent"][parent] = k_parent

        # k_geo_parent_min
        k_gpm = (
            int(np.round(np.sqrt(k_parent * k_min)))
            if k_parent > 0 and k_min > 0
            else max(k_parent, k_min)
        )
        if k_gpm > 0:
            strategy_dims["k_geo_parent_min"][parent] = k_gpm

        # k_geo3
        k_g3 = (
            int(np.round((k_parent * k_left * k_right) ** (1 / 3)))
            if k_parent > 0 and k_left > 0 and k_right > 0
            else k_gpm
        )
        if k_g3 > 0:
            strategy_dims["k_geo3"][parent] = k_g3

        # k_gamma_adj = geomean(k_parent, k_min) * sqrt(n_small / p)
        n_small = min(n_left, n_right)
        k_gadj = int(np.round(k_gpm * np.sqrt(n_small / p))) if k_gpm > 0 and n_small > 0 else 0
        k_gadj = max(k_gadj, 2) if k_gadj > 0 or k_gpm > 0 else 0
        if k_gadj > 0:
            strategy_dims["k_gamma_adj"][parent] = k_gadj

        # k_parent_gamma = k_parent * min(1, sqrt(n_parent / p))
        k_pg = int(np.round(k_parent * min(1.0, np.sqrt(n_parent / p)))) if k_parent > 0 else 0
        k_pg = max(k_pg, 2) if k_parent > 0 else 0
        if k_pg > 0:
            strategy_dims["k_parent_gamma"][parent] = k_pg

    return strategy_dims


def run_single_case(tc: dict) -> list[dict]:
    """Run all strategies × both sibling methods for one test case.

    Returns a list of result rows (one per strategy × method combination).
    """
    case_name = tc["name"]
    true_k = tc.get("n_clusters")
    category = tc.get("category", "?")

    try:
        data_t, y_t, _, _ = generate_case_data(tc)
    except Exception as e:
        print(f"  SKIP {case_name}: data generation failed — {e}")
        return []

    n, p = data_t.shape

    dist = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_t.index.tolist())
    tree.populate_node_divergences(data_t)
    base = tree.annotations_df.copy()

    # Run Gate 2 (edge annotation) once to get spectral dims
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
    )
    edge_dims = pipeline_result.annotated_df.attrs.get("_spectral_dims", {})

    # Compute strategy dims
    strategy_dims = _compute_strategy_dims(tree, edge_dims, p)

    # Also get the "default" result (what the pipeline does now = k_min_child + deflated)
    default_decomp = _decompose_from_annotations(tree, pipeline_result.annotated_df)

    rows: list[dict] = []

    for strategy_name in STRATEGY_NAMES:
        dims = strategy_dims[strategy_name]

        for mode_name, sibling_method in SIBLING_METHODS.items():
            try:
                result = run_gate_annotation_pipeline(
                    tree,
                    base.copy(),
                    alpha_local=config.EDGE_ALPHA,
                    sibling_alpha=config.SIBLING_ALPHA,
                    leaf_data=data_t,
                    spectral_method=config.SPECTRAL_METHOD,
                    minimum_projection_dimension=config.PROJECTION_MINIMUM_DIMENSION,
                    sibling_method=sibling_method,
                    sibling_whitening=config.SIBLING_WHITENING,
                    fdr_method=config.EDGE_FDR_METHOD,
                    sibling_spectral_dims=dims,
                )
                ann = result.annotated_df
                decomp = _decompose_from_annotations(tree, ann)
                k_found = decomp["num_clusters"]

                ari = float("nan")
                if y_t is not None and true_k is not None:
                    ari = _compute_ari(decomp, y_t, data_t.index, n)

                audit = ann.attrs.get("sibling_divergence_audit", {})
                c_hat = audit.get("global_inflation_factor", float("nan"))

            except Exception as e:
                k_found = -1
                ari = float("nan")
                c_hat = float("nan")
                print(f"    ERROR {case_name}/{strategy_name}/{mode_name}: {e}")

            rows.append(
                {
                    "case": case_name,
                    "category": category,
                    "true_k": true_k,
                    "n": n,
                    "p": p,
                    "strategy": strategy_name,
                    "mode": mode_name,
                    "k_found": k_found,
                    "ari": ari,
                    "c_hat": c_hat,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark k-finding strategies")
    parser.add_argument(
        "--category", type=str, default=None, help="Only run cases from this category"
    )
    parser.add_argument("--case", type=str, default=None, help="Only run a single named case")
    args = parser.parse_args()

    all_cases = get_default_test_cases()

    if args.case:
        all_cases = [c for c in all_cases if c["name"] == args.case]
        if not all_cases:
            print(f"Case '{args.case}' not found.")
            sys.exit(1)
    elif args.category:
        all_cases = [c for c in all_cases if c.get("category") == args.category]
        if not all_cases:
            print(f"No cases in category '{args.category}'.")
            print(f"Available: {list_categories()}")
            sys.exit(1)

    total = len(all_cases)
    print(
        f"Running {total} cases × {len(STRATEGY_NAMES)} strategies × {len(SIBLING_METHODS)} modes"
    )
    print(f"Strategies: {STRATEGY_NAMES}")
    print(f"Modes: {list(SIBLING_METHODS.keys())}")
    print("=" * 80)

    all_rows: list[dict] = []
    t0 = time.time()

    for i, tc in enumerate(all_cases, 1):
        case_name = tc["name"]
        true_k = tc.get("n_clusters", "?")
        t_case = time.time()
        print(f"[{i}/{total}] {case_name} (true_k={true_k}) ...", end=" ", flush=True)

        rows = run_single_case(tc)
        all_rows.extend(rows)

        elapsed = time.time() - t_case
        if rows:
            # Quick summary: deflated k_min_child vs k_parent
            deflated = {r["strategy"]: r for r in rows if r["mode"] == "deflated"}
            min_c = deflated.get("k_min_child", {})
            parent = deflated.get("k_parent", {})
            print(
                f"({elapsed:.1f}s) "
                f"min_child: K={min_c.get('k_found','?')}, ARI={min_c.get('ari',float('nan')):.3f} | "
                f"parent: K={parent.get('k_found','?')}, ARI={parent.get('ari',float('nan')):.3f}"
            )
        else:
            print(f"({elapsed:.1f}s) SKIPPED")

    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.0f}s")

    if not all_rows:
        print("No results to report.")
        return

    df = pd.DataFrame(all_rows)

    # ── Summary tables ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: Mean ARI by strategy × mode (across all cases with known true_k)")
    print("=" * 100)

    valid = df[df["true_k"].notna() & (df["true_k"] > 0) & df["ari"].notna()].copy()
    if not valid.empty:
        pivot_ari = valid.pivot_table(
            values="ari", index="strategy", columns="mode", aggfunc="mean"
        )
        pivot_ari = pivot_ari.reindex(STRATEGY_NAMES)
        print(pivot_ari.to_string(float_format="%.3f"))

        print("\n" + "-" * 100)
        print("SUMMARY: Median ARI by strategy × mode")
        print("-" * 100)
        pivot_med = valid.pivot_table(
            values="ari", index="strategy", columns="mode", aggfunc="median"
        )
        pivot_med = pivot_med.reindex(STRATEGY_NAMES)
        print(pivot_med.to_string(float_format="%.3f"))

        print("\n" + "-" * 100)
        print("SUMMARY: Exact K recovery rate (K_found == true_K)")
        print("-" * 100)
        valid["exact_k"] = valid["k_found"] == valid["true_k"]
        pivot_exact = valid.pivot_table(
            values="exact_k", index="strategy", columns="mode", aggfunc="mean"
        )
        pivot_exact = pivot_exact.reindex(STRATEGY_NAMES)
        print(pivot_exact.to_string(float_format="%.3f"))

        print("\n" + "-" * 100)
        print("SUMMARY: Mean K_found / true_K ratio")
        print("-" * 100)
        valid["k_ratio"] = valid["k_found"] / valid["true_k"]
        pivot_ratio = valid.pivot_table(
            values="k_ratio", index="strategy", columns="mode", aggfunc="mean"
        )
        pivot_ratio = pivot_ratio.reindex(STRATEGY_NAMES)
        print(pivot_ratio.to_string(float_format="%.2f"))

        # K=1 rate
        print("\n" + "-" * 100)
        print("SUMMARY: K=1 rate (collapsing to single cluster)")
        print("-" * 100)
        valid["k_is_1"] = valid["k_found"] == 1
        pivot_k1 = valid.pivot_table(
            values="k_is_1", index="strategy", columns="mode", aggfunc="mean"
        )
        pivot_k1 = pivot_k1.reindex(STRATEGY_NAMES)
        print(pivot_k1.to_string(float_format="%.3f"))

    # ── Per-case detail table (deflated only) ───────────────────────────
    print("\n" + "=" * 100)
    print("PER-CASE DETAIL (deflated mode): K_found for each strategy")
    print("=" * 100)

    deflated_df = df[df["mode"] == "deflated"].copy()
    if not deflated_df.empty:
        detail = deflated_df.pivot_table(
            values="k_found", index=["case", "category", "true_k"], columns="strategy"
        )
        detail = detail.reindex(columns=STRATEGY_NAMES)
        # Add a "best_ari" column showing which strategy had best ARI
        best_rows = []
        for case_name in deflated_df["case"].unique():
            case_rows = deflated_df[deflated_df["case"] == case_name]
            best = (
                case_rows.loc[case_rows["ari"].idxmax()]
                if not case_rows["ari"].isna().all()
                else None
            )
            if best is not None:
                best_rows.append(
                    {
                        "case": case_name,
                        "best_strategy": best["strategy"],
                        "best_ari": best["ari"],
                    }
                )
        if best_rows:
            best_df = pd.DataFrame(best_rows).set_index("case")
            detail = detail.reset_index()
            detail = detail.merge(best_df, left_on="case", right_index=True, how="left")
            detail = detail.set_index(["case", "category", "true_k"])

        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 200)
        print(detail.to_string(float_format="%.0f"))

    # ── Save CSV ────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "k_strategy_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")


if __name__ == "__main__":
    main()
