#!/usr/bin/env python3
"""Experiment 2 — Pass-through depth limit.

Tests the hypothesis: limiting the number of consecutive pass-through
decisions prevents runaway splitting while preserving the ability to
discover deep structure.

We re-implement the traversal with a max_passthrough_depth counter:
each time a node triggers pass-through, increment the counter.
When a node triggers a real SPLIT, reset the counter.
When depth exceeds the limit, force MERGE.

Usage:
    python debug_scripts/enhancement_lab/exp2_passthrough_depth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debug_scripts.enhancement_lab.lab_helpers import (
    FAILURE_CASES,
    REGRESSION_GUARD_CASES,
    build_tree_and_data,
    compute_ari,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition


def _decompose_with_pt_limit(
    tree,
    data_df,
    max_pt_depth: int,
    n_min: int = 0,
) -> dict:
    """Decompose with a pass-through depth limit.

    max_pt_depth: max consecutive pass-throughs before forcing MERGE.
                  0 = disable pass-through entirely.
    n_min:        min leaf count floor (0 = disabled).
    """
    decomposer = TreeDecomposition(
        tree=tree,
        annotations_df=tree.annotations_df.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_df,
    )

    gate = decomposer._gate
    nodes_to_visit = [decomposer._root]
    final_leaf_sets: list[set[str]] = []
    processed: set[str] = set()

    # Track pass-through depth per node
    # Key: node_id -> how many consecutive PTs led to this node
    pt_depth: dict[str, int] = {decomposer._root: 0}

    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id in processed:
            continue
        processed.add(node_id)

        leaf_count = len(gate._descendant_leaf_sets.get(node_id, ()))
        current_pt = pt_depth.get(node_id, 0)

        # Min-sample guard
        if n_min > 0 and leaf_count < n_min:
            final_leaf_sets.append(set(gate._descendant_leaf_sets[node_id]))
            continue

        if gate.should_split(node_id):
            # Real split — reset PT counter for children
            children = gate._children_map[node_id]
            left, right = children
            nodes_to_visit.append(right)
            nodes_to_visit.append(left)
            pt_depth[left] = 0
            pt_depth[right] = 0

        elif gate.should_pass_through(node_id) and current_pt < max_pt_depth:
            # Pass-through within budget — increment counter
            children = gate._children_map[node_id]
            left, right = children
            nodes_to_visit.append(right)
            nodes_to_visit.append(left)
            pt_depth[left] = current_pt + 1
            pt_depth[right] = current_pt + 1

        else:
            # MERGE: either gate fail or PT budget exhausted
            final_leaf_sets.append(set(gate._descendant_leaf_sets[node_id]))

    cluster_assignments = decomposer._build_cluster_assignments(final_leaf_sets)
    return {
        "cluster_assignments": cluster_assignments,
        "num_clusters": len(cluster_assignments),
    }


def sweep_pt_depth(
    cases: list[str],
    pt_depths: list[int],
    n_min: int = 0,
) -> pd.DataFrame:
    """Sweep max_pt_depth across cases."""
    all_rows = []
    for case_name in cases:
        try:
            tree, data_df, y_t, tc = build_tree_and_data(case_name)
            true_k = tc.get("n_clusters", None)

            for pt_max in pt_depths:
                decomp = _decompose_with_pt_limit(tree, data_df, max_pt_depth=pt_max, n_min=n_min)
                found_k = decomp["num_clusters"]
                ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")

                all_rows.append(
                    {
                        "case": case_name,
                        "max_pt_depth": pt_max,
                        "n_min": n_min,
                        "true_k": true_k,
                        "found_k": found_k,
                        "ari": round(ari, 3),
                        "delta_k": found_k - (true_k or 0),
                    }
                )
        except Exception as e:
            for pt_max in pt_depths:
                all_rows.append(
                    {
                        "case": case_name,
                        "max_pt_depth": pt_max,
                        "n_min": n_min,
                        "error": str(e),
                    }
                )
    return pd.DataFrame(all_rows)


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 2: Pass-Through Depth Limit")
    print("=" * 72)

    pt_depths = [0, 1, 2, 3, 5, 999]  # 999 ≈ unlimited

    # ── Section A: PT depth alone ──
    print("\n--- Section A: PT depth limit alone (no n_min) ---")
    print("\nFailure cases:")
    df_fail = sweep_pt_depth(FAILURE_CASES, pt_depths, n_min=0)
    found_pivot = df_fail.pivot_table(
        index="case", columns="max_pt_depth", values="found_k", aggfunc="first"
    )
    print(found_pivot.to_string())

    print("\nARI:")
    ari_pivot = df_fail.pivot_table(
        index="case", columns="max_pt_depth", values="ari", aggfunc="first"
    )
    print(ari_pivot.to_string())

    print("\nRegression guard cases:")
    df_guard = sweep_pt_depth(REGRESSION_GUARD_CASES, pt_depths, n_min=0)
    found_pivot = df_guard.pivot_table(
        index="case", columns="max_pt_depth", values="found_k", aggfunc="first"
    )
    print(found_pivot.to_string())

    # ── Section B: Combined (n_min=20, PT depth) ──
    print(f"\n{'=' * 72}")
    print("  Section B: Combined (n_min=20 + PT depth limit)")
    print(f"{'=' * 72}")

    df_combined = sweep_pt_depth(FAILURE_CASES, pt_depths, n_min=20)
    found_pivot = df_combined.pivot_table(
        index="case", columns="max_pt_depth", values="found_k", aggfunc="first"
    )
    print("\nFound K:")
    print(found_pivot.to_string())

    ari_pivot = df_combined.pivot_table(
        index="case", columns="max_pt_depth", values="ari", aggfunc="first"
    )
    print("\nARI:")
    print(ari_pivot.to_string())

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print("  Summary: Mean metrics per PT depth (failure cases, n_min=20)")
    print(f"{'=' * 72}")
    summary = df_combined.groupby("max_pt_depth").agg(
        mean_ari=("ari", "mean"),
        mean_abs_delta_k=("delta_k", lambda x: abs(x).mean()),
        exact_k=("delta_k", lambda x: (x == 0).sum()),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
