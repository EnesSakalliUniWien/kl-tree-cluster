#!/usr/bin/env python3
"""Experiment 1 — Minimum sample size guard.

Tests the hypothesis: adding a minimum leaf-count threshold to the
traversal prevents runaway splitting in overlapping data WITHOUT
regressing on clean cases.

We monkey-patch the GateEvaluator to add a leaf-count floor,
sweep n_min ∈ {10, 20, 30, 40, 50}, and measure K_found + ARI.

Usage:
    python debug_scripts/enhancement_lab/exp1_min_samples.py
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
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.traversal import (
    iterate_worklist,
    process_node,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition


def _decompose_with_min_samples(
    tree,
    data_df,
    n_min: int,
    alpha_local: float | None = None,
    sibling_alpha: float | None = None,
) -> dict:
    """Run TreeDecomposition but inject a min-sample guard into process_node."""

    # Build the decomposer normally (runs full annotation pipeline)
    decomposer = TreeDecomposition(
        tree=tree,
        annotations_df=tree.annotations_df.copy(),
        alpha_local=alpha_local or config.EDGE_ALPHA,
        sibling_alpha=sibling_alpha or config.SIBLING_ALPHA,
        leaf_data=data_df,
    )

    # Custom traversal with min-sample guard
    gate = decomposer._gate
    nodes_to_visit = [decomposer._root]
    final_leaf_sets: list[set[str]] = []
    processed: set[str] = set()

    for node in iterate_worklist(nodes_to_visit, processed):
        # GUARD: If this node has fewer leaves than n_min, force MERGE
        leaf_count = len(gate._descendant_leaf_sets.get(node, ()))
        if leaf_count < n_min:
            final_leaf_sets.append(set(gate._descendant_leaf_sets[node]))
            continue
        # Otherwise, use the standard gate logic
        process_node(node, gate, nodes_to_visit, final_leaf_sets)

    cluster_assignments = decomposer._build_cluster_assignments(final_leaf_sets)
    return {
        "cluster_assignments": cluster_assignments,
        "num_clusters": len(cluster_assignments),
    }


def sweep_min_samples(
    cases: list[str],
    n_min_values: list[int],
    label_prefix: str = "",
) -> pd.DataFrame:
    """Sweep n_min across a set of cases. Returns summary table."""
    all_rows = []
    for case_name in cases:
        try:
            tree, data_df, y_t, tc = build_tree_and_data(case_name)
            true_k = tc.get("n_clusters", None)

            for n_min in n_min_values:
                decomp = _decompose_with_min_samples(tree, data_df, n_min=n_min)
                found_k = decomp["num_clusters"]
                ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")

                all_rows.append(
                    {
                        "case": case_name,
                        "n_min": n_min,
                        "true_k": true_k,
                        "found_k": found_k,
                        "ari": round(ari, 3),
                        "delta_k": found_k - (true_k or 0),
                    }
                )
        except Exception as e:
            for n_min in n_min_values:
                all_rows.append(
                    {
                        "case": case_name,
                        "n_min": n_min,
                        "error": str(e),
                    }
                )
    return pd.DataFrame(all_rows)


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 1: Minimum Sample Size Guard")
    print("=" * 72)

    n_min_values = [10, 20, 30, 40, 50]

    # ── Failure cases sweep ──
    print("\n--- Failure cases ---")
    df_fail = sweep_min_samples(FAILURE_CASES, n_min_values)
    if "error" not in df_fail.columns:
        # Pivot for readability
        pivot = df_fail.pivot_table(
            index="case", columns="n_min", values=["found_k", "ari"], aggfunc="first"
        )
        # Add true_k column
        true_k_map = df_fail.groupby("case")["true_k"].first()
        print(f"\nTrue K values: {true_k_map.to_dict()}")
        print("\nFound K by n_min:")
        found_pivot = df_fail.pivot_table(
            index="case", columns="n_min", values="found_k", aggfunc="first"
        )
        print(found_pivot.to_string())
        print("\nARI by n_min:")
        ari_pivot = df_fail.pivot_table(
            index="case", columns="n_min", values="ari", aggfunc="first"
        )
        print(ari_pivot.to_string())
    else:
        print(df_fail.to_string(index=False))

    # ── Regression guard cases ──
    print("\n--- Regression guard cases ---")
    df_guard = sweep_min_samples(REGRESSION_GUARD_CASES, n_min_values)
    if "error" not in df_guard.columns:
        print("\nFound K by n_min:")
        found_pivot = df_guard.pivot_table(
            index="case", columns="n_min", values="found_k", aggfunc="first"
        )
        print(found_pivot.to_string())
        print("\nARI by n_min:")
        ari_pivot = df_guard.pivot_table(
            index="case", columns="n_min", values="ari", aggfunc="first"
        )
        print(ari_pivot.to_string())
    else:
        print(df_guard.to_string(index=False))

    # ── Best n_min recommendation ──
    if "ari" in df_fail.columns and df_fail["ari"].notna().any():
        print("\n--- Summary: Mean ARI across failure cases per n_min ---")
        summary = df_fail.groupby("n_min").agg(
            mean_ari=("ari", "mean"),
            exact_k=("delta_k", lambda x: (x == 0).sum()),
            mean_delta_k=("delta_k", lambda x: abs(x).mean()),
        )
        print(summary.to_string())

        if "ari" in df_guard.columns and df_guard["ari"].notna().any():
            print("\n--- Summary: Mean ARI across regression guard cases per n_min ---")
            guard_summary = df_guard.groupby("n_min").agg(
                mean_ari=("ari", "mean"),
                exact_k=("delta_k", lambda x: (x == 0).sum()),
            )
            print(guard_summary.to_string())


if __name__ == "__main__":
    main()
