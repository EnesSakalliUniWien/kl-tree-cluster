#!/usr/bin/env python3
"""Experiment 3 — Post-hoc merge safety net.

Tests the hypothesis: after the top-down decomposition, a bottom-up
merge pass can recombine spurious splits WITHOUT losing real clusters.

Algorithm:
  1. Run normal decomposition → get cluster_roots (set of node IDs).
  2. For each pair of adjacent cluster roots (sharing a parent in the tree),
     test if they are significantly different using the sibling Wald test
     with the same inflation calibration.
  3. If p_adj > alpha (siblings look the same), merge them into their
     parent node (the LCA).
  4. Repeat until no more merges are possible.

Usage:
    python debug_scripts/enhancement_lab/exp3_posthoc_merge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def _get_cluster_roots(decomp: dict, decomposer: TreeDecomposition) -> set[str]:
    """Extract the set of cluster root nodes from decomposition result."""
    roots = set()
    for cid, cinfo in decomp["cluster_assignments"].items():
        leaf_set = set(cinfo["leaves"])
        root_node = decomposer._find_cluster_root(leaf_set)
        roots.add(root_node)
    return roots


def _find_sibling_cluster_pairs(
    cluster_roots: set[str],
    tree,
) -> list[tuple[str, str, str]]:
    """Find pairs of cluster roots that are sibling children of the same parent.

    Returns list of (parent, left_root, right_root).
    """
    pairs = []
    # For each cluster root, check if its sibling is also a cluster root
    # Walk up the tree: find the lowest ancestor whose children are both
    # cluster roots or ancestors of cluster roots.

    # Simpler approach: iterate internal nodes; if both children (or
    # descendants that are cluster roots) cover exactly 2 cluster roots,
    # they're merge candidates.
    for parent in tree.nodes():
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children

        # Check if both children are cluster roots
        if left in cluster_roots and right in cluster_roots:
            pairs.append((parent, left, right))

    return pairs


def _posthoc_merge(
    tree,
    data_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    decomposer: TreeDecomposition,
    decomp: dict,
    alpha: float,
) -> tuple[dict, list[dict]]:
    """Apply bottom-up post-hoc merge.

    Returns (merged_decomp, audit_trail).
    """
    cluster_roots = _get_cluster_roots(decomp, decomposer)
    audit_trail = []
    changed = True

    while changed:
        changed = False
        pairs = _find_sibling_cluster_pairs(cluster_roots, tree)

        for parent, left, right in pairs:
            # Read the pre-computed sibling test result from the annotated stats
            if parent not in annotations_df.index:
                continue

            sib_p = (
                annotations_df.loc[parent, "Sibling_Divergence_P_Value_Corrected"]
                if "Sibling_Divergence_P_Value_Corrected" in annotations_df.columns
                else np.nan
            )

            if not np.isfinite(sib_p):
                # Can't test — skip
                audit_trail.append(
                    {
                        "parent": parent,
                        "left": left,
                        "right": right,
                        "p_value": sib_p,
                        "action": "SKIP(NaN)",
                    }
                )
                continue

            if sib_p > alpha:
                # Siblings are NOT significantly different → MERGE
                cluster_roots.discard(left)
                cluster_roots.discard(right)
                cluster_roots.add(parent)
                audit_trail.append(
                    {
                        "parent": parent,
                        "left": left,
                        "right": right,
                        "p_value": round(sib_p, 6),
                        "action": "MERGE",
                    }
                )
                changed = True
                break  # restart the loop since cluster_roots changed
            else:
                audit_trail.append(
                    {
                        "parent": parent,
                        "left": left,
                        "right": right,
                        "p_value": round(sib_p, 6),
                        "action": "KEEP",
                    }
                )

    # Rebuild cluster assignments from merged cluster roots
    desc_sets = tree.compute_descendant_sets(use_labels=True)
    final_leaf_sets = [set(desc_sets[root]) for root in cluster_roots]

    from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
        build_cluster_assignments,
    )

    cluster_assignments = build_cluster_assignments(final_leaf_sets, decomposer._find_cluster_root)
    return {
        "cluster_assignments": cluster_assignments,
        "num_clusters": len(cluster_assignments),
    }, audit_trail


def _decompose_with_posthoc(
    tree,
    data_df,
    merge_alpha: float,
    n_min: int = 0,
    max_pt_depth: int = 999,
) -> tuple[dict, dict, list[dict]]:
    """Full pipeline: decompose → post-hoc merge.

    Returns (pre_merge_decomp, post_merge_decomp, audit).
    """
    decomposer = TreeDecomposition(
        tree=tree,
        annotations_df=tree.annotations_df.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_df,
    )

    # Custom traversal with guards
    gate = decomposer._gate
    nodes_to_visit = [decomposer._root]
    final_leaf_sets: list[set[str]] = []
    processed: set[str] = set()
    pt_depth: dict[str, int] = {decomposer._root: 0}

    while nodes_to_visit:
        node_id = nodes_to_visit.pop()
        if node_id in processed:
            continue
        processed.add(node_id)

        leaf_count = len(gate._descendant_leaf_sets.get(node_id, ()))
        current_pt = pt_depth.get(node_id, 0)

        if n_min > 0 and leaf_count < n_min:
            final_leaf_sets.append(set(gate._descendant_leaf_sets[node_id]))
            continue

        if gate.should_split(node_id):
            children = gate._children_map[node_id]
            left, right = children
            nodes_to_visit.append(right)
            nodes_to_visit.append(left)
            pt_depth[left] = 0
            pt_depth[right] = 0
        elif gate.should_pass_through(node_id) and current_pt < max_pt_depth:
            children = gate._children_map[node_id]
            left, right = children
            nodes_to_visit.append(right)
            nodes_to_visit.append(left)
            pt_depth[left] = current_pt + 1
            pt_depth[right] = current_pt + 1
        else:
            final_leaf_sets.append(set(gate._descendant_leaf_sets[node_id]))

    pre_decomp = {
        "cluster_assignments": decomposer._build_cluster_assignments(final_leaf_sets),
        "num_clusters": len(final_leaf_sets),
    }

    # Post-hoc merge
    post_decomp, audit = _posthoc_merge(
        tree, data_df, decomposer.annotations_df, decomposer, pre_decomp, merge_alpha
    )

    return pre_decomp, post_decomp, audit


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 3: Post-Hoc Merge Safety Net")
    print("=" * 72)

    merge_alphas = [0.01, 0.05, 0.10, 0.20]

    all_rows = []
    for case_name in FAILURE_CASES + REGRESSION_GUARD_CASES:
        try:
            tree, data_df, y_t, tc = build_tree_and_data(case_name)
            true_k = tc.get("n_clusters", None)

            for alpha in merge_alphas:
                pre, post, audit = _decompose_with_posthoc(
                    tree,
                    data_df,
                    merge_alpha=alpha,
                    n_min=20,  # combine with best min-sample guard
                    max_pt_depth=3,  # combine with PT limit
                )
                pre_k = pre["num_clusters"]
                post_k = post["num_clusters"]
                ari_pre = compute_ari(pre, data_df, y_t) if y_t is not None else float("nan")
                ari_post = compute_ari(post, data_df, y_t) if y_t is not None else float("nan")
                n_merges = sum(1 for a in audit if a["action"] == "MERGE")

                all_rows.append(
                    {
                        "case": case_name,
                        "merge_alpha": alpha,
                        "pre_k": pre_k,
                        "post_k": post_k,
                        "true_k": true_k,
                        "ari_pre": round(ari_pre, 3),
                        "ari_post": round(ari_post, 3),
                        "n_merges": n_merges,
                        "delta_k": post_k - (true_k or 0),
                    }
                )
        except Exception as e:
            for alpha in merge_alphas:
                all_rows.append(
                    {
                        "case": case_name,
                        "merge_alpha": alpha,
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(all_rows)

    # Separate failure vs guard
    failure_set = set(FAILURE_CASES)
    df_fail = df[df["case"].isin(failure_set)]
    df_guard = df[~df["case"].isin(failure_set)]

    print("\n--- Failure cases: Post-hoc merge effect ---")
    print("\nPost-merge K by merge_alpha:")
    pivot = df_fail.pivot_table(
        index="case", columns="merge_alpha", values="post_k", aggfunc="first"
    )
    print(pivot.to_string())

    print("\nPost-merge ARI by merge_alpha:")
    pivot = df_fail.pivot_table(
        index="case", columns="merge_alpha", values="ari_post", aggfunc="first"
    )
    print(pivot.to_string())

    print("\nMerge count by merge_alpha:")
    pivot = df_fail.pivot_table(
        index="case", columns="merge_alpha", values="n_merges", aggfunc="first"
    )
    print(pivot.to_string())

    print("\n--- Regression guard cases ---")
    print("\nPost-merge K by merge_alpha:")
    pivot = df_guard.pivot_table(
        index="case", columns="merge_alpha", values="post_k", aggfunc="first"
    )
    print(pivot.to_string())

    print("\nPost-merge ARI by merge_alpha:")
    pivot = df_guard.pivot_table(
        index="case", columns="merge_alpha", values="ari_post", aggfunc="first"
    )
    print(pivot.to_string())

    # ── Audit trail for worst case ──
    worst = df_fail.loc[df_fail["delta_k"].abs().idxmax()] if len(df_fail) else None
    if worst is not None:
        case_name = worst["case"]
        alpha = worst["merge_alpha"]
        print(f"\n{'=' * 72}")
        print(f"  Audit trail: {case_name} @ merge_alpha={alpha}")
        print(f"{'=' * 72}")

        tree, data_df, y_t, tc = build_tree_and_data(case_name)
        _, _, audit = _decompose_with_posthoc(
            tree, data_df, merge_alpha=alpha, n_min=20, max_pt_depth=3
        )
        for entry in audit[:20]:
            print(f"  {entry}")
        if len(audit) > 20:
            print(f"  ... ({len(audit) - 20} more entries)")


if __name__ == "__main__":
    main()
