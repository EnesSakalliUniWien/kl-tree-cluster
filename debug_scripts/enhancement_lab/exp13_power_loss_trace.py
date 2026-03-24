#!/usr/bin/env python
"""Experiment 13 — Power Loss Trace: Where Exactly Does Each True Split Get Blocked?

For each failure case, traces every internal node in the tree and classifies it:
  - Is there a true cluster boundary here? (children have disjoint label sets)
  - If yes, WHERE does the pipeline block it?
    * Gate 1: non-binary → not tested
    * Gate 2: neither child edge-significant → blocked by edge test
    * Gate 2 partial: only ONE child significant → passes Gate 2, but weaker
    * Gate 3 skipped: sibling test skipped → blocked
    * Gate 3 BH-Same: siblings not significantly different after BH → blocked
    * Gate 3 BH-Different: siblings pass! → should split (POWER OK)
  - For blocked-at-Gate-3 nodes: what's the raw p-value? k? n_parent?

Also analyses the TREE STRUCTURE quality:
  - Does the tree even HAVE a node that cleanly separates the clusters?
  - Or are clusters interleaved across multiple subtrees?
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab_helpers import FAILURE_CASES, REGRESSION_GUARD_CASES, build_tree_and_data


from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)

ALL_CASES = FAILURE_CASES + REGRESSION_GUARD_CASES


def get_leaves(tree, node):
    """Return set of leaf labels under `node`."""
    if tree.nodes[node].get("is_leaf", False):
        return {tree.nodes[node].get("label", node)}
    return {
        tree.nodes[n].get("label", n)
        for n in nx.descendants(tree, node)
        if tree.nodes[n].get("is_leaf", False)
    }


def compute_purity(label_set_left, label_set_right, leaf_to_label):
    """Return the per-side label purity: max fraction of a single label."""
    # Not useful here — we want label overlap instead
    pass


def analyze_case(case_name: str) -> dict:
    """Full power-loss trace for one case."""
    tree, data_df, y_true, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", tc.get("true_k", "?"))
    n_samples = len(data_df)

    # Build leaf → label mapping
    leaf_to_label = {}
    if y_true is not None:
        for i, leaf_name in enumerate(data_df.index):
            leaf_to_label[leaf_name] = int(y_true[i])

    # Run full gate pipeline
    assert tree.annotations_df is not None
    annotations_df = tree.annotations_df.copy()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_df,
        spectral_method=config.SPECTRAL_METHOD,
        sibling_method=config.SIBLING_TEST_METHOD,
    )
    ann = bundle.annotated_df

    # ─── Tree structure quality ───
    # For each internal node, compute label distribution of left/right children
    root = [n for n in tree.nodes if tree.in_degree(n) == 0][0]

    # Find the "ideal" splits: nodes where children have fully disjoint label sets
    # And "boundary" nodes: where the split separates at least some labels
    node_analysis = []

    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) == 0:
            continue  # leaf — skip

        is_binary = len(children) == 2

        # Skip non-binary for detailed analysis
        if not is_binary:
            node_analysis.append(
                {
                    "node": node,
                    "gate_blocked": "Gate1_nonbinary",
                    "n_children": len(children),
                }
            )
            continue

        left, right = children
        left_leaves = get_leaves(tree, left)
        right_leaves = get_leaves(tree, right)

        # Label sets
        left_labels = {leaf_to_label.get(lf, -1) for lf in left_leaves}
        right_labels = {leaf_to_label.get(lf, -1) for lf in right_leaves}
        shared_labels = left_labels & right_labels
        is_true_split = len(shared_labels) == 0  # No label overlap

        # Label distribution — more granular
        left_label_counts = defaultdict(int)
        for lf in left_leaves:
            left_label_counts[leaf_to_label.get(lf, -1)] += 1
        right_label_counts = defaultdict(int)
        for lf in right_leaves:
            right_label_counts[leaf_to_label.get(lf, -1)] += 1

        # Purity: fraction of dominant label
        left_total = sum(left_label_counts.values())
        right_total = sum(right_label_counts.values())
        left_purity = max(left_label_counts.values()) / left_total if left_total > 0 else 0
        right_purity = max(right_label_counts.values()) / right_total if right_total > 0 else 0

        # Label overlap ratio: for boundary nodes with shared labels,
        # what fraction of leaves are "misplaced"?
        if shared_labels:
            # Count leaves whose label appears on BOTH sides
            misplaced = 0
            for lbl in shared_labels:
                misplaced += min(left_label_counts[lbl], right_label_counts[lbl])
            overlap_ratio = misplaced / (left_total + right_total)
        else:
            overlap_ratio = 0.0

        # ─── Gate 2 check ───
        left_sig = (
            bool(ann.loc[left, "Child_Parent_Divergence_Significant"])
            if left in ann.index
            else False
        )
        right_sig = (
            bool(ann.loc[right, "Child_Parent_Divergence_Significant"])
            if right in ann.index
            else False
        )
        left_p_bh = (
            float(ann.loc[left, "Child_Parent_Divergence_P_Value_BH"]) if left in ann.index else 1.0
        )
        right_p_bh = (
            float(ann.loc[right, "Child_Parent_Divergence_P_Value_BH"])
            if right in ann.index
            else 1.0
        )
        left_k = (
            float(ann.loc[left, "Child_Parent_Divergence_df"])
            if left in ann.index and "Child_Parent_Divergence_df" in ann.columns
            else 0
        )
        right_k = (
            float(ann.loc[right, "Child_Parent_Divergence_df"])
            if right in ann.index and "Child_Parent_Divergence_df" in ann.columns
            else 0
        )

        either_sig = left_sig or right_sig

        # ─── Gate 3 check ───
        sib_skipped = (
            bool(ann.loc[node, "Sibling_Divergence_Skipped"])
            if node in ann.index and "Sibling_Divergence_Skipped" in ann.columns
            else True
        )
        sib_different = (
            bool(ann.loc[node, "Sibling_BH_Different"])
            if node in ann.index and "Sibling_BH_Different" in ann.columns
            else False
        )
        sib_p_raw = (
            float(ann.loc[node, "Sibling_Divergence_P_Value"])
            if node in ann.index and "Sibling_Divergence_P_Value" in ann.columns
            else float("nan")
        )
        sib_p_corr = (
            float(ann.loc[node, "Sibling_Divergence_P_Value_Corrected"])
            if node in ann.index and "Sibling_Divergence_P_Value_Corrected" in ann.columns
            else float("nan")
        )
        sib_stat = (
            float(ann.loc[node, "Sibling_Test_Statistic"])
            if node in ann.index and "Sibling_Test_Statistic" in ann.columns
            else float("nan")
        )
        sib_df = (
            float(ann.loc[node, "Sibling_Degrees_of_Freedom"])
            if node in ann.index and "Sibling_Degrees_of_Freedom" in ann.columns
            else float("nan")
        )

        # ─── Classify gate blockage ───
        if not either_sig:
            gate_blocked = "Gate2_neither_sig"
        elif sib_skipped:
            gate_blocked = "Gate3_skipped"
        elif not sib_different:
            gate_blocked = "Gate3_BH_Same"
        else:
            gate_blocked = "PASSES_ALL"  # Would split

        node_analysis.append(
            {
                "node": node,
                "is_true_split": is_true_split,
                "n_shared_labels": len(shared_labels),
                "left_purity": left_purity,
                "right_purity": right_purity,
                "overlap_ratio": overlap_ratio,
                "n_left": left_total,
                "n_right": right_total,
                "n_parent": left_total + right_total,
                "left_labels": dict(left_label_counts),
                "right_labels": dict(right_label_counts),
                "gate_blocked": gate_blocked,
                "left_edge_sig": left_sig,
                "right_edge_sig": right_sig,
                "left_edge_p_bh": left_p_bh,
                "right_edge_p_bh": right_p_bh,
                "left_edge_k": left_k,
                "right_edge_k": right_k,
                "sib_skipped": sib_skipped,
                "sib_different": sib_different,
                "sib_p_raw": sib_p_raw,
                "sib_p_corr": sib_p_corr,
                "sib_stat": sib_stat,
                "sib_df": sib_df,
            }
        )

    return {
        "case": case_name,
        "true_k": true_k,
        "n_samples": n_samples,
        "node_analysis": node_analysis,
    }


def print_case_report(result: dict):
    """Print detailed report for one case."""
    case = result["case"]
    true_k = result["true_k"]
    nodes = result["node_analysis"]

    # Separate true-split nodes from non-true-split
    true_split_nodes = [n for n in nodes if n.get("is_true_split") is True]
    false_split_nodes = [n for n in nodes if n.get("is_true_split") is False]
    nonbinary_nodes = [n for n in nodes if n.get("gate_blocked") == "Gate1_nonbinary"]

    # Count gate blocks for true-split nodes
    true_split_gate_counts = defaultdict(int)
    for n in true_split_nodes:
        true_split_gate_counts[n["gate_blocked"]] += 1

    # Count gate blocks for all nodes
    all_gate_counts = defaultdict(int)
    for n in nodes:
        all_gate_counts[n["gate_blocked"]] += 1

    print(f"\n{'='*90}")
    print(f"  {case}  (true_k={true_k}, n={result['n_samples']})")
    print(f"{'='*90}")

    # ─── Overall gate statistics ───
    n_binary_internal = len(nodes) - len(nonbinary_nodes)
    print(
        f"\n  Internal nodes: {len(nodes)} total, {n_binary_internal} binary, {len(nonbinary_nodes)} non-binary"
    )
    print(
        f"  True-split boundary nodes: {len(true_split_nodes)} (children have disjoint label sets)"
    )
    print(f"  False-split nodes: {len(false_split_nodes)} (children share ≥1 label)")

    print("\n  Gate block distribution (ALL internal nodes):")
    for gate, count in sorted(all_gate_counts.items()):
        pct = 100 * count / len(nodes) if nodes else 0
        print(f"    {gate:>25s}: {count:>5d}  ({pct:.1f}%)")

    if true_split_nodes:
        print(
            f"\n  Gate block distribution (TRUE SPLIT nodes only — {len(true_split_nodes)} nodes):"
        )
        for gate, count in sorted(true_split_gate_counts.items()):
            pct = 100 * count / len(true_split_nodes) if true_split_nodes else 0
            print(f"    {gate:>25s}: {count:>5d}  ({pct:.1f}%)")

    # ─── Tree structure quality: do we even HAVE clean splits? ───
    # Count high-purity boundaries (both sides >80% pure)
    high_purity_splits = [n for n in true_split_nodes]
    moderate_boundary = [
        n
        for n in false_split_nodes
        if n.get("left_purity", 0) > 0.7 and n.get("right_purity", 0) > 0.7
    ]
    messy_boundary = [
        n
        for n in false_split_nodes
        if n.get("left_purity", 0) <= 0.7 or n.get("right_purity", 0) <= 0.7
    ]

    print("\n  Tree structure quality:")
    print(f"    Clean cluster boundaries (disjoint labels): {len(high_purity_splits)}")
    print(f"    Moderate boundaries (shared labels, >70% pure): {len(moderate_boundary)}")
    print(f"    Messy boundaries (≤70% pure on one side): {len(messy_boundary)}")

    # ─── Detailed trace of true-split nodes ───
    if true_split_nodes:
        print("\n  Detailed trace of TRUE SPLIT nodes (sorted by n_parent desc):")
        print(
            f"  {'node':>10s}  {'n_par':>5s}  {'gate_blocked':>20s}  "
            f"{'L_sig':>5s}  {'R_sig':>5s}  {'L_p_bh':>8s}  {'R_p_bh':>8s}  "
            f"{'sib_p_raw':>9s}  {'sib_p_bh':>9s}  {'sib_k':>5s}  {'L_k':>4s}  {'R_k':>4s}  "
            f"{'L_pur':>5s}  {'R_pur':>5s}  {'nL':>4s}  {'nR':>4s}"
        )
        print("  " + "-" * 140)
        for n in sorted(true_split_nodes, key=lambda x: x.get("n_parent", 0), reverse=True):
            print(
                f"  {n['node']:>10s}  {n.get('n_parent',0):>5d}  {n['gate_blocked']:>20s}  "
                f"{'T' if n.get('left_edge_sig') else 'F':>5s}  "
                f"{'T' if n.get('right_edge_sig') else 'F':>5s}  "
                f"{n.get('left_edge_p_bh', float('nan')):>8.4f}  "
                f"{n.get('right_edge_p_bh', float('nan')):>8.4f}  "
                f"{n.get('sib_p_raw', float('nan')):>9.4f}  "
                f"{n.get('sib_p_corr', float('nan')):>9.4f}  "
                f"{n.get('sib_df', float('nan')):>5.0f}  "
                f"{n.get('left_edge_k', float('nan')):>4.0f}  "
                f"{n.get('right_edge_k', float('nan')):>4.0f}  "
                f"{n.get('left_purity', 0):>5.2f}  "
                f"{n.get('right_purity', 0):>5.2f}  "
                f"{n.get('n_left', 0):>4d}  "
                f"{n.get('n_right', 0):>4d}"
            )

    # ─── Adjacent-to-true-splits: partial boundaries (shared labels but high purity) ───
    if moderate_boundary:
        print("\n  Top moderate boundary nodes (shared labels, >70% pure, top 10):")
        print(
            f"  {'node':>10s}  {'n_par':>5s}  {'gate_blocked':>20s}  "
            f"{'L_pur':>5s}  {'R_pur':>5s}  {'overlap':>7s}  {'shared':>6s}  "
            f"{'sib_p_raw':>9s}  {'sib_k':>5s}  {'nL':>4s}  {'nR':>4s}"
        )
        print("  " + "-" * 100)
        for n in sorted(moderate_boundary, key=lambda x: x.get("n_parent", 0), reverse=True)[:10]:
            print(
                f"  {n['node']:>10s}  {n.get('n_parent',0):>5d}  {n['gate_blocked']:>20s}  "
                f"{n.get('left_purity', 0):>5.2f}  "
                f"{n.get('right_purity', 0):>5.2f}  "
                f"{n.get('overlap_ratio', 0):>7.3f}  "
                f"{n.get('n_shared_labels', 0):>6d}  "
                f"{n.get('sib_p_raw', float('nan')):>9.4f}  "
                f"{n.get('sib_df', float('nan')):>5.0f}  "
                f"{n.get('n_left', 0):>4d}  "
                f"{n.get('n_right', 0):>4d}"
            )

    # ─── Where does the pipeline ACTUALLY split (false positives)? ───
    false_positive_splits = [n for n in false_split_nodes if n.get("gate_blocked") == "PASSES_ALL"]
    if false_positive_splits:
        print(
            f"\n  FALSE POSITIVE splits (gate passes but labels overlap): {len(false_positive_splits)}"
        )
        for n in sorted(false_positive_splits, key=lambda x: x.get("n_parent", 0), reverse=True)[
            :5
        ]:
            print(
                f"    {n['node']:>10s}  n_par={n.get('n_parent',0)}  "
                f"L_pur={n.get('left_purity', 0):.2f}  R_pur={n.get('right_purity', 0):.2f}  "
                f"overlap={n.get('overlap_ratio', 0):.3f}  "
                f"sib_p={n.get('sib_p_raw', float('nan')):.4f}  "
                f"sib_k={n.get('sib_df', float('nan')):.0f}  "
                f"L: {n.get('left_labels', {})}  R: {n.get('right_labels', {})}"
            )

    # ─── Summary: blocking stage heat map ───
    # For true-split nodes, break down by sample size
    if true_split_nodes:
        print("\n  True-split nodes by sample size and gate block:")
        size_bins = [
            (0, 10, "n ≤ 10"),
            (10, 30, "10 < n ≤ 30"),
            (30, 100, "30 < n ≤ 100"),
            (100, 10000, "n > 100"),
        ]
        for lo, hi, label in size_bins:
            subset = [n for n in true_split_nodes if lo < n.get("n_parent", 0) <= hi]
            if not subset:
                continue
            gate_counts = defaultdict(int)
            for n in subset:
                gate_counts[n["gate_blocked"]] += 1
            parts = [f"{g}: {c}" for g, c in sorted(gate_counts.items())]
            print(f"    {label:>15s} ({len(subset):>3d} nodes): {', '.join(parts)}")


def main():
    header = "=" * 90
    print(header)
    print("  EXPERIMENT 13: Power Loss Trace — Where Does Each True Split Get Blocked?")
    print(header)

    all_results = []
    for case in ALL_CASES:
        print(f"\n  Processing {case} ...", end=" ", flush=True)
        try:
            result = analyze_case(case)
            all_results.append(result)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    for result in all_results:
        print_case_report(result)

    # ─── Grand summary across all failure cases ───
    print(f"\n\n{'='*90}")
    print("  GRAND SUMMARY: Where Power Is Lost (Failure Cases Only)")
    print(f"{'='*90}")

    grand_true_split_gates = defaultdict(int)
    grand_false_split_gates = defaultdict(int)
    grand_true_split_total = 0
    grand_all_total = 0

    for result in all_results:
        if result["case"] not in FAILURE_CASES:
            continue
        for n in result["node_analysis"]:
            grand_all_total += 1
            if n.get("is_true_split") is True:
                grand_true_split_total += 1
                grand_true_split_gates[n["gate_blocked"]] += 1
            elif n.get("is_true_split") is False:
                grand_false_split_gates[n["gate_blocked"]] += 1

    print(f"\n  Total internal nodes (failure cases): {grand_all_total}")
    print(f"  True-split boundary nodes: {grand_true_split_total}")
    print("\n  Where true splits are blocked:")
    for gate, count in sorted(grand_true_split_gates.items(), key=lambda x: -x[1]):
        pct = 100 * count / grand_true_split_total if grand_true_split_total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {gate:>25s}: {count:>5d}  ({pct:5.1f}%)  {bar}")

    print("\n  Where false splits pass (over-splitting):")
    for gate, count in sorted(grand_false_split_gates.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(sum(grand_false_split_gates.values()), 1)
        print(f"    {gate:>25s}: {count:>5d}  ({pct:5.1f}%)")

    # ─── Median p-value for blocked true splits ───
    g2_blocked_pvals = []
    g3_blocked_pvals = []
    g3_blocked_ks = []
    g2_blocked_sizes = []
    g3_blocked_sizes = []

    for result in all_results:
        if result["case"] not in FAILURE_CASES:
            continue
        for n in result["node_analysis"]:
            if n.get("is_true_split") is not True:
                continue
            if n["gate_blocked"] == "Gate2_neither_sig":
                g2_blocked_pvals.append(
                    min(n.get("left_edge_p_bh", 1), n.get("right_edge_p_bh", 1))
                )
                g2_blocked_sizes.append(n.get("n_parent", 0))
            elif n["gate_blocked"] in ("Gate3_BH_Same", "Gate3_skipped"):
                g3_blocked_pvals.append(n.get("sib_p_raw", float("nan")))
                g3_blocked_ks.append(n.get("sib_df", float("nan")))
                g3_blocked_sizes.append(n.get("n_parent", 0))

    if g2_blocked_pvals:
        valid_p = [p for p in g2_blocked_pvals if np.isfinite(p)]
        valid_s = [s for s in g2_blocked_sizes if s > 0]
        print(f"\n  Gate 2 blocked true splits ({len(g2_blocked_pvals)} nodes):")
        if valid_p:
            print(f"    Best (min) edge p-value BH: {min(valid_p):.4f}")
            print(f"    Median edge p-value BH:     {np.median(valid_p):.4f}")
            print(f"    Mean edge p-value BH:       {np.mean(valid_p):.4f}")
        if valid_s:
            print(f"    Median n_parent:            {np.median(valid_s):.0f}")
            print(f"    Range n_parent:             {min(valid_s)} - {max(valid_s)}")

    if g3_blocked_pvals:
        valid_p = [p for p in g3_blocked_pvals if np.isfinite(p)]
        valid_k = [k for k in g3_blocked_ks if np.isfinite(k) and k > 0]
        valid_s = [s for s in g3_blocked_sizes if s > 0]
        print(f"\n  Gate 3 blocked true splits ({len(g3_blocked_pvals)} nodes):")
        if valid_p:
            print(f"    Best (min) raw sibling p:   {min(valid_p):.6f}")
            print(f"    Median raw sibling p:       {np.median(valid_p):.4f}")
            print(f"    Frac with raw p < 0.05:     {np.mean(np.array(valid_p) < 0.05):.1%}")
            print(f"    Frac with raw p < 0.10:     {np.mean(np.array(valid_p) < 0.10):.1%}")
        if valid_k:
            print(f"    Median sibling k:           {np.median(valid_k):.0f}")
            print(
                f"    k distribution:             {np.unique(np.round(valid_k), return_counts=True)}"
            )
        if valid_s:
            print(f"    Median n_parent:            {np.median(valid_s):.0f}")
            print(f"    Range n_parent:             {min(valid_s)} - {max(valid_s)}")

    print(f"\n{'='*90}")
    print("  Done.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
