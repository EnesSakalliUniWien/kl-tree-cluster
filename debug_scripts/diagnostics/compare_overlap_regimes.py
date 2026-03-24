#!/usr/bin/env python3
"""Compare binary and Gaussian overlap benchmark regimes.

This diagnostic script has two goals:
1. Compare representative binary-overlap and Gaussian-overlap cases side by side.
2. Trace one pathological overlap case from true labels to tree structure to
   Gate 2 and Gate 3 blockage.

Usage:
    python debug_scripts/diagnostics/compare_overlap_regimes.py
    python debug_scripts/diagnostics/compare_overlap_regimes.py --trace-case overlap_hd_4c_1k
    python debug_scripts/diagnostics/compare_overlap_regimes.py --binary-cases overlap_heavy_4c_small_feat overlap_mod_4c_small
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KL_TE_N_JOBS", "1")

from debug_scripts.enhancement_lab.lab_helpers import (  # noqa: E402
    build_tree_and_data,
    compute_ari,
    run_decomposition,
)
from kl_clustering_analysis import config  # noqa: E402
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (  # noqa: E402
    run_gate_annotation_pipeline,
)

DEFAULT_BINARY_CASES: tuple[str, ...] = (
    "overlap_heavy_4c_small_feat",
    "overlap_mod_4c_small",
    "overlap_hd_4c_1k",
    "overlap_unbal_4c_small",
)

DEFAULT_GAUSSIAN_CASES: tuple[str, ...] = (
    "gauss_overlap_3c_small",
    "gauss_overlap_4c_med",
    "gauss_overlap_3c_small_q4",
)

DEFAULT_TRACE_CASE = "overlap_heavy_4c_small_feat"


@dataclass(frozen=True)
class NodeTrace:
    node: str
    gate_blocked: str
    n_parent: int
    n_left: int
    n_right: int
    is_true_split: bool
    left_purity: float
    right_purity: float
    overlap_ratio: float
    left_edge_sig: bool
    right_edge_sig: bool
    left_edge_p_bh: float
    right_edge_p_bh: float
    left_edge_k: float
    right_edge_k: float
    sib_skipped: bool
    sib_different: bool
    sib_p_raw: float
    sib_p_corr: float
    sib_df: float


@dataclass(frozen=True)
class CaseSummary:
    case_name: str
    regime: str
    true_k: int | None
    found_k: int
    ari: float
    n_binary_internal: int
    n_true_split: int
    largest_true_split_parent: int
    median_true_split_parent: float
    gate2_blocks: int
    gate3_blocks: int
    passes_all: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-cases", nargs="+", default=list(DEFAULT_BINARY_CASES))
    parser.add_argument("--gaussian-cases", nargs="+", default=list(DEFAULT_GAUSSIAN_CASES))
    parser.add_argument("--trace-case", default=DEFAULT_TRACE_CASE)
    parser.add_argument("--trace-limit", type=int, default=12)
    return parser.parse_args()


def _safe_float(df: pd.DataFrame, index: str, column: str, default: float) -> float:
    if index not in df.index or column not in df.columns:
        return default
    value = df.loc[index, column]
    if pd.isna(value):
        return default
    return float(value)


def _safe_bool(df: pd.DataFrame, index: str, column: str, default: bool) -> bool:
    if index not in df.index or column not in df.columns:
        return default
    value = df.loc[index, column]
    if pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(value)


def _leaf_labels(tree: nx.DiGraph, node: str) -> set[str]:
    if tree.nodes[node].get("is_leaf", False):
        return {tree.nodes[node].get("label", node)}
    return {
        tree.nodes[desc].get("label", desc)
        for desc in nx.descendants(tree, node)
        if tree.nodes[desc].get("is_leaf", False)
    }


def _node_label_counts(
    tree: nx.DiGraph,
    node: str,
    leaf_to_label: dict[str, int],
) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for leaf in _leaf_labels(tree, node):
        counts[leaf_to_label[leaf]] += 1
    return dict(counts)


def _purity(label_counts: dict[int, int]) -> float:
    total = sum(label_counts.values())
    if total <= 0:
        return 0.0
    return max(label_counts.values()) / total


def _overlap_ratio(left_counts: dict[int, int], right_counts: dict[int, int]) -> float:
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    shared_labels = set(left_counts) & set(right_counts)
    if left_total + right_total <= 0 or not shared_labels:
        return 0.0
    misplaced = sum(min(left_counts[label], right_counts[label]) for label in shared_labels)
    return misplaced / (left_total + right_total)


def _classify_gate_block(
    left_sig: bool,
    right_sig: bool,
    sib_skipped: bool,
    sib_different: bool,
) -> str:
    if not (left_sig or right_sig):
        return "Gate2_neither_sig"
    if sib_skipped:
        return "Gate3_skipped"
    if not sib_different:
        return "Gate3_BH_Same"
    return "PASSES_ALL"


def analyze_case(case_name: str, regime: str) -> tuple[CaseSummary, list[NodeTrace]]:
    tree, data_df, y_true, test_case = build_tree_and_data(case_name)
    if y_true is None:
        raise ValueError(f"Case {case_name!r} has no ground-truth labels.")

    leaf_to_label = {leaf_name: int(y_true[idx]) for idx, leaf_name in enumerate(data_df.index)}

    if tree.annotations_df is None:
        raise ValueError(f"Tree for case {case_name!r} has no initial annotations_df.")

    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=data_df,
        spectral_method=config.SPECTRAL_METHOD,
        sibling_method=config.SIBLING_TEST_METHOD,
    )
    annotated_df = gate_bundle.annotated_df

    decomp = run_decomposition(tree, data_df)
    ari = compute_ari(decomp, data_df, y_true)

    traces: list[NodeTrace] = []
    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) != 2:
            continue

        left, right = children
        left_counts = _node_label_counts(tree, left, leaf_to_label)
        right_counts = _node_label_counts(tree, right, leaf_to_label)
        is_true_split = len(set(left_counts) & set(right_counts)) == 0

        left_sig = _safe_bool(annotated_df, left, "Child_Parent_Divergence_Significant", False)
        right_sig = _safe_bool(annotated_df, right, "Child_Parent_Divergence_Significant", False)
        sib_skipped = _safe_bool(annotated_df, node, "Sibling_Divergence_Skipped", True)
        sib_different = _safe_bool(annotated_df, node, "Sibling_BH_Different", False)

        traces.append(
            NodeTrace(
                node=node,
                gate_blocked=_classify_gate_block(left_sig, right_sig, sib_skipped, sib_different),
                n_parent=sum(left_counts.values()) + sum(right_counts.values()),
                n_left=sum(left_counts.values()),
                n_right=sum(right_counts.values()),
                is_true_split=is_true_split,
                left_purity=_purity(left_counts),
                right_purity=_purity(right_counts),
                overlap_ratio=_overlap_ratio(left_counts, right_counts),
                left_edge_sig=left_sig,
                right_edge_sig=right_sig,
                left_edge_p_bh=_safe_float(
                    annotated_df, left, "Child_Parent_Divergence_P_Value_BH", 1.0
                ),
                right_edge_p_bh=_safe_float(
                    annotated_df, right, "Child_Parent_Divergence_P_Value_BH", 1.0
                ),
                left_edge_k=_safe_float(annotated_df, left, "Child_Parent_Divergence_df", 0.0),
                right_edge_k=_safe_float(annotated_df, right, "Child_Parent_Divergence_df", 0.0),
                sib_skipped=sib_skipped,
                sib_different=sib_different,
                sib_p_raw=_safe_float(
                    annotated_df, node, "Sibling_Divergence_P_Value", float("nan")
                ),
                sib_p_corr=_safe_float(
                    annotated_df,
                    node,
                    "Sibling_Divergence_P_Value_Corrected",
                    float("nan"),
                ),
                sib_df=_safe_float(annotated_df, node, "Sibling_Degrees_of_Freedom", 0.0),
            )
        )

    true_split_nodes = [trace for trace in traces if trace.is_true_split]
    parent_sizes = [trace.n_parent for trace in true_split_nodes]

    summary = CaseSummary(
        case_name=case_name,
        regime=regime,
        true_k=test_case.get("n_clusters", test_case.get("true_k")),
        found_k=decomp["num_clusters"],
        ari=float(ari),
        n_binary_internal=len(traces),
        n_true_split=len(true_split_nodes),
        largest_true_split_parent=max(parent_sizes) if parent_sizes else 0,
        median_true_split_parent=float(median(parent_sizes)) if parent_sizes else 0.0,
        gate2_blocks=sum(
            1 for trace in true_split_nodes if trace.gate_blocked == "Gate2_neither_sig"
        ),
        gate3_blocks=sum(
            1
            for trace in true_split_nodes
            if trace.gate_blocked in {"Gate3_skipped", "Gate3_BH_Same"}
        ),
        passes_all=sum(1 for trace in true_split_nodes if trace.gate_blocked == "PASSES_ALL"),
    )
    return summary, traces


def _format_summary_table(summaries: list[CaseSummary]) -> str:
    frame = pd.DataFrame(
        [
            {
                "case": summary.case_name,
                "K": f"{summary.found_k}/{summary.true_k}",
                "ARI": f"{summary.ari:.3f}",
                "true_split_nodes": summary.n_true_split,
                "largest_n_parent": summary.largest_true_split_parent,
                "median_n_parent": f"{summary.median_true_split_parent:.1f}",
                "Gate2_blocks": summary.gate2_blocks,
                "Gate3_blocks": summary.gate3_blocks,
                "passes_all": summary.passes_all,
            }
            for summary in summaries
        ]
    )
    if frame.empty:
        return "<no cases>"
    return frame.to_string(index=False)


def _print_regime_snapshot(label: str, summaries: list[CaseSummary]) -> None:
    if not summaries:
        return
    aris = [summary.ari for summary in summaries]
    largest_sizes = [summary.largest_true_split_parent for summary in summaries]
    gate2 = sum(summary.gate2_blocks for summary in summaries)
    gate3 = sum(summary.gate3_blocks for summary in summaries)
    print(
        f"{label}: mean ARI={np.mean(aris):.3f}, "
        f"median largest true-split n_parent={float(np.median(largest_sizes)):.1f}, "
        f"true-split Gate2 blocks={gate2}, true-split Gate3 blocks={gate3}"
    )


def print_regime_comparison(
    binary_summaries: list[CaseSummary],
    gaussian_summaries: list[CaseSummary],
) -> None:
    print("=" * 110)
    print("BINARY OVERLAP CASES")
    print("=" * 110)
    print(_format_summary_table(binary_summaries))

    print("\n" + "=" * 110)
    print("GAUSSIAN OVERLAP CASES")
    print("=" * 110)
    print(_format_summary_table(gaussian_summaries))

    print("\n" + "-" * 110)
    _print_regime_snapshot("Binary overlap", binary_summaries)
    _print_regime_snapshot("Gaussian overlap", gaussian_summaries)


def print_trace(case_name: str, summary: CaseSummary, traces: list[NodeTrace], limit: int) -> None:
    print("\n" + "=" * 110)
    print(f"PATHOLOGICAL TRACE: {case_name}")
    print("=" * 110)
    print(
        f"K={summary.found_k}/{summary.true_k} ARI={summary.ari:.3f} | "
        f"true-split nodes={summary.n_true_split} | "
        f"largest true-split n_parent={summary.largest_true_split_parent}"
    )
    print(
        "Interpretation: Gate2_neither_sig means the edge test died first; "
        "Gate3_* means the boundary survived Gate 2 but the sibling stage failed."
    )

    true_split = [trace for trace in traces if trace.is_true_split]
    true_split.sort(key=lambda trace: trace.n_parent, reverse=True)

    print(
        f"{'node':>10s}  {'n_par':>5s}  {'gate_blocked':>18s}  "
        f"{'L_p_bh':>8s}  {'R_p_bh':>8s}  {'sib_p':>8s}  {'sib_bh':>8s}  "
        f"{'sib_k':>5s}  {'L_k':>4s}  {'R_k':>4s}"
    )
    print("-" * 110)
    for trace in true_split[:limit]:
        print(
            f"{trace.node:>10s}  {trace.n_parent:>5d}  {trace.gate_blocked:>18s}  "
            f"{trace.left_edge_p_bh:>8.4f}  {trace.right_edge_p_bh:>8.4f}  "
            f"{trace.sib_p_raw:>8.4f}  {trace.sib_p_corr:>8.4f}  {trace.sib_df:>5.0f}  "
            f"{trace.left_edge_k:>4.0f}  {trace.right_edge_k:>4.0f}"
        )

    gate_counter = Counter(trace.gate_blocked for trace in true_split)
    print("\nTrue-split blockage counts:")
    for gate_name, count in sorted(gate_counter.items()):
        print(f"  {gate_name:>18s}: {count}")


def main() -> None:
    args = parse_args()

    binary_results = [analyze_case(case_name, regime="binary") for case_name in args.binary_cases]
    gaussian_results = [
        analyze_case(case_name, regime="gaussian") for case_name in args.gaussian_cases
    ]

    binary_summaries = [summary for summary, _ in binary_results]
    gaussian_summaries = [summary for summary, _ in gaussian_results]

    print(
        f"Config: SPECTRAL_METHOD={config.SPECTRAL_METHOD}, "
        f"SIBLING_METHOD={config.SIBLING_TEST_METHOD}, "
        f"TREE_DISTANCE={config.TREE_DISTANCE_METRIC}, "
        f"TREE_LINKAGE={config.TREE_LINKAGE_METHOD}"
    )
    print_regime_comparison(binary_summaries, gaussian_summaries)

    trace_lookup: dict[str, tuple[CaseSummary, list[NodeTrace]]] = {
        summary.case_name: (summary, traces)
        for summary, traces in binary_results + gaussian_results
    }
    if args.trace_case not in trace_lookup:
        trace_lookup[args.trace_case] = analyze_case(args.trace_case, regime="trace")

    trace_summary, trace_rows = trace_lookup[args.trace_case]
    print_trace(args.trace_case, trace_summary, trace_rows, limit=args.trace_limit)


if __name__ == "__main__":
    main()
