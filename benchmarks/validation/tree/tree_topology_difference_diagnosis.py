#!/usr/bin/env python3
"""Diagnose why topology cells differ and what null model changes follow.

This module is evidence-only. It reads the BranchArchitect tree comparison,
fail-closed p-value audit, alpha sweep, and literature replay artifacts. It does
not change production topology selection, alpha defaults, traversal, or fallback
behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "tree_topology_difference_diagnosis/v1"
GENERATED_BY = "benchmarks.validation.tree.tree_topology_difference_diagnosis"

DEFAULT_TREE_DIR = Path("reports/tree_consensus_brancharchitect_tree_comparison_20260709")
DEFAULT_PVALUE_DIR = Path("reports/tree_consensus_fail_closed_pvalues_20260709")
DEFAULT_ALPHA_DIR = Path("reports/tree_consensus_alpha_sensitivity_20260709")
DEFAULT_LITERATURE_DIR = Path("reports/tree_consensus_literature_policy_replay_20260709")
DEFAULT_OUTPUT_DIR = Path("reports/tree_consensus_topology_difference_diagnosis_20260709")

METHOD_SUMMARY_NAME = "topology_difference_method_summary.csv"
PAIR_FAMILY_SUMMARY_NAME = "topology_difference_pair_family_summary.csv"
SUBTREE_SUPPORT_NAME = "topology_difference_subtree_support.csv"
SUBTREE_HIGHLIGHTS_NAME = "topology_difference_subtree_highlights.csv"
CASE_DIAGNOSIS_NAME = "topology_difference_case_diagnosis.csv"
NULL_RECOMMENDATIONS_NAME = "topology_difference_null_recommendations.csv"
REPORT_NAME = "topology_difference_diagnosis_report.md"
MANIFEST_NAME = "topology_difference_diagnosis_manifest.json"

EDGE_ALPHA = 0.001
SIBLING_ALPHA = 0.01


@dataclass(frozen=True)
class DiagnosisInputs:
    """Resolved artifact paths for topology-difference diagnosis."""

    tree_cells: Path
    tree_pairwise: Path
    tree_newick: Path
    loss_taxonomy: Path
    pvalue_cells: Path
    traversal_trace: Path
    alpha_best_by_case: Path | None = None
    literature_case_summary: Path | None = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required diagnosis input does not exist: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _nanmin(values: Sequence[object]) -> float:
    finite = [_finite(value) for value in values if math.isfinite(_finite(value))]
    return min(finite) if finite else math.nan


def _nanmax(values: Sequence[object]) -> float:
    finite = [_finite(value) for value in values if math.isfinite(_finite(value))]
    return max(finite) if finite else math.nan


def _ratio(max_value: float, min_value: float) -> float:
    if not math.isfinite(max_value) or not math.isfinite(min_value) or min_value <= 0.0:
        return math.nan
    return max_value / min_value


def _bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    return series.map(
        lambda value: (
            bool(value)
            if isinstance(value, bool)
            else str(value).strip().lower() in {"true", "1", "yes", "y"}
        )
    )


def _format_float(value: object, digits: int = 4) -> str:
    number = _finite(value)
    if not math.isfinite(number):
        return ""
    if number != 0.0 and abs(number) < 10 ** (-(digits - 1)):
        return f"{number:.2e}"
    return f"{number:.{digits}g}"


@dataclass(frozen=True)
class ParsedNewickNode:
    """Minimal parsed Newick node used for split-support diagnostics."""

    leaves: frozenset[str]
    branch_length: float
    children: tuple["ParsedNewickNode", ...] = ()


class _NewickParser:
    """Small Newick parser for the branch-length exports produced by this repo."""

    def __init__(self, text: str) -> None:
        self.text = str(text).strip()
        self.index = 0
        sys.setrecursionlimit(max(sys.getrecursionlimit(), self.text.count("(") * 4 + 1000))

    def parse(self) -> ParsedNewickNode:
        node = self._parse_node()
        self._skip_ws()
        if self._peek() == ";":
            self.index += 1
        self._skip_ws()
        if self.index != len(self.text):
            raise ValueError(f"Unexpected Newick suffix at byte {self.index}.")
        return node

    def _peek(self) -> str:
        return self.text[self.index] if self.index < len(self.text) else ""

    def _skip_ws(self) -> None:
        while self._peek().isspace():
            self.index += 1

    def _parse_node(self) -> ParsedNewickNode:
        self._skip_ws()
        children: tuple[ParsedNewickNode, ...] = ()
        if self._peek() == "(":
            self.index += 1
            parsed_children = [self._parse_node()]
            while True:
                self._skip_ws()
                char = self._peek()
                if char == ",":
                    self.index += 1
                    parsed_children.append(self._parse_node())
                    continue
                if char == ")":
                    self.index += 1
                    break
                raise ValueError(f"Expected ',' or ')' in Newick at byte {self.index}.")
            self._parse_label(optional=True)
            branch_length = self._parse_branch_length()
            leaves = frozenset().union(*(child.leaves for child in parsed_children))
            children = tuple(parsed_children)
        else:
            label = self._parse_label(optional=False)
            if not label:
                raise ValueError(f"Expected Newick leaf label at byte {self.index}.")
            branch_length = self._parse_branch_length()
            leaves = frozenset({label})
        return ParsedNewickNode(
            leaves=frozenset(str(leaf) for leaf in leaves),
            branch_length=branch_length,
            children=children,
        )

    def _parse_label(self, *, optional: bool) -> str:
        self._skip_ws()
        char = self._peek()
        if char in {":", ",", ")", ";", ""}:
            if optional:
                return ""
            raise ValueError(f"Expected Newick label at byte {self.index}.")
        if char == "'":
            self.index += 1
            parts: list[str] = []
            while self.index < len(self.text):
                current = self.text[self.index]
                self.index += 1
                if current == "\\" and self.index < len(self.text):
                    parts.append(self.text[self.index])
                    self.index += 1
                    continue
                if current == "'":
                    return "".join(parts)
                parts.append(current)
            raise ValueError("Unterminated quoted Newick label.")

        start = self.index
        while self._peek() not in {":", ",", ")", ";", ""}:
            self.index += 1
        return self.text[start : self.index].strip()

    def _parse_branch_length(self) -> float:
        self._skip_ws()
        if self._peek() != ":":
            return 0.0
        self.index += 1
        start = self.index
        while self._peek() not in {",", ")", ";", ""}:
            self.index += 1
        raw = self.text[start : self.index].strip()
        if not raw:
            return 0.0
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid Newick branch length {raw!r}.") from exc
        return value if math.isfinite(value) and value >= 0.0 else 0.0


def _newick_internal_split_lengths(
    newick: str,
) -> tuple[frozenset[str], dict[frozenset[str], float]]:
    root = _NewickParser(newick).parse()
    n_leaves = len(root.leaves)
    splits: dict[frozenset[str], float] = {}

    def visit(node: ParsedNewickNode, *, is_root: bool) -> None:
        if not is_root and 1 < len(node.leaves) < n_leaves:
            splits[node.leaves] = splits.get(node.leaves, 0.0) + float(node.branch_length)
        for child in node.children:
            visit(child, is_root=False)

    visit(root, is_root=True)
    return root.leaves, splits


def _split_id(leaves: frozenset[str]) -> str:
    raw = "\0".join(sorted(leaves))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _leaf_preview(leaves: Sequence[str] | frozenset[str], *, limit: int = 12) -> str:
    ordered = sorted(str(leaf) for leaf in leaves)
    if len(ordered) <= limit:
        return ",".join(ordered)
    return ",".join(ordered[:limit]) + f",...(+{len(ordered) - limit})"


def _subtree_contrast_type(
    *,
    supporting_methods: set[str],
    absent_methods: set[str],
    support_count: int,
    method_count: int,
) -> str:
    nonmonotone = {"centroid", "median"}
    stable_core = {"average", "weighted", "complete", "ward"}
    if supporting_methods and supporting_methods <= nonmonotone:
        return "nonmonotone_specific_subtree"
    if absent_methods and absent_methods <= nonmonotone and supporting_methods & stable_core:
        return "stable_core_subtree_missing_in_nonmonotone"
    if supporting_methods == {"neighbor_joining"}:
        return "neighbor_joining_specific_subtree"
    if absent_methods == {"neighbor_joining"}:
        return "linkage_subtree_missing_in_neighbor_joining"
    if support_count == 1:
        return "single_method_specific_subtree"
    if support_count == method_count - 1:
        return "one_method_missing_subtree"
    return "mixed_topology_subtree"


def default_inputs(
    *,
    tree_dir: Path = DEFAULT_TREE_DIR,
    pvalue_dir: Path = DEFAULT_PVALUE_DIR,
    alpha_dir: Path = DEFAULT_ALPHA_DIR,
    literature_dir: Path = DEFAULT_LITERATURE_DIR,
) -> DiagnosisInputs:
    """Return default artifact paths used by the 2026-07-09 diagnosis."""

    return DiagnosisInputs(
        tree_cells=tree_dir / "brancharchitect_tree_cells.csv",
        tree_pairwise=tree_dir / "brancharchitect_tree_pairwise.csv",
        tree_newick=tree_dir / "brancharchitect_tree_newick.csv",
        loss_taxonomy=pvalue_dir / "fail_closed_loss_taxonomy.csv",
        pvalue_cells=pvalue_dir / "fail_closed_pvalue_cells.csv",
        traversal_trace=pvalue_dir / "fail_closed_traversal_trace.csv",
        alpha_best_by_case=alpha_dir / "sibling_alpha_sweep_best_by_case.csv",
        literature_case_summary=literature_dir / "literature_policy_replay_case_summary.csv",
    )


def load_diagnosis_frames(inputs: DiagnosisInputs) -> dict[str, pd.DataFrame]:
    """Load diagnosis frames from artifact CSVs."""

    return {
        "tree_cells": _read_csv(inputs.tree_cells),
        "tree_pairwise": _read_csv(inputs.tree_pairwise),
        "tree_newick": _read_csv(inputs.tree_newick),
        "loss_taxonomy": _read_csv(inputs.loss_taxonomy),
        "pvalue_cells": _read_csv(inputs.pvalue_cells),
        "traversal_trace": _read_csv(inputs.traversal_trace),
        "alpha_best_by_case": _read_csv(inputs.alpha_best_by_case, required=False)
        if inputs.alpha_best_by_case is not None
        else pd.DataFrame(),
        "literature_case_summary": _read_csv(inputs.literature_case_summary, required=False)
        if inputs.literature_case_summary is not None
        else pd.DataFrame(),
    }


def build_pair_family_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize topology/path divergence for each unordered method pair."""

    if pairwise.empty:
        return pd.DataFrame()
    frame = pairwise.copy()
    frame["method_pair"] = frame.apply(
        lambda row: " vs ".join(
            sorted([str(row["left_tree_inference"]), str(row["right_tree_inference"])])
        ),
        axis=1,
    )
    summary = (
        frame.groupby("method_pair", dropna=False)
        .agg(
            schema_version=("schema_version", "first"),
            comparisons=("case_id", "count"),
            median_rooted_internal_rf_relative=("rooted_internal_rf_relative", "median"),
            max_rooted_internal_rf_relative=("rooted_internal_rf_relative", "max"),
            median_rooted_weighted_split_l1=("rooted_weighted_split_l1", "median"),
            max_rooted_weighted_split_l1=("rooted_weighted_split_l1", "max"),
            median_leaf_path_rmse=("leaf_path_rmse", "median"),
            max_leaf_path_rmse=("leaf_path_rmse", "max"),
            min_predicted_label_ari_between_topologies=(
                "predicted_label_ari_between_topologies",
                "min",
            ),
        )
        .reset_index()
    )
    summary.insert(0, "diagnosis_schema_version", SCHEMA_VERSION)
    return summary.sort_values(
        ["median_rooted_internal_rf_relative", "median_leaf_path_rmse", "method_pair"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_subtree_support(tree_newick: pd.DataFrame) -> pd.DataFrame:
    """Summarize rooted subtree support across topology methods per case."""

    if tree_newick.empty:
        return pd.DataFrame()
    required = {"case_id", "tree_inference", "newick"}
    missing = sorted(required - set(tree_newick.columns))
    if missing:
        raise ValueError(f"tree_newick frame is missing required columns: {missing!r}")

    rows: list[dict[str, object]] = []
    for case_id, case_frame in tree_newick.groupby("case_id", dropna=False):
        method_names = tuple(sorted(case_frame["tree_inference"].astype(str).unique()))
        method_count = len(method_names)
        split_methods: dict[frozenset[str], set[str]] = {}
        split_lengths: dict[frozenset[str], list[float]] = {}
        split_leaves: dict[frozenset[str], frozenset[str]] = {}
        all_leaves: frozenset[str] = frozenset()
        case_category = str(case_frame["case_category"].dropna().iloc[0])
        test_case = int(case_frame["test_case"].dropna().iloc[0])

        for row in case_frame.itertuples(index=False):
            method = str(row.tree_inference)
            leaf_universe, lengths = _newick_internal_split_lengths(str(row.newick))
            all_leaves = all_leaves | leaf_universe
            for leaves, branch_length in lengths.items():
                split_methods.setdefault(leaves, set()).add(method)
                split_lengths.setdefault(leaves, []).append(float(branch_length))
                split_leaves[leaves] = leaves

        n_case_leaves = len(all_leaves)
        for leaves, supporting in split_methods.items():
            support_count = len(supporting)
            absent = set(method_names) - supporting
            complement = all_leaves - leaves
            leaf_count = len(leaves)
            complement_count = len(complement)
            smaller_side = min(leaf_count, complement_count)
            lengths = split_lengths[leaves]
            is_unstable = 0 < support_count < method_count
            rows.append(
                {
                    "diagnosis_schema_version": SCHEMA_VERSION,
                    "case_id": str(case_id),
                    "test_case": test_case,
                    "case_category": case_category,
                    "rooted_subtree_id": _split_id(leaves),
                    "subtree_leaf_count": int(leaf_count),
                    "complement_leaf_count": int(complement_count),
                    "smaller_side_leaf_count": int(smaller_side),
                    "smaller_side_fraction": (
                        float(smaller_side) / float(n_case_leaves)
                        if n_case_leaves > 0
                        else math.nan
                    ),
                    "support_count": int(support_count),
                    "method_count": int(method_count),
                    "support_fraction": (
                        float(support_count) / float(method_count) if method_count > 0 else math.nan
                    ),
                    "is_unstable": bool(is_unstable),
                    "supporting_methods": ",".join(sorted(supporting)),
                    "absent_methods": ",".join(sorted(absent)),
                    "contrast_type": _subtree_contrast_type(
                        supporting_methods=supporting,
                        absent_methods=absent,
                        support_count=support_count,
                        method_count=method_count,
                    ),
                    "median_supporting_branch_length": float(np.median(lengths)),
                    "min_supporting_branch_length": float(np.min(lengths)),
                    "max_supporting_branch_length": float(np.max(lengths)),
                    "subtree_leaf_preview": _leaf_preview(split_leaves[leaves]),
                    "complement_leaf_preview": _leaf_preview(complement),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["case_id", "is_unstable", "smaller_side_leaf_count", "support_count"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )


def build_subtree_highlights(
    subtree_support: pd.DataFrame,
    *,
    max_subtrees_per_case: int = 12,
) -> pd.DataFrame:
    """Return the highest-impact unstable subtrees for each case."""

    if subtree_support.empty:
        return pd.DataFrame()
    unstable = subtree_support[subtree_support["is_unstable"].astype(bool)].copy()
    if unstable.empty:
        return pd.DataFrame(columns=list(subtree_support.columns))
    unstable["absent_count"] = unstable["method_count"] - unstable["support_count"]
    unstable["subtree_difference_score"] = (
        unstable[["support_count", "absent_count"]].min(axis=1)
        * unstable["smaller_side_leaf_count"]
    )
    unstable = unstable.sort_values(
        [
            "case_id",
            "subtree_difference_score",
            "smaller_side_leaf_count",
            "median_supporting_branch_length",
            "rooted_subtree_id",
        ],
        ascending=[True, False, False, False, True],
    )
    unstable["subtree_difference_rank"] = unstable.groupby("case_id").cumcount().astype(int) + 1
    return unstable[unstable["subtree_difference_rank"] <= int(max_subtrees_per_case)].reset_index(
        drop=True
    )


def _method_centrality(pairwise: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(
        set(pairwise.get("left_tree_inference", pd.Series(dtype=str)).astype(str))
        | set(pairwise.get("right_tree_inference", pd.Series(dtype=str)).astype(str))
    )
    rows: list[dict[str, object]] = []
    for method in methods:
        subset = pairwise[
            (pairwise["left_tree_inference"].astype(str) == method)
            | (pairwise["right_tree_inference"].astype(str) == method)
        ]
        rows.append(
            {
                "tree_inference": method,
                "pairwise_comparisons": int(len(subset)),
                "median_rf_to_other_methods": subset["rooted_internal_rf_relative"].median(),
                "mean_rf_to_other_methods": subset["rooted_internal_rf_relative"].mean(),
                "median_weighted_split_l1_to_other_methods": subset[
                    "rooted_weighted_split_l1"
                ].median(),
                "median_leaf_path_rmse_to_other_methods": subset["leaf_path_rmse"].median(),
            }
        )
    return pd.DataFrame(rows)


def _method_role(method: str, median_rf: float) -> str:
    if method == "neighbor_joining":
        return "additive_distance_mad_rooting_outlier"
    if method in {"centroid", "median"} and math.isfinite(median_rf) and median_rf >= 0.75:
        return "nonmonotone_linkage_topology_outlier"
    if math.isfinite(median_rf) and median_rf >= 0.5:
        return "high_topology_divergence"
    return "stable_linkage_core"


def build_method_summary(
    tree_cells: pd.DataFrame,
    pvalue_cells: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize tree-shape, branch-time, and p-value behavior by topology method."""

    cells = tree_cells.copy()
    pvalues = pvalue_cells.copy()
    ok_cells = cells[cells["status"].astype(str).eq("ok")].copy()
    ok_pvalues = pvalues[pvalues["status"].astype(str).eq("ok")].copy()
    centrality = _method_centrality(pairwise)

    cell_summary = (
        ok_cells.groupby("tree_inference", dropna=False)
        .agg(
            ok_cells=("case_id", "count"),
            median_internal_split_count=("internal_split_count", "median"),
            median_total_branch_length=("total_branch_length", "median"),
            median_mean_branch_length=("mean_branch_length", "median"),
            median_max_branch_length=("max_branch_length", "median"),
            median_found_clusters=("found_clusters", "median"),
            median_largest_cluster_fraction=("largest_cluster_fraction", "median"),
        )
        .reset_index()
    )
    skipped = (
        cells[~cells["status"].astype(str).eq("ok")]
        .groupby("tree_inference", dropna=False)
        .size()
        .rename("skipped_cells")
        .reset_index()
    )
    pvalue_summary = (
        ok_pvalues.groupby("tree_inference", dropna=False)
        .agg(
            edge_open_cells=("edge_gate_open", lambda values: int(_bool_series(values).sum())),
            sibling_open_cells=(
                "sibling_gate_open",
                lambda values: int(_bool_series(values).sum()),
            ),
            min_root_edge_bh_min=("min_root_edge_p_value_bh", "min"),
            min_root_edge_bh_median=("min_root_edge_p_value_bh", "median"),
            active_sibling_corrected_min=("sibling_p_value_corrected", "min"),
            active_sibling_corrected_median=("sibling_p_value_corrected", "median"),
            median_left_root_branch_length=("left_branch_length", "median"),
            median_right_root_branch_length=("right_branch_length", "median"),
        )
        .reset_index()
    )
    summary = (
        cell_summary.merge(skipped, on="tree_inference", how="left")
        .merge(pvalue_summary, on="tree_inference", how="left")
        .merge(centrality, on="tree_inference", how="left")
    )
    summary["skipped_cells"] = summary["skipped_cells"].fillna(0).astype(int)
    summary.insert(0, "diagnosis_schema_version", SCHEMA_VERSION)
    summary["method_role"] = [
        _method_role(str(row.tree_inference), _finite(row.median_rf_to_other_methods))
        for row in summary.itertuples()
    ]
    return summary.sort_values(
        ["median_rf_to_other_methods", "tree_inference"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _trace_summary(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame(columns=["case_id"])
    full = trace[
        trace["trace_type"].astype(str).eq("full_edge_traversal_trace")
        & _bool_series(trace["actual_visited"])
    ].copy()
    if full.empty:
        return pd.DataFrame(columns=["case_id"])
    full["branch_sum"] = pd.to_numeric(full["left_branch_length"], errors="coerce") + pd.to_numeric(
        full["right_branch_length"], errors="coerce"
    )
    return (
        full.groupby("case_id", dropna=False)
        .agg(
            visited_full_edge_nodes=("node_id", "count"),
            full_edge_open_nodes=("edge_gate_open", lambda values: int(_bool_series(values).sum())),
            full_active_sibling_finite=("sibling_p_value_corrected", "count"),
            full_min_active_sibling_corrected=("sibling_p_value_corrected", "min"),
            full_min_sparse_p_value=("sibling_sparse_p_value", "min"),
            full_min_dense_p_value=("sibling_dense_p_value", "min"),
            full_min_fixed_coordinate_bh_p_value=(
                "sibling_fixed_coordinate_bh_p_value",
                "min",
            ),
            full_min_fixed_global_p_value=("sibling_fixed_global_p_value", "min"),
            full_min_branch_length_sum=("branch_sum", "min"),
            full_median_branch_length_sum=("branch_sum", "median"),
        )
        .reset_index()
    )


def _case_tree_summary(tree_cells: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    ok_cells = tree_cells[tree_cells["status"].astype(str).eq("ok")].copy()
    cell_summary = (
        ok_cells.groupby("case_id", dropna=False)
        .agg(
            ok_tree_cells=("tree_inference", "count"),
            successful_topologies=(
                "tree_inference",
                lambda values: ",".join(sorted(map(str, values))),
            ),
            tree_found_clusters_min=("found_clusters", "min"),
            tree_found_clusters_max=("found_clusters", "max"),
            median_total_branch_length=("total_branch_length", "median"),
            total_branch_length_min=("total_branch_length", "min"),
            total_branch_length_max=("total_branch_length", "max"),
            median_mean_branch_length=("mean_branch_length", "median"),
            median_max_branch_length=("max_branch_length", "median"),
        )
        .reset_index()
    )
    skipped = (
        tree_cells[~tree_cells["status"].astype(str).eq("ok")]
        .groupby("case_id", dropna=False)
        .agg(
            skipped_tree_cells=("tree_inference", "count"),
            skipped_topologies_tree_comparison=(
                "tree_inference",
                lambda values: ",".join(sorted(map(str, values))),
            ),
            tree_comparison_skip_reasons=(
                "skip_reason",
                lambda values: " | ".join(sorted({str(value) for value in values if str(value)})),
            ),
        )
        .reset_index()
    )
    pair_summary = (
        pairwise.groupby("case_id", dropna=False)
        .agg(
            pairwise_comparisons=("case_id", "count"),
            median_rooted_internal_rf_relative=("rooted_internal_rf_relative", "median"),
            max_rooted_internal_rf_relative=("rooted_internal_rf_relative", "max"),
            median_rooted_weighted_split_l1=("rooted_weighted_split_l1", "median"),
            max_rooted_weighted_split_l1=("rooted_weighted_split_l1", "max"),
            median_leaf_path_rmse=("leaf_path_rmse", "median"),
            max_leaf_path_rmse=("leaf_path_rmse", "max"),
            min_predicted_label_ari_between_topologies=(
                "predicted_label_ari_between_topologies",
                "min",
            ),
        )
        .reset_index()
    )
    return cell_summary.merge(skipped, on="case_id", how="left").merge(
        pair_summary,
        on="case_id",
        how="left",
    )


def _case_pvalue_summary(pvalue_cells: pd.DataFrame) -> pd.DataFrame:
    ok = pvalue_cells[pvalue_cells["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(columns=["case_id"])
    ok["root_branch_min_cell"] = ok[["left_branch_length", "right_branch_length"]].min(axis=1)
    ok["root_branch_max_cell"] = ok[["left_branch_length", "right_branch_length"]].max(axis=1)
    return (
        ok.groupby("case_id", dropna=False)
        .agg(
            root_edge_open_cells=("edge_gate_open", lambda values: int(_bool_series(values).sum())),
            root_sibling_open_cells=(
                "sibling_gate_open",
                lambda values: int(_bool_series(values).sum()),
            ),
            root_edge_bh_min=("min_root_edge_p_value_bh", "min"),
            root_edge_bh_median=("min_root_edge_p_value_bh", "median"),
            root_edge_bh_max=("min_root_edge_p_value_bh", "max"),
            root_active_sibling_corrected_min=("sibling_p_value_corrected", "min"),
            root_active_sibling_corrected_median=("sibling_p_value_corrected", "median"),
            root_sparse_p_min=("sibling_sparse_p_value", "min"),
            root_dense_p_min=("sibling_dense_p_value", "min"),
            root_fixed_coordinate_bh_min=("sibling_fixed_coordinate_bh_p_value", "min"),
            root_fixed_global_p_min=("sibling_fixed_global_p_value", "min"),
            root_branch_length_min=("root_branch_min_cell", "min"),
            root_branch_length_max=("root_branch_max_cell", "max"),
            root_branch_length_median=("left_branch_length", "median"),
        )
        .reset_index()
    )


def _alpha_summary(alpha_best_by_case: pd.DataFrame) -> pd.DataFrame:
    if alpha_best_by_case.empty:
        return pd.DataFrame(columns=["case_id"])
    return alpha_best_by_case.rename(
        columns={
            "sibling_alpha": "best_sweep_sibling_alpha",
            "best_ari": "best_sweep_ari",
            "best_found_clusters": "best_sweep_found_clusters",
            "min_largest_cluster_fraction": "best_sweep_min_largest_cluster_fraction",
        }
    )[
        [
            "case_id",
            "best_sweep_sibling_alpha",
            "best_sweep_ari",
            "best_sweep_found_clusters",
            "best_sweep_min_largest_cluster_fraction",
        ]
    ]


def _literature_adaptive_summary(literature_case_summary: pd.DataFrame) -> pd.DataFrame:
    if literature_case_summary.empty:
        return pd.DataFrame(columns=["case_id"])
    adaptive = literature_case_summary[
        literature_case_summary["policy_id"].astype(str).eq("trace_adaptive_alpha_spending_proxy")
    ].copy()
    if adaptive.empty:
        return pd.DataFrame(columns=["case_id"])
    return adaptive[
        [
            "case_id",
            "split_cells",
            "max_cluster_count",
            "min_largest_cluster_fraction",
            "max_local_alpha",
            "min_rejected_p_value",
            "rescue_status",
        ]
    ].rename(
        columns={
            "split_cells": "adaptive_proxy_split_cells",
            "max_cluster_count": "adaptive_proxy_max_cluster_count",
            "min_largest_cluster_fraction": "adaptive_proxy_min_largest_cluster_fraction",
            "max_local_alpha": "adaptive_proxy_max_local_alpha",
            "min_rejected_p_value": "adaptive_proxy_min_rejected_p_value",
            "rescue_status": "adaptive_proxy_rescue_status",
        }
    )


def _subtree_case_summary(subtree_support: pd.DataFrame) -> pd.DataFrame:
    if subtree_support is None or subtree_support.empty:
        return pd.DataFrame(columns=["case_id"])
    unstable = subtree_support[subtree_support["is_unstable"].astype(bool)].copy()
    if unstable.empty:
        return pd.DataFrame(columns=["case_id"])
    large = unstable[unstable["smaller_side_fraction"] >= 0.05]
    summary = (
        unstable.groupby("case_id", dropna=False)
        .agg(
            unstable_subtree_count=("rooted_subtree_id", "count"),
            max_unstable_subtree_smaller_side_leaf_count=(
                "smaller_side_leaf_count",
                "max",
            ),
            median_unstable_subtree_smaller_side_leaf_count=(
                "smaller_side_leaf_count",
                "median",
            ),
            unstable_subtree_contrast_types=(
                "contrast_type",
                lambda values: ",".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    large_summary = (
        large.groupby("case_id", dropna=False)
        .size()
        .rename("large_unstable_subtree_count")
        .reset_index()
    )
    return summary.merge(large_summary, on="case_id", how="left")


def _top_subtree_summary(subtree_highlights: pd.DataFrame) -> pd.DataFrame:
    if subtree_highlights is None or subtree_highlights.empty:
        return pd.DataFrame(columns=["case_id"])
    top = subtree_highlights[subtree_highlights["subtree_difference_rank"] == 1].copy()
    if top.empty:
        return pd.DataFrame(columns=["case_id"])
    return top[
        [
            "case_id",
            "rooted_subtree_id",
            "smaller_side_leaf_count",
            "smaller_side_fraction",
            "supporting_methods",
            "absent_methods",
            "contrast_type",
            "subtree_leaf_preview",
            "complement_leaf_preview",
        ]
    ].rename(
        columns={
            "rooted_subtree_id": "top_unstable_subtree_id",
            "smaller_side_leaf_count": "top_unstable_subtree_smaller_side_leaf_count",
            "smaller_side_fraction": "top_unstable_subtree_smaller_side_fraction",
            "supporting_methods": "top_unstable_subtree_supporting_methods",
            "absent_methods": "top_unstable_subtree_absent_methods",
            "contrast_type": "top_unstable_subtree_contrast_type",
            "subtree_leaf_preview": "top_unstable_subtree_leaf_preview",
            "complement_leaf_preview": "top_unstable_subtree_complement_preview",
        }
    )


def _topology_driver(row: Mapping[str, object]) -> str:
    max_rf = _finite(row.get("max_rooted_internal_rf_relative"))
    root_ratio = _finite(row.get("root_branch_length_ratio"))
    skip_text = str(row.get("tree_comparison_skip_reasons", ""))
    if math.isfinite(max_rf) and max_rf >= 0.9 and math.isfinite(root_ratio) and root_ratio >= 10.0:
        return "nonmonotone_or_selected_root_branch_time_instability"
    if math.isfinite(max_rf) and max_rf >= 0.9:
        return "nonmonotone_linkage_topology_instability"
    if "MAD rooting requires positive distances" in skip_text:
        return "nj_mad_positive_distance_precondition"
    if math.isfinite(max_rf) and max_rf >= 0.65:
        return "linkage_update_rule_topology_instability"
    if math.isfinite(root_ratio) and root_ratio >= 10.0:
        return "branch_time_instability_on_similar_rooted_splits"
    return "moderate_topology_variation"


def _null_implication(row: Mapping[str, object]) -> str:
    loss_bucket = str(row.get("loss_bucket", ""))
    sibling_min = _finite(row.get("root_active_sibling_corrected_min"))
    diagnostic_min = _nanmin(
        [
            row.get("root_sparse_p_min"),
            row.get("root_dense_p_min"),
            row.get("root_fixed_coordinate_bh_min"),
            row.get("root_fixed_global_p_min"),
            row.get("full_min_sparse_p_value"),
            row.get("full_min_dense_p_value"),
            row.get("full_min_fixed_coordinate_bh_p_value"),
            row.get("full_min_fixed_global_p_value"),
        ]
    )
    if loss_bucket == "edge_gate_closed_global":
        return (
            "selected_tree_edge_null_first; sibling_null_is_not_reached_until "
            "edge_support_is_calibrated_for_selected_topology_and_nnls_time"
        )
    if math.isfinite(sibling_min) and sibling_min <= 0.025:
        return (
            "near_threshold_selected_sibling_null; calibrate local selected-node "
            "p_values and alpha spending with topology-stability guard"
        )
    if math.isfinite(diagnostic_min) and diagnostic_min <= 1e-3:
        return (
            "diagnostic_signal_suppressed_by_active_empirical_null; replace blunt "
            "alpha_relaxation with branch_length_and_topology_conditioned sibling null"
        )
    return (
        "active_sibling_support_weak; require selected_hierarchy null calibration "
        "before any adaptive alpha opening"
    )


def _development_action(row: Mapping[str, object]) -> str:
    loss_bucket = str(row.get("loss_bucket", ""))
    topology_driver = str(row.get("topology_difference_driver", ""))
    if loss_bucket == "edge_gate_closed_global":
        return (
            "Estimate an edge null over the full selected pipeline "
            "(adaptive KNN diffusion, topology method, NNLS branch lengths, selected root)."
        )
    if "topology" in topology_driver:
        return (
            "Estimate sibling p-values conditional on selected topology/branch lengths and "
            "spend alpha only across topology-stable candidate splits."
        )
    return (
        "Keep fail-closed until active sibling p-values, diagnostic p-values, and "
        "cluster-structure guards agree under a selected-hierarchy null."
    )


def build_case_diagnosis(
    tree_cells: pd.DataFrame,
    pairwise: pd.DataFrame,
    loss_taxonomy: pd.DataFrame,
    pvalue_cells: pd.DataFrame,
    traversal_trace: pd.DataFrame,
    alpha_best_by_case: pd.DataFrame,
    literature_case_summary: pd.DataFrame,
    subtree_support: pd.DataFrame | None = None,
    subtree_highlights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-case mechanistic diagnosis rows."""

    diagnosis = (
        loss_taxonomy.merge(_case_tree_summary(tree_cells, pairwise), on="case_id", how="left")
        .merge(_case_pvalue_summary(pvalue_cells), on="case_id", how="left")
        .merge(_trace_summary(traversal_trace), on="case_id", how="left")
        .merge(_alpha_summary(alpha_best_by_case), on="case_id", how="left")
        .merge(_literature_adaptive_summary(literature_case_summary), on="case_id", how="left")
        .merge(_subtree_case_summary(subtree_support), on="case_id", how="left")
        .merge(_top_subtree_summary(subtree_highlights), on="case_id", how="left")
    )
    diagnosis["skipped_tree_cells"] = diagnosis["skipped_tree_cells"].fillna(0).astype(int)
    for column in ["unstable_subtree_count", "large_unstable_subtree_count"]:
        if column in diagnosis.columns:
            diagnosis[column] = diagnosis[column].fillna(0).astype(int)
    diagnosis["root_branch_length_ratio"] = [
        _ratio(_finite(row.root_branch_length_max), _finite(row.root_branch_length_min))
        for row in diagnosis.itertuples()
    ]
    diagnosis["root_edge_p_value_ratio"] = [
        _ratio(_finite(row.root_edge_bh_max), _finite(row.root_edge_bh_min))
        for row in diagnosis.itertuples()
    ]
    diagnosis["minimum_diagnostic_sibling_p_value"] = [
        _nanmin(
            [
                row.root_sparse_p_min,
                row.root_dense_p_min,
                row.root_fixed_coordinate_bh_min,
                row.root_fixed_global_p_min,
                row.full_min_sparse_p_value,
                row.full_min_dense_p_value,
                row.full_min_fixed_coordinate_bh_p_value,
                row.full_min_fixed_global_p_value,
            ]
        )
        for row in diagnosis.itertuples()
    ]
    diagnosis["active_vs_diagnostic_p_value_ratio"] = [
        _ratio(
            _finite(row.root_active_sibling_corrected_min),
            _finite(row.minimum_diagnostic_sibling_p_value),
        )
        for row in diagnosis.itertuples()
    ]
    diagnosis["topology_difference_driver"] = [
        _topology_driver(row._asdict()) for row in diagnosis.itertuples()
    ]
    diagnosis["null_hypothesis_change"] = [
        _null_implication(row._asdict()) for row in diagnosis.itertuples()
    ]
    diagnosis["development_action"] = [
        _development_action(row._asdict()) for row in diagnosis.itertuples()
    ]
    diagnosis.insert(0, "diagnosis_schema_version", SCHEMA_VERSION)
    return diagnosis.sort_values(
        ["loss_bucket", "max_rooted_internal_rf_relative", "root_branch_length_ratio"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_null_recommendations(case_diagnosis: pd.DataFrame) -> pd.DataFrame:
    """Summarize recommended null-model changes by failure family."""

    rows = [
        {
            "diagnosis_schema_version": SCHEMA_VERSION,
            "recommendation_id": "selected_tree_edge_null",
            "applies_to_loss_bucket": "edge_gate_closed_global",
            "cases": ",".join(
                case_diagnosis[
                    case_diagnosis["loss_bucket"].astype(str).eq("edge_gate_closed_global")
                ]["case_id"].astype(str)
            ),
            "current_failure": (
                "The sibling statistic is not reached because every selected topology "
                "fails the child-parent edge prerequisite."
            ),
            "required_null_hypothesis": (
                "H0_edge_selected: after adaptive KNN diffusion, topology selection, "
                "rooting, and NNLS branch-length fitting, child-parent contrasts at the "
                "selected root are exchangeable under a no-split hierarchy."
            ),
            "required_statistic_change": (
                "Calibrate edge p-values over the selected pipeline or fail closed; do "
                "not reuse fixed-tree p-values as if the root were pre-specified."
            ),
            "production_guard": (
                "Only let sibling testing run after selected-tree edge calibration "
                "opens the root at the production edge alpha."
            ),
        },
        {
            "diagnosis_schema_version": SCHEMA_VERSION,
            "recommendation_id": "branch_length_conditioned_sibling_null",
            "applies_to_loss_bucket": "sibling_gate_closed_after_edge_open",
            "cases": ",".join(
                case_diagnosis[
                    case_diagnosis["loss_bucket"]
                    .astype(str)
                    .eq("sibling_gate_closed_after_edge_open")
                ]["case_id"].astype(str)
            ),
            "current_failure": (
                "Edge support can be strong, but active sibling p-values remain above "
                "alpha while dense/sparse diagnostics often show signal."
            ),
            "required_null_hypothesis": (
                "H0_sibling_selected: conditional on the selected topology, selected "
                "node, root-to-child branch lengths, and parent feature covariance, the "
                "two child subtrees are exchangeable up to branch-time-scaled noise."
            ),
            "required_statistic_change": (
                "Use a selected-node sibling p-value calibrated with branch-length sum, "
                "local covariance, and empirical-null support; combine dense/sparse "
                "diagnostics only through a predeclared closed or union-intersection rule."
            ),
            "production_guard": (
                "Require topology stability, largest-cluster guard, cluster-count "
                "plausibility, and fragmentation guard before spending extra alpha."
            ),
        },
        {
            "diagnosis_schema_version": SCHEMA_VERSION,
            "recommendation_id": "topology_stability_alpha_spending",
            "applies_to_loss_bucket": "all_fail_closed_under_splits",
            "cases": ",".join(case_diagnosis["case_id"].astype(str)),
            "current_failure": (
                "Different topology builders often choose very different rooted splits, "
                "but the resulting labels remain identical because traversal fails closed."
            ),
            "required_null_hypothesis": (
                "H0_consensus: the selected split remains null after accounting for the "
                "family of accessible topologies and their NNLS branch lengths."
            ),
            "required_statistic_change": (
                "Treat topology as a model-selection dimension. Spend alpha on a split "
                "only when its leaf partition is stable across a predeclared topology "
                "family or calibrated by a selected-family min-p procedure."
            ),
            "production_guard": (
                "If topology agreement is low and selected-family p-values are absent, "
                "report unsupported instead of raising global alpha."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "\n"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows)
    return subset.to_markdown(index=False)


def write_report(
    *,
    output_dir: Path,
    method_summary: pd.DataFrame,
    pair_family_summary: pd.DataFrame,
    subtree_highlights: pd.DataFrame,
    case_diagnosis: pd.DataFrame,
    null_recommendations: pd.DataFrame,
    tree_pairwise: pd.DataFrame,
    inputs: DiagnosisInputs,
    generated_at: str,
) -> str:
    """Return the Markdown diagnosis report body."""

    edge_cases = int(case_diagnosis["loss_bucket"].astype(str).eq("edge_gate_closed_global").sum())
    sibling_cases = int(
        case_diagnosis["loss_bucket"].astype(str).eq("sibling_gate_closed_after_edge_open").sum()
    )
    all_one_cluster = bool(
        (case_diagnosis["tree_found_clusters_min"].fillna(1) == 1).all()
        and (case_diagnosis["tree_found_clusters_max"].fillna(1) == 1).all()
    )
    max_rf = case_diagnosis["max_rooted_internal_rf_relative"].max()
    overall_pairwise_median_rf = tree_pairwise["rooted_internal_rf_relative"].median()
    median_case_median_rf = case_diagnosis["median_rooted_internal_rf_relative"].median()
    max_root_ratio = case_diagnosis["root_branch_length_ratio"].replace(np.inf, np.nan).max()
    method_roles = method_summary[["tree_inference", "method_role"]].copy()

    lines = [
        "# Tree Topology Difference Diagnosis",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- generated_by: `{GENERATED_BY}`",
        f"- generated_at_utc: `{generated_at}`",
        f"- git_revision: `{_git_revision()}`",
        f"- output_dir: `{output_dir}`",
        "",
        "## Executive Finding",
        "",
        (
            "The current fail-closed evidence separates two effects. The accessible "
            "topology builders can produce very different rooted split sets and NNLS "
            "branch-time assignments, but those differences do not currently change "
            "the cluster structure because every successful cell still returns one "
            "cluster."
        ),
        "",
        (
            f"Across all pairwise topology-cell comparisons, the median rooted RF "
            f"difference is `{_format_float(overall_pairwise_median_rf)}`; the "
            f"median of per-case medians is `{_format_float(median_case_median_rf)}` "
            f"because the stable linkage core often shares topology. The maximum "
            f"rooted RF difference is `{_format_float(max_rf)}`. The largest root "
            f"branch-length ratio across topology cells is "
            f"`{_format_float(max_root_ratio)}`. "
            f"All successful cells one-cluster: `{all_one_cluster}`."
        ),
        "",
        (
            f"The failure split is `{edge_cases}` edge-gate cases and `{sibling_cases}` "
            "sibling-gate-after-edge cases. This means the next method change should "
            "not be a global alpha increase or another selector tie-breaker. It should "
            "be a selected-topology, branch-length-conditioned null model for the edge "
            "and sibling statistics."
        ),
        "",
        "## What Makes The Trees Different",
        "",
        (
            "Topology is inferred before NNLS. The linkage or neighbor-joining step "
            "chooses the tree; fixed-topology NNLS then fits nonnegative branch "
            "lengths to pairwise squared standardized distances on that already "
            "selected topology. Therefore NNLS explains branch-time and path-distance "
            "changes, but not topology changes."
        ),
        "",
        (
            "The method-family pattern is clear. Average, weighted, complete, single, "
            "and ward form the stable linkage core in many cases. Centroid and median "
            "are the major nonmonotone-linkage outliers. Neighbor joining is a separate "
            "additive-distance/MAD-rooted outlier and is also limited by positive "
            "distance preconditions."
        ),
        "",
        _markdown_table(
            method_roles,
            ["tree_inference", "method_role"],
        ),
        "",
        "The largest method-pair differences are:",
        "",
        _markdown_table(
            pair_family_summary,
            [
                "method_pair",
                "comparisons",
                "median_rooted_internal_rf_relative",
                "median_rooted_weighted_split_l1",
                "median_leaf_path_rmse",
                "min_predicted_label_ari_between_topologies",
            ],
            max_rows=12,
        ),
        "",
        "## Why Topology Changes Do Not Rescue Clusters",
        "",
        (
            "The topology changes occur below or around a gate that still closes. In "
            "edge-gate cases, child-parent support closes before sibling evidence can "
            "matter. In sibling-gate cases, full-edge traversal can visit descendants, "
            "but the active corrected sibling p-value never opens a split at the "
            "production sibling alpha."
        ),
        "",
        "Representative case diagnoses:",
        "",
        _markdown_table(
            case_diagnosis,
            [
                "case_id",
                "loss_bucket",
                "max_rooted_internal_rf_relative",
                "root_branch_length_ratio",
                "unstable_subtree_count",
                "large_unstable_subtree_count",
                "top_unstable_subtree_smaller_side_leaf_count",
                "top_unstable_subtree_supporting_methods",
                "root_edge_bh_min",
                "root_active_sibling_corrected_min",
                "minimum_diagnostic_sibling_p_value",
                "topology_difference_driver",
                "null_hypothesis_change",
            ],
            max_rows=20,
        ),
        "",
        "## Subtrees Showing The Differences",
        "",
        (
            "The subtree table parses the branch-length Newick exports and counts "
            "rooted internal leaf sets across topology methods. A high-impact unstable "
            "subtree is a leaf set present in some topology cells and absent in others, "
            "ranked by the size of the smaller side and by method-support balance."
        ),
        "",
        (
            "These subtrees explain why RF distance can be high while cluster labels "
            "stay unchanged: the unstable clades are real topology differences, but "
            "the active edge/sibling gates still collapse the selected hierarchy to "
            "the root boundary."
        ),
        "",
        _markdown_table(
            subtree_highlights,
            [
                "case_id",
                "subtree_difference_rank",
                "rooted_subtree_id",
                "smaller_side_leaf_count",
                "support_count",
                "method_count",
                "supporting_methods",
                "absent_methods",
                "contrast_type",
                "subtree_leaf_preview",
            ],
            max_rows=24,
        ),
        "",
        "## Null-Hypothesis Implications",
        "",
        (
            "The current fixed-tree style sibling evidence is not enough because the "
            "same data select adaptive KNN diffusion, topology, root, branch lengths, "
            "and the tested node. Under that pipeline, a production-valid null must "
            "condition on selection or explicitly replay the selection under null "
            "resampling."
        ),
        "",
        _markdown_table(
            null_recommendations,
            [
                "recommendation_id",
                "applies_to_loss_bucket",
                "required_null_hypothesis",
                "required_statistic_change",
                "production_guard",
            ],
            max_rows=10,
        ),
        "",
        "## Development Direction",
        "",
        "1. Keep the current fail-closed alpha defaults while the selected null is absent.",
        (
            "2. Add selected-pipeline calibration for edge support first; edge-closed "
            "cases do not reach a sibling null."
        ),
        (
            "3. For edge-supported overlap/SBM/low-rank cases, calibrate sibling "
            "p-values conditional on the selected topology, selected node, NNLS "
            "branch lengths, and parent covariance."
        ),
        (
            "4. Spend additional alpha only along a topology-stable chain with "
            "predeclared structural guards: largest-cluster fraction, plausible "
            "cluster count, fragmentation penalty, and topology-family agreement."
        ),
        (
            "5. If selected-family p-values are absent or topology agreement is low, "
            "return unsupported rather than raising global alpha."
        ),
        "",
        "## Inputs",
        "",
        f"- `{inputs.tree_cells}`",
        f"- `{inputs.tree_pairwise}`",
        f"- `{inputs.tree_newick}`",
        f"- `{inputs.loss_taxonomy}`",
        f"- `{inputs.pvalue_cells}`",
        f"- `{inputs.traversal_trace}`",
        f"- `{inputs.alpha_best_by_case}`",
        f"- `{inputs.literature_case_summary}`",
        "",
        "## Generated Artifacts",
        "",
        f"- `{output_dir / METHOD_SUMMARY_NAME}`",
        f"- `{output_dir / PAIR_FAMILY_SUMMARY_NAME}`",
        f"- `{output_dir / SUBTREE_SUPPORT_NAME}`",
        f"- `{output_dir / SUBTREE_HIGHLIGHTS_NAME}`",
        f"- `{output_dir / CASE_DIAGNOSIS_NAME}`",
        f"- `{output_dir / NULL_RECOMMENDATIONS_NAME}`",
        f"- `{output_dir / REPORT_NAME}`",
        f"- `{output_dir / MANIFEST_NAME}`",
        "",
    ]
    return "\n".join(lines)


def write_topology_difference_diagnosis_artifacts(
    *,
    frames: Mapping[str, pd.DataFrame],
    output_dir: Path,
    inputs: DiagnosisInputs,
) -> dict[str, Path]:
    """Write diagnosis CSV, Markdown, and manifest artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _timestamp()

    method_summary = build_method_summary(
        frames["tree_cells"],
        frames["pvalue_cells"],
        frames["tree_pairwise"],
    )
    pair_family_summary = build_pair_family_summary(frames["tree_pairwise"])
    subtree_support = build_subtree_support(frames["tree_newick"])
    subtree_highlights = build_subtree_highlights(subtree_support)
    case_diagnosis = build_case_diagnosis(
        frames["tree_cells"],
        frames["tree_pairwise"],
        frames["loss_taxonomy"],
        frames["pvalue_cells"],
        frames["traversal_trace"],
        frames["alpha_best_by_case"],
        frames["literature_case_summary"],
        subtree_support,
        subtree_highlights,
    )
    null_recommendations = build_null_recommendations(case_diagnosis)

    outputs = {
        "method_summary": output_dir / METHOD_SUMMARY_NAME,
        "pair_family_summary": output_dir / PAIR_FAMILY_SUMMARY_NAME,
        "subtree_support": output_dir / SUBTREE_SUPPORT_NAME,
        "subtree_highlights": output_dir / SUBTREE_HIGHLIGHTS_NAME,
        "case_diagnosis": output_dir / CASE_DIAGNOSIS_NAME,
        "null_recommendations": output_dir / NULL_RECOMMENDATIONS_NAME,
        "report": output_dir / REPORT_NAME,
        "manifest": output_dir / MANIFEST_NAME,
    }
    method_summary.to_csv(outputs["method_summary"], index=False)
    pair_family_summary.to_csv(outputs["pair_family_summary"], index=False)
    subtree_support.to_csv(outputs["subtree_support"], index=False)
    subtree_highlights.to_csv(outputs["subtree_highlights"], index=False)
    case_diagnosis.to_csv(outputs["case_diagnosis"], index=False)
    null_recommendations.to_csv(outputs["null_recommendations"], index=False)
    outputs["report"].write_text(
        write_report(
            output_dir=output_dir,
            method_summary=method_summary,
            pair_family_summary=pair_family_summary,
            subtree_highlights=subtree_highlights,
            case_diagnosis=case_diagnosis,
            null_recommendations=null_recommendations,
            tree_pairwise=frames["tree_pairwise"],
            inputs=inputs,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at_utc": generated_at,
        "git_revision": _git_revision(),
        "inputs": {
            key: str(value) if value is not None else None for key, value in inputs.__dict__.items()
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "row_counts": {
            "method_summary": int(len(method_summary)),
            "pair_family_summary": int(len(pair_family_summary)),
            "subtree_support": int(len(subtree_support)),
            "subtree_highlights": int(len(subtree_highlights)),
            "case_diagnosis": int(len(case_diagnosis)),
            "null_recommendations": int(len(null_recommendations)),
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", type=Path, default=DEFAULT_TREE_DIR)
    parser.add_argument("--pvalue-dir", type=Path, default=DEFAULT_PVALUE_DIR)
    parser.add_argument("--alpha-dir", type=Path, default=DEFAULT_ALPHA_DIR)
    parser.add_argument("--literature-dir", type=Path, default=DEFAULT_LITERATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = default_inputs(
        tree_dir=args.tree_dir,
        pvalue_dir=args.pvalue_dir,
        alpha_dir=args.alpha_dir,
        literature_dir=args.literature_dir,
    )
    outputs = write_topology_difference_diagnosis_artifacts(
        frames=load_diagnosis_frames(inputs),
        output_dir=args.output_dir,
        inputs=inputs,
    )
    print(outputs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
