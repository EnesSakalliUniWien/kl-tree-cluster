"""Path-conditioned traversal tuple diagnostics.

This module is diagnostic-only. It annotates fixed traversal tuple rows with
upper-chain state, descendant support state, and benchmark truth labels. It
does not change traversal decisions or route between methods.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.traversal.branch_length_traversal_audit import (
    DEFAULT_CASE_NAMES,
    DEFAULT_METHODS,
    _json_default,
    _log,
    _select_cases,
    _tuple_rows_from_computed,
    _validate_methods,
)
from benchmarks.shared.result_records import benchmark_rows_to_dataframe
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import parse_names

SCHEMA_VERSION = "path_conditioned_traversal_audit/v1"
STUDY_ROLE = "diagnostic_path_conditioned_traversal_audit_not_policy"
GENERATED_BY = "benchmarks.diagnostics.calibration.traversal.path_conditioned_traversal_audit"

TUPLES_OUTPUT = "path_conditioned_traversal_tuples.csv"
CASE_SUMMARY_OUTPUT = "path_conditioned_traversal_case_summary.csv"
PASS_THROUGH_SUMMARY_OUTPUT = "path_conditioned_pass_through_summary.csv"
BOUNDARY_SUMMARY_OUTPUT = "path_conditioned_boundary_summary.csv"
METHOD_ROWS_OUTPUT = "method_rows.csv"
MANIFEST_OUTPUT = "manifest.json"
RUN_LOG_OUTPUT = "run.log"
VERIFICATION_LOG_OUTPUT = "verification.log"

PURE_BOUNDARY = "pure_boundary"
MIXED_BOUNDARY = "mixed_boundary"
TRUTH_COHERENT_SPLIT_PATH = "truth_coherent_split_path"
UNRESOLVED_TUPLE = "unresolved_tuple"


@dataclass(frozen=True)
class PathConditionedTraversalAuditConfig:
    """Configuration for the path-conditioned traversal audit."""

    output_dir: Path
    suite: str = "full"
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES
    methods: tuple[str, ...] = DEFAULT_METHODS
    significance_level: float = DEFAULT_SIBLING_ALPHA
    edge_alpha: float = DEFAULT_EDGE_ALPHA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="full")
    parser.add_argument("--case-names", type=parse_names, default=DEFAULT_CASE_NAMES)
    parser.add_argument("--methods", type=parse_names, default=DEFAULT_METHODS)
    parser.add_argument("--significance-level", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    return parser.parse_args()


def _path_nodes(path: object) -> tuple[str, ...]:
    """Return normalized node ids from a tuple row path."""
    if isinstance(path, (tuple, list)):
        return tuple(str(part).strip() for part in path if str(part).strip())
    if path is None or pd.isna(path):
        return ()
    return tuple(part.strip() for part in str(path).split(">") if part.strip())


def _is_true(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _finite_or_nan(value: object) -> float:
    if value is None or pd.isna(value):
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _path_prefix(path: object) -> str:
    nodes = _path_nodes(path)
    if not nodes:
        return ""
    return " > ".join(nodes) + " > "


def add_path_covariates(tuple_rows: pd.DataFrame) -> pd.DataFrame:
    """Add incoming-edge and ancestor-chain covariates to tuple rows."""
    out = tuple_rows.copy()
    if out.empty:
        return out

    for column in ("case_id", "method", "node_id", "left_child", "right_child", "path"):
        out[column] = out[column].fillna("").astype(str)

    row_by_key = {
        (str(row.case_id), str(row.method), str(row.node_id)): row
        for row in out.itertuples(index=False)
    }
    records: list[dict[str, object]] = []

    for row in out.itertuples(index=False):
        nodes = _path_nodes(row.path)
        ancestors = nodes[:-1]
        ancestor_rows = []
        missing_ancestors = []
        path_branch_lengths: list[float] = []

        for ancestor in ancestors:
            ancestor_row = row_by_key.get((row.case_id, row.method, ancestor))
            if ancestor_row is None:
                missing_ancestors.append(ancestor)
                continue
            ancestor_rows.append(ancestor_row)

        for parent, child in zip(nodes[:-1], nodes[1:]):
            parent_row = row_by_key.get((row.case_id, row.method, parent))
            if parent_row is None:
                continue
            if str(parent_row.left_child) == child:
                branch_length = _finite_or_nan(parent_row.left_branch_length)
            elif str(parent_row.right_child) == child:
                branch_length = _finite_or_nan(parent_row.right_branch_length)
            else:
                branch_length = math.nan
            if math.isfinite(branch_length):
                path_branch_lengths.append(branch_length)

        incoming_parent_id = str(ancestors[-1]) if ancestors else ""
        incoming_child_side = ""
        incoming_branch_length = math.nan
        incoming_edge_open = False
        incoming_parent_sibling_gate_open = False
        incoming_parent_decision = ""

        incoming_parent_row = (
            row_by_key.get((row.case_id, row.method, incoming_parent_id))
            if incoming_parent_id
            else None
        )
        if incoming_parent_row is not None:
            incoming_parent_decision = str(incoming_parent_row.actual_decision)
            incoming_parent_sibling_gate_open = _is_true(incoming_parent_row.sibling_gate_open)
            if str(incoming_parent_row.left_child) == row.node_id:
                incoming_child_side = "left"
                incoming_branch_length = _finite_or_nan(incoming_parent_row.left_branch_length)
                incoming_edge_open = _is_true(incoming_parent_row.left_edge_open)
            elif str(incoming_parent_row.right_child) == row.node_id:
                incoming_child_side = "right"
                incoming_branch_length = _finite_or_nan(incoming_parent_row.right_branch_length)
                incoming_edge_open = _is_true(incoming_parent_row.right_edge_open)

        records.append(
            {
                "incoming_parent_id": incoming_parent_id,
                "incoming_child_side": incoming_child_side,
                "incoming_branch_length": incoming_branch_length,
                "incoming_edge_open": incoming_edge_open,
                "incoming_parent_sibling_gate_open": (incoming_parent_sibling_gate_open),
                "incoming_parent_decision": incoming_parent_decision,
                "ancestor_split_count": sum(
                    str(ancestor.actual_decision) == "split" for ancestor in ancestor_rows
                ),
                "ancestor_pass_through_count": sum(
                    str(ancestor.actual_decision) == "pass_through" for ancestor in ancestor_rows
                ),
                "ancestor_sibling_closed_edge_open_count": sum(
                    (not _is_true(ancestor.sibling_gate_open))
                    and str(ancestor.edge_traversal_action) == "continue"
                    for ancestor in ancestor_rows
                ),
                "ancestor_chain_complete": not missing_ancestors,
                "missing_ancestor_count": len(missing_ancestors),
                "min_incoming_branch_length_on_path": (
                    min(path_branch_lengths) if path_branch_lengths else math.nan
                ),
                "median_incoming_branch_length_on_path": (
                    float(np.median(path_branch_lengths)) if path_branch_lengths else math.nan
                ),
            }
        )

    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def add_descendant_covariates(tuple_rows: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive descendant-support counts using path-prefix matching."""
    out = tuple_rows.copy()
    if out.empty:
        return out

    records: list[dict[str, object]] = []
    grouped = {
        (str(case_id), str(method)): group.copy()
        for (case_id, method), group in out.groupby(["case_id", "method"], sort=False)
    }

    for row in out.itertuples(index=False):
        group = grouped[(str(row.case_id), str(row.method))]
        prefix = _path_prefix(row.path)
        if prefix:
            descendants = group[group["path"].fillna("").astype(str).str.startswith(prefix)]
        else:
            descendants = group.iloc[0:0]
        live_descendants = descendants[descendants["actual_visited"].map(_is_true)]
        records.append(
            {
                "descendant_edge_open_continue_count": int(
                    (descendants["edge_traversal_action"] == "continue").sum()
                ),
                "descendant_sibling_open_count": int(
                    descendants["sibling_gate_open"].map(_is_true).sum()
                ),
                "descendant_live_split_count": int(
                    (live_descendants["actual_decision"] == "split").sum()
                ),
                "descendant_live_boundary_count": int(
                    (live_descendants["actual_decision"] == "boundary").sum()
                ),
            }
        )

    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def _truth_counts_json(leaves: list[str], truth_by_leaf: dict[str, object]) -> str:
    counts = Counter(str(truth_by_leaf[leaf]) for leaf in leaves if leaf in truth_by_leaf)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _truth_summary(
    leaves: list[str],
    truth_by_leaf: dict[str, object],
) -> dict[str, object]:
    counts = Counter(str(truth_by_leaf[leaf]) for leaf in leaves if leaf in truth_by_leaf)
    total = sum(counts.values())
    if total == 0:
        return {
            "truth_label_count": 0,
            "truth_sample_count": 0,
            "majority_truth_label": "",
            "majority_truth_count": 0,
            "descendant_purity": math.nan,
            "truth_counts_json": "{}",
        }
    majority_label, majority_count = counts.most_common(1)[0]
    return {
        "truth_label_count": len(counts),
        "truth_sample_count": int(total),
        "majority_truth_label": majority_label,
        "majority_truth_count": int(majority_count),
        "descendant_purity": float(majority_count / total),
        "truth_counts_json": json.dumps(dict(sorted(counts.items())), sort_keys=True),
    }


def _truth_audit_label(
    *,
    actual_visited: bool,
    actual_decision: str,
    tuple_truth_label_count: int,
    left_majority: str,
    right_majority: str,
) -> str:
    if not actual_visited:
        return UNRESOLVED_TUPLE
    if actual_decision == "boundary":
        return PURE_BOUNDARY if tuple_truth_label_count <= 1 else MIXED_BOUNDARY
    if actual_decision in {"split", "pass_through"}:
        if left_majority and right_majority and left_majority != right_majority:
            return TRUTH_COHERENT_SPLIT_PATH
    return UNRESOLVED_TUPLE


def add_truth_covariates(
    tuple_rows: pd.DataFrame,
    *,
    descendant_leaf_sets: dict[object, object],
    truth_by_leaf: dict[str, object],
) -> pd.DataFrame:
    """Add benchmark truth-label composition diagnostics to tuple rows."""
    out = tuple_rows.copy()
    if out.empty:
        return out

    records: list[dict[str, object]] = []
    for row in out.itertuples(index=False):
        node_leaves = sorted(str(leaf) for leaf in descendant_leaf_sets.get(row.node_id, []))
        left_leaves = sorted(str(leaf) for leaf in descendant_leaf_sets.get(row.left_child, []))
        right_leaves = sorted(str(leaf) for leaf in descendant_leaf_sets.get(row.right_child, []))

        tuple_summary = _truth_summary(node_leaves, truth_by_leaf)
        left_summary = _truth_summary(left_leaves, truth_by_leaf)
        right_summary = _truth_summary(right_leaves, truth_by_leaf)
        left_majority = str(left_summary["majority_truth_label"])
        right_majority = str(right_summary["majority_truth_label"])
        child_majority_differ = bool(
            left_majority and right_majority and left_majority != right_majority
        )
        label = _truth_audit_label(
            actual_visited=_is_true(row.actual_visited),
            actual_decision=str(row.actual_decision),
            tuple_truth_label_count=int(tuple_summary["truth_label_count"]),
            left_majority=left_majority,
            right_majority=right_majority,
        )
        records.append(
            {
                "tuple_truth_label_count": int(tuple_summary["truth_label_count"]),
                "tuple_truth_sample_count": int(tuple_summary["truth_sample_count"]),
                "tuple_majority_truth_label": tuple_summary["majority_truth_label"],
                "tuple_majority_truth_count": int(tuple_summary["majority_truth_count"]),
                "tuple_descendant_purity": tuple_summary["descendant_purity"],
                "tuple_truth_counts_json": tuple_summary["truth_counts_json"],
                "left_truth_label_count": int(left_summary["truth_label_count"]),
                "left_truth_sample_count": int(left_summary["truth_sample_count"]),
                "left_majority_truth_label": left_summary["majority_truth_label"],
                "left_descendant_purity": left_summary["descendant_purity"],
                "left_truth_counts_json": left_summary["truth_counts_json"],
                "right_truth_label_count": int(right_summary["truth_label_count"]),
                "right_truth_sample_count": int(right_summary["truth_sample_count"]),
                "right_majority_truth_label": right_summary["majority_truth_label"],
                "right_descendant_purity": right_summary["descendant_purity"],
                "right_truth_counts_json": right_summary["truth_counts_json"],
                "children_majority_truth_labels_differ": child_majority_differ,
                "truth_audit_label": label,
            }
        )

    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    return add_descendant_truth_coherent_counts(out)


def add_descendant_truth_coherent_counts(tuple_rows: pd.DataFrame) -> pd.DataFrame:
    """Count truth-coherent live descendant splits under each tuple."""
    out = tuple_rows.copy()
    if out.empty or "truth_audit_label" not in out.columns:
        out["descendant_truth_coherent_live_split_count"] = 0
        return out

    records: list[dict[str, object]] = []
    grouped = {
        (str(case_id), str(method)): group.copy()
        for (case_id, method), group in out.groupby(["case_id", "method"], sort=False)
    }
    for row in out.itertuples(index=False):
        group = grouped[(str(row.case_id), str(row.method))]
        prefix = _path_prefix(row.path)
        if prefix:
            descendants = group[group["path"].fillna("").astype(str).str.startswith(prefix)]
        else:
            descendants = group.iloc[0:0]
        records.append(
            {
                "descendant_truth_coherent_live_split_count": int(
                    (
                        descendants["actual_visited"].map(_is_true)
                        & (descendants["actual_decision"] == "split")
                        & (descendants["truth_audit_label"] == TRUTH_COHERENT_SPLIT_PATH)
                    ).sum()
                )
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def annotate_path_conditioned_tuples(
    tuple_rows: pd.DataFrame,
    *,
    descendant_leaf_sets: dict[object, object],
    truth_by_leaf: dict[str, object],
) -> pd.DataFrame:
    """Return path-conditioned traversal rows with truth diagnostics."""
    out = tuple_rows.copy()
    out["schema_version"] = SCHEMA_VERSION
    out["study_role"] = STUDY_ROLE
    out = add_path_covariates(out)
    out = add_descendant_covariates(out)
    return add_truth_covariates(
        out,
        descendant_leaf_sets=descendant_leaf_sets,
        truth_by_leaf=truth_by_leaf,
    )


def _skip_summary(
    *,
    case_idx: int,
    case_name: str,
    case: dict[str, object],
    method_id: str,
    row: Any,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "test_case": int(case_idx),
        "case_id": str(case_name),
        "case_category": str(case.get("category", "")),
        "method": str(method_id),
        "status": row.status.value,
        "skip_reason": row.skip_reason,
        "true_clusters": int(row.true_clusters),
        "found_clusters": int(row.found_clusters),
        "ari": float(row.ari) if math.isfinite(float(row.ari)) else math.nan,
    }


def _case_summary_from_tuples(
    *,
    base_summary: dict[str, object],
    tuple_rows: pd.DataFrame,
) -> dict[str, object]:
    live = tuple_rows[tuple_rows["actual_visited"].map(_is_true)]
    pass_through = live[live["actual_decision"] == "pass_through"]
    boundaries = live[live["actual_decision"] == "boundary"]
    edge_open_closed_sibling = boundaries[
        (boundaries["edge_traversal_action"] == "continue")
        & (~boundaries["sibling_gate_open"].map(_is_true))
    ]
    out = dict(base_summary)
    out.update(
        {
            "path_conditioned_tuple_rows": int(len(tuple_rows)),
            "path_conditioned_live_rows": int(len(live)),
            "path_conditioned_pass_through_rows": int(len(pass_through)),
            "stacked_pass_through_rows": int(
                (pass_through["ancestor_pass_through_count"] > 0).sum()
            ),
            "edge_open_closed_sibling_boundary_rows": int(len(edge_open_closed_sibling)),
            "pure_boundary_rows": int((tuple_rows["truth_audit_label"] == PURE_BOUNDARY).sum()),
            "mixed_boundary_rows": int((tuple_rows["truth_audit_label"] == MIXED_BOUNDARY).sum()),
            "truth_coherent_split_path_rows": int(
                (tuple_rows["truth_audit_label"] == TRUTH_COHERENT_SPLIT_PATH).sum()
            ),
            "median_pass_through_ancestor_pass_through_count": (
                float(pass_through["ancestor_pass_through_count"].median())
                if not pass_through.empty
                else math.nan
            ),
            "median_pass_through_incoming_branch_length": (
                float(pass_through["incoming_branch_length"].median())
                if not pass_through.empty
                else math.nan
            ),
            "pass_through_with_truth_coherent_descendant_split_rows": int(
                (pass_through["descendant_truth_coherent_live_split_count"] > 0).sum()
            ),
        }
    )
    return out


def _summarize_pass_through(tuple_rows: pd.DataFrame) -> pd.DataFrame:
    live = tuple_rows[tuple_rows["actual_visited"].map(_is_true)]
    pass_through = live[live["actual_decision"] == "pass_through"]
    if pass_through.empty:
        return pd.DataFrame()
    rows = []
    for (method, case_id), group in pass_through.groupby(["method", "case_id"]):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "method": method,
                "case_id": case_id,
                "pass_through_rows": int(len(group)),
                "stacked_pass_through_rows": int((group["ancestor_pass_through_count"] > 0).sum()),
                "median_depth": float(group["depth"].median()),
                "max_depth": int(group["depth"].max()),
                "median_ancestor_pass_through_count": float(
                    group["ancestor_pass_through_count"].median()
                ),
                "median_ancestor_sibling_closed_edge_open_count": float(
                    group["ancestor_sibling_closed_edge_open_count"].median()
                ),
                "median_incoming_branch_length": float(group["incoming_branch_length"].median()),
                "rows_with_truth_coherent_descendant_split": int(
                    (group["descendant_truth_coherent_live_split_count"] > 0).sum()
                ),
                "truth_coherent_pass_through_rows": int(
                    (group["truth_audit_label"] == TRUTH_COHERENT_SPLIT_PATH).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_boundaries(tuple_rows: pd.DataFrame) -> pd.DataFrame:
    live = tuple_rows[tuple_rows["actual_visited"].map(_is_true)]
    boundaries = live[live["actual_decision"] == "boundary"]
    if boundaries.empty:
        return pd.DataFrame()
    rows = []
    for (method, case_id), group in boundaries.groupby(["method", "case_id"]):
        edge_open_closed = group[
            (group["edge_traversal_action"] == "continue")
            & (~group["sibling_gate_open"].map(_is_true))
        ]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "method": method,
                "case_id": case_id,
                "boundary_rows": int(len(group)),
                "pure_boundary_rows": int((group["truth_audit_label"] == PURE_BOUNDARY).sum()),
                "mixed_boundary_rows": int((group["truth_audit_label"] == MIXED_BOUNDARY).sum()),
                "edge_open_closed_sibling_boundary_rows": int(len(edge_open_closed)),
                "median_descendant_edge_open_continue_count": (
                    float(edge_open_closed["descendant_edge_open_continue_count"].median())
                    if not edge_open_closed.empty
                    else math.nan
                ),
                "median_ancestor_pass_through_count": float(
                    group["ancestor_pass_through_count"].median()
                ),
                "median_incoming_branch_length": float(group["incoming_branch_length"].median()),
            }
        )
    return pd.DataFrame(rows)


def _truth_by_leaf_from_computed(computed: Any) -> dict[str, object]:
    labels = np.asarray(computed.y_true)
    return {str(sample_id): labels[index] for index, sample_id in enumerate(computed.data.index)}


def run_path_conditioned_traversal_audit(
    config: PathConditionedTraversalAuditConfig,
) -> dict[str, Path]:
    """Run the path-conditioned traversal audit and write artifacts."""
    methods = _validate_methods(config.methods)
    cases = _select_cases(suite=config.suite, case_names=config.case_names)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / RUN_LOG_OUTPUT
    run_log_path.write_text("", encoding="utf-8")
    _log(run_log_path, f"generated_by={GENERATED_BY}")
    _log(run_log_path, f"cwd={Path.cwd()}")
    _log(run_log_path, f"python={sys.version.split()[0]} platform={platform.platform()}")
    _log(run_log_path, f"methods={list(methods)}")
    _log(run_log_path, f"cases={[str(case['name']) for case in cases]}")
    _log(run_log_path, f"argv={sys.argv}")

    method_rows = []
    tuple_frames: list[pd.DataFrame] = []
    case_summary_rows: list[dict[str, object]] = []

    for case_idx, case in enumerate(cases, start=1):
        case = dict(case)
        case["test_case_num"] = case_idx
        case_name = str(case["name"])
        _log(run_log_path, f"case_start index={case_idx} case={case_name}")
        inputs = prepare_case_inputs(case, list(methods))
        for method_id in methods:
            spec = METHOD_SPECS[method_id]
            for params in spec.param_grid:
                row, computed, _method_audit = run_single_method_once(
                    method_id=method_id,
                    spec=spec,
                    params=params,
                    case_idx=case_idx,
                    case_name=case_name,
                    tc_seed=case["seed"],
                    significance_level=config.significance_level,
                    edge_alpha=config.edge_alpha,
                    data_t=inputs.data,
                    y_t=inputs.labels,
                    x_original=inputs.original_features,
                    meta=inputs.metadata,
                    distance_matrix=inputs.distance_matrix,
                    distance_condensed=inputs.distance_condensed,
                    matrix_audit=False,
                )
                method_rows.append(row)
                if computed is None:
                    case_summary_rows.append(
                        _skip_summary(
                            case_idx=case_idx,
                            case_name=case_name,
                            case=case,
                            method_id=method_id,
                            row=row,
                        )
                    )
                    _log(
                        run_log_path,
                        (
                            f"method_done case={case_name} method={method_id} "
                            f"status={row.status.value} skip={row.skip_reason!r}"
                        ),
                    )
                    continue

                raw_tuple_rows = _tuple_rows_from_computed(
                    case_idx=case_idx,
                    case_name=case_name,
                    case=case,
                    method_id=method_id,
                    row=row,
                    computed=computed,
                )
                descendant_leaf_sets = computed.tree.compute_descendant_sets(use_labels=True)
                path_rows = annotate_path_conditioned_tuples(
                    pd.DataFrame(raw_tuple_rows),
                    descendant_leaf_sets=descendant_leaf_sets,
                    truth_by_leaf=_truth_by_leaf_from_computed(computed),
                )
                tuple_frames.append(path_rows)

                counters = (computed.decomposition or {}).get("traversal_counters", {})
                base_summary = _skip_summary(
                    case_idx=case_idx,
                    case_name=case_name,
                    case=case,
                    method_id=method_id,
                    row=row,
                )
                base_summary.update({key: int(value) for key, value in counters.items()})
                case_summary_rows.append(
                    _case_summary_from_tuples(
                        base_summary=base_summary,
                        tuple_rows=path_rows,
                    )
                )
                _log(
                    run_log_path,
                    (
                        f"method_done case={case_name} method={method_id} "
                        f"status={row.status.value} "
                        f"path_tuple_rows={len(path_rows)} "
                        f"pass_through_rows="
                        f"{int((path_rows['actual_visited'].map(_is_true) & (path_rows['actual_decision'] == 'pass_through')).sum())}"
                    ),
                )

    all_tuples = pd.concat(tuple_frames, ignore_index=True) if tuple_frames else pd.DataFrame()
    case_summary = pd.DataFrame(case_summary_rows)
    pass_through_summary = _summarize_pass_through(all_tuples)
    boundary_summary = _summarize_boundaries(all_tuples)

    outputs = {
        "path_conditioned_traversal_tuples": output_dir / TUPLES_OUTPUT,
        "path_conditioned_traversal_case_summary": output_dir / CASE_SUMMARY_OUTPUT,
        "path_conditioned_pass_through_summary": output_dir / PASS_THROUGH_SUMMARY_OUTPUT,
        "path_conditioned_boundary_summary": output_dir / BOUNDARY_SUMMARY_OUTPUT,
        "method_rows": output_dir / METHOD_ROWS_OUTPUT,
        "manifest": output_dir / MANIFEST_OUTPUT,
        "run_log": run_log_path,
        "verification_log": output_dir / VERIFICATION_LOG_OUTPUT,
    }
    all_tuples.to_csv(outputs["path_conditioned_traversal_tuples"], index=False)
    case_summary.to_csv(outputs["path_conditioned_traversal_case_summary"], index=False)
    pass_through_summary.to_csv(outputs["path_conditioned_pass_through_summary"], index=False)
    boundary_summary.to_csv(outputs["path_conditioned_boundary_summary"], index=False)
    benchmark_rows_to_dataframe(method_rows).to_csv(outputs["method_rows"], index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "output_dir": str(output_dir),
        "suite": config.suite,
        "case_names": [str(case["name"]) for case in cases],
        "methods": list(methods),
        "significance_level": float(config.significance_level),
        "edge_alpha": float(config.edge_alpha),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "environment": {
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "TBS_N_JOBS": os.environ.get("TBS_N_JOBS", ""),
        },
        "argv": sys.argv,
        "counts": {
            "method_rows": len(method_rows),
            "path_conditioned_tuple_rows": len(all_tuples),
            "summary_rows": len(case_summary),
            "pass_through_summary_rows": len(pass_through_summary),
            "boundary_summary_rows": len(boundary_summary),
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _log(run_log_path, f"wrote_outputs={outputs}")
    return outputs


def main() -> None:
    args = parse_args()
    config = PathConditionedTraversalAuditConfig(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=tuple(str(name) for name in args.case_names),
        methods=tuple(str(method) for method in args.methods),
        significance_level=float(args.significance_level),
        edge_alpha=float(args.edge_alpha),
    )
    run_path_conditioned_traversal_audit(config)


if __name__ == "__main__":
    main()
