"""Audit fixed branch-length candidate traversal tuples.

This diagnostic is evidence-only. It compares the live TBS traversal with an
edge-reachable traversal that keeps walking until child-parent edge tests close.
It does not route between methods and does not promote an adaptive policy.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

from benchmarks.diagnostics.calibration.traversal.cli import parse_traversal_audit_args
from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.result_records import benchmark_rows_to_dataframe
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import run_single_method_once
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "branch_length_traversal_audit/v1"
STUDY_ROLE = "diagnostic_fixed_candidate_traversal_audit_not_policy"
GENERATED_BY = "benchmarks.diagnostics.calibration.traversal.branch_length_traversal_audit"

DEFAULT_METHODS = ("tbs", "tbs_internal_filter_branch_length_v1")
DEFAULT_CASE_NAMES = (
    "phylo_dna_8taxa_low_mut",
    "phylo_protein_8taxa",
    "cat_mod_4cat_6c",
    "binary_2clusters",
    "gauss_outlier_cluster_4c",
    "gauss_noisy_many",
    "gauss_clear_medium",
    "dim_consolidated_4c_24f",
    "phylo_dna_4taxa_low_mut",
    "phylo_protein_4taxa",
    "phylo_dna_8taxa_med_mut",
    "gauss_single_outlier_4c",
    "binary_unbalanced_med",
    "overlap_extreme_4c",
    "traversal_deep_signal_under_same_parent",
    "traversal_deep_branch_recovery_stress",
)

TUPLES_OUTPUT = "traversal_tuples.csv"
EDGES_OUTPUT = "traversal_edges.csv"
SUMMARY_OUTPUT = "traversal_case_summary.csv"
METHOD_ROWS_OUTPUT = "method_rows.csv"
MANIFEST_OUTPUT = "manifest.json"
RUN_LOG_OUTPUT = "run.log"


@dataclass(frozen=True)
class BranchLengthTraversalAuditConfig:
    """Configuration for the fixed-candidate traversal audit."""

    output_dir: Path
    suite: str = "full"
    case_names: tuple[str, ...] = DEFAULT_CASE_NAMES
    methods: tuple[str, ...] = DEFAULT_METHODS
    significance_level: float = DEFAULT_SIBLING_ALPHA
    edge_alpha: float = DEFAULT_EDGE_ALPHA


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _node_to_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _path_to_str(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return " > ".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value)


def _finite_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _validate_methods(methods: tuple[str, ...]) -> tuple[str, ...]:
    missing = [method for method in methods if method not in METHOD_SPECS]
    if missing:
        raise ValueError(f"Unknown method id(s): {missing!r}")
    return methods


def _select_cases(*, suite: str, case_names: tuple[str, ...]) -> list[dict[str, object]]:
    cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing!r}")
    return [dict(by_name[name]) for name in case_names]


def _log(log_path: Path, message: str) -> None:
    line = f"[{format_timestamp_utc()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _case_summary_from_skip(
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


def _counter_summary(
    *,
    case_idx: int,
    case_name: str,
    case: dict[str, object],
    method_id: str,
    row: Any,
    counters: dict[str, object],
) -> dict[str, object]:
    base = _case_summary_from_skip(
        case_idx=case_idx,
        case_name=case_name,
        case=case,
        method_id=method_id,
        row=row,
    )
    base.update({key: int(value) for key, value in counters.items()})
    return base


def _tuple_rows_from_computed(
    *,
    case_idx: int,
    case_name: str,
    case: dict[str, object],
    method_id: str,
    row: Any,
    computed: Any,
) -> list[dict[str, object]]:
    decomposition = computed.decomposition or {}
    trace = decomposition.get("full_edge_traversal_trace", [])
    out: list[dict[str, object]] = []
    for order, trace_row in enumerate(trace, start=1):
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "test_case": int(case_idx),
                "case_id": str(case_name),
                "case_category": str(case.get("category", "")),
                "method": str(method_id),
                "status": row.status.value,
                "true_clusters": int(row.true_clusters),
                "found_clusters": int(row.found_clusters),
                "ari": _finite_float(row.ari),
                "tuple_order": int(order),
                "node_id": _node_to_str(trace_row.get("node_id")),
                "left_child": _node_to_str(trace_row.get("left_child")),
                "right_child": _node_to_str(trace_row.get("right_child")),
                "depth": int(trace_row.get("depth", 0)),
                "path": _path_to_str(trace_row.get("path")),
                "actual_visited": bool(trace_row.get("actual_visited", False)),
                "actual_decision": str(trace_row.get("actual_decision", "")),
                "edge_traversal_action": str(trace_row.get("edge_traversal_action", "")),
                "edge_traversal_stop_reason": str(trace_row.get("edge_traversal_stop_reason", "")),
                "is_leaf": bool(trace_row.get("is_leaf", False)),
                "n_children": int(trace_row.get("n_children", 0)),
                "n_descendant_leaves": int(trace_row.get("n_descendant_leaves", 0)),
                "left_edge_open": bool(trace_row.get("left_edge_open", False)),
                "right_edge_open": bool(trace_row.get("right_edge_open", False)),
                "edge_gate_open": bool(trace_row.get("edge_gate_open", False)),
                "sibling_different": bool(trace_row.get("sibling_different", False)),
                "sibling_skipped": bool(trace_row.get("sibling_skipped", False)),
                "sibling_gate_open": bool(trace_row.get("sibling_gate_open", False)),
                "left_branch_length": _finite_float(trace_row.get("left_branch_length")),
                "right_branch_length": _finite_float(trace_row.get("right_branch_length")),
                "left_branch_length_missing": bool(
                    trace_row.get("left_branch_length_missing", False)
                ),
                "right_branch_length_missing": bool(
                    trace_row.get("right_branch_length_missing", False)
                ),
            }
        )
    return out


def _edge_rows_from_tuples(tuple_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for tuple_row in tuple_rows:
        for side in ("left", "right"):
            child_id = str(tuple_row[f"{side}_child"])
            if not child_id:
                continue
            out.append(
                {
                    "schema_version": tuple_row["schema_version"],
                    "study_role": tuple_row["study_role"],
                    "test_case": tuple_row["test_case"],
                    "case_id": tuple_row["case_id"],
                    "case_category": tuple_row["case_category"],
                    "method": tuple_row["method"],
                    "status": tuple_row["status"],
                    "tuple_order": tuple_row["tuple_order"],
                    "parent_id": tuple_row["node_id"],
                    "child_id": child_id,
                    "child_side": side,
                    "depth": tuple_row["depth"],
                    "actual_visited": tuple_row["actual_visited"],
                    "actual_decision": tuple_row["actual_decision"],
                    "edge_traversal_action": tuple_row["edge_traversal_action"],
                    "edge_traversal_stop_reason": tuple_row["edge_traversal_stop_reason"],
                    "child_edge_open": bool(tuple_row[f"{side}_edge_open"]),
                    "branch_length": tuple_row[f"{side}_branch_length"],
                    "branch_length_missing": bool(tuple_row[f"{side}_branch_length_missing"]),
                }
            )
    return out


def run_branch_length_traversal_audit(
    config: BranchLengthTraversalAuditConfig,
) -> dict[str, Path]:
    """Run the traversal audit and write artifacts."""
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
    tuple_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

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
                    summary_rows.append(
                        _case_summary_from_skip(
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

                decomposition = computed.decomposition or {}
                counters = decomposition.get("traversal_counters", {})
                summary_rows.append(
                    _counter_summary(
                        case_idx=case_idx,
                        case_name=case_name,
                        case=case,
                        method_id=method_id,
                        row=row,
                        counters=counters,
                    )
                )
                current_tuple_rows = _tuple_rows_from_computed(
                    case_idx=case_idx,
                    case_name=case_name,
                    case=case,
                    method_id=method_id,
                    row=row,
                    computed=computed,
                )
                tuple_rows.extend(current_tuple_rows)
                edge_rows.extend(_edge_rows_from_tuples(current_tuple_rows))
                _log(
                    run_log_path,
                    (
                        f"method_done case={case_name} method={method_id} "
                        f"status={row.status.value} "
                        f"live_nodes={counters.get('live_nodes_visited')} "
                        f"full_edge_nodes={counters.get('full_edge_nodes_visited')}"
                    ),
                )

    outputs = {
        "traversal_tuples": output_dir / TUPLES_OUTPUT,
        "traversal_edges": output_dir / EDGES_OUTPUT,
        "traversal_case_summary": output_dir / SUMMARY_OUTPUT,
        "method_rows": output_dir / METHOD_ROWS_OUTPUT,
        "manifest": output_dir / MANIFEST_OUTPUT,
        "run_log": run_log_path,
    }

    pd.DataFrame(tuple_rows).to_csv(outputs["traversal_tuples"], index=False)
    pd.DataFrame(edge_rows).to_csv(outputs["traversal_edges"], index=False)
    pd.DataFrame(summary_rows).to_csv(outputs["traversal_case_summary"], index=False)
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
            "traversal_tuple_rows": len(tuple_rows),
            "traversal_edge_rows": len(edge_rows),
            "summary_rows": len(summary_rows),
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _log(run_log_path, f"wrote_outputs={outputs}")
    return outputs


def main() -> None:
    args = parse_traversal_audit_args(
        description=__doc__,
        default_case_names=DEFAULT_CASE_NAMES,
        default_methods=DEFAULT_METHODS,
        default_significance_level=DEFAULT_SIBLING_ALPHA,
        default_edge_alpha=DEFAULT_EDGE_ALPHA,
    )
    config = BranchLengthTraversalAuditConfig(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=tuple(str(name) for name in args.case_names),
        methods=tuple(str(method) for method in args.methods),
        significance_level=float(args.significance_level),
        edge_alpha=float(args.edge_alpha),
    )
    run_branch_length_traversal_audit(config)


if __name__ == "__main__":
    main()
