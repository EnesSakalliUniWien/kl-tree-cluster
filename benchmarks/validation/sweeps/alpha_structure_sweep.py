#!/usr/bin/env python3
"""Structural alpha-sweep diagnostic for TBS benchmark gates.

This runner treats edge alpha and sibling alpha as methodological axes. It
reruns selected benchmark cases over an explicit alpha-pair grid and records
how p-value boundaries, traversal decisions, and final partitions change.

The output is validation evidence only. It does not change production alpha
defaults and does not claim selected-tree Type-I error calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from benchmarks.shared.cases import get_default_test_cases, get_test_cases_by_suite
from benchmarks.shared.result_records import BenchmarkResultRow, benchmark_rows_to_dataframe
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_run import run_single_case
from benchmarks.shared.util.method_selection import resolve_selected_methods_and_param_sets
from benchmarks.validation.sweeps.alpha_grid_search import (
    DEFAULT_EDGE_ALPHA_GRID,
    DEFAULT_SIBLING_ALPHA_GRID,
    alpha_pair_id,
    build_alpha_pairs,
    parse_float_grid,
)

SCHEMA_VERSION = "alpha_structure_sweep/v1"
GENERATED_BY = "benchmarks.validation.sweeps.alpha_structure_sweep"
DEFAULT_METHODS = ("tbs", "tbs_diffusion_adaptive")
PVALUE_COLUMNS = (
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_Fixed_Coordinate_BH_P_Value",
    "Sibling_Fixed_Block_BH_P_Value",
    "Sibling_Fixed_Global_P_Value",
)
ANNOTATION_COLUMNS = (
    "Child_Parent_Divergence_Test_Statistic",
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_Tested",
    "Child_Parent_Divergence_Ancestor_Blocked",
    "Sibling_Divergence_Skipped",
    "Sibling_Test_Statistic",
    "Sibling_Degrees_of_Freedom",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_BH_Different",
    "Sibling_BH_Same",
    "Sibling_Test_Method",
    "Sibling_Gate_P_Value_Calibration",
    "Sibling_Gate_P_Value_Role",
    "Sibling_Projection_Dimension",
    "Sibling_Fixed_Coordinate_BH_P_Value",
    "Sibling_Fixed_Block_BH_P_Value",
    "Sibling_Fixed_Global_P_Value",
    "parent_node",
    "branch_length",
)
TRACE_COLUMNS = (
    "node_id",
    "left_child",
    "right_child",
    "left_edge_p_value",
    "left_edge_p_value_bh",
    "right_edge_p_value",
    "right_edge_p_value_bh",
    "sibling_p_value",
    "sibling_p_value_corrected",
    "depth",
    "actual_visited",
    "actual_decision",
    "edge_traversal_action",
    "edge_traversal_stop_reason",
    "n_descendant_leaves",
    "left_edge_open",
    "right_edge_open",
    "edge_gate_open",
    "sibling_different",
    "sibling_skipped",
    "sibling_gate_open",
    "passthrough_candidate",
    "passthrough_supported",
    "passthrough_decision_reason",
)


@dataclass(frozen=True)
class AlphaStructureSweepConfig:
    suite: str
    case_names: tuple[str, ...]
    methods: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alphas: tuple[float, ...]
    output_dir: Path
    resume: bool
    baseline_edge_alpha: float
    baseline_sibling_alpha: float
    alpha_pairs: tuple[tuple[float, float], ...] | None = None


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _parse_alpha_pairs(raw: str | None) -> tuple[tuple[float, float], ...] | None:
    if raw is None or not raw.strip():
        return None
    pairs: list[tuple[float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Alpha pair specs must have form edge_alpha:sibling_alpha; got {item!r}."
            )
        edge_raw, sibling_raw = item.split(":", 1)
        edge_alpha = float(edge_raw)
        sibling_alpha = float(sibling_raw)
        if not 0.0 < edge_alpha < 1.0 or not 0.0 < sibling_alpha < 1.0:
            raise ValueError(f"Alpha pair values must lie in (0, 1): {item!r}")
        pairs.append((edge_alpha, sibling_alpha))
    if not pairs:
        raise ValueError("At least one alpha pair is required.")
    return tuple(pairs)


def _select_cases(*, suite: str, case_names: Sequence[str]) -> list[dict[str, Any]]:
    if suite == "default":
        cases = get_default_test_cases()
    else:
        cases = get_test_cases_by_suite(suite)
    if not case_names:
        return [dict(case) for case in cases]
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing!r}")
    return [dict(by_name[name]) for name in case_names]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _finite_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else math.nan


def _bool_sum(series: pd.Series) -> int:
    if series.empty:
        return 0
    return int(series.fillna(False).astype(bool).sum())


def _pvalue_margin(p_value: Any, alpha: float) -> float:
    p = _finite_float(p_value)
    if not math.isfinite(p) or p < 0.0:
        return math.nan
    if p == 0.0:
        return math.inf
    return float(math.log10(float(alpha) / p))


def _near_boundary(p_values: pd.Series, alpha: float, factor: float = 3.0) -> int:
    numeric = pd.to_numeric(p_values, errors="coerce").dropna()
    if numeric.empty:
        return 0
    lower = float(alpha) / float(factor)
    upper = float(alpha) * float(factor)
    return int(((numeric >= lower) & (numeric <= upper)).sum())


def _quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(numeric.quantile(q))


def _current_git_state() -> dict[str, object]:
    state: dict[str, object] = {}
    commands = {
        "commit": ("git", "rev-parse", "HEAD"),
        "branch": ("git", "branch", "--show-current"),
        "status_short": ("git", "status", "--short"),
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            state[key] = f"unavailable:{exc}"
            continue
        if completed.returncode == 0:
            state[key] = completed.stdout.strip()
        else:
            state[key] = {
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            }
    return state


def _annotations_for_output(record: Any) -> pd.DataFrame:
    annotations = record.annotations
    if annotations is None:
        return pd.DataFrame()
    table = annotations.copy()
    table.insert(0, "node_id", table.index.astype(str))
    keep = ["node_id"] + [column for column in ANNOTATION_COLUMNS if column in table.columns]
    table = table[keep].copy()
    for column in PVALUE_COLUMNS:
        if column in table.columns:
            table[f"{column}_neg_log10"] = -np.log10(
                pd.to_numeric(table[column], errors="coerce").clip(lower=np.nextafter(0, 1))
            )
    table["edge_bh_log10_alpha_over_p"] = table.get(
        "Child_Parent_Divergence_P_Value_BH",
        pd.Series(index=table.index, dtype=float),
    ).map(lambda value: _pvalue_margin(value, record.meta["edge_alpha"]))
    table["sibling_corrected_log10_alpha_over_p"] = table.get(
        "Sibling_Divergence_P_Value_Corrected",
        pd.Series(index=table.index, dtype=float),
    ).map(lambda value: _pvalue_margin(value, record.meta["sibling_alpha"]))
    return table


def _trace_for_output(record: Any) -> pd.DataFrame:
    decomposition = record.decomposition if isinstance(record.decomposition, dict) else {}
    trace = decomposition.get("full_edge_traversal_trace", [])
    table = pd.DataFrame(trace)
    if table.empty:
        return table
    keep = [column for column in TRACE_COLUMNS if column in table.columns]
    table = table[keep].copy()
    table["edge_pair_min_bh_log10_alpha_over_p"] = [
        min(
            _pvalue_margin(row.get("left_edge_p_value_bh"), record.meta["edge_alpha"]),
            _pvalue_margin(row.get("right_edge_p_value_bh"), record.meta["edge_alpha"]),
        )
        for _, row in table.iterrows()
    ]
    table["sibling_corrected_log10_alpha_over_p"] = table.get(
        "sibling_p_value_corrected",
        pd.Series(index=table.index, dtype=float),
    ).map(lambda value: _pvalue_margin(value, record.meta["sibling_alpha"]))
    return table


def _structural_summary_rows(
    *,
    alpha_pair: tuple[float, float],
    results: pd.DataFrame,
    computed_results: Sequence[Any],
    elapsed_sec: float,
) -> list[dict[str, object]]:
    edge_alpha, sibling_alpha = alpha_pair
    rows: list[dict[str, object]] = []
    for record in computed_results:
        result_match = results[
            (results["test_case"] == int(record.test_case_num))
            & (results["method"] == str(record.method))
        ]
        result_row = result_match.iloc[0].to_dict() if not result_match.empty else {}
        annotations = (
            record.annotations.copy() if record.annotations is not None else pd.DataFrame()
        )
        decomposition = record.decomposition if isinstance(record.decomposition, dict) else {}
        full_trace = pd.DataFrame(decomposition.get("full_edge_traversal_trace", []))
        traversal_trace = pd.DataFrame(decomposition.get("traversal_trace", []))
        edge_bh = (
            pd.to_numeric(annotations["Child_Parent_Divergence_P_Value_BH"], errors="coerce")
            if "Child_Parent_Divergence_P_Value_BH" in annotations.columns
            else pd.Series(dtype=float)
        )
        sibling_corrected = (
            pd.to_numeric(
                annotations["Sibling_Divergence_P_Value_Corrected"],
                errors="coerce",
            )
            if "Sibling_Divergence_P_Value_Corrected" in annotations.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "grid_edge_alpha": float(edge_alpha),
                "grid_sibling_alpha": float(sibling_alpha),
                "alpha_pair_id": alpha_pair_id(edge_alpha, sibling_alpha),
                "test_case": int(record.test_case_num),
                "case_id": str(result_row.get("case_id", record.meta.get("name", ""))),
                "case_category": str(
                    result_row.get("case_category", record.meta.get("category", ""))
                ),
                "method": str(record.method),
                "method_name": str(record.method_name),
                "true_clusters": int(
                    result_row.get("true_clusters", record.meta.get("n_clusters", 0))
                ),
                "found_clusters": int(
                    result_row.get("found_clusters", record.meta.get("found_clusters", 0))
                ),
                "ari": float(result_row.get("ari", record.ari)),
                "nmi": float(result_row.get("nmi", record.nmi)),
                "purity": float(result_row.get("purity", record.purity)),
                "n_annotation_nodes": int(len(annotations)),
                "n_full_trace_rows": int(len(full_trace)),
                "n_actual_traversal_rows": int(len(traversal_trace)),
                "n_edge_bh_tested": int(edge_bh.notna().sum()),
                "n_edge_bh_below_alpha": int((edge_bh <= float(edge_alpha)).sum()),
                "n_edge_bh_near_alpha_x3": _near_boundary(edge_bh, edge_alpha),
                "edge_bh_q01": _quantile(edge_bh, 0.01),
                "edge_bh_q05": _quantile(edge_bh, 0.05),
                "edge_bh_q50": _quantile(edge_bh, 0.50),
                "n_sibling_corrected_tested": int(sibling_corrected.notna().sum()),
                "n_sibling_corrected_below_alpha": int(
                    (sibling_corrected <= float(sibling_alpha)).sum()
                ),
                "n_sibling_corrected_near_alpha_x3": _near_boundary(
                    sibling_corrected,
                    sibling_alpha,
                ),
                "sibling_corrected_q01": _quantile(sibling_corrected, 0.01),
                "sibling_corrected_q05": _quantile(sibling_corrected, 0.05),
                "sibling_corrected_q50": _quantile(sibling_corrected, 0.50),
                "n_trace_edge_open": (
                    _bool_sum(full_trace["edge_gate_open"])
                    if "edge_gate_open" in full_trace.columns
                    else 0
                ),
                "n_trace_sibling_open": (
                    _bool_sum(full_trace["sibling_gate_open"])
                    if "sibling_gate_open" in full_trace.columns
                    else 0
                ),
                "n_trace_pass_through": (
                    int((full_trace["actual_decision"].astype(str) == "pass_through").sum())
                    if "actual_decision" in full_trace.columns
                    else 0
                ),
                "n_trace_boundaries": (
                    int((full_trace["actual_decision"].astype(str) == "boundary").sum())
                    if "actual_decision" in full_trace.columns
                    else 0
                ),
                "elapsed_alpha_pair_sec": float(elapsed_sec),
            }
        )
    return rows


def _partition_transition_rows(
    *,
    computed_by_key: dict[tuple[str, str, str], Any],
    baseline_edge_alpha: float,
    baseline_sibling_alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_pair = alpha_pair_id(baseline_edge_alpha, baseline_sibling_alpha)
    case_method_keys = sorted({(case_id, method) for case_id, method, _ in computed_by_key})
    for case_id, method in case_method_keys:
        baseline = computed_by_key.get((case_id, method, baseline_pair))
        if baseline is None:
            continue
        baseline_labels = np.asarray(baseline.labels)
        baseline_clusters = len(pd.Series(baseline_labels).astype(str).unique())
        for _, _, pair_id in sorted(
            key for key in computed_by_key if key[0] == case_id and key[1] == method
        ):
            record = computed_by_key[(case_id, method, pair_id)]
            labels = np.asarray(record.labels)
            if len(labels) != len(baseline_labels):
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "baseline_alpha_pair_id": baseline_pair,
                    "alpha_pair_id": pair_id,
                    "baseline_edge_alpha": float(baseline_edge_alpha),
                    "baseline_sibling_alpha": float(baseline_sibling_alpha),
                    "edge_alpha": float(record.meta["edge_alpha"]),
                    "sibling_alpha": float(record.meta["sibling_alpha"]),
                    "baseline_found_clusters": int(baseline_clusters),
                    "found_clusters": int(len(pd.Series(labels).astype(str).unique())),
                    "ari_vs_baseline": float(adjusted_rand_score(baseline_labels, labels)),
                    "nmi_vs_baseline": float(normalized_mutual_info_score(baseline_labels, labels)),
                }
            )
    return rows


def _write_pair_outputs(
    *,
    pair_dir: Path,
    results: pd.DataFrame,
    computed_results: Sequence[Any],
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(pair_dir / "benchmark_metrics.csv", index=False)
    node_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    for record in computed_results:
        case_id = str(record.meta.get("name", record.test_case_num))
        prefix = {
            "test_case": int(record.test_case_num),
            "case_id": case_id,
            "method": str(record.method),
            "edge_alpha": float(record.meta["edge_alpha"]),
            "sibling_alpha": float(record.meta["sibling_alpha"]),
        }
        annotations = _annotations_for_output(record)
        if not annotations.empty:
            for key, value in prefix.items():
                annotations.insert(0, key, value)
            node_frames.append(annotations)
        trace = _trace_for_output(record)
        if not trace.empty:
            for key, value in prefix.items():
                trace.insert(0, key, value)
            trace_frames.append(trace)
        labels = np.asarray(record.labels)
        assignment = pd.DataFrame(
            {
                "sample_id": record.data.index.astype(str).to_numpy(),
                "true_label": np.asarray(record.y_true).astype(str),
                "cluster_id": labels.astype(str),
            }
        )
        for key, value in prefix.items():
            assignment.insert(0, key, value)
        assignment_frames.append(assignment)
    if node_frames:
        pd.concat(node_frames, ignore_index=True).to_csv(
            pair_dir / "node_pvalue_margins.csv", index=False
        )
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(
            pair_dir / "traversal_pvalue_margins.csv",
            index=False,
        )
    if assignment_frames:
        pd.concat(assignment_frames, ignore_index=True).to_csv(
            pair_dir / "partition_assignments.csv",
            index=False,
        )


def run_alpha_structure_sweep(config: AlphaStructureSweepConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _select_cases(suite=config.suite, case_names=config.case_names)
    selected_methods, param_sets = resolve_selected_methods_and_param_sets(
        methods=list(config.methods),
        method_params=None,
        default_methods=list(DEFAULT_METHODS),
        method_specs=METHOD_SPECS,
    )
    alpha_pairs = (
        config.alpha_pairs
        if config.alpha_pairs is not None
        else build_alpha_pairs(config.edge_alphas, config.sibling_alphas)
    )
    summaries: list[dict[str, object]] = []
    all_metrics: list[pd.DataFrame] = []
    computed_by_key: dict[tuple[str, str, str], Any] = {}

    for edge_alpha, sibling_alpha in alpha_pairs:
        pair_id = alpha_pair_id(edge_alpha, sibling_alpha)
        pair_dir = config.output_dir / pair_id
        metrics_path = pair_dir / "benchmark_metrics.csv"
        summary_path = pair_dir / "structural_summary.csv"
        if config.resume and metrics_path.exists() and summary_path.exists():
            metrics = pd.read_csv(metrics_path)
            summary = pd.read_csv(summary_path)
            all_metrics.append(metrics)
            summaries.extend(summary.to_dict("records"))
            continue

        start = perf_counter()
        result_rows: list[BenchmarkResultRow] = []
        computed_results: list[Any] = []
        for case_index, case in enumerate(cases, start=1):
            case = dict(case)
            case["test_case_num"] = case_index
            case_rows, case_computed = run_single_case(
                tc=case,
                total_cases=len(cases),
                selected_methods=selected_methods,
                param_sets=param_sets,
                significance_level=float(sibling_alpha),
                edge_alpha=float(edge_alpha),
                output_pdf=pair_dir / "_audit_anchor.pdf",
                plots_root=pair_dir / "plots",
                matrix_audit=False,
                verbose=False,
            )
            result_rows.extend(case_rows)
            computed_results.extend(case_computed)
        elapsed_sec = perf_counter() - start

        metrics = benchmark_rows_to_dataframe(result_rows)
        metrics.insert(0, "grid_edge_alpha", float(edge_alpha))
        metrics.insert(1, "grid_sibling_alpha", float(sibling_alpha))
        metrics.insert(2, "alpha_pair_id", pair_id)
        for record in computed_results:
            record.meta["edge_alpha"] = float(edge_alpha)
            record.meta["sibling_alpha"] = float(sibling_alpha)
        summary_rows = _structural_summary_rows(
            alpha_pair=(edge_alpha, sibling_alpha),
            results=metrics,
            computed_results=computed_results,
            elapsed_sec=elapsed_sec,
        )
        summary = pd.DataFrame(summary_rows)

        _write_pair_outputs(pair_dir=pair_dir, results=metrics, computed_results=computed_results)
        summary.to_csv(summary_path, index=False)
        all_metrics.append(metrics)
        summaries.extend(summary_rows)
        for record in computed_results:
            case_id = str(record.meta.get("name", record.test_case_num))
            computed_by_key[(case_id, str(record.method), pair_id)] = record

        print(
            "alpha-structure "
            f"{pair_id} cases={len(cases)} methods={len(selected_methods)} "
            f"elapsed={elapsed_sec:.3f}s",
            flush=True,
        )

    combined_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    combined_summary = pd.DataFrame(summaries)
    combined_metrics.to_csv(config.output_dir / "alpha_structure_metrics.csv", index=False)
    combined_summary.to_csv(config.output_dir / "alpha_structure_summary.csv", index=False)
    transitions = pd.DataFrame(
        _partition_transition_rows(
            computed_by_key=computed_by_key,
            baseline_edge_alpha=config.baseline_edge_alpha,
            baseline_sibling_alpha=config.baseline_sibling_alpha,
        )
    )
    transitions.to_csv(config.output_dir / "alpha_partition_transitions.csv", index=False)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": config.suite,
        "case_names": [str(case["name"]) for case in cases],
        "methods": list(selected_methods),
        "edge_alphas": list(config.edge_alphas),
        "sibling_alphas": list(config.sibling_alphas),
        "selected_alpha_pairs": [
            {"edge_alpha": edge_alpha, "sibling_alpha": sibling_alpha}
            for edge_alpha, sibling_alpha in alpha_pairs
        ],
        "baseline_edge_alpha": float(config.baseline_edge_alpha),
        "baseline_sibling_alpha": float(config.baseline_sibling_alpha),
        "output_dir": config.output_dir,
        "outputs": {
            "alpha_structure_metrics": config.output_dir / "alpha_structure_metrics.csv",
            "alpha_structure_summary": config.output_dir / "alpha_structure_summary.csv",
            "alpha_partition_transitions": config.output_dir / "alpha_partition_transitions.csv",
            "per_pair_outputs": [
                {
                    "alpha_pair_id": alpha_pair_id(edge_alpha, sibling_alpha),
                    "benchmark_metrics": config.output_dir
                    / alpha_pair_id(edge_alpha, sibling_alpha)
                    / "benchmark_metrics.csv",
                    "structural_summary": config.output_dir
                    / alpha_pair_id(edge_alpha, sibling_alpha)
                    / "structural_summary.csv",
                    "node_pvalue_margins": config.output_dir
                    / alpha_pair_id(edge_alpha, sibling_alpha)
                    / "node_pvalue_margins.csv",
                    "traversal_pvalue_margins": config.output_dir
                    / alpha_pair_id(edge_alpha, sibling_alpha)
                    / "traversal_pvalue_margins.csv",
                }
                for edge_alpha, sibling_alpha in alpha_pairs
            ],
        },
        "note": (
            "Diagnostic alpha sweep only. Edge and sibling alpha are varied as "
            "methodological axes while data, method, and case seed are fixed. "
            "P-value margins are descriptive boundary distances, not selected-tree "
            "calibration guarantees."
        ),
        "git": _current_git_state(),
    }
    (config.output_dir / "alpha_structure_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"metrics": combined_metrics, "summary": combined_summary, "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="categorical")
    parser.add_argument(
        "--case-names",
        default=None,
        help="Comma-separated case subset within the selected suite.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated TBS-family method ids to sweep.",
    )
    parser.add_argument(
        "--edge-alphas",
        default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
    )
    parser.add_argument(
        "--sibling-alphas",
        default=",".join(str(value) for value in DEFAULT_SIBLING_ALPHA_GRID),
    )
    parser.add_argument(
        "--alpha-pairs",
        default=None,
        help=(
            "Optional comma-separated edge:sibling pairs. When provided, this "
            "overrides the Cartesian edge/sibling grids."
        ),
    )
    parser.add_argument(
        "--baseline-edge-alpha",
        type=float,
        default=0.001,
        help="Baseline edge alpha for partition transition comparisons.",
    )
    parser.add_argument(
        "--baseline-sibling-alpha",
        type=float,
        default=0.01,
        help="Baseline sibling alpha for partition transition comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/alpha_structure_sweep"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing per-pair metrics and summaries. Transition tables "
            "are complete only for pairs recomputed in this invocation."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AlphaStructureSweepConfig(
        suite=str(args.suite),
        case_names=_parse_csv(args.case_names),
        methods=_parse_csv(args.methods),
        edge_alphas=parse_float_grid(str(args.edge_alphas)),
        sibling_alphas=parse_float_grid(str(args.sibling_alphas)),
        output_dir=args.output_dir,
        resume=bool(args.resume),
        baseline_edge_alpha=float(args.baseline_edge_alpha),
        baseline_sibling_alpha=float(args.baseline_sibling_alpha),
        alpha_pairs=_parse_alpha_pairs(args.alpha_pairs),
    )
    run_alpha_structure_sweep(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
