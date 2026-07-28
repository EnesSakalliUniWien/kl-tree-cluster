#!/usr/bin/env python3
"""Diagnose empirical-null inflation on oracle-recoverable sibling blockers."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[3]

from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition

from benchmarks.diagnostics.calibration.sibling.nulls.sibling_inflation_diagnostic import (
    build_sibling_inflation_diagnostic_tables,
    collect_sibling_inflation_inputs,
)
from benchmarks.diagnostics.oracle.gate_path_trace import build_gate_path_trace_dataframe
from benchmarks.diagnostics.oracle.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    oracle_subtree_cut,
)
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.cases.regression_gate import get_regression_gate_test_cases
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build target and calibration-contributor tables for the sibling "
            "empirical-null inflation estimator."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("regression_gate", "full"),
        default="full",
        help="Benchmark suite containing the classified case names.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=None,
        help=(
            "Oracle recoverability CSV with failure_class. Defaults to the latest "
            "oracle_tree_recoverability CSV containing failure_class."
        ),
    )
    parser.add_argument(
        "--failure-classes",
        default=FAILURE_CLASS_GATE_UNDER_SPLIT,
        help="Comma-separated failure classes to diagnose.",
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Optional comma-separated case names after failure-class filtering.",
    )
    parser.add_argument(
        "--max-contributors",
        type=int,
        default=10,
        help="Maximum local calibration contributor rows per target.",
    )
    parser.add_argument(
        "--contributors-for-all-crossings",
        action="store_true",
        help=(
            "Emit contributor rows for every focal test whose raw p-value crosses "
            "alpha after inflation or sibling FDR, not only oracle blocker nodes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/sibling_inflation_diagnostic_<timestamp>/."
        ),
    )
    return parser.parse_args()


def _configure_runtime_defaults() -> None:
    for env_var in _THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")
    os.environ.setdefault("TBS_N_JOBS", "1")


def _load_cases(suite: str) -> list[dict[str, object]]:
    if suite == "regression_gate":
        return get_regression_gate_test_cases()
    if suite == "full":
        return get_default_test_cases()
    raise ValueError(f"Unknown suite {suite!r}.")


def _latest_classification_csv() -> Path:
    candidates = sorted(
        repo_root.glob(
            "benchmarks/results/oracle_tree_recoverability_*/oracle_tree_recoverability.csv"
        )
    )
    for candidate in reversed(candidates):
        columns = pd.read_csv(candidate, nrows=0).columns
        if "failure_class" in columns:
            return candidate
    raise FileNotFoundError(
        "No oracle_tree_recoverability CSV with a failure_class column was found."
    )


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_classification(path: Path | None) -> tuple[pd.DataFrame, Path]:
    resolved = _latest_classification_csv() if path is None else path
    df = pd.read_csv(resolved)
    required = {
        "case_id",
        "failure_class",
        "tbs_ari",
        "oracle_true_k_subtree_ari",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Classification CSV {resolved} is missing columns: {sorted(missing)}.")
    return df, resolved


def _select_cases(
    cases: list[dict[str, object]],
    classification_df: pd.DataFrame,
    *,
    failure_classes: list[str],
    case_names: list[str],
) -> list[tuple[dict[str, object], pd.Series]]:
    case_by_name = {str(case["name"]): case for case in cases}
    selected = classification_df[
        classification_df["failure_class"].astype(str).isin(failure_classes)
    ].copy()
    if case_names:
        requested = set(case_names)
        selected = selected[selected["case_id"].astype(str).isin(requested)]
    if selected.empty:
        raise ValueError("No cases matched the requested failure class/name filters.")

    missing = [
        case_id for case_id in selected["case_id"].astype(str) if case_id not in case_by_name
    ]
    if missing:
        raise ValueError(
            f"Selected cases are not present in the {len(cases)}-case suite: {missing}."
        )
    return [
        (case_by_name[str(row.case_id)].copy(), row) for row in selected.itertuples(index=False)
    ]


def _make_output_dir(explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        output_dir = explicit_output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = repo_root / "benchmarks" / "results" / f"sibling_inflation_diagnostic_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _diagnose_case(
    case: dict[str, object],
    classification_row,
    *,
    max_contributors: int,
    contributors_for_all_crossings: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = build_tbs_tree_context(case, populate_node_distributions=True)
    gate_annotation_bundle = run_gate_annotation_pipeline(
        context.tree,
        context.tree.annotations_df,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    decomposer = TreeDecomposition(
        tree=context.tree,
        gate_annotation_bundle=gate_annotation_bundle,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
        passthrough=True,
        trace_level="full",
    )
    decomposition = decomposer.decompose_tree()

    true_k = int(context.metadata["n_clusters"])
    oracle_any = oracle_subtree_cut(
        context.tree,
        sample_index=context.data.index,
        true_labels=context.true_labels,
    )
    oracle_true_k = oracle_subtree_cut(
        context.tree,
        sample_index=context.data.index,
        true_labels=context.true_labels,
        exact_k=true_k,
    )
    trace_df = build_gate_path_trace_dataframe(
        tree=context.tree,
        annotations_df=gate_annotation_bundle.annotated_df,
        decomposition=decomposition,
        oracle_true_k_boundary_nodes=oracle_true_k.selected_nodes,
        oracle_any_k_boundary_nodes=oracle_any.selected_nodes,
        sibling_inflation_trace_by_parent={},
        case_id=str(context.metadata["name"]),
        failure_class=str(classification_row.failure_class),
        tbs_ari=float(classification_row.tbs_ari),
        oracle_true_k_ari=float(oracle_true_k.ari),
        oracle_any_k_ari=float(oracle_any.ari),
        passthrough=True,
    )

    inputs = collect_sibling_inflation_inputs(
        context.tree,
        gate_annotation_bundle,
        feature_space=context.feature_space,
    )
    if inputs.model is None:
        raise ValueError(
            f"Case {context.metadata['name']!r} has no focal sibling records to diagnose."
        )
    tables = build_sibling_inflation_diagnostic_tables(
        records=inputs.records,
        model=inputs.model,
        trace_df=trace_df,
        sibling_alpha=DEFAULT_SIBLING_ALPHA,
        max_contributors=max_contributors,
        contributors_for_all_crossings=contributors_for_all_crossings,
    )
    for frame in (tables.targets, tables.contributors, tables.summary):
        if frame.empty:
            continue
        frame.insert(2, "tree_distance_metric", context.tree_distance_metric)
        frame.insert(3, "tree_distance_source", context.tree_distance_source)
        frame.insert(4, "tree_linkage_method", context.tree_linkage_method)
    return tables.targets, tables.contributors, tables.summary


def main() -> None:
    args = _parse_args()
    _configure_runtime_defaults()

    classification_df, classification_path = _load_classification(args.classification_csv)
    selected = _select_cases(
        _load_cases(args.suite),
        classification_df,
        failure_classes=_parse_csv_list(args.failure_classes),
        case_names=_parse_csv_list(args.case_names),
    )
    output_dir = _make_output_dir(args.output_dir)
    targets_csv = output_dir / "sibling_inflation_targets.csv"
    contributors_csv = output_dir / "sibling_inflation_contributors.csv"
    summary_csv = output_dir / "sibling_inflation_summary.csv"
    metadata_json = output_dir / "sibling_inflation_metadata.json"

    started_at = time.perf_counter()
    target_frames: list[pd.DataFrame] = []
    contributor_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for index, (case, classification_row) in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {case['name']} ({classification_row.failure_class})",
            flush=True,
        )
        targets, contributors, summary = _diagnose_case(
            case,
            classification_row,
            max_contributors=int(args.max_contributors),
            contributors_for_all_crossings=bool(args.contributors_for_all_crossings),
        )
        target_frames.append(targets)
        contributor_frames.append(contributors)
        summary_frames.append(summary)

    targets_df = pd.concat(target_frames, ignore_index=True)
    contributors_df = pd.concat(contributor_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    targets_df.to_csv(targets_csv, index=False)
    contributors_df.to_csv(contributors_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "classification_csv": str(classification_path),
        "failure_classes": _parse_csv_list(args.failure_classes),
        "case_names": [str(case["name"]) for case, _row in selected],
        "n_cases": len(selected),
        "sibling_alpha": float(DEFAULT_SIBLING_ALPHA),
        "max_contributors": int(args.max_contributors),
        "contributors_for_all_crossings": bool(args.contributors_for_all_crossings),
        "elapsed_sec": round(elapsed_sec, 6),
        "targets_csv": str(targets_csv),
        "contributors_csv": str(contributors_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Sibling inflation diagnostic complete in {elapsed_sec:.2f}s")
    print(f"Targets: {targets_csv}")
    print(f"Contributors: {contributors_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Metadata: {metadata_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
