#!/usr/bin/env python3
"""Trace gate decisions against oracle subtree cuts for classified failures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.oracle.gate_path_trace import (
    build_prepared_gate_path_trace,
    collect_sibling_inflation_trace,
    prepare_gate_path_case,
    summarize_gate_path_trace,
)
from benchmarks.diagnostics.oracle.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_OVER_SPLIT,
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE,
)
from benchmarks.diagnostics.oracle.statistical_decision_trace import (
    statistical_decision_trace_from_gate_path_trace,
)
from benchmarks.diagnostics.runner_support import (
    configure_serial_runtime,
    create_result_directory,
    parse_csv_values,
    resolve_classified_cases,
)

DEFAULT_FAILURE_CLASSES = (
    FAILURE_CLASS_GATE_OVER_SPLIT,
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build node-level traces that compare actual TBS gate decisions "
            "against oracle subtree-cut boundaries."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("regression_gate", "full"),
        default="full",
        help="Benchmark case suite containing the classified case names.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=None,
        help=(
            "Oracle recoverability CSV with failure_class and TBS comparison columns. "
            "Defaults to the latest oracle_tree_recoverability CSV containing "
            "failure_class."
        ),
    )
    parser.add_argument(
        "--failure-classes",
        default=",".join(DEFAULT_FAILURE_CLASSES),
        help="Comma-separated failure classes to trace.",
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Optional comma-separated case name override after failure-class filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/gate_path_trace_<timestamp>/.",
    )
    return parser.parse_args()


def _trace_case(
    case: dict[str, object],
    classification: dict[str, object],
) -> pd.DataFrame:
    prepared = prepare_gate_path_case(case)
    context = prepared.context
    sibling_inflation_trace = collect_sibling_inflation_trace(
        context.tree,
        prepared.gate_annotation_bundle,
        feature_space=context.feature_space,
    )
    return build_prepared_gate_path_trace(
        prepared,
        classification,
        sibling_inflation_trace_by_parent=sibling_inflation_trace,
    )


def main() -> None:
    args = _parse_args()
    configure_serial_runtime()

    selection = resolve_classified_cases(
        suite=args.suite,
        classification_csv=args.classification_csv,
        failure_classes=parse_csv_values(args.failure_classes),
        case_names=parse_csv_values(args.case_names),
        required_columns=(
            "tbs_ari",
            "tbs_found_clusters",
            "true_clusters",
            "oracle_subtree_ari",
            "oracle_true_k_subtree_ari",
        ),
    )
    output_dir = create_result_directory(
        args.output_dir,
        study_slug="gate_path_trace",
    )
    trace_csv = output_dir / "gate_path_trace.csv"
    statistical_trace_csv = output_dir / "statistical_decision_trace.csv"
    summary_csv = output_dir / "gate_path_trace_summary.csv"
    metadata_json = output_dir / "gate_path_trace_metadata.json"

    started_at = time.perf_counter()
    traces: list[pd.DataFrame] = []
    for index, selected_case in enumerate(selection.cases, 1):
        case = selected_case.case
        classification = selected_case.classification
        print(
            f"[{index}/{len(selection.cases)}] "
            f"{case['name']} ({classification['failure_class']})",
            flush=True,
        )
        traces.append(_trace_case(case, dict(classification)))

    trace_df = pd.concat(traces, ignore_index=True)
    statistical_trace_df = statistical_decision_trace_from_gate_path_trace(trace_df)
    summary_df = summarize_gate_path_trace(trace_df)
    trace_df.to_csv(trace_csv, index=False)
    statistical_trace_df.to_csv(statistical_trace_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "classification_csv": str(selection.classification_path),
        "failure_classes": list(parse_csv_values(args.failure_classes)),
        "case_names": [str(item.case["name"]) for item in selection.cases],
        "n_cases": len(selection.cases),
        "elapsed_sec": round(elapsed_sec, 6),
        "trace_csv": str(trace_csv),
        "statistical_trace_csv": str(statistical_trace_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Gate-path trace complete in {elapsed_sec:.2f}s")
    print(f"Trace: {trace_csv}")
    print(f"Statistical trace: {statistical_trace_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Metadata: {metadata_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
