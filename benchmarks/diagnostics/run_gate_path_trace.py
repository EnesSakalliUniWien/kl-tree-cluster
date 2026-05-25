#!/usr/bin/env python3
"""Trace gate decisions against oracle subtree cuts for classified failures."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_script_path = Path(__file__).resolve()
_benchmarks_root = (
    _script_path.parent if _script_path.parent.name == "benchmarks" else _script_path.parents[1]
)
if str(_benchmarks_root) not in sys.path:
    sys.path.insert(0, str(_benchmarks_root))
from _bootstrap import ensure_repo_root_on_path

repo_root = ensure_repo_root_on_path(__file__)

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.cases.regression_gate import get_regression_gate_test_cases
from benchmarks.diagnostics.gate_path_trace import (
    build_gate_path_trace_dataframe,
    collect_sibling_inflation_trace,
    summarize_gate_path_trace,
)
from benchmarks.shared.kl_tree_context import build_kl_tree_context
from benchmarks.diagnostics.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_OVER_SPLIT,
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE,
    oracle_subtree_cut,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

DEFAULT_FAILURE_CLASSES = (
    FAILURE_CLASS_GATE_OVER_SPLIT,
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build node-level traces that compare actual KL gate decisions "
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
            "Oracle recoverability CSV with failure_class and KL comparison columns. "
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


def _configure_runtime_defaults() -> None:
    for env_var in _THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")
    os.environ.setdefault("KL_TE_N_JOBS", "1")


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
        "kl_ari",
        "kl_found_clusters",
        "true_clusters",
        "oracle_subtree_ari",
        "oracle_true_k_subtree_ari",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Classification CSV {resolved} is missing columns: {sorted(missing)}."
        )
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
        case_id
        for case_id in selected["case_id"].astype(str)
        if case_id not in case_by_name
    ]
    if missing:
        raise ValueError(f"Selected cases are not present in the {len(cases)}-case suite: {missing}.")
    return [
        (case_by_name[str(row.case_id)].copy(), row)
        for row in selected.itertuples(index=False)
    ]


def _make_output_dir(explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        output_dir = explicit_output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = repo_root / "benchmarks" / "results" / f"gate_path_trace_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _trace_case(case: dict[str, object], classification_row) -> pd.DataFrame:
    context = build_kl_tree_context(case, populate_node_distributions=True)
    gate_annotation_bundle = run_gate_annotation_pipeline(
        context.tree,
        context.tree.annotations_df,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    decomposer = TreeDecomposition(
        tree=context.tree,
        gate_annotation_bundle=gate_annotation_bundle,
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
        passthrough=config.PASSTHROUGH,
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
    sibling_inflation_trace = collect_sibling_inflation_trace(
        context.tree,
        gate_annotation_bundle,
        feature_space=context.feature_space,
    )

    trace_df = build_gate_path_trace_dataframe(
        tree=context.tree,
        annotations_df=gate_annotation_bundle.annotated_df,
        decomposition=decomposition,
        oracle_true_k_boundary_nodes=oracle_true_k.selected_nodes,
        oracle_any_k_boundary_nodes=oracle_any.selected_nodes,
        sibling_inflation_trace_by_parent=sibling_inflation_trace,
        case_id=str(context.metadata["name"]),
        failure_class=str(classification_row.failure_class),
        kl_ari=float(classification_row.kl_ari),
        oracle_true_k_ari=float(oracle_true_k.ari),
        oracle_any_k_ari=float(oracle_any.ari),
        passthrough=config.PASSTHROUGH,
    )
    trace_df.insert(2, "tree_distance_metric", context.tree_distance_metric)
    trace_df.insert(3, "tree_distance_source", context.tree_distance_source)
    trace_df.insert(4, "tree_linkage_method", context.tree_linkage_method)
    return trace_df


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
    trace_csv = output_dir / "gate_path_trace.csv"
    summary_csv = output_dir / "gate_path_trace_summary.csv"
    metadata_json = output_dir / "gate_path_trace_metadata.json"

    started_at = time.perf_counter()
    traces: list[pd.DataFrame] = []
    for index, (case, classification_row) in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] {case['name']} "
            f"({classification_row.failure_class})",
            flush=True,
        )
        traces.append(_trace_case(case, classification_row))

    trace_df = pd.concat(traces, ignore_index=True)
    summary_df = summarize_gate_path_trace(trace_df)
    trace_df.to_csv(trace_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "classification_csv": str(classification_path),
        "failure_classes": _parse_csv_list(args.failure_classes),
        "case_names": [str(case["name"]) for case, _row in selected],
        "n_cases": len(selected),
        "elapsed_sec": round(elapsed_sec, 6),
        "trace_csv": str(trace_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Gate-path trace complete in {elapsed_sec:.2f}s")
    print(f"Trace: {trace_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Metadata: {metadata_json}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
