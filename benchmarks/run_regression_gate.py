#!/usr/bin/env python3
"""Run the fast regression-gate benchmark suite.

The gate is a fixed, historically sensitive subset of the full benchmark and
is intended for fast regression detection rather than exhaustive evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Load shared path bootstrap helper from benchmarks root.
_script_path = Path(__file__).resolve()
_benchmarks_root = (
    _script_path.parent if _script_path.parent.name == "benchmarks" else _script_path.parents[1]
)
if str(_benchmarks_root) not in sys.path:
    sys.path.insert(0, str(_benchmarks_root))
from _bootstrap import ensure_repo_root_on_path

# Ensure repo root is importable before any benchmarks/kl_clustering imports.
repo_root = ensure_repo_root_on_path(__file__)

from benchmarks.shared.cases.regression_gate import (
    get_regression_gate_case_names,
    get_regression_gate_test_cases,
)
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.relationship_analysis import normalize_results_dataframe
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.time import format_timestamp_utc

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed regression-gate benchmark suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to benchmarks/results/regression_gate_<timestamp>/",
    )
    parser.add_argument(
        "--methods",
        default="kl",
        help="Comma-separated method ids. Defaults to 'kl'.",
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Optional comma-separated override of case names from the regression gate.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the regression-gate case names and exit.",
    )
    return parser.parse_args()


def _configure_runtime_defaults() -> None:
    for env_var in _THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")
    os.environ.setdefault("KL_TE_N_JOBS", "1")


def _parse_methods(raw: str) -> list[str]:
    methods = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one method id must be provided.")

    unknown = [method for method in methods if method not in METHOD_SPECS]
    if unknown:
        available = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"Unknown methods: {unknown}. Available: {available}")
    return methods


def _resolve_case_list(raw_case_names: str) -> list[dict]:
    default_cases = get_regression_gate_test_cases()
    if not raw_case_names.strip():
        return default_cases

    requested_names = [part.strip() for part in raw_case_names.split(",") if part.strip()]
    if not requested_names:
        raise ValueError("--case-names was provided but no valid case names were parsed.")

    case_index = {case["name"]: case for case in default_cases}
    missing = [name for name in requested_names if name not in case_index]
    if missing:
        available = ", ".join(get_regression_gate_case_names())
        raise ValueError(
            "Unknown regression-gate case names: "
            + ", ".join(missing)
            + f". Available: {available}"
        )
    return [case_index[name].copy() for name in requested_names]


def _normalize_results(df_results: pd.DataFrame) -> pd.DataFrame:
    df_out = normalize_results_dataframe(df_results)

    name_map = {spec.name: method_id for method_id, spec in METHOD_SPECS.items()}
    df_out["method"] = df_out["method"].map(lambda value: name_map.get(str(value), str(value)))

    return df_out


def _sort_results(
    df_results: pd.DataFrame,
    test_cases: list[dict],
    methods: list[str],
) -> pd.DataFrame:
    case_order = {case["name"]: index for index, case in enumerate(test_cases)}
    method_order = {method: index for index, method in enumerate(methods)}

    df_sorted = df_results.copy()
    df_sorted["_case_order"] = df_sorted["case_id"].map(case_order)
    df_sorted["_method_order"] = df_sorted["method"].map(method_order)
    df_sorted = df_sorted.sort_values(
        by=["_case_order", "_method_order", "test_case"],
        kind="stable",
    )
    return df_sorted.drop(columns=["_case_order", "_method_order"])


def _print_summary(df_results: pd.DataFrame, methods: list[str]) -> None:
    print(f"Methods: {methods}")
    print(f"Rows: {len(df_results)}")

    for method in methods:
        rows = df_results[df_results["method"] == method].copy()
        if rows.empty:
            continue

        valid_truth = rows["true_clusters"] > 0
        exact_k = int((rows.loc[valid_truth, "found_clusters"] == rows.loc[valid_truth, "true_clusters"]).sum())
        ari = rows["ari"].dropna()
        mean_ari = float(ari.mean()) if not ari.empty else float("nan")
        median_ari = float(ari.median()) if not ari.empty else float("nan")
        print(
            f"{method}: mean_ari={mean_ari:.4f} median_ari={median_ari:.4f} "
            f"exact_k={exact_k}/{int(valid_truth.sum())}"
        )


def main() -> None:
    args = _parse_args()
    _configure_runtime_defaults()

    if args.list_cases:
        for name in get_regression_gate_case_names():
            print(name)
        return

    methods = _parse_methods(args.methods)
    test_cases = _resolve_case_list(args.case_names)

    timestamp = format_timestamp_utc()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else repo_root / "benchmarks" / "results" / f"regression_gate_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "regression_gate_comparison.csv"
    metadata_json = output_dir / "regression_gate_metadata.json"

    started_at = time.perf_counter()
    df_results, _ = benchmark_cluster_algorithm(
        test_cases=test_cases,
        verbose=False,
        plot_umap=False,
        plot_manifold=False,
        methods=methods,
        concat_plots_pdf=False,
    )
    elapsed_sec = time.perf_counter() - started_at

    normalized = _normalize_results(df_results)
    normalized = _sort_results(normalized, test_cases, methods)
    normalized.to_csv(output_csv, index=False)

    metadata = {
        "suite": "regression_gate",
        "generated_at_utc": timestamp,
        "elapsed_sec": round(elapsed_sec, 6),
        "methods": methods,
        "case_names": [case["name"] for case in test_cases],
        "n_cases": len(test_cases),
        "n_rows": len(normalized),
        "kl_te_n_jobs": os.environ.get("KL_TE_N_JOBS"),
        "thread_env": {env_var: os.environ.get(env_var) for env_var in _THREAD_ENV_VARS},
        "results_csv": str(output_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Regression gate complete in {elapsed_sec:.2f}s")
    print(f"Results: {output_csv}")
    print(f"Metadata: {metadata_json}")
    _print_summary(normalized, methods)


if __name__ == "__main__":
    main()
