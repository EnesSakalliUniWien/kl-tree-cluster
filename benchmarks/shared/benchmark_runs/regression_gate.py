"""Regression-gate benchmark execution module."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.benchmark_runs.runtime import (
    THREAD_ENV_VARS,
    apply_single_thread_runtime_defaults,
)
from benchmarks.shared.cases.regression_gate import (
    get_regression_gate_case_names,
    get_regression_gate_test_cases,
)
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.relationship_analysis import normalize_results_dataframe
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.time import format_timestamp_utc

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RegressionGateRunResult:
    """Artifacts produced by one regression-gate benchmark run."""

    output_dir: Path
    results_csv: Path
    metadata_json: Path
    elapsed_sec: float
    results: pd.DataFrame


def parse_methods(raw: str) -> list[str]:
    """Parse and validate comma-separated benchmark method IDs."""
    methods = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one method id must be provided.")

    unknown = [method for method in methods if method not in METHOD_SPECS]
    if unknown:
        available = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"Unknown methods: {unknown}. Available: {available}")
    return methods


def resolve_case_list(raw_case_names: str) -> list[dict]:
    """Resolve an optional comma-separated case list inside the regression gate."""
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


def normalize_regression_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """Normalize benchmark rows and use method IDs instead of display names."""
    df_out = normalize_results_dataframe(df_results)

    name_map = {spec.name: method_id for method_id, spec in METHOD_SPECS.items()}
    df_out["method"] = df_out["method"].map(lambda value: name_map.get(str(value), str(value)))

    return df_out


def sort_regression_results(
    df_results: pd.DataFrame,
    test_cases: list[dict],
    methods: list[str],
) -> pd.DataFrame:
    """Sort regression rows by configured case and method order."""
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


def print_regression_summary(df_results: pd.DataFrame, methods: list[str]) -> None:
    """Print a compact quality summary for selected regression methods."""
    print(f"Methods: {methods}")
    print(f"Rows: {len(df_results)}")

    for method in methods:
        rows = df_results[df_results["method"] == method].copy()
        if rows.empty:
            continue

        valid_truth = rows["true_clusters"] > 0
        exact_k = int(
            (
                rows.loc[valid_truth, "found_clusters"] == rows.loc[valid_truth, "true_clusters"]
            ).sum()
        )
        ari = rows["ari"].dropna()
        mean_ari = float(ari.mean()) if not ari.empty else float("nan")
        median_ari = float(ari.median()) if not ari.empty else float("nan")
        print(
            f"{method}: mean_ari={mean_ari:.4f} median_ari={median_ari:.4f} "
            f"exact_k={exact_k}/{int(valid_truth.sum())}"
        )


def run_regression_gate(
    *,
    methods: list[str],
    test_cases: list[dict],
    output_dir: Path | None = None,
) -> RegressionGateRunResult:
    """Run the fixed regression-gate benchmark suite and write canonical artifacts."""
    apply_single_thread_runtime_defaults()
    timestamp = format_timestamp_utc()
    resolved_output_dir = (
        output_dir
        if output_dir is not None
        else REPO_ROOT / "benchmarks" / "results" / f"regression_gate_{timestamp}"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = resolved_output_dir / "regression_gate_comparison.csv"
    metadata_json = resolved_output_dir / "regression_gate_metadata.json"

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

    normalized = normalize_regression_results(df_results)
    normalized = sort_regression_results(normalized, test_cases, methods)
    normalized.to_csv(output_csv, index=False)

    metadata = {
        "suite": "regression_gate",
        "generated_at_utc": timestamp,
        "elapsed_sec": round(elapsed_sec, 6),
        "methods": methods,
        "case_names": [case["name"] for case in test_cases],
        "n_cases": len(test_cases),
        "n_rows": len(normalized),
        "tbs_n_jobs": os.environ.get("TBS_N_JOBS"),
        "thread_env": {env_var: os.environ.get(env_var) for env_var in THREAD_ENV_VARS},
        "results_csv": str(output_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return RegressionGateRunResult(
        output_dir=resolved_output_dir,
        results_csv=output_csv,
        metadata_json=metadata_json,
        elapsed_sec=elapsed_sec,
        results=normalized,
    )


__all__ = [
    "RegressionGateRunResult",
    "normalize_regression_results",
    "parse_methods",
    "print_regression_summary",
    "resolve_case_list",
    "run_regression_gate",
    "sort_regression_results",
]
