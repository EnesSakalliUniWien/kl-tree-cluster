#!/usr/bin/env python3
"""Run oracle subtree-cut recoverability diagnostics for benchmark trees."""

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
from benchmarks.shared.kl_tree_context import build_kl_tree_context
from benchmarks.shared.oracle_tree_recoverability import (
    classify_tree_recoverability_failure,
    oracle_subtree_cut,
)

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
            "Compute the best ARI recoverable from each KL hierarchy when the "
            "statistical gate decisions are replaced by an oracle subtree cut."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("regression_gate", "full"),
        default="regression_gate",
        help="Benchmark case suite to diagnose.",
    )
    parser.add_argument(
        "--case-names",
        default="",
        help="Optional comma-separated case names from the selected suite.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="Optional KL benchmark CSV to join current KL ARI/found_clusters into the output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/oracle_tree_recoverability_<timestamp>/.",
    )
    parser.add_argument(
        "--solved-ari-threshold",
        type=float,
        default=0.95,
        help="KL ARI at or above this value is classified as solved.",
    )
    parser.add_argument(
        "--recoverable-ari-threshold",
        type=float,
        default=0.8,
        help=(
            "Oracle true-K ARI at or above this value means the tree is treated "
            "as recoverable for failure classification."
        ),
    )
    parser.add_argument(
        "--oracle-gap-tolerance",
        type=float,
        default=1e-9,
        help=(
            "Tolerance for classifying KL as matching the exact-K oracle when "
            "both are below the solved threshold."
        ),
    )
    return parser.parse_args()


def _configure_runtime_defaults() -> None:
    for env_var in _THREAD_ENV_VARS:
        os.environ.setdefault(env_var, "1")
    os.environ.setdefault("KL_TE_N_JOBS", "1")


def _load_cases(suite: str) -> list[dict]:
    if suite == "regression_gate":
        return get_regression_gate_test_cases()
    if suite == "full":
        return get_default_test_cases()
    raise ValueError(f"Unknown suite {suite!r}.")


def _filter_cases(cases: list[dict], raw_case_names: str) -> list[dict]:
    if not raw_case_names.strip():
        return cases
    requested = [part.strip() for part in raw_case_names.split(",") if part.strip()]
    case_by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in requested if name not in case_by_name]
    if missing:
        raise ValueError(f"Unknown case names for selected suite: {missing}.")
    return [case_by_name[name].copy() for name in requested]


def _make_output_dir(explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        output_dir = explicit_output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_dir = repo_root / "benchmarks" / "results" / f"oracle_tree_recoverability_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_kl_benchmark_rows(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method" in df.columns:
        df = df[df["method"].astype(str).eq("kl")].copy()
    return df


def _run_case(case_index: int, case: dict) -> dict[str, object]:
    context = build_kl_tree_context(case, populate_node_distributions=False)

    true_k = int(context.metadata["n_clusters"])
    any_k = oracle_subtree_cut(
        context.tree,
        sample_index=context.data.index,
        true_labels=context.true_labels,
    )
    true_k_cut = oracle_subtree_cut(
        context.tree,
        sample_index=context.data.index,
        true_labels=context.true_labels,
        exact_k=true_k,
    )

    return {
        "test_case": case_index,
        "case_id": str(case["name"]),
        "category": str(context.metadata["category"]),
        "generator": str(context.metadata["generator"]),
        "samples": int(context.metadata["n_samples"]),
        "features": int(context.metadata["n_features"]),
        "true_clusters": true_k,
        "tree_distance_metric": context.tree_distance_metric,
        "tree_distance_source": context.tree_distance_source,
        "tree_linkage_method": context.tree_linkage_method,
        "oracle_subtree_ari": any_k.ari,
        "oracle_subtree_found_clusters": any_k.found_clusters,
        "oracle_subtree_iterations": any_k.dinkelbach_iterations,
        "oracle_true_k_subtree_ari": true_k_cut.ari,
        "oracle_true_k_found_clusters": true_k_cut.found_clusters,
        "oracle_true_k_iterations": true_k_cut.dinkelbach_iterations,
    }


def _attach_benchmark_comparison(
    oracle_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    solved_ari_threshold: float,
    recoverable_ari_threshold: float,
    oracle_gap_tolerance: float,
) -> pd.DataFrame:
    if benchmark_df.empty:
        return oracle_df
    required = {"case_id", "ari", "found_clusters"}
    missing = required - set(benchmark_df.columns)
    if missing:
        raise ValueError(f"benchmark CSV is missing required columns: {sorted(missing)}.")
    comparison = benchmark_df[["case_id", "ari", "found_clusters"]].rename(
        columns={
            "ari": "kl_ari",
            "found_clusters": "kl_found_clusters",
        }
    )
    merged = oracle_df.merge(comparison, on="case_id", how="left", validate="one_to_one")
    missing_comparison = merged["kl_ari"].isna() | merged["kl_found_clusters"].isna()
    if missing_comparison.any():
        missing_cases = merged.loc[missing_comparison, "case_id"].astype(str).tolist()
        raise ValueError(
            "benchmark CSV does not contain matching KL rows for every oracle case. "
            f"Missing={missing_cases[:10]}."
        )
    merged["kl_to_oracle_subtree_gap"] = (
        merged["oracle_subtree_ari"] - merged["kl_ari"]
    )
    merged["kl_to_oracle_true_k_gap"] = (
        merged["oracle_true_k_subtree_ari"] - merged["kl_ari"]
    )
    merged["failure_class"] = [
        classify_tree_recoverability_failure(
            kl_ari=float(row.kl_ari),
            kl_found_clusters=int(row.kl_found_clusters),
            true_clusters=int(row.true_clusters),
            oracle_true_k_subtree_ari=float(row.oracle_true_k_subtree_ari),
            solved_ari_threshold=solved_ari_threshold,
            recoverable_ari_threshold=recoverable_ari_threshold,
            oracle_gap_tolerance=oracle_gap_tolerance,
        )
        for row in merged.itertuples(index=False)
    ]
    return merged


def main() -> None:
    args = _parse_args()
    _configure_runtime_defaults()

    cases = _filter_cases(_load_cases(args.suite), args.case_names)
    output_dir = _make_output_dir(args.output_dir)
    output_csv = output_dir / "oracle_tree_recoverability.csv"
    metadata_json = output_dir / "oracle_tree_recoverability_metadata.json"

    started_at = time.perf_counter()
    rows: list[dict[str, object]] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['name']}", flush=True)
        rows.append(_run_case(i, case))

    oracle_df = pd.DataFrame(rows)
    benchmark_df = _load_kl_benchmark_rows(args.benchmark_csv)
    oracle_df = _attach_benchmark_comparison(
        oracle_df,
        benchmark_df,
        solved_ari_threshold=float(args.solved_ari_threshold),
        recoverable_ari_threshold=float(args.recoverable_ari_threshold),
        oracle_gap_tolerance=float(args.oracle_gap_tolerance),
    )
    oracle_df.to_csv(output_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "n_cases": len(cases),
        "elapsed_sec": round(elapsed_sec, 6),
        "benchmark_csv": str(args.benchmark_csv) if args.benchmark_csv else "",
        "oracle_gap_tolerance": float(args.oracle_gap_tolerance),
        "recoverable_ari_threshold": float(args.recoverable_ari_threshold),
        "results_csv": str(output_csv),
        "solved_ari_threshold": float(args.solved_ari_threshold),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"Oracle tree recoverability complete in {elapsed_sec:.2f}s")
    print(f"Results: {output_csv}")
    print(f"Metadata: {metadata_json}")
    print(
        "oracle_subtree_ari: "
        f"mean={oracle_df['oracle_subtree_ari'].mean():.4f} "
        f"median={oracle_df['oracle_subtree_ari'].median():.4f}"
    )
    print(
        "oracle_true_k_subtree_ari: "
        f"mean={oracle_df['oracle_true_k_subtree_ari'].mean():.4f} "
        f"median={oracle_df['oracle_true_k_subtree_ari'].median():.4f}"
    )
    if "kl_ari" in oracle_df.columns:
        print(
            "kl_to_oracle_subtree_gap: "
            f"mean={oracle_df['kl_to_oracle_subtree_gap'].mean():.4f} "
            f"median={oracle_df['kl_to_oracle_subtree_gap'].median():.4f}"
        )
    if "failure_class" in oracle_df.columns:
        print("failure_class counts:")
        print(oracle_df["failure_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()
