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

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

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
from benchmarks.shared.oracle_tree_recoverability import oracle_subtree_cut
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.method_execution import (
    KL_TREE_DISTANCE_SOURCE_FEATURE_METRIC,
    KL_TREE_DISTANCE_SOURCE_PRECOMPUTED,
    _require_precomputed_kl_distance_metric,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

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


def _resolve_kl_tree_distance(
    *,
    data_t: pd.DataFrame,
    meta: dict[str, object],
    distance_condensed: np.ndarray | None,
    precomputed_distance_condensed: object,
) -> tuple[np.ndarray, str, str, str]:
    params = METHOD_SPECS["kl"].param_grid[0]
    tree_linkage_method = str(params["tree_linkage_method"])
    configured_metric = str(params["tree_distance_metric"])

    requires_precomputed = bool(meta["requires_precomputed_kl_distance"])
    if requires_precomputed or precomputed_distance_condensed is not None:
        if distance_condensed is None:
            raise ValueError(
                f"Case '{meta['name']}' requires/provides precomputed KL distance, "
                "but no condensed distance was prepared."
            )
        return (
            np.asarray(distance_condensed, dtype=float),
            _require_precomputed_kl_distance_metric(meta=meta, case_name=str(meta["name"])),
            KL_TREE_DISTANCE_SOURCE_PRECOMPUTED,
            tree_linkage_method,
        )

    return (
        pdist(data_t.values, metric=configured_metric),
        configured_metric,
        KL_TREE_DISTANCE_SOURCE_FEATURE_METRIC,
        tree_linkage_method,
    )


def _run_case(case_index: int, case: dict) -> dict[str, object]:
    (
        data_t,
        y_t,
        _x_original,
        meta,
        distance_condensed,
        _distance_matrix,
        precomputed_distance_condensed,
    ) = prepare_case_inputs(case, ["kl"])
    (
        distance_for_tree,
        tree_distance_metric,
        tree_distance_source,
        tree_linkage_method,
    ) = _resolve_kl_tree_distance(
        data_t=data_t,
        meta=meta,
        distance_condensed=distance_condensed,
        precomputed_distance_condensed=precomputed_distance_condensed,
    )
    linkage_matrix = linkage(distance_for_tree, method=tree_linkage_method)
    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_t.index.tolist())

    true_k = int(meta["n_clusters"])
    any_k = oracle_subtree_cut(tree, sample_index=data_t.index, true_labels=np.asarray(y_t))
    true_k_cut = oracle_subtree_cut(
        tree,
        sample_index=data_t.index,
        true_labels=np.asarray(y_t),
        exact_k=true_k,
    )

    return {
        "test_case": case_index,
        "case_id": str(case["name"]),
        "category": str(meta["category"]),
        "generator": str(meta["generator"]),
        "samples": int(meta["n_samples"]),
        "features": int(meta["n_features"]),
        "true_clusters": true_k,
        "tree_distance_metric": tree_distance_metric,
        "tree_distance_source": tree_distance_source,
        "tree_linkage_method": tree_linkage_method,
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
    merged["kl_to_oracle_subtree_gap"] = (
        merged["oracle_subtree_ari"] - merged["kl_ari"]
    )
    merged["kl_to_oracle_true_k_gap"] = (
        merged["oracle_true_k_subtree_ari"] - merged["kl_ari"]
    )
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
    oracle_df = _attach_benchmark_comparison(oracle_df, benchmark_df)
    oracle_df.to_csv(output_csv, index=False)

    elapsed_sec = time.perf_counter() - started_at
    metadata = {
        "suite": args.suite,
        "n_cases": len(cases),
        "elapsed_sec": round(elapsed_sec, 6),
        "benchmark_csv": str(args.benchmark_csv) if args.benchmark_csv else "",
        "results_csv": str(output_csv),
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


if __name__ == "__main__":
    main()
