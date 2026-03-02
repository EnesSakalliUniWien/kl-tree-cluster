"""Run the full benchmark suite with per-case process isolation.

Why this script exists
----------------------
The default in-process full benchmark can hit native segmentation faults in long
runs on some environments. This script runs each case in an isolated subprocess,
so a single-case crash does not kill the whole run.

Usage
-----
    /Users/berksakalli/Projects/kl-te-cluster/.venv/bin/python \
        scripts/run_full_benchmark_isolated.py

Optional flags
--------------
    --timeout-sec 3600       Timeout per case (default: 3600)
    --retry-count 2          Retries per failed case (default: 2)
    --methods kl,kmeans      Comma-separated method IDs
    --run-dir <path>         Explicit output directory
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_execution import run_case_isolated


def _parse_methods(raw: str | None) -> list[str]:
    if not raw:
        return list(METHOD_SPECS.keys())
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown method IDs: {unknown}. Valid IDs: {sorted(METHOD_SPECS)}")
    return methods


def _make_run_dir(explicit_dir: str | None) -> Path:
    if explicit_dir:
        run_dir = Path(explicit_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        run_dir = Path("benchmarks/results") / f"run_{stamp}_isolated_cases"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resume_coverage(
    df_existing: pd.DataFrame,
    expected_methods: list[str],
) -> tuple[set[int], dict[int, list[str]], int]:
    if df_existing.empty:
        return set(), {}, 0

    case_col = "test_case" if "test_case" in df_existing.columns else "Test"
    if case_col not in df_existing.columns:
        return set(), {}, 0

    method_col = "method" if "method" in df_existing.columns else "Method"
    if method_col not in df_existing.columns:
        return set(), {}, 0

    expected_set = set(expected_methods)
    name_to_id = {spec.name: method_id for method_id, spec in METHOD_SPECS.items()}

    case_keys = pd.to_numeric(df_existing[case_col], errors="coerce")
    methods_raw = df_existing[method_col].astype(str)

    def _normalize_method(value: str) -> str | None:
        if value in expected_set:
            return value
        mapped = name_to_id.get(value)
        if mapped in expected_set:
            return mapped
        return None

    methods_norm = methods_raw.map(_normalize_method)
    progress = pd.DataFrame({"case_key": case_keys, "method_id": methods_norm})
    progress = progress.dropna(subset=["case_key", "method_id"])
    if progress.empty:
        return set(), {}, 0

    progress["case_key"] = progress["case_key"].astype(int)
    methods_by_case = (
        progress.groupby("case_key")["method_id"].agg(lambda s: set(s.tolist())).to_dict()
    )

    completed_cases: set[int] = set()
    missing_methods_by_case: dict[int, list[str]] = {}
    for case_key, seen_methods in methods_by_case.items():
        missing = [method_id for method_id in expected_methods if method_id not in seen_methods]
        if not missing:
            completed_cases.add(int(case_key))
        else:
            missing_methods_by_case[int(case_key)] = missing

    return completed_cases, missing_methods_by_case, len(methods_by_case)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark with per-case isolation.")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    test_cases = get_default_test_cases()

    run_dir = _make_run_dir(args.run_dir)
    out_csv = run_dir / "full_benchmark_comparison.csv"

    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        done, missing_methods_by_case, tracked_case_count = _resume_coverage(existing_df, methods)
        partial_case_count = max(0, tracked_case_count - len(done))
        print(f"Resuming existing run: {run_dir}")
        print(f"Fully completed cases: {len(done)}")
        print(f"Partial cases to resume: {partial_case_count}")
    else:
        done = set()
        missing_methods_by_case = {}
        print(f"Starting new isolated run: {run_dir}")

    all_rows: list[pd.DataFrame] = []

    for i, case in enumerate(test_cases, 1):
        case_key = int(case.get("test_case_num", i))
        case_id = str(case.get("name", case.get("id", f"case_{i}")))
        if case_key in done:
            print(f"[{i}/{len(test_cases)}] Skip {case_id} (already completed)")
            continue

        methods_for_case = missing_methods_by_case.get(case_key, methods)
        if case_key in missing_methods_by_case:
            print(
                f"[{i}/{len(test_cases)}] Resume partial {case_id} "
                f"(missing methods: {methods_for_case})",
                flush=True,
            )

        print(f"[{i}/{len(test_cases)}] Running {case_id}", flush=True)
        case_with_num = dict(case)
        case_with_num["test_case_num"] = case_key

        case_df = None
        last_exc: Exception | None = None
        for attempt in range(1, max(1, args.retry_count) + 2):
            try:
                case_df = run_case_isolated(
                    case=case_with_num,
                    methods_to_test=methods_for_case,
                    case_plot_umap=False,
                    case_plot_manifold=False,
                    enable_plots=False,
                    pdf_path=None,
                    timeout_sec=max(1, args.timeout_sec),
                )
                break
            except Exception as exc:
                last_exc = exc
                print(f"  attempt {attempt}/{args.retry_count + 1} failed: {exc}", flush=True)

        if case_df is None:
            print(f"  FAILED after retries: {last_exc}", flush=True)
            continue

        all_rows.append(case_df)
        write_header = not out_csv.exists()
        case_df.to_csv(out_csv, mode="a", header=write_header, index=False)

        if not case_df.empty and "Method" in case_df.columns and "ARI" in case_df.columns:
            for method_id in methods_for_case:
                method_name = METHOD_SPECS[method_id].name
                row = case_df[case_df["Method"] == method_name]
                if not row.empty:
                    ari = float(row["ARI"].iloc[0])
                    print(f"    {method_id}: ARI={ari:.4f}", flush=True)

    print("-" * 60)
    if not out_csv.exists():
        print("No results were written.")
        return

    final_df = pd.read_csv(out_csv)
    print("Isolated benchmark run complete.")
    print(f"CSV: {out_csv}")

    if "Method" in final_df.columns and "ARI" in final_df.columns:
        print("\nMean ARI by Method:")
        print(final_df.groupby("Method")["ARI"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
