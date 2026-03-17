#!/usr/bin/env python3
"""Run the lab baseline and compare with the most recent full benchmark.

Runs all three case tiers (FAILURE, INTERMEDIATE, GUARD) with current config,
then compares against the benchmark CSV to identify regressions.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from debug_scripts.enhancement_lab.lab_helpers import (
    FAILURE_CASES,
    INTERMEDIATE_CASES,
    REGRESSION_GUARD_CASES,
    print_summary,
    run_case_battery,
)


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("  LAB BASELINE: running all 3 tiers with current config")
    print("=" * 72)

    # 1. Regression guards (most important — must stay perfect)
    print("\n=== REGRESSION GUARD CASES (25) — must remain ARI=1.0 ===")
    df_guard = run_case_battery(REGRESSION_GUARD_CASES, label="guard")
    print_summary(df_guard)
    regressions = df_guard[(df_guard["ari"].notna()) & (df_guard["ari"] < 1.0)]
    if len(regressions):
        print(f"\n*** {len(regressions)} REGRESSIONS detected! ***")
        print(regressions[["case", "true_k", "found_k", "ari"]].to_string(index=False))
    else:
        print("\n  All guards PASS.")

    # 2. Failure cases
    print("\n=== FAILURE CASES (18) — current failures ===")
    df_fail = run_case_battery(FAILURE_CASES, label="failure")
    print_summary(df_fail)

    # 3. Intermediate cases
    print("\n=== INTERMEDIATE CASES (30) — partial successes ===")
    df_inter = run_case_battery(INTERMEDIATE_CASES, label="intermediate")
    print_summary(df_inter)

    # 4. Aggregate stats
    elapsed = time.time() - t0
    all_df = pd.concat([df_guard, df_fail, df_inter], ignore_index=True)
    all_df = all_df[all_df["ari"].notna()]
    print(f"\n{'=' * 72}")
    print(f"  AGGREGATE ({len(all_df)} cases, {elapsed:.0f}s)")
    print(f"{'=' * 72}")
    print(f"  Mean ARI:   {all_df['ari'].mean():.3f}")
    print(f"  Median ARI: {all_df['ari'].median():.3f}")
    print(f"  Exact K:    {(all_df['delta_k'] == 0).sum()}/{len(all_df)}")
    print(f"  K=1 (collapse): {(all_df['found_k'] == 1).sum()}")
    print(f"  Over-split (delta_k > 2): {(all_df['delta_k'] > 2).sum()}")
    print(f"  Under-split (delta_k < -1): {(all_df['delta_k'] < -1).sum()}")

    # 5. Compare with Feb 15 benchmark baseline (if available)
    csv_latest = sorted(Path("benchmarks/results").glob("run_*/full_benchmark_comparison.csv"))
    if csv_latest:
        bench = pd.read_csv(csv_latest[-1])
        kl_bench = bench[bench["method"] == "kl"]
        print(f"\n--- Comparison with benchmark ({csv_latest[-1].parent.name}) ---")
        print(f"  Benchmark KL mean ARI: {kl_bench['ari'].mean():.3f}")
        print(
            f"  Benchmark KL exact K:  {(kl_bench['found_clusters'] == kl_bench['true_clusters']).sum()}/{len(kl_bench)}"
        )


if __name__ == "__main__":
    main()
