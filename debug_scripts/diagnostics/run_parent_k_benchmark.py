#!/usr/bin/env python3
"""Run a quick tree-method benchmark summary for a narrow method set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.shared.cases import get_default_test_cases  # noqa: E402
from benchmarks.shared.pipeline import benchmark_cluster_algorithm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["kl"],
        help="Methods to benchmark (default: kl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = get_default_test_cases()
    print(f"Running {len(cases)} cases with methods={args.methods}...", flush=True)
    df, _ = benchmark_cluster_algorithm(test_cases=cases, methods=args.methods)

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Cases: {len(df)}")
    print(f"Mean ARI: {df['ARI'].mean():.3f}")
    print(f"Median ARI: {df['ARI'].median():.3f}")
    exact_k = (df["Found"] == df["True"]).sum()
    print(f"Exact K: {exact_k}/{len(df)} ({100 * exact_k / len(df):.1f}%)")
    under = (df["Found"] < df["True"]).sum()
    over = (df["Found"] > df["True"]).sum()
    k1 = (df["Found"] == 1).sum()
    print(f"Under-split: {under}/{len(df)} ({100 * under / len(df):.1f}%)")
    print(f"Over-split: {over}/{len(df)} ({100 * over / len(df):.1f}%)")
    print(f"K=1 collapses: {k1}")

    for category, group_df in df.groupby("Case_Category"):
        exact = int((group_df["Found"] == group_df["True"]).sum())
        print(f"  {category}: ARI={group_df['ARI'].mean():.3f}, exactK={exact}/{len(group_df)}")

    print("\nWorst 15 cases (by ARI):")
    worst = df.nsmallest(15, "ARI")[["Case_Name", "True", "Found", "ARI"]]
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
