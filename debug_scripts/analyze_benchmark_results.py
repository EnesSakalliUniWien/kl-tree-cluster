#!/usr/bin/env python3
"""Analyze the latest full benchmark results CSV and print summary tables."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def find_latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("run_*"), reverse=True)
    for r in runs:
        csv = r / "full_benchmark_comparison.csv"
        if csv.exists():
            return csv
    raise FileNotFoundError(f"No benchmark CSV found under {results_dir}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    csv_path = (
        sys.argv[1] if len(sys.argv) > 1 else str(find_latest_run(repo / "benchmarks" / "results"))
    )
    print(f"Reading: {csv_path}")

    df = pd.read_csv(csv_path)
    valid = df[df["true_clusters"].notna() & (df["true_clusters"] > 0)].copy()

    print(f"\nTotal rows: {len(df)}, Valid (true_k>0): {len(valid)}")
    methods = sorted(valid["method"].unique())
    print(f"Methods: {methods}")

    rows = []
    for m in methods:
        mv = valid[valid["method"] == m]
        exact = (mv["found_clusters"] == mv["true_clusters"]).sum()
        k1 = (mv["found_clusters"] == 1).sum()
        k1_wrong = ((mv["found_clusters"] == 1) & (mv["true_clusters"] > 1)).sum()
        rows.append(
            {
                "method": m,
                "n_cases": len(mv),
                "mean_ari": mv["ari"].mean(),
                "median_ari": mv["ari"].median(),
                "exact_k": exact,
                "exact_k_pct": exact / len(mv) * 100,
                "k1_total": k1,
                "k1_wrong": k1_wrong,
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_ari", ascending=False)

    print("\n" + "=" * 100)
    print("SUMMARY: All methods ranked by Mean ARI")
    print("=" * 100)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False, float_format="%.4f"))

    # KL-specific detail
    kl = valid[valid["method"] == "kl"]
    if not kl.empty:
        print("\n" + "=" * 100)
        print("KL METHOD — Per-case detail")
        print("=" * 100)
        detail = kl[["test_case", "true_clusters", "found_clusters", "ari"]].copy()
        detail = detail.sort_values("ari")
        detail.columns = ["case", "true_k", "K_found", "ARI"]
        print(detail.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
    main()
