#!/usr/bin/env python3
"""Analyze tree-method benchmark results with a focus on `kl_diffusion`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CSV = REPO_ROOT / "benchmarks" / "results" / "run_20260323_133042Z" / "full_benchmark_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Benchmark comparison CSV (default: {DEFAULT_CSV})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["exact_k"] = (df["found_clusters"] == df["true_clusters"]).astype(int)
    df["k1_wrong"] = ((df["found_clusters"] == 1) & (df["true_clusters"] > 1)).astype(int)

    valid = df[df["Status"] == "ok"].copy()

    summary = (
        valid.groupby("method")
        .agg(
            n_cases=("ari", "count"),
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            exact_k_rate=("exact_k", "mean"),
            exact_k_count=("exact_k", "sum"),
            k1_wrong_count=("k1_wrong", "sum"),
            mean_found_k=("found_clusters", "mean"),
        )
        .round(3)
        .sort_values("mean_ari", ascending=False)
    )
    print("=== METHOD SUMMARY (sorted by mean ARI) ===")
    print(summary.to_string())
    print()

    kl = valid[valid["method"] == "kl"][
        ["test_case", "ari", "found_clusters", "true_clusters"]
    ].set_index("test_case")
    kld = valid[valid["method"] == "kl_diffusion"][
        ["test_case", "ari", "found_clusters"]
    ].set_index("test_case")
    merged = kl.join(kld, rsuffix="_diff")
    merged["delta"] = merged["ari_diff"] - merged["ari"]

    print("=== HEAD-TO-HEAD: kl_diffusion vs kl ===")
    print(f"kl_diffusion wins (delta > 0.01): {(merged['delta'] > 0.01).sum()}")
    print(f"kl wins          (delta < -0.01): {(merged['delta'] < -0.01).sum()}")
    print(f"ties             (|delta| <= 0.01): {(merged['delta'].abs() <= 0.01).sum()}")
    print(f"Mean delta ARI: {merged['delta'].mean():.4f}")
    print()

    print("=== Top 10 kl_diffusion WINS ===")
    cols = ["ari", "ari_diff", "true_clusters", "found_clusters", "found_clusters_diff", "delta"]
    print(merged.nlargest(10, "delta")[cols].to_string())
    print()

    print("=== Top 10 kl_diffusion REGRESSIONS ===")
    print(merged.nsmallest(10, "delta")[cols].to_string())
    print()

    kl_cat = valid[valid["method"] == "kl"][["test_case", "Case_Category", "ari"]].copy()
    kld_cat = valid[valid["method"] == "kl_diffusion"][["test_case", "ari"]].copy()
    cat_merged = kl_cat.merge(kld_cat, on="test_case", suffixes=("_kl", "_diff"))

    cat_summary = (
        cat_merged.groupby("Case_Category")
        .agg(
            n=("ari_kl", "count"),
            kl_mean=("ari_kl", "mean"),
            diff_mean=("ari_diff", "mean"),
        )
        .round(3)
    )
    cat_summary["delta"] = (cat_summary["diff_mean"] - cat_summary["kl_mean"]).round(3)
    cat_summary = cat_summary.sort_values("delta", ascending=False)
    print("=== CATEGORY BREAKDOWN: kl_diffusion - kl ===")
    print(cat_summary.to_string())


if __name__ == "__main__":
    main()
