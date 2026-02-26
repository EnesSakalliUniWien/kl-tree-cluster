#!/usr/bin/env python3
"""
Analyze the edge calibration benchmark regression.

Compares:
  - BASELINE (no edge calibration): benchmarks/results/run_20260218_114417Z/
  - CALIBRATED (edge calibration ON): benchmarks/results/run_20260218_130036Z/

Baseline Mean ARI = 0.844, Calibrated Mean ARI = 0.691 → regression of -0.153.
Goal: identify which cases regressed, why, and what the calibration ĉ values were.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASELINE_CSV = ROOT / "benchmarks/results/run_20260218_114417Z/full_benchmark_comparison.csv"
CALIBRATED_CSV = ROOT / "benchmarks/results/run_20260218_130036Z/full_benchmark_comparison.csv"


def load_kl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["method"] == "kl"][
        ["case_id", "true_clusters", "found_clusters", "ari"]
    ].reset_index(drop=True)


def main() -> None:
    base = load_kl(BASELINE_CSV).rename(
        columns={"found_clusters": "base_K", "ari": "base_ari"}
    )
    cal = load_kl(CALIBRATED_CSV).rename(
        columns={"found_clusters": "cal_K", "ari": "cal_ari"}
    )
    merged = base.merge(cal[["case_id", "cal_K", "cal_ari"]], on="case_id", how="outer")
    merged["ari_diff"] = merged["cal_ari"] - merged["base_ari"]
    merged["k_diff"] = merged["cal_K"] - merged["base_K"]

    print("=" * 80)
    print("EDGE CALIBRATION BENCHMARK REGRESSION ANALYSIS")
    print("=" * 80)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Calibrated':>12} {'Delta':>12}")
    print("-" * 66)
    print(f"{'Mean ARI':<30} {merged.base_ari.mean():>12.4f} {merged.cal_ari.mean():>12.4f} {(merged.cal_ari.mean() - merged.base_ari.mean()):>+12.4f}")
    print(f"{'Median ARI':<30} {merged.base_ari.median():>12.4f} {merged.cal_ari.median():>12.4f} {(merged.cal_ari.median() - merged.base_ari.median()):>+12.4f}")
    exact_base = (merged.base_K == merged.true_clusters).sum()
    exact_cal = (merged.cal_K == merged.true_clusters).sum()
    n = len(merged)
    print(f"{'Exact K':<30} {exact_base:>10}/{n} {exact_cal:>10}/{n} {exact_cal - exact_base:>+12d}")
    k1_base = (merged.base_K == 1).sum()
    k1_cal = (merged.cal_K == 1).sum()
    print(f"{'K=1 collapses':<30} {k1_base:>12d} {k1_cal:>12d} {k1_cal - k1_base:>+12d}")

    # ── K=1 collapses ───────────────────────────────────────────────────────
    k1 = merged[merged.cal_K == 1].sort_values("base_ari", ascending=False)
    print(f"\n{'─'*80}")
    print(f"K=1 COLLAPSES (calibrated): {len(k1)} cases")
    print(f"{'─'*80}")
    print(f"  {'Case':<40} {'True K':>6} {'Base K':>7} {'Base ARI':>9}")
    print(f"  {'─'*62}")
    for _, r in k1.iterrows():
        marker = " *** NEW" if r.base_K > 1 else ""
        print(
            f"  {r.case_id:<40} {int(r.true_clusters):>6} {int(r.base_K):>7} {r.base_ari:>9.3f}{marker}"
        )

    # ── Regressed cases ─────────────────────────────────────────────────────
    regressed = merged[merged.ari_diff < -0.01].sort_values("ari_diff")
    print(f"\n{'─'*80}")
    print(f"REGRESSED CASES (ARI dropped > 0.01): {len(regressed)}")
    print(f"{'─'*80}")
    print(f"  {'Case':<35} {'True K':>6} {'Base K':>7} {'Cal K':>6} {'Base ARI':>9} {'Cal ARI':>8} {'Δ ARI':>8}")
    print(f"  {'─'*79}")
    for _, r in regressed.iterrows():
        print(
            f"  {r.case_id:<35} {int(r.true_clusters):>6} {int(r.base_K):>7} {int(r.cal_K):>6}"
            f" {r.base_ari:>9.3f} {r.cal_ari:>8.3f} {r.ari_diff:>+8.3f}"
        )

    # ── Improved cases ───────────────────────────────────────────────────────
    improved = merged[merged.ari_diff > 0.01].sort_values("ari_diff", ascending=False)
    print(f"\n{'─'*80}")
    print(f"IMPROVED CASES (ARI gained > 0.01): {len(improved)}")
    print(f"{'─'*80}")
    if len(improved) > 0:
        print(f"  {'Case':<35} {'True K':>6} {'Base K':>7} {'Cal K':>6} {'Base ARI':>9} {'Cal ARI':>8} {'Δ ARI':>8}")
        print(f"  {'─'*79}")
        for _, r in improved.iterrows():
            print(
                f"  {r.case_id:<35} {int(r.true_clusters):>6} {int(r.base_K):>7} {int(r.cal_K):>6}"
                f" {r.base_ari:>9.3f} {r.cal_ari:>8.3f} {r.ari_diff:>+8.3f}"
            )
    else:
        print("  (none)")

    # ── Unchanged cases ──────────────────────────────────────────────────────
    unchanged = merged[merged.ari_diff.abs() <= 0.01]
    print(f"\n{'─'*80}")
    print(f"UNCHANGED CASES (|Δ ARI| ≤ 0.01): {len(unchanged)}")
    print(f"{'─'*80}")

    # ── Category breakdown ───────────────────────────────────────────────────
    merged["category"] = merged.case_id.str.extract(r"^([a-z_]+?)_\d|^([a-z_]+)$").bfill(axis=1).iloc[:, 0]
    # Fallback: extract prefix before first digit
    merged["category"] = merged.case_id.str.replace(r"_?\d+[a-z_]*$", "", regex=True)
    cat_summary = (
        merged.groupby("category")
        .agg(
            n=("case_id", "count"),
            base_mean_ari=("base_ari", "mean"),
            cal_mean_ari=("cal_ari", "mean"),
            ari_diff=("ari_diff", "mean"),
            k1_base=("base_K", lambda x: (x == 1).sum()),
            k1_cal=("cal_K", lambda x: (x == 1).sum()),
        )
        .sort_values("ari_diff")
    )
    print(f"\n{'─'*80}")
    print("CATEGORY BREAKDOWN")
    print(f"{'─'*80}")
    print(f"  {'Category':<30} {'N':>3} {'Base ARI':>9} {'Cal ARI':>8} {'Δ ARI':>8} {'K1 base':>8} {'K1 cal':>7}")
    print(f"  {'─'*73}")
    for cat, r in cat_summary.iterrows():
        print(
            f"  {cat:<30} {int(r.n):>3} {r.base_mean_ari:>9.3f} {r.cal_mean_ari:>8.3f}"
            f" {r.ari_diff:>+8.3f} {int(r.k1_base):>8} {int(r.k1_cal):>7}"
        )

    # ── Direction summary ────────────────────────────────────────────────────
    over_split_base = (merged.base_K > merged.true_clusters).sum()
    under_split_base = (merged.base_K < merged.true_clusters).sum()
    over_split_cal = (merged.cal_K > merged.true_clusters).sum()
    under_split_cal = (merged.cal_K < merged.true_clusters).sum()
    print(f"\n{'─'*80}")
    print("SPLITTING DIRECTION")
    print(f"{'─'*80}")
    print(f"  {'Direction':<25} {'Baseline':>10} {'Calibrated':>12} {'Delta':>8}")
    print(f"  {'─'*55}")
    print(f"  {'Over-split (K > true)' :<25} {over_split_base:>10} {over_split_cal:>12} {over_split_cal - over_split_base:>+8}")
    print(f"  {'Under-split (K < true)':<25} {under_split_base:>10} {under_split_cal:>12} {under_split_cal - under_split_base:>+8}")
    print(f"  {'Exact K':<25} {exact_base:>10} {exact_cal:>12} {exact_cal - exact_base:>+8}")

    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
