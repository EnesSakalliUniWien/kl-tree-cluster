#!/usr/bin/env python3
"""Analyze the impact of switching SPECTRAL_METHOD to marchenko_pastur.

Compares benchmark runs across all cases and methods. Originally used to
compare the legacy effective_rank baseline against the Marchenko-Pastur
default.

Usage:
    python benchmarks/analyze_mp_switch.py [--baseline RUN_DIR] [--mp RUN_DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def load_run(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "full_benchmark_comparison.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No CSV at {csv}")
    df = pd.read_csv(csv)
    return df


def method_summary(df: pd.DataFrame, method: str) -> dict:
    """Compute summary stats for a single method."""
    m = df[df["method"] == method].copy()
    mt = m[m["true_clusters"] > 0].copy()
    ari_valid = m["ari"].dropna()
    exact_k = int((mt["found_clusters"] == mt["true_clusters"]).sum())
    k1 = int((mt["found_clusters"] == 1).sum())
    over_split = int((mt["found_clusters"] > mt["true_clusters"]).sum())
    under_split = int((mt["found_clusters"] < mt["true_clusters"]).sum())
    return {
        "n_cases": len(m),
        "n_with_truth": len(mt),
        "mean_ari": float(ari_valid.mean()) if len(ari_valid) > 0 else float("nan"),
        "median_ari": float(ari_valid.median()) if len(ari_valid) > 0 else float("nan"),
        "exact_k": exact_k,
        "k1": k1,
        "over_split": over_split,
        "under_split": under_split,
    }


def per_case_comparison(baseline: pd.DataFrame, mp: pd.DataFrame, method: str = "kl"):
    """Compare per-case ARI and K between baseline and MP for a given method."""
    b = baseline[baseline["method"] == method][
        ["case_id", "ari", "true_clusters", "found_clusters"]
    ].copy()
    m = mp[mp["method"] == method][["case_id", "ari", "true_clusters", "found_clusters"]].copy()

    b = b.rename(columns={"ari": "ari_base", "found_clusters": "found_base"})
    m = m.rename(columns={"ari": "ari_mp", "found_clusters": "found_mp"})

    merged = b.merge(m, on=["case_id", "true_clusters"], how="outer", suffixes=("", ""))
    merged["ari_delta"] = merged["ari_mp"] - merged["ari_base"]
    return merged


def main():
    parser = argparse.ArgumentParser(description="Analyze MP vs effective_rank benchmark results")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline run directory name")
    parser.add_argument("--mp", type=str, default=None, help="MP run directory name")
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"

    # Auto-detect: latest two full runs (931+ lines)
    if args.mp is None or args.baseline is None:
        full_runs = []
        for d in sorted(results_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("run_"):
                csv = d / "full_benchmark_comparison.csv"
                if csv.exists():
                    n_lines = sum(1 for _ in open(csv))
                    if n_lines > 500:  # full run threshold
                        full_runs.append(d)
            if len(full_runs) >= 2:
                break

        if len(full_runs) < 2:
            print("ERROR: Need at least 2 full benchmark runs to compare.")
            print(f"Found: {[d.name for d in full_runs]}")
            sys.exit(1)

        mp_dir = full_runs[0]
        base_dir = full_runs[1]
    else:
        mp_dir = results_dir / args.mp
        base_dir = results_dir / args.baseline

    print(f"Baseline (effective_rank): {base_dir.name}")
    print(f"Current  (marchenko_pastur): {mp_dir.name}")
    print()

    baseline = load_run(base_dir)
    mp = load_run(mp_dir)

    # ── 1. High-level comparison across all methods ──────────────────────
    print("=" * 90)
    print("1. METHOD COMPARISON: Mean ARI")
    print("=" * 90)

    methods = sorted(set(baseline["method"].unique()) & set(mp["method"].unique()))

    print(
        f"\n  {'Method':<15s} {'Base ARI':>9s} {'MP ARI':>9s} {'Delta':>7s}  "
        f"{'Base K':>7s} {'MP K':>7s} {'Base K=1':>8s} {'MP K=1':>7s}"
    )
    print(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*7}  {'-'*7} {'-'*7} {'-'*8} {'-'*7}")

    for method in methods:
        bs = method_summary(baseline, method)
        ms = method_summary(mp, method)
        delta = ms["mean_ari"] - bs["mean_ari"]
        sign = "+" if delta > 0 else ""
        ek_b = f"{bs['exact_k']}/{bs['n_with_truth']}"
        ek_m = f"{ms['exact_k']}/{ms['n_with_truth']}"
        print(
            f"  {method:<15s} {bs['mean_ari']:>9.3f} {ms['mean_ari']:>9.3f} {sign}{delta:>6.3f}  "
            f"{ek_b:>7s} {ek_m:>7s} {bs['k1']:>8d} {ms['k1']:>7d}"
        )

    # ── 2. Detailed KL comparison ────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("2. KL DIVERGENCE METHOD: Detailed Before/After")
    print("=" * 90)

    for method in ["kl", "kl_complete", "kl_single"]:
        if method not in methods:
            continue
        bs = method_summary(baseline, method)
        ms = method_summary(mp, method)
        delta_ari = ms["mean_ari"] - bs["mean_ari"]
        delta_k = ms["exact_k"] - bs["exact_k"]

        print(f"\n  --- {method} ---")
        print(f"  {'Metric':<25s} {'Baseline':>10s} {'MP':>10s} {'Delta':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        print(
            f"  {'Mean ARI':<25s} {bs['mean_ari']:>10.3f} {ms['mean_ari']:>10.3f} {delta_ari:>+10.3f}"
        )
        print(
            f"  {'Median ARI':<25s} {bs['median_ari']:>10.3f} {ms['median_ari']:>10.3f} {ms['median_ari'] - bs['median_ari']:>+10.3f}"
        )
        print(f"  {'Exact K':<25s} {bs['exact_k']:>10d} {ms['exact_k']:>10d} {delta_k:>+10d}")
        print(
            f"  {'K=1 (under-merge)':<25s} {bs['k1']:>10d} {ms['k1']:>10d} {ms['k1'] - bs['k1']:>+10d}"
        )
        print(
            f"  {'Over-split':<25s} {bs['over_split']:>10d} {ms['over_split']:>10d} {ms['over_split'] - bs['over_split']:>+10d}"
        )
        print(
            f"  {'Under-split':<25s} {bs['under_split']:>10d} {ms['under_split']:>10d} {ms['under_split'] - bs['under_split']:>+10d}"
        )

    # ── 3. Per-case ARI changes (KL method) ──────────────────────────────
    print(f"\n\n{'='*90}")
    print("3. PER-CASE ARI CHANGES (kl method)")
    print("=" * 90)

    comp = per_case_comparison(baseline, mp, "kl")
    comp = comp.sort_values("ari_delta", ascending=True)

    # Improved cases
    improved = comp[comp["ari_delta"] > 0.01].copy()
    regressed = comp[comp["ari_delta"] < -0.01].copy()
    unchanged = comp[(comp["ari_delta"].abs() <= 0.01) | comp["ari_delta"].isna()].copy()

    print(f"\n  Improved:  {len(improved)} cases")
    print(f"  Regressed: {len(regressed)} cases")
    print(f"  Unchanged: {len(unchanged)} cases (|delta| <= 0.01)")

    if len(improved) > 0:
        print("\n  --- IMPROVED (ARI increased > 0.01) ---")
        print(
            f"  {'Case':<42s} {'Base':>7s} {'MP':>7s} {'Delta':>7s} {'K_true':>6s} {'K_b':>4s} {'K_mp':>5s}"
        )
        print(f"  {'-'*42} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*4} {'-'*5}")
        for _, r in improved.sort_values("ari_delta", ascending=False).iterrows():
            ari_b = (
                f"{r['ari_base']:.3f}" if not np.isnan(r.get("ari_base", float("nan"))) else "  N/A"
            )
            ari_m = f"{r['ari_mp']:.3f}" if not np.isnan(r.get("ari_mp", float("nan"))) else "  N/A"
            d = (
                f"{r['ari_delta']:+.3f}"
                if not np.isnan(r.get("ari_delta", float("nan")))
                else "  N/A"
            )
            kt = f"{int(r['true_clusters'])}" if r["true_clusters"] > 0 else "N/A"
            kb = (
                f"{int(r['found_base'])}"
                if not np.isnan(r.get("found_base", float("nan")))
                else "?"
            )
            km = f"{int(r['found_mp'])}" if not np.isnan(r.get("found_mp", float("nan"))) else "?"
            print(
                f"  {r['case_id']:<42s} {ari_b:>7s} {ari_m:>7s} {d:>7s} {kt:>6s} {kb:>4s} {km:>5s}"
            )

    if len(regressed) > 0:
        print("\n  --- REGRESSED (ARI decreased > 0.01) ---")
        print(
            f"  {'Case':<42s} {'Base':>7s} {'MP':>7s} {'Delta':>7s} {'K_true':>6s} {'K_b':>4s} {'K_mp':>5s}"
        )
        print(f"  {'-'*42} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*4} {'-'*5}")
        for _, r in regressed.sort_values("ari_delta").iterrows():
            ari_b = (
                f"{r['ari_base']:.3f}" if not np.isnan(r.get("ari_base", float("nan"))) else "  N/A"
            )
            ari_m = f"{r['ari_mp']:.3f}" if not np.isnan(r.get("ari_mp", float("nan"))) else "  N/A"
            d = (
                f"{r['ari_delta']:+.3f}"
                if not np.isnan(r.get("ari_delta", float("nan")))
                else "  N/A"
            )
            kt = f"{int(r['true_clusters'])}" if r["true_clusters"] > 0 else "N/A"
            kb = (
                f"{int(r['found_base'])}"
                if not np.isnan(r.get("found_base", float("nan")))
                else "?"
            )
            km = f"{int(r['found_mp'])}" if not np.isnan(r.get("found_mp", float("nan"))) else "?"
            print(
                f"  {r['case_id']:<42s} {ari_b:>7s} {ari_m:>7s} {d:>7s} {kt:>6s} {kb:>4s} {km:>5s}"
            )

    # ── 4. Category breakdown ────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("4. ARI BY CATEGORY (kl method)")
    print("=" * 90)

    b_kl = baseline[baseline["method"] == "kl"][
        ["case_id", "Case_Category", "ari", "true_clusters", "found_clusters"]
    ].copy()
    m_kl = mp[mp["method"] == "kl"][
        ["case_id", "Case_Category", "ari", "true_clusters", "found_clusters"]
    ].copy()
    b_kl = b_kl.rename(columns={"ari": "ari_base", "found_clusters": "found_base"})
    m_kl = m_kl.rename(columns={"ari": "ari_mp", "found_clusters": "found_mp"})
    cat_merged = b_kl.merge(m_kl, on=["case_id", "Case_Category", "true_clusters"], how="outer")

    print(
        f"\n  {'Category':<30s} {'n':>3s} {'Base ARI':>9s} {'MP ARI':>9s} {'Delta':>7s} {'Base ExK':>8s} {'MP ExK':>7s}"
    )
    print(f"  {'-'*30} {'-'*3} {'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*7}")

    for cat in sorted(cat_merged["Case_Category"].dropna().unique()):
        sub = cat_merged[cat_merged["Case_Category"] == cat]
        n = len(sub)
        ari_b = sub["ari_base"].dropna().mean()
        ari_m = sub["ari_mp"].dropna().mean()
        delta = ari_m - ari_b if not (np.isnan(ari_b) or np.isnan(ari_m)) else float("nan")
        st = sub[sub["true_clusters"] > 0]
        ek_b = int((st["found_base"] == st["true_clusters"]).sum()) if len(st) > 0 else 0
        ek_m = int((st["found_mp"] == st["true_clusters"]).sum()) if len(st) > 0 else 0
        nt = len(st)
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "   N/A"
        print(
            f"  {cat:<30s} {n:>3d} {ari_b:>9.3f} {ari_m:>9.3f} {delta_str:>7s} "
            f"{ek_b:>4d}/{nt:<3d} {ek_m:>3d}/{nt:<3d}"
        )

    # ── 5. Statistical summary ───────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("5. STATISTICAL SUMMARY")
    print("=" * 90)

    comp_valid = comp.dropna(subset=["ari_delta"])
    if len(comp_valid) > 0:
        delta = comp_valid["ari_delta"]
        print(f"\n  ARI delta (MP − baseline) across {len(comp_valid)} cases:")
        print(f"    Mean:   {delta.mean():+.4f}")
        print(f"    Median: {delta.median():+.4f}")
        print(f"    Std:    {delta.std():.4f}")
        print(f"    Min:    {delta.min():+.4f}  ({comp_valid.loc[delta.idxmin(), 'case_id']})")
        print(f"    Max:    {delta.max():+.4f}  ({comp_valid.loc[delta.idxmax(), 'case_id']})")

        # Wilcoxon signed-rank test
        try:
            from scipy.stats import wilcoxon

            nonzero = delta[delta.abs() > 1e-10]
            if len(nonzero) >= 10:
                stat, pval = wilcoxon(nonzero)
                print(f"\n  Wilcoxon signed-rank test (n={len(nonzero)} non-zero deltas):")
                print(f"    Statistic: {stat:.1f}")
                print(f"    p-value:   {pval:.4f}")
                direction = (
                    "MP is significantly better"
                    if pval < 0.05 and delta.mean() > 0
                    else (
                        "MP is significantly worse"
                        if pval < 0.05 and delta.mean() < 0
                        else "No significant difference"
                    )
                )
                print(f"    Verdict:   {direction}")
        except ImportError:
            pass

    # ── 6. K recovery analysis ───────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("6. K RECOVERY ANALYSIS (kl method)")
    print("=" * 90)

    comp_k = comp[comp["true_clusters"] > 0].copy()
    if len(comp_k) > 0:
        comp_k["base_exact"] = comp_k["found_base"] == comp_k["true_clusters"]
        comp_k["mp_exact"] = comp_k["found_mp"] == comp_k["true_clusters"]
        comp_k["base_k1"] = comp_k["found_base"] == 1
        comp_k["mp_k1"] = comp_k["found_mp"] == 1

        gained = comp_k[~comp_k["base_exact"] & comp_k["mp_exact"]]
        lost = comp_k[comp_k["base_exact"] & ~comp_k["mp_exact"]]
        k1_fixed = comp_k[comp_k["base_k1"] & ~comp_k["mp_k1"]]
        k1_new = comp_k[~comp_k["base_k1"] & comp_k["mp_k1"]]

        print(f"\n  Cases that GAINED exact K with MP:  {len(gained)}")
        if len(gained) > 0:
            for _, r in gained.iterrows():
                print(
                    f"    {r['case_id']:<40s}  K_true={int(r['true_clusters'])}  base→{int(r['found_base'])}  MP→{int(r['found_mp'])}"
                )

        print(f"\n  Cases that LOST exact K with MP:    {len(lost)}")
        if len(lost) > 0:
            for _, r in lost.iterrows():
                print(
                    f"    {r['case_id']:<40s}  K_true={int(r['true_clusters'])}  base→{int(r['found_base'])}  MP→{int(r['found_mp'])}"
                )

        print(f"\n  K=1 collapses FIXED by MP:          {len(k1_fixed)}")
        if len(k1_fixed) > 0:
            for _, r in k1_fixed.iterrows():
                print(
                    f"    {r['case_id']:<40s}  K_true={int(r['true_clusters'])}  MP→{int(r['found_mp'])}"
                )

        print(f"\n  New K=1 collapses from MP:           {len(k1_new)}")
        if len(k1_new) > 0:
            for _, r in k1_new.iterrows():
                print(
                    f"    {r['case_id']:<40s}  K_true={int(r['true_clusters'])}  base→{int(r['found_base'])}"
                )

    print()


if __name__ == "__main__":
    main()
