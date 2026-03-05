"""Analyze effect-size diagnostic results.

Reads the CSVs produced by effect_size_impact.py and prints clean statistics:
  1. Threshold sweep — aggregate K and ARI at each sibling-effect threshold
  2. Per-case changes — which cases improve or degrade at each threshold
  3. Node-level distributions — effect sizes at SPLIT vs MERGE nodes
  4. Separation analysis — overlap between SPLIT and MERGE effect ranges
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
SWEEP_CSV = RESULTS_DIR / "effect_size_threshold_sweep.csv"
NODES_CSV = RESULTS_DIR / "effect_size_node_records.csv"


def load_sweep() -> pd.DataFrame:
    df = pd.read_csv(SWEEP_CSV)
    # Drop cases without ground-truth ARI (e.g. real-data / preloaded)
    df = df.dropna(subset=["ari"])
    return df


def load_nodes() -> pd.DataFrame:
    return pd.read_csv(NODES_CSV)


# ── Section 1: Threshold sweep ───────────────────────────────────────────────


def report_threshold_sweep(df: pd.DataFrame) -> None:
    n_cases = df["case_name"].nunique()
    thresholds = sorted(df["threshold"].unique())

    print("=" * 78)
    print(f"SECTION 1: SIBLING EFFECT-SIZE THRESHOLD SWEEP  ({n_cases} cases)")
    print("=" * 78)
    print()
    header = (
        f"{'Thr':>8}  {'MeanARI':>8}  {'MedARI':>8}  "
        f"{'MeanΔK':>8}  {'ExactK':>7}  {'K=1':>4}  {'Over':>5}  {'Under':>6}"
    )
    print(header)
    print("-" * len(header))

    for thr in thresholds:
        s = df[df["threshold"] == thr]
        mean_ari = s["ari"].mean()
        med_ari = s["ari"].median()
        dk = (s["k_found"] - s["true_k"]).mean()
        exact = int((s["k_found"] == s["true_k"]).sum())
        k1 = int((s["k_found"] == 1).sum())
        over = int((s["k_found"] > s["true_k"]).sum())
        under = int((s["k_found"] < s["true_k"]).sum())
        print(
            f"{thr:>8.3f}  {mean_ari:>8.4f}  {med_ari:>8.4f}  "
            f"{dk:>+8.2f}  {exact:>7}  {k1:>4}  {over:>5}  {under:>6}"
        )

    print()


# ── Section 2: Per-case changes at selected thresholds ───────────────────────


def report_per_case_changes(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SECTION 2: PER-CASE CHANGES AT KEY THRESHOLDS")
    print("=" * 78)

    baseline = (
        df[df["threshold"] == 0.0][["case_name", "ari", "k_found", "true_k"]]
        .rename(columns={"ari": "ari_base", "k_found": "k_base"})
        .copy()
    )

    for thr in [0.03, 0.05, 0.075, 0.10]:
        thr_df = (
            df[df["threshold"] == thr][["case_name", "ari", "k_found"]]
            .rename(columns={"ari": "ari_thr", "k_found": "k_thr"})
            .copy()
        )
        merged = baseline.merge(thr_df, on="case_name")
        merged["delta_ari"] = merged["ari_thr"] - merged["ari_base"]
        merged["delta_k"] = merged["k_thr"] - merged["k_base"]

        changed = merged[merged["delta_ari"].abs() > 0.005].sort_values("delta_ari")

        print(f"\n  Threshold = {thr:.3f}:")
        if len(changed) == 0:
            print("    No cases changed ARI by more than 0.005")
        else:
            for _, r in changed.iterrows():
                sign = "+" if r["delta_ari"] > 0 else ""
                print(
                    f"    {r['case_name']:40s}  "
                    f"K: {int(r['k_base']):>3d} → {int(r['k_thr']):>3d} "
                    f"(true={int(r['true_k']):>2d})  "
                    f"ARI: {r['ari_base']:.3f} → {r['ari_thr']:.3f} "
                    f"({sign}{r['delta_ari']:.3f})"
                )

    print()


# ── Section 3: Best threshold per case ───────────────────────────────────────


def report_best_threshold_per_case(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SECTION 3: PER-CASE OPTIMAL THRESHOLD")
    print("=" * 78)

    baseline = (
        df[df["threshold"] == 0.0][["case_name", "category", "ari", "k_found", "true_k"]]
        .rename(columns={"ari": "ari_base", "k_found": "k_base"})
        .copy()
    )

    improvements = []
    for case_name in baseline["case_name"].unique():
        case_rows = df[df["case_name"] == case_name].sort_values("threshold")
        base_ari = case_rows[case_rows["threshold"] == 0.0]["ari"].iloc[0]
        base_k = int(case_rows[case_rows["threshold"] == 0.0]["k_found"].iloc[0])
        true_k = int(case_rows["true_k"].iloc[0])
        category = case_rows["category"].iloc[0]

        best_row = case_rows.loc[case_rows["ari"].idxmax()]
        best_ari = best_row["ari"]
        best_thr = best_row["threshold"]
        best_k = int(best_row["k_found"])

        if best_thr > 0 and best_ari > base_ari + 0.01:
            improvements.append(
                {
                    "case": case_name,
                    "category": category,
                    "true_k": true_k,
                    "k_base": base_k,
                    "k_best": best_k,
                    "ari_base": base_ari,
                    "ari_best": best_ari,
                    "delta": best_ari - base_ari,
                    "best_thr": best_thr,
                }
            )

    if improvements:
        imp_df = pd.DataFrame(improvements).sort_values("delta", ascending=False)
        print(f"\n  Cases improved by > 0.01 ARI ({len(imp_df)}):\n")
        for _, r in imp_df.iterrows():
            print(
                f"    {r['case']:40s}  "
                f"K: {r['k_base']:>3d} → {r['k_best']:>3d} (true={r['true_k']:>2d})  "
                f"ARI: {r['ari_base']:.3f} → {r['ari_best']:.3f} (+{r['delta']:.3f})  "
                f"@ thr={r['best_thr']:.3f}"
            )
    else:
        print("\n  No cases improved by > 0.01 ARI at any threshold.")

    print()


# ── Section 4: Node-level effect-size distributions ──────────────────────────


def report_node_distributions(nodes_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SECTION 4: EFFECT-SIZE DISTRIBUTIONS AT SPLIT vs MERGE NODES")
    print("=" * 78)

    binary = nodes_df[nodes_df["decision"].isin(["SPLIT", "MERGE"])].copy()
    if len(binary) == 0:
        print("  No binary-children nodes found.")
        return

    for decision in ["SPLIT", "MERGE"]:
        subset = binary[binary["decision"] == decision]
        n = len(subset)
        if n == 0:
            continue

        print(f"\n  {decision} nodes (n={n}):")
        for col, label in [
            ("sibling_effect_mean", "Sibling mean |Δp|"),
            ("sibling_effect_max", "Sibling max  |Δp|"),
            ("edge_effect_left", "Edge L mean  |Δp|"),
            ("edge_effect_right", "Edge R mean  |Δp|"),
        ]:
            vals = subset[col].dropna()
            if len(vals) == 0:
                continue
            print(
                f"    {label}:  "
                f"min={vals.min():.4f}  "
                f"p10={vals.quantile(0.10):.4f}  "
                f"p25={vals.quantile(0.25):.4f}  "
                f"med={vals.median():.4f}  "
                f"p75={vals.quantile(0.75):.4f}  "
                f"p90={vals.quantile(0.90):.4f}  "
                f"max={vals.max():.4f}"
            )

    print()


# ── Section 5: Overlap analysis ──────────────────────────────────────────────


def report_overlap_analysis(nodes_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SECTION 5: SPLIT vs MERGE OVERLAP ON SIBLING EFFECT (mean |Δp|)")
    print("=" * 78)

    binary = nodes_df[nodes_df["decision"].isin(["SPLIT", "MERGE"])].copy()
    split_vals = binary.loc[binary["decision"] == "SPLIT", "sibling_effect_mean"].dropna()
    merge_vals = binary.loc[binary["decision"] == "MERGE", "sibling_effect_mean"].dropna()

    if len(split_vals) == 0 or len(merge_vals) == 0:
        print("  Insufficient data for overlap analysis.")
        return

    print(
        f"\n  SPLIT range: [{split_vals.min():.4f}, {split_vals.max():.4f}]  (n={len(split_vals)})"
    )
    print(f"  MERGE range: [{merge_vals.min():.4f}, {merge_vals.max():.4f}]  (n={len(merge_vals)})")

    overlap_lo = max(split_vals.min(), merge_vals.min())
    overlap_hi = min(split_vals.max(), merge_vals.max())

    if overlap_lo < overlap_hi:
        n_split_overlap = int(((split_vals >= overlap_lo) & (split_vals <= overlap_hi)).sum())
        n_merge_overlap = int(((merge_vals >= overlap_lo) & (merge_vals <= overlap_hi)).sum())
        print(f"\n  Overlap zone: [{overlap_lo:.4f}, {overlap_hi:.4f}]")
        print(
            f"    SPLIT nodes in overlap: {n_split_overlap}/{len(split_vals)} ({100*n_split_overlap/len(split_vals):.1f}%)"
        )
        print(
            f"    MERGE nodes in overlap: {n_merge_overlap}/{len(merge_vals)} ({100*n_merge_overlap/len(merge_vals):.1f}%)"
        )
    else:
        print(f"\n  No overlap — clean separation at ~{overlap_lo:.4f}")

    # Histogram
    bins = [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    print(f"\n  {'Bin':>18s}  {'SPLIT':>7s} {'%':>6s}   {'MERGE':>7s} {'%':>6s}")
    print("  " + "-" * 52)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n_s = int(((split_vals >= lo) & (split_vals < hi)).sum())
        n_m = int(((merge_vals >= lo) & (merge_vals < hi)).sum())
        pct_s = 100 * n_s / len(split_vals) if len(split_vals) else 0
        pct_m = 100 * n_m / len(merge_vals) if len(merge_vals) else 0
        print(f"  [{lo:>5.3f}, {hi:>5.3f})  {n_s:>7d} {pct_s:>5.1f}%   {n_m:>7d} {pct_m:>5.1f}%")

    print()


# ── Section 6: Potential false splits ────────────────────────────────────────


def report_low_effect_splits(nodes_df: pd.DataFrame) -> None:
    print("=" * 78)
    print("SECTION 6: LOW-EFFECT SPLIT NODES (potential micro-splits to suppress)")
    print("=" * 78)

    splits = nodes_df[nodes_df["decision"] == "SPLIT"].copy()
    if len(splits) == 0:
        print("  No SPLIT nodes.")
        return

    for cutoff in [0.03, 0.05, 0.075]:
        low = splits[splits["sibling_effect_mean"] < cutoff]
        print(
            f"\n  Sibling mean |Δp| < {cutoff:.3f}:  {len(low)}/{len(splits)} SPLIT nodes ({100*len(low)/len(splits):.1f}%)"
        )
        if len(low) > 0:
            cases_affected = low["case_name"].unique()
            print(f"    Affecting {len(cases_affected)} case(s): {', '.join(cases_affected[:10])}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if not SWEEP_CSV.exists():
        print(f"Missing: {SWEEP_CSV}")
        print("Run effect_size_impact.py first.")
        sys.exit(1)

    sweep_df = load_sweep()
    nodes_df = load_nodes() if NODES_CSV.exists() else pd.DataFrame()

    report_threshold_sweep(sweep_df)
    report_per_case_changes(sweep_df)
    report_best_threshold_per_case(sweep_df)

    if len(nodes_df) > 0:
        report_node_distributions(nodes_df)
        report_overlap_analysis(nodes_df)
        report_low_effect_splits(nodes_df)

    print("Done.")


if __name__ == "__main__":
    main()
