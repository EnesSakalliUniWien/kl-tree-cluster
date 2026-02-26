#!/usr/bin/env python3
"""Compare edge calibration benchmark runs against baseline.

Usage:
    python debug_scripts/diagnostics/compare_edge_calibration_runs.py

Compares up to three runs:
  - BASELINE (no edge calibration): run_20260218_114417Z
  - GLOBAL calibration:             run_20260218_130036Z
  - SOFT LOCAL calibration:         run_20260218_133637Z
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"

# ── Configure runs to compare ────────────────────────────────────────────
RUNS = {
    "baseline": RESULTS / "run_20260218_114417Z" / "full_benchmark_comparison.csv",
    "global": RESULTS / "run_20260218_130036Z" / "full_benchmark_comparison.csv",
    "soft_local": RESULTS / "run_20260218_133637Z" / "full_benchmark_comparison.csv",
}


def load_kl(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [skip] {label}: {path} not found")
        return None
    df = pd.read_csv(path)
    kl = df[df["method"] == "kl"][["case_id", "true_clusters", "found_clusters", "ari"]].copy()
    kl = kl.rename(columns={"found_clusters": f"{label}_K", "ari": f"{label}_ari"})
    return kl.reset_index(drop=True)


def main() -> None:
    frames = {}
    for label, path in RUNS.items():
        f = load_kl(path, label)
        if f is not None:
            frames[label] = f

    if not frames:
        print("No benchmark CSVs found.")
        return

    # Merge all on case_id + true_clusters
    base_label = list(frames.keys())[0]
    merged = frames[base_label]
    for label in list(frames.keys())[1:]:
        cols = ["case_id", f"{label}_K", f"{label}_ari"]
        merged = merged.merge(frames[label][cols], on="case_id", how="outer")

    # ── Summary table ────────────────────────────────────────────────────
    print("=" * 80)
    print("EDGE CALIBRATION COMPARISON — SUMMARY")
    print("=" * 80)

    labels = list(frames.keys())
    header = f"{'Metric':<25}" + "".join(f"{l:>15}" for l in labels)
    print(f"\n{header}")
    print("-" * (25 + 15 * len(labels)))

    # Mean ARI
    vals = [merged[f"{l}_ari"].mean() for l in labels]
    row = f"{'Mean ARI':<25}" + "".join(f"{v:>15.4f}" for v in vals)
    print(row)

    # Median ARI
    vals = [merged[f"{l}_ari"].median() for l in labels]
    row = f"{'Median ARI':<25}" + "".join(f"{v:>15.4f}" for v in vals)
    print(row)

    n = len(merged)
    # Exact K
    vals = [int((merged[f"{l}_K"] == merged["true_clusters"]).sum()) for l in labels]
    row = f"{'Exact K':<25}" + "".join(f"{v:>13}/{n}" for v in vals)
    print(row)

    # K=1
    vals = [int((merged[f"{l}_K"] == 1).sum()) for l in labels]
    row = f"{'K=1 collapses':<25}" + "".join(f"{v:>15}" for v in vals)
    print(row)

    # Over-split
    vals = [int((merged[f"{l}_K"] > merged["true_clusters"]).sum()) for l in labels]
    row = f"{'Over-split (K>true)':<25}" + "".join(f"{v:>15}" for v in vals)
    print(row)

    # Under-split
    vals = [int((merged[f"{l}_K"] < merged["true_clusters"]).sum()) for l in labels]
    row = f"{'Under-split (K<true)':<25}" + "".join(f"{v:>15}" for v in vals)
    print(row)

    # ── Per-run pairwise vs baseline ─────────────────────────────────────
    if len(labels) > 1:
        base = labels[0]
        for comp in labels[1:]:
            merged[f"delta_{comp}"] = merged[f"{comp}_ari"] - merged[f"{base}_ari"]

            regressed = merged[merged[f"delta_{comp}"] < -0.01].sort_values(f"delta_{comp}")
            improved = merged[merged[f"delta_{comp}"] > 0.01].sort_values(
                f"delta_{comp}", ascending=False
            )

            print(f"\n{'─' * 80}")
            print(f"{comp.upper()} vs {base.upper()}")
            print(f"{'─' * 80}")

            print(f"\n  REGRESSED ({len(regressed)} cases):")
            if len(regressed) > 0:
                print(
                    f"  {'Case':<35} {'True K':>6} "
                    f"{base + ' K':>8} {comp + ' K':>8} "
                    f"{base + ' ARI':>10} {comp + ' ARI':>10} {'Δ ARI':>8}"
                )
                print(f"  {'─' * 85}")
                for _, r in regressed.iterrows():
                    print(
                        f"  {r.case_id:<35} {int(r.true_clusters):>6} "
                        f"{int(r[f'{base}_K']):>8} {int(r[f'{comp}_K']):>8} "
                        f"{r[f'{base}_ari']:>10.3f} {r[f'{comp}_ari']:>10.3f} "
                        f"{r[f'delta_{comp}']:>+8.3f}"
                    )
            else:
                print("  (none)")

            print(f"\n  IMPROVED ({len(improved)} cases):")
            if len(improved) > 0:
                print(
                    f"  {'Case':<35} {'True K':>6} "
                    f"{base + ' K':>8} {comp + ' K':>8} "
                    f"{base + ' ARI':>10} {comp + ' ARI':>10} {'Δ ARI':>8}"
                )
                print(f"  {'─' * 85}")
                for _, r in improved.iterrows():
                    print(
                        f"  {r.case_id:<35} {int(r.true_clusters):>6} "
                        f"{int(r[f'{base}_K']):>8} {int(r[f'{comp}_K']):>8} "
                        f"{r[f'{base}_ari']:>10.3f} {r[f'{comp}_ari']:>10.3f} "
                        f"{r[f'delta_{comp}']:>+8.3f}"
                    )
            else:
                print("  (none)")

    # ── K=1 collapses detail ─────────────────────────────────────────────
    for comp in labels[1:]:
        k1 = merged[merged[f"{comp}_K"] == 1].sort_values(f"{labels[0]}_ari", ascending=False)
        new_k1 = k1[k1[f"{labels[0]}_K"] > 1]
        if len(new_k1) > 0:
            print(f"\n{'─' * 80}")
            print(f"NEW K=1 COLLAPSES in {comp.upper()} ({len(new_k1)} cases)")
            print(f"{'─' * 80}")
            print(f"  {'Case':<40} {'True K':>6} {labels[0] + ' K':>8} {labels[0] + ' ARI':>10}")
            print(f"  {'─' * 64}")
            for _, r in new_k1.iterrows():
                print(
                    f"  {r.case_id:<40} {int(r.true_clusters):>6} "
                    f"{int(r[f'{labels[0]}_K']):>8} {r[f'{labels[0]}_ari']:>10.3f}"
                )

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
