#!/usr/bin/env python3
"""Analyze why the inflation factor ĉ=1.0 (no deflation happening).

Inspects the T/k ratio distribution and weight distribution for
null-like vs focal pairs in the worst over-splitting case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debug_scripts.enhancement_lab.lab_helpers import quick_eval


def analyze_calibration(case_name: str) -> None:
    print(f"{'=' * 72}")
    print(f"  Calibration Analysis: {case_name}")
    print(f"{'=' * 72}")

    result = quick_eval(case_name)
    stats = result["annotations_df"]
    tree = result["tree"]
    audit = stats.attrs.get("sibling_divergence_audit", {})

    # ── Section 1: Full audit ──
    print("\n--- Calibration audit ---")
    for k, v in audit.items():
        if k == "diagnostics" and isinstance(v, dict):
            for dk, dv in v.items():
                print(f"  diagnostics.{dk}: {dv}")
        else:
            print(f"  {k}: {v}")

    # ── Section 2: T/k by null-like vs focal ──
    edge_p_bh = stats["Child_Parent_Divergence_P_Value_BH"]
    rows = []
    for nd in stats.index:
        children = list(tree.successors(nd))
        if len(children) != 2:
            continue
        l, r = children
        t = stats.loc[nd, "Sibling_Test_Statistic"]
        k = stats.loc[nd, "Sibling_Degrees_of_Freedom"]
        pl = float(edge_p_bh.get(l, 1.0)) if l in edge_p_bh.index else 1.0
        pr = float(edge_p_bh.get(r, 1.0)) if r in edge_p_bh.index else 1.0
        w = min(pl, pr)
        l_sig = bool(stats.loc[l, "Child_Parent_Divergence_Significant"])
        r_sig = bool(stats.loc[r, "Child_Parent_Divergence_Significant"])
        if np.isfinite(t) and np.isfinite(k) and k > 0:
            rows.append(
                {
                    "node": nd,
                    "T": t,
                    "k": k,
                    "ratio": t / k,
                    "w": w,
                    "l_sig": l_sig,
                    "r_sig": r_sig,
                    "null_like": not l_sig and not r_sig,
                }
            )

    bdf = pd.DataFrame(rows)
    null = bdf[bdf["null_like"]]
    focal = bdf[~bdf["null_like"]]

    print(f"\n--- T/k distribution (n={len(bdf)}) ---")
    print(
        f"Overall       : mean={bdf['ratio'].mean():.3f}  median={bdf['ratio'].median():.3f}  max={bdf['ratio'].max():.3f}"
    )
    print(
        f"Null-like ({len(null):3d}): mean={null['ratio'].mean():.3f}  median={null['ratio'].median():.3f}  max={null['ratio'].max():.3f}"
    )
    print(
        f"Focal     ({len(focal):3d}): mean={focal['ratio'].mean():.3f}  median={focal['ratio'].median():.3f}  max={focal['ratio'].max():.3f}"
    )

    # ── Section 3: Weight distribution ──
    print("\n--- Weight distribution ---")
    print(f"Mean w:    {bdf['w'].mean():.6f}")
    print(f"Median w:  {bdf['w'].median():.6e}")
    print(f"Max w:     {bdf['w'].max():.6f}")
    print(f"w >= 0.99: {(bdf['w'] >= 0.99).sum()}")
    print(f"w < 0.01:  {(bdf['w'] < 0.01).sum()}")
    print(f"w < 1e-10: {(bdf['w'] < 1e-10).sum()}")

    # ── Section 4: Weighted mean check ──
    wsum = (bdf["w"] * bdf["ratio"]).sum()
    wtotal = bdf["w"].sum()
    weighted_mean = wsum / wtotal if wtotal > 0 else float("nan")
    print("\n--- Weighted mean recalculation ---")
    print(f"Sum(w*ratio):  {wsum:.4f}")
    print(f"Sum(w):        {wtotal:.4f}")
    print(f"Weighted mean: {weighted_mean:.4f}")
    print(f"Clamped [1, max_ratio]: {max(1.0, min(weighted_mean, bdf['ratio'].max())):.4f}")

    # ── Section 5: Top-10 highest-weight pairs ──
    top_w = bdf.nlargest(10, "w")
    print("\n--- Top-10 highest-weight pairs ---")
    print(top_w[["node", "T", "k", "ratio", "w", "null_like"]].to_string(index=False))

    # ── Section 6: Top-10 null-like pairs with highest ratio ──
    if len(null):
        top_null = null.nlargest(10, "ratio")
        print("\n--- Top-10 null-like pairs by T/k ratio ---")
        print(top_null[["node", "T", "k", "ratio", "w"]].to_string(index=False))


def main() -> None:
    for case in ["overlap_heavy_4c_small_feat", "binary_perfect_4c", "gauss_clear_small"]:
        analyze_calibration(case)
        print()


if __name__ == "__main__":
    main()
