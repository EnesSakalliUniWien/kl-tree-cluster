#!/usr/bin/env python3
"""Experiment 6 — Independent alpha sweep (2D grid).

exp4 swept EDGE_ALPHA and SIBLING_ALPHA in lockstep, which confounds
Gate 2 (signal detection) and Gate 3 (cluster separation) effects.

This experiment sweeps them independently on a 2D grid:
  EDGE_ALPHA   ∈ {0.001, 0.005, 0.01, 0.05}
  SIBLING_ALPHA ∈ {0.001, 0.005, 0.01, 0.05}

  → 16 cells per case.

Produces:
  - Heatmap of found_k(edge_alpha, sibling_alpha) per case
  - Heatmap of ARI(edge_alpha, sibling_alpha) per case
  - Mean-ARI summary across failure / regression-guard groups

Usage:
    python debug_scripts/enhancement_lab/exp6_independent_alpha.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debug_scripts.enhancement_lab.lab_helpers import (
    FAILURE_CASES,
    REGRESSION_GUARD_CASES,
    build_tree_and_data,
    compute_ari,
)


def _decompose_with_alphas(
    case_name: str,
    edge_alpha: float,
    sibling_alpha: float,
) -> dict:
    """Build tree and decompose with specific alpha pair.

    Rebuilds the tree each time so annotation cache doesn't interfere.
    """
    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=edge_alpha,
        sibling_alpha=sibling_alpha,
    )
    true_k = tc.get("n_clusters", None)
    found_k = decomp["num_clusters"]
    ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")
    return {
        "case": case_name,
        "edge_alpha": edge_alpha,
        "sibling_alpha": sibling_alpha,
        "true_k": true_k,
        "found_k": found_k,
        "ari": round(ari, 3),
        "delta_k": found_k - (true_k or 0),
    }


def alpha_grid_sweep(
    cases: list[str],
    edge_alphas: list[float],
    sibling_alphas: list[float],
) -> pd.DataFrame:
    """Run the full 2D alpha grid across all cases."""
    rows: list[dict] = []
    total = len(cases) * len(edge_alphas) * len(sibling_alphas)
    done = 0

    for case_name in cases:
        for ea in edge_alphas:
            for sa in sibling_alphas:
                done += 1
                try:
                    result = _decompose_with_alphas(case_name, ea, sa)
                    rows.append(result)
                except Exception as e:
                    rows.append(
                        {
                            "case": case_name,
                            "edge_alpha": ea,
                            "sibling_alpha": sa,
                            "error": str(e),
                        }
                    )
                if done % 10 == 0:
                    print(f"  [{done}/{total}]", end="", flush=True)

    print()
    return pd.DataFrame(rows)


def print_heatmap(df: pd.DataFrame, case_name: str, metric: str) -> None:
    """Print a 2D pivot of edge_alpha × sibling_alpha for one case."""
    sub = df[df["case"] == case_name]
    if sub.empty or metric not in sub.columns:
        return
    pivot = sub.pivot_table(
        index="edge_alpha",
        columns="sibling_alpha",
        values=metric,
        aggfunc="first",
    )
    print(f"\n  {case_name} — {metric}:")
    print(f"  {'':>12s}", end="")
    for sa in pivot.columns:
        print(f"  sib={sa:<6}", end="")
    print()
    for ea, row in pivot.iterrows():
        print(f"  edge={ea:<6}", end="")
        for val in row:
            if pd.isna(val):
                print(f"  {'NaN':>7}", end="")
            elif isinstance(val, float):
                print(f"  {val:>7.3f}", end="")
            else:
                print(f"  {val:>7}", end="")
        print()


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 6: Independent Alpha Sweep (2D Grid)")
    print("=" * 72)

    edge_alphas = [0.001, 0.005, 0.01, 0.05]
    sibling_alphas = [0.001, 0.005, 0.01, 0.05]

    # ── Section A: Failure cases ──
    print("\n--- Section A: Failure cases ---")
    print(
        f"  Grid: {len(edge_alphas)} × {len(sibling_alphas)} = "
        f"{len(edge_alphas) * len(sibling_alphas)} cells per case"
    )
    print(f"  Cases: {len(FAILURE_CASES)}")
    df_fail = alpha_grid_sweep(FAILURE_CASES, edge_alphas, sibling_alphas)

    for case_name in FAILURE_CASES:
        print_heatmap(df_fail, case_name, "found_k")
        print_heatmap(df_fail, case_name, "ari")

    # ── Section B: Regression guard cases ──
    print(f"\n{'=' * 72}")
    print("  Section B: Regression guard cases")
    print(f"{'=' * 72}")
    df_guard = alpha_grid_sweep(REGRESSION_GUARD_CASES, edge_alphas, sibling_alphas)

    for case_name in REGRESSION_GUARD_CASES:
        print_heatmap(df_guard, case_name, "found_k")
        print_heatmap(df_guard, case_name, "ari")

    # ── Section C: Aggregate summary ──
    print(f"\n{'=' * 72}")
    print("  Section C: Aggregate summary")
    print(f"{'=' * 72}")

    combined = pd.concat([df_fail, df_guard], ignore_index=True)
    valid = combined.dropna(subset=["ari"])

    # Mean ARI per (edge_alpha, sibling_alpha) combination
    agg = (
        valid.groupby(["edge_alpha", "sibling_alpha"])
        .agg(
            mean_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            exact_k=("delta_k", lambda x: (x == 0).sum()),
            mean_abs_delta_k=("delta_k", lambda x: x.abs().mean()),
            n_cases=("case", "count"),
        )
        .reset_index()
    )

    print("\n  Full grid summary (all cases):")
    pivot_ari = agg.pivot_table(
        index="edge_alpha",
        columns="sibling_alpha",
        values="mean_ari",
        aggfunc="first",
    )
    print("\n  Mean ARI:")
    print(pivot_ari.to_string())

    pivot_exact = agg.pivot_table(
        index="edge_alpha",
        columns="sibling_alpha",
        values="exact_k",
        aggfunc="first",
    )
    print("\n  Exact K count:")
    print(pivot_exact.to_string())

    pivot_delta = agg.pivot_table(
        index="edge_alpha",
        columns="sibling_alpha",
        values="mean_abs_delta_k",
        aggfunc="first",
    )
    print("\n  Mean |ΔK|:")
    print(pivot_delta.to_string())

    # ── Failure vs guard breakdown ──
    print(f"\n{'=' * 72}")
    print("  Failure cases only:")
    print(f"{'=' * 72}")
    fail_valid = df_fail.dropna(subset=["ari"])
    fail_agg = (
        fail_valid.groupby(["edge_alpha", "sibling_alpha"])
        .agg(
            mean_ari=("ari", "mean"),
            exact_k=("delta_k", lambda x: (x == 0).sum()),
            mean_abs_delta_k=("delta_k", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    pivot = fail_agg.pivot_table(
        index="edge_alpha",
        columns="sibling_alpha",
        values="mean_ari",
        aggfunc="first",
    )
    print("\n  Mean ARI (failure cases):")
    print(pivot.to_string())

    print(f"\n{'=' * 72}")
    print("  Guard cases only:")
    print(f"{'=' * 72}")
    guard_valid = df_guard.dropna(subset=["ari"])
    guard_agg = (
        guard_valid.groupby(["edge_alpha", "sibling_alpha"])
        .agg(
            mean_ari=("ari", "mean"),
            exact_k=("delta_k", lambda x: (x == 0).sum()),
            mean_abs_delta_k=("delta_k", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    pivot = guard_agg.pivot_table(
        index="edge_alpha",
        columns="sibling_alpha",
        values="mean_ari",
        aggfunc="first",
    )
    print("\n  Mean ARI (guard cases):")
    print(pivot.to_string())

    # ── Best cell ──
    best = agg.loc[agg["mean_ari"].idxmax()]
    print(
        f"\n  Best cell: edge_alpha={best['edge_alpha']}, "
        f"sibling_alpha={best['sibling_alpha']} → "
        f"mean ARI={best['mean_ari']:.3f}, "
        f"exact K={int(best['exact_k'])}/{int(best['n_cases'])}"
    )


if __name__ == "__main__":
    main()
