#!/usr/bin/env python3
"""Experiment 4 — Alpha sensitivity and calibration diagnostics.

Tests two things:
  A. Alpha sweep: does lowering alpha (stricter) help over-splitting
     without hurting clean cases?
  B. Calibration transparency: for each case, show the inflation factor
     ĉ, effective calibration n, and how they relate to performance.

Usage:
    python debug_scripts/enhancement_lab/exp4_alpha_calibration.py
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


def alpha_sweep(
    cases: list[str],
    alpha_values: list[float],
) -> pd.DataFrame:
    """Sweep both alpha_local and sibling_alpha together."""
    all_rows = []
    for case_name in cases:
        try:
            tree, data_df, y_t, tc = build_tree_and_data(case_name)
            true_k = tc.get("n_clusters", None)

            for alpha in alpha_values:
                # Need fresh tree for each alpha (annotations are cached)
                tree2, _, _, _ = build_tree_and_data(case_name)
                decomp = tree2.decompose(
                    leaf_data=data_df,
                    alpha_local=alpha,
                    sibling_alpha=alpha,
                )
                found_k = decomp["num_clusters"]
                ari = compute_ari(decomp, data_df, y_t) if y_t is not None else float("nan")

                # Extract calibration info
                stats = tree2.annotations_df
                audit = stats.attrs.get("sibling_divergence_audit", {})
                c_hat = audit.get("global_inflation_factor", None)
                n_cal = audit.get("calibration_n", None)
                diag = audit.get("diagnostics", {})
                eff_n = diag.get("effective_n", None)

                all_rows.append(
                    {
                        "case": case_name,
                        "alpha": alpha,
                        "true_k": true_k,
                        "found_k": found_k,
                        "ari": round(ari, 3),
                        "delta_k": found_k - (true_k or 0),
                        "c_hat": round(c_hat, 4) if c_hat else None,
                        "n_cal": n_cal,
                        "eff_n": round(eff_n, 1) if eff_n else None,
                    }
                )
        except Exception as e:
            for alpha in alpha_values:
                all_rows.append(
                    {
                        "case": case_name,
                        "alpha": alpha,
                        "error": str(e),
                    }
                )
    return pd.DataFrame(all_rows)


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 4: Alpha Sensitivity & Calibration Diagnostics")
    print("=" * 72)

    alphas = [0.001, 0.005, 0.01, 0.05, 0.10]

    # ── Section A: Failure cases ──
    print("\n--- Failure cases ---")
    df_fail = alpha_sweep(FAILURE_CASES, alphas)

    print("\nFound K by alpha:")
    pivot = df_fail.pivot_table(index="case", columns="alpha", values="found_k", aggfunc="first")
    print(pivot.to_string())

    print("\nARI by alpha:")
    pivot = df_fail.pivot_table(index="case", columns="alpha", values="ari", aggfunc="first")
    print(pivot.to_string())

    # ── Section B: Regression guard ──
    print("\n--- Regression guard cases ---")
    df_guard = alpha_sweep(REGRESSION_GUARD_CASES, alphas)

    print("\nFound K by alpha:")
    pivot = df_guard.pivot_table(index="case", columns="alpha", values="found_k", aggfunc="first")
    print(pivot.to_string())

    print("\nARI by alpha:")
    pivot = df_guard.pivot_table(index="case", columns="alpha", values="ari", aggfunc="first")
    print(pivot.to_string())

    # ── Section C: Calibration diagnostics ──
    print(f"\n{'=' * 72}")
    print("  Calibration Diagnostics (at alpha=0.01)")
    print(f"{'=' * 72}")

    # Show calibration metrics at default alpha
    df_default = df_fail[df_fail["alpha"] == 0.01]
    if len(df_default):
        print(
            df_default[["case", "true_k", "found_k", "ari", "c_hat", "n_cal", "eff_n"]].to_string(
                index=False
            )
        )

    df_guard_default = df_guard[df_guard["alpha"] == 0.01]
    if len(df_guard_default):
        print("\nRegression guard:")
        print(
            df_guard_default[
                ["case", "true_k", "found_k", "ari", "c_hat", "n_cal", "eff_n"]
            ].to_string(index=False)
        )

    # ── Summary ──
    print(f"\n{'=' * 72}")
    print("  Summary: Mean ARI per alpha")
    print(f"{'=' * 72}")

    combined = pd.concat([df_fail, df_guard])
    summary = combined.groupby("alpha").agg(
        mean_ari=("ari", "mean"),
        exact_k=("delta_k", lambda x: (x == 0).sum()),
        mean_abs_delta=("delta_k", lambda x: abs(x).mean()),
        total_cases=("case", "count"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
