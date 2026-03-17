#!/usr/bin/env python3
"""Experiment 0 — Baseline measurement and failure diagnosis.

Runs all failure cases + regression guard cases under current config,
prints a summary table, then shows per-node gate diagnostics for the
worst over-splitting case.

Usage:
    python debug_scripts/enhancement_lab/exp0_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debug_scripts.enhancement_lab.lab_helpers import (
    FAILURE_CASES,
    REGRESSION_GUARD_CASES,
    collect_node_stats,
    print_summary,
    quick_eval,
    run_case_battery,
)


def main() -> None:
    # ── Section 1: Summary table ──
    print("=" * 72)
    print("  EXPERIMENT 0: BASELINE (current config)")
    print("=" * 72)

    print("\n--- Failure cases ---")
    df_fail = run_case_battery(FAILURE_CASES, label="baseline")
    print_summary(df_fail)

    print("\n--- Regression guard cases (should remain correct) ---")
    df_guard = run_case_battery(REGRESSION_GUARD_CASES, label="baseline")
    print_summary(df_guard)

    # ── Section 2: Deep dive on worst over-split ──
    worst = df_fail.loc[df_fail["delta_k"].idxmax()] if df_fail["delta_k"].notna().any() else None
    if worst is not None and worst["delta_k"] > 0:
        case_name = worst["case"]
        print(f"\n{'=' * 72}")
        print(f"  DEEP DIVE: {case_name} (found K={worst['found_k']}, true K={worst['true_k']})")
        print(f"{'=' * 72}")

        result = quick_eval(case_name)
        node_df = collect_node_stats(result["tree"], result["stats_df"])

        print(f"\nTotal internal binary nodes: {len(node_df)}")
        print(f"SPLIT decisions: {(node_df['decision'] == 'SPLIT').sum()}")
        print(f"MERGE(G2) decisions: {(node_df['decision'] == 'MERGE(G2)').sum()}")
        print(f"MERGE(G3) decisions: {(node_df['decision'] == 'MERGE(G3)').sum()}")
        print(f"MERGE(SKIP) decisions: {(node_df['decision'] == 'MERGE(SKIP)').sum()}")

        # Show split nodes by depth
        splits = node_df[node_df["decision"] == "SPLIT"].copy()
        if len(splits):
            print("\n--- SPLIT nodes by depth (showing leaf_count) ---")
            depth_stats = splits.groupby("depth").agg(
                count=("node", "count"),
                mean_leaves=("leaf_count", "mean"),
                min_leaves=("leaf_count", "min"),
                max_leaves=("leaf_count", "max"),
            )
            print(depth_stats.to_string())

            print("\n--- Smallest SPLIT nodes (leaf_count < 30) ---")
            small_splits = splits[splits["leaf_count"] < 30].sort_values("leaf_count")
            if len(small_splits):
                print(
                    small_splits[
                        ["node", "depth", "leaf_count", "sib_stat", "sib_df", "sib_p_corr"]
                    ]
                    .head(15)
                    .to_string(index=False)
                )
            else:
                print("  (none)")

        # Calibration audit
        audit = result["stats_df"].attrs.get("sibling_divergence_audit", {})
        if audit:
            print("\n--- Calibration audit ---")
            for k, v in audit.items():
                if k == "diagnostics" and isinstance(v, dict):
                    for dk, dv in v.items():
                        print(f"  diagnostics.{dk}: {dv}")
                else:
                    print(f"  {k}: {v}")

    # ── Section 3: Deep dive on NaN under-split ──
    nan_cases = df_fail[df_fail["found_k"] == 1]
    if len(nan_cases):
        for _, row in nan_cases.iterrows():
            case_name = row["case"]
            print(f"\n{'=' * 72}")
            print(f"  DEEP DIVE (under-split): {case_name}")
            print(f"{'=' * 72}")

            result = quick_eval(case_name)
            stats = result["stats_df"]
            tree = result["tree"]
            root = next(n for n, d in tree.in_degree() if d == 0)
            children = list(tree.successors(root))

            print(f"Root: {root}, children: {children}")
            for c in children:
                sig = stats.loc[c, "Child_Parent_Divergence_Significant"]
                pval = stats.loc[c, "Child_Parent_Divergence_P_Value_BH"]
                print(f"  {c}: edge_sig={sig}, edge_p_BH={pval}")

            sib_stat = (
                stats.loc[root, "Sibling_Test_Statistic"]
                if "Sibling_Test_Statistic" in stats.columns
                else "N/A"
            )
            sib_p = (
                stats.loc[root, "Sibling_Divergence_P_Value"]
                if "Sibling_Divergence_P_Value" in stats.columns
                else "N/A"
            )
            print(f"  Root sibling stat: {sib_stat}")
            print(f"  Root sibling p-value: {sib_p}")
            print(f"  Root sibling skipped: {stats.loc[root, 'Sibling_Divergence_Skipped']}")


if __name__ == "__main__":
    main()
