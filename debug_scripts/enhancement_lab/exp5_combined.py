#!/usr/bin/env python3
"""Experiment 5 — Combined enhancement: best parameters from exp1-4.

Integrates the three interventions simultaneously:
  1. Minimum sample size guard (from exp1)
  2. Pass-through depth limit (from exp2)
  3. Post-hoc merge (from exp3)

Then runs the FULL benchmark suite to measure real-world impact.

Usage:
    python debug_scripts/enhancement_lab/exp5_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debug_scripts.enhancement_lab.exp3_posthoc_merge import _decompose_with_posthoc
from debug_scripts.enhancement_lab.lab_helpers import (
    FAILURE_CASES,
    REGRESSION_GUARD_CASES,
    build_tree_and_data,
    compute_ari,
)


def run_enhanced_pipeline(
    case_name: str,
    n_min: int = 20,
    max_pt_depth: int = 3,
    merge_alpha: float = 0.05,
) -> dict:
    """Run the full enhanced pipeline on one case.

    Returns metrics dict.
    """
    tree, data_df, y_t, tc = build_tree_and_data(case_name)
    true_k = tc.get("n_clusters", None)

    pre, post, audit = _decompose_with_posthoc(
        tree,
        data_df,
        merge_alpha=merge_alpha,
        n_min=n_min,
        max_pt_depth=max_pt_depth,
    )

    ari_pre = compute_ari(pre, data_df, y_t) if y_t is not None else float("nan")
    ari_post = compute_ari(post, data_df, y_t) if y_t is not None else float("nan")

    return {
        "case": case_name,
        "true_k": true_k,
        "baseline_k": pre["num_clusters"],
        "enhanced_k": post["num_clusters"],
        "baseline_ari": round(ari_pre, 3),
        "enhanced_ari": round(ari_post, 3),
        "n_merges": sum(1 for a in audit if a["action"] == "MERGE"),
        "delta_k": post["num_clusters"] - (true_k or 0),
    }


def main() -> None:
    print("=" * 72)
    print("  EXPERIMENT 5: Combined Enhancement Pipeline")
    print("=" * 72)

    # ── Configuration grid ──
    configs = [
        {
            "label": "A: n_min=20, pt=3, merge_α=0.05",
            "n_min": 20,
            "max_pt_depth": 3,
            "merge_alpha": 0.05,
        },
        {
            "label": "B: n_min=30, pt=2, merge_α=0.05",
            "n_min": 30,
            "max_pt_depth": 2,
            "merge_alpha": 0.05,
        },
        {
            "label": "C: n_min=20, pt=3, merge_α=0.10",
            "n_min": 20,
            "max_pt_depth": 3,
            "merge_alpha": 0.10,
        },
        {
            "label": "D: n_min=40, pt=5, merge_α=0.05",
            "n_min": 40,
            "max_pt_depth": 5,
            "merge_alpha": 0.05,
        },
    ]

    cases = FAILURE_CASES + REGRESSION_GUARD_CASES

    all_rows = []
    for cfg in configs:
        print(f"\n--- Config {cfg['label']} ---")
        for case_name in cases:
            try:
                result = run_enhanced_pipeline(
                    case_name,
                    n_min=cfg["n_min"],
                    max_pt_depth=cfg["max_pt_depth"],
                    merge_alpha=cfg["merge_alpha"],
                )
                result["config"] = cfg["label"]
                all_rows.append(result)
            except Exception as e:
                all_rows.append(
                    {
                        "case": case_name,
                        "config": cfg["label"],
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(all_rows)

    # ── Best config analysis ──
    print(f"\n{'=' * 72}")
    print("  Results by Configuration")
    print(f"{'=' * 72}")

    for cfg in configs:
        label = cfg["label"]
        sub = df[df["config"] == label]
        if "error" in sub.columns:
            sub = sub[sub["error"].isna()] if "error" in sub.columns else sub

        if len(sub) == 0:
            print(f"\n{label}: No results")
            continue

        fail_sub = sub[sub["case"].isin(FAILURE_CASES)]
        guard_sub = sub[~sub["case"].isin(set(FAILURE_CASES))]

        print(f"\n{label}:")
        print(
            f"  Failure cases:    mean ARI={fail_sub['enhanced_ari'].mean():.3f}, "
            f"exact K={int((fail_sub['delta_k'] == 0).sum())}/{len(fail_sub)}, "
            f"mean |ΔK|={fail_sub['delta_k'].abs().mean():.1f}"
        )
        print(
            f"  Guard cases:      mean ARI={guard_sub['enhanced_ari'].mean():.3f}, "
            f"exact K={int((guard_sub['delta_k'] == 0).sum())}/{len(guard_sub)}, "
            f"mean |ΔK|={guard_sub['delta_k'].abs().mean():.1f}"
        )

    # ── Detailed table for best config ──
    print(f"\n{'=' * 72}")
    print("  Detailed Results: Config A")
    print(f"{'=' * 72}")

    cfg_a = df[df["config"].str.startswith("A")]
    if "error" in cfg_a.columns:
        cfg_a = cfg_a[cfg_a["error"].isna()] if cfg_a["error"].notna().any() else cfg_a
    cols = [
        "case",
        "true_k",
        "baseline_k",
        "enhanced_k",
        "baseline_ari",
        "enhanced_ari",
        "n_merges",
    ]
    available_cols = [c for c in cols if c in cfg_a.columns]
    print(cfg_a[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
