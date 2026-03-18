#!/usr/bin/env python3
"""Bisect which config change caused the regression.

Tests 4 config permutations on sentinel cases that regressed heavily
between Feb 15 (ARI=0.823) and Mar 17 (ARI=0.671) benchmarks.

Config changes at commit 4971059 (2026-03-04):
  SIBLING_ALPHA: 0.05 → 0.01
  SIBLING_TEST_METHOD: weighted Wald lineage → cousin_adjusted_wald

The exact weighted-Wald baseline is no longer available on current HEAD, so this
script compares the supported sibling methods and alpha settings that remain
executable.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from debug_scripts.enhancement_lab.lab_helpers import temporary_config
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Sentinel cases: perfect in Feb 15, regressed heavily by Mar 17
SENTINEL_CASES = [
    "binary_balanced_low_noise__2",  # 1.0 → 0.0
    "cat_clear_4cat_4c",  # 1.0 → 0.0
    "overlap_hd_4c_1k",  # 0.99 → 0.0
    "gauss_clear_small",  # 1.0 → 0.554
    "binary_hard_4c",  # 1.0 → 0.708
    "binary_perfect_8c",  # 1.0 → 0.757
    "binary_low_noise_12c",  # 1.0 → 0.614
    "gauss_moderate_5c",  # 1.0 → 0.362 (guard regression)
]

# Config permutations to test
CONFIGS = {
    "wald_alpha_005": {
        "SIBLING_ALPHA": 0.05,
        "SIBLING_TEST_METHOD": "wald",
    },
    "wald_alpha_001": {
        "SIBLING_ALPHA": 0.01,
        "SIBLING_TEST_METHOD": "wald",
    },
    "adjusted_alpha_005": {
        "SIBLING_ALPHA": 0.05,
        "SIBLING_TEST_METHOD": "cousin_adjusted_wald",
    },
    "adjusted_alpha_001": {
        "SIBLING_ALPHA": 0.01,
        "SIBLING_TEST_METHOD": "cousin_adjusted_wald",
    },
}


def run_case(name: str, sibling_alpha: float, sibling_method: str) -> dict:
    """Run a single case with specified config."""
    all_cases = get_default_test_cases()
    tc = next(c for c in all_cases if c["name"] == name)
    if "n_clusters" not in tc and "sizes" in tc:
        tc["n_clusters"] = len(tc["sizes"])

    data_df, y_true, _, _ = generate_case_data(tc)
    dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)

    # Override config for this run
    with temporary_config(SIBLING_TEST_METHOD=sibling_method):
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=sibling_alpha,  # benchmark uses same alpha for both
            sibling_alpha=sibling_alpha,
        )

    n = len(data_df)
    y_pred = np.full(n, -1, dtype=int)
    for cid, cinfo in decomp["cluster_assignments"].items():
        for leaf in cinfo["leaves"]:
            y_pred[data_df.index.get_loc(leaf)] = cid

    true_k = tc.get("n_clusters")
    found_k = decomp["num_clusters"]
    ari = adjusted_rand_score(y_true, y_pred) if y_true is not None else float("nan")
    return {"true_k": true_k, "found_k": found_k, "ari": round(ari, 3)}


def main():
    rows = []
    for case_name in SENTINEL_CASES:
        for cfg_label, cfg in CONFIGS.items():
            try:
                r = run_case(case_name, cfg["SIBLING_ALPHA"], cfg["SIBLING_TEST_METHOD"])
                rows.append(
                    {
                        "case": case_name,
                        "config": cfg_label,
                        "true_k": r["true_k"],
                        "found_k": r["found_k"],
                        "ari": r["ari"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "case": case_name,
                        "config": cfg_label,
                        "error": str(e)[:60],
                    }
                )

    df = pd.DataFrame(rows)

    # Pivot for readable comparison
    pivot = df.pivot(index="case", columns="config", values="ari")
    pivot = pivot[["wald_alpha_005", "wald_alpha_001", "adjusted_alpha_005", "adjusted_alpha_001"]]
    print("=" * 80)
    print("  CONFIG BISECT: ARI by supported config permutation")
    print("  wald_alpha_005     = alpha=0.05 + wald")
    print("  wald_alpha_001     = alpha=0.01 + wald")
    print("  adjusted_alpha_005 = alpha=0.05 + adjusted_wald")
    print("  adjusted_alpha_001 = alpha=0.01 + adjusted_wald (current)")
    print("=" * 80)
    print(pivot.to_string())

    print("\n--- Mean ARI ---")
    for col in pivot.columns:
        print(f"  {col}: {pivot[col].mean():.3f}")

    # Also show K found
    pivot_k = df.pivot(index="case", columns="config", values="found_k")
    pivot_k = pivot_k[
        ["wald_alpha_005", "wald_alpha_001", "adjusted_alpha_005", "adjusted_alpha_001"]
    ]
    print("\n--- K found ---")
    print(pivot_k.to_string())


if __name__ == "__main__":
    main()
