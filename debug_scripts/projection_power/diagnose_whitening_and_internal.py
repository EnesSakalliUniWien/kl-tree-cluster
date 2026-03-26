#!/usr/bin/env python3
"""A/B diagnosis: EIGENVALUE_WHITENING × INCLUDE_INTERNAL_IN_SPECTRAL.

Tests all 4 combinations of these two config flags across a representative
set of benchmark cases (including null cases) and reports:
  - Found K, True K, ARI
  - Root spectral dimension (k)
  - Gate 2 edge-significant count at root
  - Regressions (K changed for the worse) highlighted

Usage:
  python debug_scripts/projection_power/diagnose_whitening_and_internal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

import kl_clustering_analysis.config as config
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Representative cases covering: easy, moderate, hard, null, many-cluster, sparse
TARGET_CASES = [
    "gauss_clear_small",
    "gauss_clear_large",
    "gauss_moderate_3c",
    "gauss_moderate_5c",
    "gauss_null_small",
    "gauss_null_large",
    "binary_perfect_2c",
    "binary_perfect_4c",
    "binary_perfect_8c",
    "binary_low_noise_4c",
    "binary_low_noise_8c",
    "binary_moderate_4c",
    "binary_null_small",
    "binary_null_medium",
    "sparse_features_72x72",
    "sparse_features_100x500",
    "binary_many_features",
]

# The 4 config combinations to test
CONFIGS = [
    {"whitening": True, "include_internal": True, "label": "W=T I=T (old default)"},
    {"whitening": True, "include_internal": False, "label": "W=T I=F"},
    {"whitening": False, "include_internal": True, "label": "W=F I=T"},
    {"whitening": False, "include_internal": False, "label": "W=F I=F (new default)"},
]


def find_case(name: str) -> dict | None:
    for c in get_default_test_cases():
        if c.get("name") == name:
            return c
    return None


def run_case(case_config: dict, whitening: bool, include_internal: bool):
    """Run decomposition with specific config and return (K, ARI, root_k)."""
    # Set config
    config.EIGENVALUE_WHITENING = whitening
    config.INCLUDE_INTERNAL_IN_SPECTRAL = include_internal

    data, labels, _x_orig, _meta = generate_case_data(case_config)
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        n, p = data.shape
        df = pd.DataFrame(
            data, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)]
        )

    Z = linkage(
        pdist(df.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())
    tree.populate_node_divergences(df)

    results = tree.decompose(leaf_data=df, alpha_local=0.05, sibling_alpha=0.05)
    found_k = results["num_clusters"]

    # Compute ARI
    cluster_assignments = results["cluster_assignments"]
    leaf_names = df.index.tolist()
    label_map = {}
    for cid, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cid
    pred = np.array([label_map.get(leaf_names[i], -1) for i in range(len(leaf_names))])
    ari = adjusted_rand_score(labels, pred) if labels is not None else float("nan")

    # Get root spectral dim from annotations_df
    annotations_df = tree.annotations_df
    spectral_dims = annotations_df.attrs.get("_spectral_dims", {})
    root = tree.root()
    root_k = spectral_dims.get(root) if spectral_dims else None

    # Count edge-significant nodes
    n_edge_sig = 0
    if "Child_Parent_Divergence_Significant" in annotations_df.columns:
        n_edge_sig = int(annotations_df["Child_Parent_Divergence_Significant"].sum())

    return {
        "found_k": found_k,
        "ari": ari,
        "root_k": root_k,
        "n_edge_sig": n_edge_sig,
    }


def main():
    print("=" * 110)
    print("WHITENING × INCLUDE_INTERNAL A/B DIAGNOSIS")
    print(
        "SPECTRAL_DIMENSION_ESTIMATOR=marchenko_pastur (fixed), "
        f"SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}"
    )
    print("=" * 110)

    # Collect all case configs first
    cases = []
    for name in TARGET_CASES:
        case = find_case(name)
        if case is None:
            print(f"  SKIP: {name} not found")
            continue
        true_k = case.get("n_clusters", case.get("true_k", "?"))
        cases.append((name, case, true_k))

    # Run all combinations
    # results[case_name][config_label] = {found_k, ari, root_k, n_edge_sig}
    all_results: dict[str, dict[str, dict]] = {}

    for name, case, true_k in cases:
        all_results[name] = {"true_k": true_k}
        for cfg in CONFIGS:
            try:
                r = run_case(case, cfg["whitening"], cfg["include_internal"])
                all_results[name][cfg["label"]] = r
            except Exception as e:
                all_results[name][cfg["label"]] = {
                    "found_k": "ERR",
                    "ari": float("nan"),
                    "root_k": None,
                    "n_edge_sig": 0,
                }
                print(f"  ERROR on {name} [{cfg['label']}]: {e}")

    # ====== SUMMARY TABLE: K found ======
    print(f"\n{'='*110}")
    print("FOUND K COMPARISON (bold = regression from old default)")
    print(f"{'='*110}")

    header = f"{'Case':<30} {'TrK':>4}"
    for cfg in CONFIGS:
        header += f"  {cfg['label']:>18}"
    print(header)
    print("-" * 110)

    n_regressions = {cfg["label"]: 0 for cfg in CONFIGS}
    for name, case, true_k in cases:
        row = f"{name:<30} {true_k:>4}"
        old_k = all_results[name].get(CONFIGS[0]["label"], {}).get("found_k", "?")
        for cfg in CONFIGS:
            r = all_results[name].get(cfg["label"], {})
            k = r.get("found_k", "?")
            # Mark regressions: K moved further from true_k compared to old default
            marker = ""
            if isinstance(k, int) and isinstance(old_k, int) and isinstance(true_k, int):
                old_err = abs(old_k - true_k)
                new_err = abs(k - true_k)
                if new_err > old_err:
                    marker = " !"
                    n_regressions[cfg["label"]] += 1
                elif new_err < old_err:
                    marker = " +"
            row += f"  {str(k) + marker:>18}"
        print(row)

    print("-" * 110)
    rrow = f"{'Regressions':<30} {'':>4}"
    for cfg in CONFIGS:
        rrow += f"  {n_regressions[cfg['label']]:>18}"
    print(rrow)

    # ====== SUMMARY TABLE: ARI ======
    print(f"\n{'='*110}")
    print("ARI COMPARISON")
    print(f"{'='*110}")

    header = f"{'Case':<30} {'TrK':>4}"
    for cfg in CONFIGS:
        header += f"  {cfg['label']:>18}"
    print(header)
    print("-" * 110)

    ari_sums = {cfg["label"]: [] for cfg in CONFIGS}
    for name, case, true_k in cases:
        row = f"{name:<30} {true_k:>4}"
        for cfg in CONFIGS:
            r = all_results[name].get(cfg["label"], {})
            ari = r.get("ari", float("nan"))
            ari_str = f"{ari:.3f}" if isinstance(ari, float) and np.isfinite(ari) else "N/A"
            row += f"  {ari_str:>18}"
            if isinstance(ari, float) and np.isfinite(ari):
                ari_sums[cfg["label"]].append(ari)
        print(row)

    print("-" * 110)
    mean_row = f"{'Mean ARI':<30} {'':>4}"
    for cfg in CONFIGS:
        vals = ari_sums[cfg["label"]]
        m = np.mean(vals) if vals else float("nan")
        mean_row += f"  {m:>18.3f}"
    print(mean_row)

    median_row = f"{'Median ARI':<30} {'':>4}"
    for cfg in CONFIGS:
        vals = ari_sums[cfg["label"]]
        m = np.median(vals) if vals else float("nan")
        median_row += f"  {m:>18.3f}"
    print(median_row)

    # ====== SUMMARY TABLE: Root spectral k ======
    print(f"\n{'='*110}")
    print("ROOT SPECTRAL DIMENSION (k)")
    print(f"{'='*110}")

    header = f"{'Case':<30} {'TrK':>4}"
    for cfg in CONFIGS:
        header += f"  {cfg['label']:>18}"
    print(header)
    print("-" * 110)

    for name, case, true_k in cases:
        row = f"{name:<30} {true_k:>4}"
        for cfg in CONFIGS:
            r = all_results[name].get(cfg["label"], {})
            rk = r.get("root_k")
            rk_str = str(rk) if rk is not None else "?"
            row += f"  {rk_str:>18}"
        print(row)

    # ====== SUMMARY TABLE: Edge-significant count ======
    print(f"\n{'='*110}")
    print("EDGE-SIGNIFICANT COUNT (Gate 2 rejections)")
    print(f"{'='*110}")

    header = f"{'Case':<30} {'TrK':>4}"
    for cfg in CONFIGS:
        header += f"  {cfg['label']:>18}"
    print(header)
    print("-" * 110)

    for name, case, true_k in cases:
        row = f"{name:<30} {true_k:>4}"
        for cfg in CONFIGS:
            r = all_results[name].get(cfg["label"], {})
            n_sig = r.get("n_edge_sig", 0)
            row += f"  {n_sig:>18}"
        print(row)

    # ====== HIGHLIGHT REGRESSIONS ======
    print(f"\n{'='*110}")
    print("REGRESSION DETAILS (cases where new config is worse than old default)")
    print(f"{'='*110}")
    found_any = False
    old_label = CONFIGS[0]["label"]
    for cfg in CONFIGS[1:]:
        for name, case, true_k in cases:
            old_r = all_results[name].get(old_label, {})
            new_r = all_results[name].get(cfg["label"], {})
            old_k = old_r.get("found_k", "?")
            new_k = new_r.get("found_k", "?")
            if isinstance(old_k, int) and isinstance(new_k, int) and isinstance(true_k, int):
                old_err = abs(old_k - true_k)
                new_err = abs(new_k - true_k)
                if new_err > old_err:
                    found_any = True
                    print(
                        f"  {cfg['label']:>18}: {name:<30} true_k={true_k}, "
                        f"old K={old_k} (err={old_err}), new K={new_k} (err={new_err})"
                    )
    if not found_any:
        print("  None — all configs are equal or better than old default.")

    print()


if __name__ == "__main__":
    main()
