"""Diagnose the K=1 over-deflation problem in cousin_adjusted_wald.

Traces the exact failure path for cases where adj_wald collapses to K=1
while raw wald finds the correct K. Shows:
  - How many null-like vs focal pairs exist
  - Regression coefficients (β₀, β₁, β₂) and R²
  - Predicted ĉ at root vs at deeper nodes
  - What the raw Wald T and adjusted T_adj are at the root split
  - Whether capping ĉ or dropping β₂ would fix it
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import logging
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald import (
    _collect_all_pairs,
    _fit_inflation_model,
    _predict_c,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)

logging.basicConfig(level=logging.WARNING)


def run_case_three_methods(case):
    """Run one case with all three methods, return K for each."""
    data_df, true_labels, x_original, metadata = generate_case_data(case)

    case_type = case.get("type", "gaussian")
    if case_type == "sbm" and metadata.get("distance_condensed") is not None:
        dist_c = metadata["distance_condensed"]
    else:
        dist_c = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)

    Z = linkage(dist_c, method=config.TREE_LINKAGE_METHOD)

    results = {}
    for method in ["wald", "cousin_adjusted_wald"]:
        config.SIBLING_TEST_METHOD = method
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
        results[method] = {
            "k": decomp["num_clusters"],
            "tree": tree,
            "decomp": decomp,
        }

    return data_df, true_labels, Z, results


def diagnose_adjusted_wald(tree, data_df):
    """Deep dive into the adj_wald calibration for a given tree."""
    sdf = tree.stats_df
    if sdf is None:
        print("  No stats_df available")
        return

    audit = sdf.attrs.get("sibling_divergence_audit", {})
    diag = audit.get("diagnostics", {})

    print(f"  Calibration method: {audit.get('calibration_method', 'N/A')}")
    print(f"  Null-like pairs:    {audit.get('null_like_pairs', 'N/A')}")
    print(f"  Focal pairs:        {audit.get('focal_pairs', 'N/A')}")
    print(f"  Global ĉ (median):  {audit.get('global_c_hat', 'N/A')}")
    print(f"  R²:                 {diag.get('r_squared', 'N/A')}")
    print(f"  β coefficients:     {diag.get('beta', 'N/A')}")

    # Re-collect pairs to inspect per-node predictions
    mean_bl = compute_mean_branch_length(tree)
    records = _collect_all_pairs(tree, sdf, mean_bl)
    model = _fit_inflation_model(records)

    print(f"\n  --- Per-node ĉ predictions (regression model) ---")
    print(f"  {'Parent':<10} {'n_parent':>8} {'BL_sum':>8} {'T_raw':>8} {'k':>4} {'ĉ':>8} {'T_adj':>8} {'p_adj':>8} {'null?':>6}")
    print(f"  {'-'*80}")

    # Sort by n_parent descending (root first)
    records_sorted = sorted(records, key=lambda r: r.n_parent, reverse=True)
    for rec in records_sorted:
        c_hat = _predict_c(model, rec.bl_sum, rec.n_parent)
        t_adj = rec.stat / c_hat if c_hat > 0 else rec.stat
        p_adj = float(chi2.sf(t_adj, df=rec.df)) if rec.df > 0 else np.nan
        marker = "NULL" if rec.is_null_like else ""
        print(
            f"  {rec.parent:<10} {rec.n_parent:>8d} {rec.bl_sum:>8.4f} "
            f"{rec.stat:>8.2f} {rec.df:>4d} {c_hat:>8.3f} {t_adj:>8.2f} {p_adj:>8.4f} {marker:>6}"
        )

    # Simulate fix: cap ĉ at max observed ratio
    if model.method == "regression" and model.beta is not None:
        null_records = [r for r in records if r.is_null_like and np.isfinite(r.stat) and r.df > 0]
        if null_records:
            ratios = [r.stat / r.df for r in null_records]
            max_observed_c = max(ratios)
            median_c = float(np.median(ratios))
            p95_c = float(np.percentile(ratios, 95))

            print(f"\n  --- Calibration pair statistics ---")
            print(f"  Null-like ratios (T/k): {[f'{r:.2f}' for r in sorted(ratios)]}")
            print(f"  Median: {median_c:.3f}, Max: {max_observed_c:.3f}, P95: {p95_c:.3f}")

            print(f"\n  --- Simulated fixes for focal pairs ---")
            print(f"  {'Parent':<10} {'n_parent':>8} {'T_raw':>8} {'ĉ_reg':>8} {'ĉ_cap':>8} {'ĉ_noB2':>8} {'p_reg':>8} {'p_cap':>8} {'p_noB2':>8}")
            print(f"  {'-'*90}")

            for rec in records_sorted:
                if rec.is_null_like:
                    continue
                if not np.isfinite(rec.stat) or rec.df <= 0:
                    continue

                # Original regression ĉ
                c_reg = _predict_c(model, rec.bl_sum, rec.n_parent)

                # Fix 1: Cap ĉ at max observed
                c_cap = min(c_reg, max_observed_c)
                c_cap = max(c_cap, 1.0)

                # Fix 2: Drop β₂ (n_parent), use only β₀ + β₁·log(BL)
                if rec.bl_sum > 0:
                    log_c_noB2 = model.beta[0] + model.beta[1] * np.log(rec.bl_sum)
                    c_noB2 = max(float(np.exp(log_c_noB2)), 1.0)
                else:
                    c_noB2 = max(model.global_c_hat, 1.0)

                t_reg = rec.stat / c_reg
                t_cap = rec.stat / c_cap
                t_noB2 = rec.stat / c_noB2

                p_reg = float(chi2.sf(t_reg, df=rec.df))
                p_cap = float(chi2.sf(t_cap, df=rec.df))
                p_noB2 = float(chi2.sf(t_noB2, df=rec.df))

                print(
                    f"  {rec.parent:<10} {rec.n_parent:>8d} {rec.stat:>8.2f} "
                    f"{c_reg:>8.3f} {c_cap:>8.3f} {c_noB2:>8.3f} "
                    f"{p_reg:>8.4f} {p_cap:>8.4f} {p_noB2:>8.4f}"
                )


def main():
    print("=" * 90)
    print("DIAGNOSING K=1 OVER-DEFLATION IN cousin_adjusted_wald")
    print("=" * 90)

    test_cases = get_default_test_cases()

    # Find cases where adj_wald gives K=1 but wald gives K>1
    k1_cases = []

    print("\nPhase 1: Scanning all cases for K=1 collapses...\n")
    for i, case in enumerate(test_cases):
        name = case.get("name", f"case_{i}")
        true_k = case.get("n_clusters", "?")

        try:
            data_df, true_labels, Z, results = run_case_three_methods(case)
        except Exception as e:
            print(f"  [{i+1}] {name}: ERROR - {e}")
            continue

        k_wald = results["wald"]["k"]
        k_adj = results["cousin_adjusted_wald"]["k"]

        if k_adj == 1 and k_wald > 1:
            print(f"  [{i+1}] {name}: TRUE_K={true_k}, wald=K{k_wald}, adj_wald=K{k_adj} *** K=1 COLLAPSE ***")
            k1_cases.append((case, data_df, true_labels, Z, results))
        elif k_adj != k_wald:
            print(f"  [{i+1}] {name}: TRUE_K={true_k}, wald=K{k_wald}, adj_wald=K{k_adj}")

    print(f"\n{'=' * 90}")
    print(f"Found {len(k1_cases)} cases with K=1 collapse")
    print(f"{'=' * 90}")

    # Deep dive into each K=1 case
    for case, data_df, true_labels, Z, results in k1_cases:
        name = case.get("name", "unknown")
        true_k = case.get("n_clusters", "?")
        print(f"\n{'─' * 90}")
        print(f"CASE: {name} (true_k={true_k}, n={len(data_df)}, p={data_df.shape[1]})")
        print(f"  wald K={results['wald']['k']}, adj_wald K={results['cousin_adjusted_wald']['k']}")
        print(f"{'─' * 90}")

        tree = results["cousin_adjusted_wald"]["tree"]
        diagnose_adjusted_wald(tree, data_df)


if __name__ == "__main__":
    main()
