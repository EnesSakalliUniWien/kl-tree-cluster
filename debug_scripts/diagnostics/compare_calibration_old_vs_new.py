#!/usr/bin/env python3
"""Compare old vs new cousin_weighted_wald calibration on the 'clear' test case.

Identifies exactly which node decisions differ, and why.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from kl_clustering_analysis import config

# Import internals for low-level inspection
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
    _collect_weighted_pairs,
    _fit_weighted_inflation_model,
    _weighted_median,
    annotate_sibling_divergence_weighted,
    predict_weighted_inflation_factor,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree

warnings.filterwarnings("ignore")


def create_clear_case():
    """Reproduce the 'clear' SMALL_TEST_CASE."""
    rng = np.random.RandomState(42)
    n_per_cluster = 30
    p = 50
    n = n_per_cluster * 3
    X = np.zeros((n, p), dtype=int)
    # 3 clusters with clear separation
    for c in range(3):
        start = c * n_per_cluster
        end = (c + 1) * n_per_cluster
        feature_start = c * (p // 3)
        feature_end = (c + 1) * (p // 3)
        X[start:end, feature_start:feature_end] = rng.binomial(
            1, 0.8, size=(n_per_cluster, feature_end - feature_start)
        )
        # Noise in other features
        other_cols = list(range(feature_start)) + list(range(feature_end, p))
        X[start:end, other_cols] = rng.binomial(1, 0.2, size=(n_per_cluster, len(other_cols)))
    data = pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])
    return data


def simulate_old_fit(records):
    """Simulate OLD _fit_weighted_inflation_model behavior.

    Key differences from new code:
    - max_c from ALL pairs (not just null-like)
    - global_c from weighted MEDIAN (not weighted mean)
    - WLS regression on log(ratios) (not Gamma GLM)
    """
    valid = [r for r in records if np.isfinite(r.stat) and r.df > 0 and r.stat / r.df > 0]
    if not valid:
        return None

    ratios = np.array([r.stat / r.df for r in valid])
    weights = np.array([r.weight for r in valid])
    bl_sums = np.array([r.bl_sum for r in valid])
    n_parents = np.array([r.n_parent for r in valid])

    n_cal = len(ratios)
    old_global_c = float(_weighted_median(ratios, weights))
    old_max_c = float(np.max(ratios))  # OLD: from ALL pairs

    if n_cal < 5:
        return {
            "method": "weighted_median" if n_cal >= 3 else "none",
            "global_c": old_global_c,
            "max_c": old_max_c,
            "beta": None,
        }

    # OLD: WLS on log(ratios)
    log_r = np.log(ratios)
    X = np.column_stack(
        [
            np.ones(n_cal),
            np.log(bl_sums),
            np.log(n_parents.astype(float)),
        ]
    )
    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, np.newaxis]
    yw = log_r * sqrt_w

    try:
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "method": "weighted_median",
            "global_c": old_global_c,
            "max_c": old_max_c,
            "beta": None,
        }

    fitted = X @ beta
    ss_res = float(np.sum(weights * (log_r - fitted) ** 2))
    weighted_mean_logr = float(np.average(log_r, weights=weights))
    ss_tot = float(np.sum(weights * (log_r - weighted_mean_logr) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "method": "weighted_regression",
        "global_c": old_global_c,
        "max_c": old_max_c,
        "beta": beta,
        "r_squared": r_squared,
    }


def predict_old(old_result, bl_sum, n_parent):
    """Predict ĉ using OLD code logic."""
    if old_result is None or old_result["method"] == "none":
        return 1.0
    if old_result["beta"] is None or old_result["method"] == "weighted_median":
        return max(old_result["global_c"], 1.0)
    if bl_sum <= 0 or n_parent <= 0:
        return max(old_result["global_c"], 1.0)

    log_c = (
        old_result["beta"][0]
        + old_result["beta"][1] * np.log(bl_sum)
        + old_result["beta"][2] * np.log(float(n_parent))
    )
    c_hat = float(np.exp(log_c))
    c_hat = min(c_hat, old_result["max_c"])  # OLD: clamp at ALL-pairs max
    return max(c_hat, 1.0)


def main():
    data = create_clear_case()
    print(f"Data shape: {data.shape}")

    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)

    # Run annotation steps
    from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
        annotate_child_parent_divergence,
    )

    results_df = tree.stats_df
    results_df = annotate_child_parent_divergence(
        tree, results_df, significance_level_alpha=config.SIGNIFICANCE_ALPHA
    )

    # Compute mean branch length
    from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
        compute_mean_branch_length,
    )

    mean_bl = compute_mean_branch_length(tree)

    # Collect raw pairs for comparison
    records = _collect_weighted_pairs(tree, results_df, mean_bl)
    print(f"\nCollected {len(records)} weighted pairs")

    # Simulate OLD fit
    old_result = simulate_old_fit(records)
    print(f"\n{'='*80}")
    print("OLD calibration (WLS, all-pairs max_c, weighted median global_c):")
    print(f"  method:   {old_result['method']}")
    print(f"  global_c: {old_result['global_c']:.6f}")
    print(f"  max_c:    {old_result['max_c']:.6f}")
    if old_result["beta"] is not None:
        print(
            f"  beta:     [{old_result['beta'][0]:.4f}, {old_result['beta'][1]:.4f}, {old_result['beta'][2]:.4f}]"
        )
        print(f"  R²:       {old_result['r_squared']:.4f}")

    # NEW fit
    new_model = _fit_weighted_inflation_model(records)
    print("\nNEW calibration (Gamma GLM, null-like max_c, weighted mean global_c):")
    print(f"  method:   {new_model.method}")
    print(f"  global_c: {new_model.global_c_hat:.6f}")
    print(f"  max_c:    {new_model.max_observed_ratio:.6f}")
    if new_model.beta is not None:
        print(
            f"  beta:     [{new_model.beta[0]:.4f}, {new_model.beta[1]:.4f}, {new_model.beta[2]:.4f}]"
        )
        print(f"  R²:       {new_model.diagnostics.get('r_squared', 0):.4f}")

    # Compare predictions for ALL focal pairs
    print(f"\n{'='*80}")
    print("Per-pair comparison of predictions:")
    print(
        f"{'Parent':<10} {'null?':<6} {'T/k':<8} {'BL_sum':<10} {'n_par':<8} {'old_ĉ':<10} {'new_ĉ':<10} {'old_T_adj':<12} {'new_T_adj':<12} {'old_p':<12} {'new_p':<12}"
    )
    print("-" * 120)

    from scipy.stats import chi2

    focal_records = [r for r in records if not r.is_null_like]
    for r in sorted(records, key=lambda x: x.parent):
        ratio = r.stat / r.df if r.df > 0 else float("nan")
        old_c = predict_old(old_result, r.bl_sum, r.n_parent)
        new_c = predict_weighted_inflation_factor(new_model, r.bl_sum, r.n_parent)

        old_t_adj = r.stat / old_c
        new_t_adj = r.stat / new_c

        old_p = float(chi2.sf(old_t_adj, df=r.df))
        new_p = float(chi2.sf(new_t_adj, df=r.df))

        marker = " <-- DIFFERENT" if abs(old_p - new_p) > 0.01 else ""
        null_str = "NULL" if r.is_null_like else "FOCAL"

        print(
            f"{r.parent:<10} {null_str:<6} {ratio:<8.3f} {r.bl_sum:<10.4f} {r.n_parent:<8} "
            f"{old_c:<10.4f} {new_c:<10.4f} "
            f"{old_t_adj:<12.4f} {new_t_adj:<12.4f} "
            f"{old_p:<12.6f} {new_p:<12.6f}{marker}"
        )

    # Run full annotation to see final decisions
    print(f"\n{'='*80}")
    print("Running full annotation with NEW code:")
    config.SIBLING_TEST_METHOD = "cousin_weighted_wald"
    results_df_new = annotate_sibling_divergence_weighted(
        tree, results_df, significance_level_alpha=config.SIBLING_ALPHA
    )

    sibling_cols = [c for c in results_df_new.columns if "Sibling" in c]
    internal_nodes = [n for n in results_df_new.index if not str(n).startswith("L")]
    relevant = results_df_new.loc[internal_nodes, sibling_cols]
    print(relevant.to_string())

    # Now run decomposition
    print(f"\n{'='*80}")
    print("Decomposition results:")
    for method in ["cousin_weighted_wald", "wald"]:
        config.SIBLING_TEST_METHOD = method
        results = tree.decompose(
            leaf_data=data,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=config.SIBLING_ALPHA,
            posthoc_merge=False,  # Isolate the decomposition
        )
        K = results.get("n_clusters", results.get("k", "?"))
        print(f"  {method}: K={K}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
