"""
Purpose: Diagnostic: Sibling test calibration when run unconditionally at ALL binary parents.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/pipeline_gates/q_unconditional_sibling_calibration__sibling_tests__null_sim.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _get_binary_children,
    _get_sibling_data,
    annotate_sibling_divergence,
    sibling_divergence_test,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def generate_null_data(n_samples=200, n_features=50, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    return pd.DataFrame(
        X,
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )


def run_unconditional_sibling_trial(seed=42, n=200, p=50, verbose=True):
    """Run sibling tests at ALL binary parents, regardless of edge test."""
    data = generate_null_data(n_samples=n, n_features=p, seed=seed)

    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)

    mean_bl = compute_mean_branch_length(tree)

    # Collect ALL binary parents and run sibling tests unconditionally
    all_parents = []
    all_pvals = []
    all_stats = []
    n_ratios = []

    for node in tree.nodes():
        children = _get_binary_children(tree, node)
        if children is None:
            continue

        left, right = children
        left_dist, right_dist, n_left, n_right, bl_left, bl_right = _get_sibling_data(
            tree, node, left, right
        )

        if n_left < 2 or n_right < 2:
            continue

        stat, df, pval = sibling_divergence_test(
            left_dist,
            right_dist,
            n_left,
            n_right,
            bl_left,
            bl_right,
            mean_bl,
            test_id=f"sibling:{node}",
        )

        all_parents.append(node)
        all_pvals.append(pval if np.isfinite(pval) else 1.0)
        all_stats.append(stat if np.isfinite(stat) else 0.0)

        n_parent = n_left + n_right
        ratio = min(n_left, n_right) / n_parent
        n_ratios.append(ratio)

    pvals_arr = np.array(all_pvals)
    m = len(pvals_arr)

    # BH correction over ALL binary parents
    reject_bh, padj, _ = benjamini_hochberg_correction(pvals_arr, alpha=0.05)

    # Also run the CURRENT pipeline (edge-gated) for comparison
    results_df = tree.stats_df.copy()
    results_df_edge = annotate_child_parent_divergence(
        tree, results_df.copy(), significance_level_alpha=0.05, fdr_method="tree_bh"
    )
    results_df_full = annotate_sibling_divergence(
        tree, results_df_edge, significance_level_alpha=0.05
    )
    gated_tested = results_df_full[
        results_df_full["Sibling_Divergence_P_Value"].notna()
        & (results_df_full["Sibling_Divergence_P_Value"] < 1.0)
    ]
    n_gated_tested = len(gated_tested)
    n_gated_reject = int(gated_tested["Sibling_BH_Different"].sum()) if n_gated_tested > 0 else 0

    raw_lt_05 = (pvals_arr < 0.05).sum()

    if verbose:
        print(
            f"  Unconditional sibling: m={m}, raw p<0.05={raw_lt_05} ({raw_lt_05/m:.1%}), "
            f"BH reject={reject_bh.sum()} ({reject_bh.sum()/m:.1%}) | "
            f"Edge-gated: tested={n_gated_tested}/{m}, reject={n_gated_reject}"
        )

    return {
        "m_total": m,
        "raw_lt_05": int(raw_lt_05),
        "raw_rate": raw_lt_05 / m,
        "bh_reject": int(reject_bh.sum()),
        "bh_rate": reject_bh.sum() / m,
        "gated_tested": n_gated_tested,
        "gated_reject": n_gated_reject,
        "n_ratios": n_ratios,
        "pvals": pvals_arr,
    }


def run_multi_unconditional(n_trials=20, n=200, p=50):
    """Run multiple trials and aggregate unconditional vs gated sibling results."""
    print("=" * 80)
    print(f"UNCONDITIONAL SIBLING TEST: {n_trials} null trials, n={n}, p={p}")
    print("=" * 80)

    records = []
    all_pvals = []
    all_ratios = []

    for i in range(n_trials):
        r = run_unconditional_sibling_trial(seed=3000 + i, n=n, p=p, verbose=True)
        records.append(r)
        all_pvals.extend(r["pvals"].tolist())
        all_ratios.extend(r["n_ratios"])

    df = pd.DataFrame(records)
    print()
    print("AGGREGATE:")
    print(f"  Total binary parents per tree:       {df['m_total'].mean():.0f}")
    print()
    print("  UNCONDITIONAL SIBLING (proposed):")
    print(
        f"    Raw p < 0.05:                      {df['raw_lt_05'].mean():.1f}/{df['m_total'].mean():.0f} "
        f"({df['raw_rate'].mean():.1%})"
    )
    print(
        f"    BH rejections (m={df['m_total'].mean():.0f}):            "
        f"{df['bh_reject'].mean():.1f} ({df['bh_rate'].mean():.1%})"
    )
    print()
    print("  EDGE-GATED SIBLING (current):")
    print(
        f"    Tested:                            {df['gated_tested'].mean():.1f}/{df['m_total'].mean():.0f} "
        f"({df['gated_tested'].mean()/df['m_total'].mean():.1%})"
    )
    print(
        f"    Rejections:                        {df['gated_reject'].mean():.1f} "
        f"({df['gated_reject'].mean()/max(df['gated_tested'].mean(), 1):.1%} of tested)"
    )

    # Stratify unconditional sibling raw p-values by n_ratio
    all_pvals_arr = np.array(all_pvals)
    all_ratios_arr = np.array(all_ratios)

    print()
    print("  UNCONDITIONAL SIBLING: raw p<0.05 by min(n_L,n_R)/n_P:")
    bins = [
        (0, 0.05, "<0.05"),
        (0.05, 0.1, "0.05-0.1"),
        (0.1, 0.2, "0.1-0.2"),
        (0.2, 0.3, "0.2-0.3"),
        (0.3, 0.4, "0.3-0.4"),
        (0.4, 0.5, "0.4-0.5"),
    ]
    for lo, hi, label in bins:
        mask = (all_ratios_arr > lo) & (all_ratios_arr <= hi)
        if mask.sum() == 0:
            continue
        sub = all_pvals_arr[mask]
        rate = (sub < 0.05).mean()
        print(f"    {label:>10}:  {rate:.1%}  (n={mask.sum()})")

    # P-value uniformity check
    print()
    print("  P-VALUE UNIFORMITY (Kolmogorov-Smirnov vs Uniform[0,1]):")
    from scipy.stats import kstest

    finite_pvals = all_pvals_arr[np.isfinite(all_pvals_arr)]
    ks_stat, ks_pval = kstest(finite_pvals, "uniform")
    print(f"    KS statistic: {ks_stat:.4f}, KS p-value: {ks_pval:.4e}")
    print(f"    Median p-value: {np.median(finite_pvals):.4f} (should be ~0.5)")
    print(f"    Mean p-value:   {np.mean(finite_pvals):.4f} (should be ~0.5)")

    # Deciles
    deciles = np.percentile(finite_pvals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    print(f"    Deciles: {', '.join(f'{d:.3f}' for d in deciles)}")
    print("    Expected: 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900")

    return df


if __name__ == "__main__":
    run_multi_unconditional(n_trials=20, n=200, p=50)
