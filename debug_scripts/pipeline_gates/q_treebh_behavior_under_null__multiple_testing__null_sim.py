"""
Purpose: Diagnostic: Verify BH correction behavior under null hypothesis.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/pipeline_gates/q_treebh_behavior_under_null__multiple_testing__null_sim.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.tree_utils import compute_node_depths
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base import (
    benjamini_hochberg_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.flat_correction import (
    flat_bh_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.tree_bh_correction import (
    _get_families_by_parent,
    tree_bh_correction,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _collect_test_arguments,
    annotate_sibling_divergence,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def generate_null_data(n_samples: int = 200, n_features: int = 50, seed: int = 42):
    """Generate pure null data: all rows i.i.d. Bernoulli(0.5)."""
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    data = pd.DataFrame(
        X,
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    return data


def run_single_null_trial(seed: int = 42, n: int = 200, p: int = 50, verbose: bool = True):
    """Run one null trial and return diagnostic info."""
    data = generate_null_data(n_samples=n, n_features=p, seed=seed)

    # Build tree
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    tree.populate_node_divergences(data)
    results_df = tree.stats_df.copy()

    # --- Part 1: TreeBH family structure analysis ---
    if verbose:
        print("=" * 70)
        print("PART 1: TreeBH FAMILY STRUCTURE IN BINARY TREE")
        print("=" * 70)

    edge_list = list(tree.edges())
    child_ids = [c for _, c in edge_list]
    families = _get_families_by_parent(tree, child_ids)

    family_sizes = [len(indices) for indices in families.values()]
    if verbose:
        print(f"Total families (parents): {len(families)}")
        print(
            f"Family sizes: min={min(family_sizes)}, max={max(family_sizes)}, "
            f"median={np.median(family_sizes):.0f}"
        )
        print(
            f"Family size distribution: {dict(zip(*np.unique(family_sizes, return_counts=True)))}"
        )
        print("  → In a binary tree, ALL families should be size 2")
        print()

    # --- Part 2: Run edge test with tree_bh and inspect thresholds ---
    if verbose:
        print("=" * 70)
        print("PART 2: EDGE TEST (tree_bh) — THRESHOLDS AND REJECTIONS")
        print("=" * 70)

    results_df_treebh = annotate_child_parent_divergence(
        tree, results_df.copy(), significance_level_alpha=0.05, fdr_method="tree_bh"
    )

    edge_nodes = results_df_treebh[results_df_treebh["Child_Parent_Divergence_P_Value"].notna()]
    n_edges = len(edge_nodes)
    n_reject_treebh = edge_nodes["Child_Parent_Divergence_Significant"].sum()
    raw_pvals = edge_nodes["Child_Parent_Divergence_P_Value"].values
    n_raw_lt_05 = (raw_pvals < 0.05).sum()

    if verbose:
        print(f"Total edges tested: {n_edges}")
        print(f"Raw p < 0.05: {n_raw_lt_05} ({n_raw_lt_05/n_edges:.1%})")
        print(f"TreeBH rejections: {n_reject_treebh} ({n_reject_treebh/n_edges:.1%})")
        print(
            "  → Under null, raw p<0.05 should be ~5%. If much higher, the test statistic is inflated."
        )
        print()

    # --- Part 3: Compare all three correction methods on SAME raw p-values ---
    if verbose:
        print("=" * 70)
        print("PART 3: CORRECTION METHOD COMPARISON (same raw p-values)")
        print("=" * 70)

    # Re-extract raw p-values for all edges
    from kl_clustering_analysis.core_utils.data_utils import extract_leaf_counts
    from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
        _compute_p_values_via_projection,
    )

    parent_ids = [p for p, _ in edge_list]
    child_leaf_counts = extract_leaf_counts(results_df, child_ids)
    parent_leaf_counts = extract_leaf_counts(results_df, parent_ids)

    stats, dfs, pvals, invalid = _compute_p_values_via_projection(
        tree, child_ids, parent_ids, child_leaf_counts, parent_leaf_counts
    )

    # Replace NaN with 1.0 for correction
    pvals_clean = np.where(np.isfinite(pvals), pvals, 1.0)
    n_total = len(pvals_clean)

    # Flat BH
    reject_flat, padj_flat = flat_bh_correction(pvals_clean, alpha=0.05)

    # Level-wise BH
    from kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.level_wise_correction import (
        level_wise_bh_correction,
    )

    node_depths = compute_node_depths(tree)
    child_depths = np.array([node_depths.get(cid, 0) for cid in child_ids])
    reject_level, padj_level = level_wise_bh_correction(pvals_clean, child_depths, 0.05)

    # TreeBH
    result_tbh = tree_bh_correction(tree, pvals_clean, child_ids, alpha=0.05)
    reject_tbh = result_tbh.reject

    if verbose:
        print(
            f"Raw p < 0.05: {(pvals_clean < 0.05).sum()}/{n_total} "
            f"({(pvals_clean < 0.05).mean():.1%})"
        )
        print(f"Flat BH rejections: {reject_flat.sum()}/{n_total} " f"({reject_flat.mean():.1%})")
        print(
            f"Level-wise BH rejections: {reject_level.sum()}/{n_total} "
            f"({reject_level.mean():.1%})"
        )
        print(f"TreeBH rejections: {reject_tbh.sum()}/{n_total} " f"({reject_tbh.mean():.1%})")
        print()

    # --- Part 3b: Inspect TreeBH per-family thresholds ---
    if verbose and result_tbh.family_results:
        print("TreeBH per-family detail:")
        for parent_id, info in sorted(
            result_tbh.family_results.items(), key=lambda x: x[1]["level"]
        )[:10]:
            print(
                f"  {parent_id}: level={info['level']}, "
                f"α_adj={info['adjusted_alpha']:.4f}, "
                f"n_tests={info['n_tests']}, "
                f"n_rej={info['n_rejections']}, "
                f"p_values={[f'{p:.4f}' for p in info['p_values']]}"
            )
        if len(result_tbh.family_results) > 10:
            print(f"  ... ({len(result_tbh.family_results) - 10} more families)")
        print()

    # --- Part 4: Sibling test selection bias ---
    if verbose:
        print("=" * 70)
        print("PART 4: SIBLING TEST SELECTION BIAS")
        print("=" * 70)

    # Count total binary parents
    total_binary_parents = sum(1 for node in tree.nodes() if len(list(tree.successors(node))) == 2)

    # Run edge annotation first (required for sibling test)
    results_df_annotated = annotate_child_parent_divergence(
        tree, results_df.copy(), significance_level_alpha=0.05, fdr_method="tree_bh"
    )

    # How many parents qualify for sibling testing?
    parents_tested, args, skipped = _collect_test_arguments(tree, results_df_annotated)

    if verbose:
        print(f"Total binary parents in tree: {total_binary_parents}")
        print(f"Parents qualifying for sibling test: {len(parents_tested)}")
        print(f"Parents skipped (no edge signal): {len(skipped)}")
        print(
            f"Selection ratio: {len(parents_tested)}/{total_binary_parents} "
            f"= {len(parents_tested)/total_binary_parents:.1%}"
        )
        print(
            f"  → Under null at α=0.05, only ~{0.05*total_binary_parents:.0f} edge rejections expected"
        )
        print("  → If selection ratio >> 5%, the edge test is inflated")
        print()

    # --- Part 5: Sibling test rejection rate ---
    if verbose:
        print("=" * 70)
        print("PART 5: SIBLING TEST REJECTION RATE")
        print("=" * 70)

    results_df_full = annotate_sibling_divergence(
        tree, results_df_annotated, significance_level_alpha=0.05
    )

    sibling_tested = results_df_full[
        results_df_full["Sibling_Divergence_P_Value"].notna()
        & (results_df_full["Sibling_Divergence_P_Value"] < 1.0)
    ]
    n_sib_tested = len(sibling_tested)
    if n_sib_tested > 0:
        n_sib_reject = sibling_tested["Sibling_BH_Different"].sum()
        sib_raw_pvals = sibling_tested["Sibling_Divergence_P_Value"].values
        n_sib_raw_lt_05 = (sib_raw_pvals < 0.05).sum()

        if verbose:
            print(f"Sibling tests run: {n_sib_tested}")
            print(
                f"BH correction m = {n_sib_tested} (should be {total_binary_parents} for proper correction)"
            )
            print(
                f"Sibling raw p < 0.05: {n_sib_raw_lt_05}/{n_sib_tested} "
                f"({n_sib_raw_lt_05/n_sib_tested:.1%})"
            )
            print(
                f"Sibling BH rejections: {n_sib_reject}/{n_sib_tested} "
                f"({n_sib_reject/n_sib_tested:.1%})"
            )
            print("  → Under null, raw p<0.05 should be ~5%")
            print("  → But this is conditional on edge rejection, so it's biased")
    else:
        if verbose:
            print("No sibling tests were run (no edges rejected)")
    print()

    # --- Part 6: What if sibling BH used m = all binary parents? ---
    if verbose and n_sib_tested > 0:
        print("=" * 70)
        print("PART 6: SIBLING BH WITH m = ALL BINARY PARENTS (counterfactual)")
        print("=" * 70)

        # Pretend we tested all binary parents, pad with p=1.0 for untested
        all_parents_pvals = np.ones(total_binary_parents)
        all_parents_pvals[:n_sib_tested] = sib_raw_pvals

        reject_full_m, _, _ = benjamini_hochberg_correction(all_parents_pvals, alpha=0.05)
        n_reject_full_m = reject_full_m[:n_sib_tested].sum()

        if verbose:
            print(f"Current BH (m={n_sib_tested}): {n_sib_reject} rejections")
            print(f"Corrected BH (m={total_binary_parents}): {n_reject_full_m} rejections")
            print("  → Proper m reduces rejections because correction is stronger")

    return {
        "n_edges": n_edges,
        "raw_p_lt_05": n_raw_lt_05,
        "treebh_reject": int(n_reject_treebh),
        "flat_reject": int(reject_flat.sum()),
        "level_reject": int(reject_level.sum()),
        "total_binary_parents": total_binary_parents,
        "sibling_tested": n_sib_tested if n_sib_tested > 0 else 0,
        "sibling_reject": int(n_sib_reject) if n_sib_tested > 0 else 0,
    }


def run_multi_trial(n_trials: int = 20, n: int = 200, p: int = 50):
    """Run multiple null trials and aggregate."""
    print("=" * 70)
    print(f"MULTI-TRIAL: {n_trials} null datasets, n={n}, p={p}")
    print("=" * 70)

    records = []
    for i in range(n_trials):
        r = run_single_null_trial(seed=1000 + i, n=n, p=p, verbose=False)
        records.append(r)
        print(
            f"  Trial {i+1:2d}: raw_p<0.05={r['raw_p_lt_05']:3d}/{r['n_edges']}, "
            f"treebh={r['treebh_reject']:3d}, flat={r['flat_reject']:3d}, "
            f"sib_tested={r['sibling_tested']:3d}/{r['total_binary_parents']}, "
            f"sib_reject={r['sibling_reject']:3d}"
        )

    df = pd.DataFrame(records)
    print()
    print("AGGREGATE (mean ± std):")
    print(f"  n_edges:              {df['n_edges'].mean():.0f}")
    print(
        f"  Raw p < 0.05:         {df['raw_p_lt_05'].mean():.1f} ± {df['raw_p_lt_05'].std():.1f} "
        f"({df['raw_p_lt_05'].mean()/df['n_edges'].mean():.1%})"
    )
    print(
        f"  TreeBH rejections:    {df['treebh_reject'].mean():.1f} ± {df['treebh_reject'].std():.1f} "
        f"({df['treebh_reject'].mean()/df['n_edges'].mean():.1%})"
    )
    print(
        f"  Flat BH rejections:   {df['flat_reject'].mean():.1f} ± {df['flat_reject'].std():.1f} "
        f"({df['flat_reject'].mean()/df['n_edges'].mean():.1%})"
    )
    print(
        f"  Level BH rejections:  {df['level_reject'].mean():.1f} ± {df['level_reject'].std():.1f} "
        f"({df['level_reject'].mean()/df['n_edges'].mean():.1%})"
    )
    print(
        f"  Sibling tested/total: {df['sibling_tested'].mean():.1f}/{df['total_binary_parents'].mean():.0f} "
        f"({df['sibling_tested'].mean()/df['total_binary_parents'].mean():.1%})"
    )
    print(
        f"  Sibling rejections:   {df['sibling_reject'].mean():.1f} ± {df['sibling_reject'].std():.1f} "
        f"({df['sibling_reject'].mean()/max(df['sibling_tested'].mean(), 1):.1%} of tested)"
    )
    return df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SINGLE DETAILED TRIAL")
    print("=" * 70 + "\n")
    run_single_null_trial(seed=42, n=200, p=50)

    print("\n" + "=" * 70)
    print("MULTI-TRIAL AGGREGATION")
    print("=" * 70 + "\n")
    run_multi_trial(n_trials=20, n=200, p=50)
