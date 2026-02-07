"""Explore why noisy data has so many significant edges.

Diagnose the child-parent test behavior on clean vs noisy data.
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_standardized_z,
    _compute_mean_branch_length,
)


def analyze_z_scores(tree, stats_df, entropy_label):
    """Analyze z-score distributions for edges in the tree."""
    print(f"\n{'=' * 60}")
    print(f"Z-Score Analysis: entropy={entropy_label}")
    print("=" * 60)

    mean_bl = _compute_mean_branch_length(tree)
    print(f"Mean branch length: {mean_bl:.4f}")

    z_magnitudes = []
    sample_sizes = []
    branch_lengths = []

    for parent, child in tree.edges():
        child_dist = tree.nodes[child].get("distribution")
        parent_dist = tree.nodes[parent].get("distribution")

        if child_dist is None or parent_dist is None:
            continue

        n_child = int(stats_df.loc[child, "leaf_count"])
        n_parent = int(stats_df.loc[parent, "leaf_count"])
        bl = tree.edges[parent, child].get("branch_length", 0)

        z = _compute_standardized_z(
            np.asarray(child_dist),
            np.asarray(parent_dist),
            n_child,
            n_parent,
            bl,
            mean_bl,
        )

        z_mag = np.sqrt(np.sum(z**2))  # L2 norm of z-vector
        z_magnitudes.append(z_mag)
        sample_sizes.append((n_child, n_parent))
        branch_lengths.append(bl)

    z_magnitudes = np.array(z_magnitudes)

    print(f"\nZ-score magnitude (||z||₂) statistics:")
    print(f"  Min:    {z_magnitudes.min():.2f}")
    print(f"  Median: {np.median(z_magnitudes):.2f}")
    print(f"  Mean:   {z_magnitudes.mean():.2f}")
    print(f"  Max:    {z_magnitudes.max():.2f}")
    print(f"  Std:    {z_magnitudes.std():.2f}")

    # How many edges have "large" z-scores?
    thresholds = [5, 10, 20, 50]
    print(f"\nEdges with ||z||₂ > threshold:")
    for t in thresholds:
        count = (z_magnitudes > t).sum()
        print(f"  > {t:2d}: {count:3d} edges ({100 * count / len(z_magnitudes):.1f}%)")

    # Analyze by sample size
    print(f"\nSample size breakdown (n_child):")
    n_childs = np.array([s[0] for s in sample_sizes])
    for thresh in [10, 20, 50, 100]:
        mask = n_childs < thresh
        if mask.sum() > 0:
            print(
                f"  n_child < {thresh:3d}: {mask.sum():3d} edges, mean ||z|| = {z_magnitudes[mask].mean():.2f}"
            )

    return z_magnitudes


def analyze_pvalues(stats_df, entropy_label):
    """Analyze p-value distribution."""
    print(f"\nP-value Analysis:")

    pval_col = "Child_Parent_Divergence_P_Value"
    pval_corr_col = "Child_Parent_Divergence_P_Value_BH"

    if pval_col in stats_df.columns:
        pvals = stats_df[pval_col].dropna()
        print(f"  Raw p-values (n={len(pvals)}):")
        print(f"    Min:    {pvals.min():.2e}")
        print(f"    Median: {pvals.median():.2e}")
        print(f"    Mean:   {pvals.mean():.2e}")

        for alpha in [0.05, 0.01, 0.001, 1e-5, 1e-10]:
            count = (pvals < alpha).sum()
            print(f"    p < {alpha:.0e}: {count} edges")
    else:
        print(f"  Column '{pval_col}' not found!")

    if pval_corr_col in stats_df.columns:
        pvals_corr = stats_df[pval_corr_col].dropna()
        print(f"\n  BH-corrected p-values (n={len(pvals_corr)}):")
        for alpha in [0.05, 0.01, 0.001]:
            count = (pvals_corr < alpha).sum()
            print(f"    p_BH < {alpha}: {count} edges")
    else:
        print(f"  Column '{pval_corr_col}' not found!")


def analyze_variance_components(tree, stats_df, entropy_label):
    """Break down variance into components."""
    print(f"\nVariance Component Analysis:")

    mean_bl = _compute_mean_branch_length(tree)

    nested_factors = []
    theta_vars = []
    bl_multipliers = []

    for parent, child in tree.edges():
        child_dist = tree.nodes[child].get("distribution")
        parent_dist = tree.nodes[parent].get("distribution")

        if child_dist is None or parent_dist is None:
            continue

        n_child = int(stats_df.loc[child, "leaf_count"])
        n_parent = int(stats_df.loc[parent, "leaf_count"])
        bl = tree.edges[parent, child].get("branch_length", 0)

        # Component 1: Nested factor
        nested_factor = 1 / n_child - 1 / n_parent
        nested_factors.append(nested_factor)

        # Component 2: θ(1-θ) at each feature
        theta_var = np.mean(parent_dist * (1 - parent_dist))
        theta_vars.append(theta_var)

        # Component 3: BL multiplier
        bl_mult = 1.0 + bl / mean_bl
        bl_multipliers.append(bl_mult)

    print(f"  Nested factor (1/n_c - 1/n_p):")
    print(f"    Min:    {np.min(nested_factors):.6f}")
    print(f"    Median: {np.median(nested_factors):.6f}")
    print(f"    Max:    {np.max(nested_factors):.6f}")

    print(f"\n  θ(1-θ) term (mean across features):")
    print(f"    Min:    {np.min(theta_vars):.4f}")
    print(f"    Median: {np.median(theta_vars):.4f}")
    print(f"    Max:    {np.max(theta_vars):.4f}")

    print(f"\n  BL multiplier (1 + BL/mean_BL):")
    print(f"    Min:    {np.min(bl_multipliers):.4f}")
    print(f"    Median: {np.median(bl_multipliers):.4f}")
    print(f"    Max:    {np.max(bl_multipliers):.4f}")


def main():
    np.random.seed(42)

    # Compare clean vs noisy
    for entropy in [0.1, 0.2, 0.3]:
        print(f"\n{'#' * 70}")
        print(f"# ENTROPY = {entropy}")
        print("#" * 70)

        data_dict, labels = generate_random_feature_matrix(
            n_rows=200, n_cols=50, n_clusters=3, entropy_param=entropy, random_seed=42
        )
        df = pd.DataFrame.from_dict(data_dict, orient="index")

        # Build tree
        Z = linkage(pdist(df.values, metric="hamming"), method="average")
        tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())

        # Populate distributions and run tests
        tree.populate_node_divergences(df)

        from kl_clustering_analysis.hierarchy_analysis.statistics import (
            annotate_child_parent_divergence,
        )

        stats_df = annotate_child_parent_divergence(
            tree, tree.stats_df, significance_level_alpha=0.01
        )

        analyze_z_scores(tree, stats_df, entropy)
        analyze_pvalues(stats_df, entropy)
        analyze_variance_components(tree, stats_df, entropy)


if __name__ == "__main__":
    main()
