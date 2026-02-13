"""Debug script to verify branch-length adjustment in sibling divergence test.

Tests the Felsenstein (1985) phylogenetic independent contrasts integration
where variance is scaled by (b_L + b_R) to account for expected divergence
over topological distance.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
    _get_sibling_data,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import (
    standardize_proportion_difference,
)


def test_branch_length_extraction():
    """Test that branch lengths are correctly extracted from tree edges."""
    print("=" * 60)
    print("TEST 1: Branch Length Extraction from Tree Edges")
    print("=" * 60)

    # Create simple test data
    np.random.seed(42)
    n_leaves = 6
    n_features = 5

    # Generate leaf data
    leaf_data = np.random.dirichlet(np.ones(3), size=(n_leaves, n_features))
    leaf_names = [f"leaf_{i}" for i in range(n_leaves)]
    data_df = pd.DataFrame(
        [leaf_data[i].flatten() for i in range(n_leaves)], index=leaf_names
    )

    # Build tree from linkage
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=leaf_names)

    print(f"\nTree has {len(tree.nodes)} nodes and {len(tree.edges)} edges")

    # Check edge attributes
    print("\nEdge branch lengths:")
    for parent, child in tree.edges():
        branch_length = tree.edges[parent, child].get("branch_length", "NOT FOUND")
        print(
            f"  {parent} -> {child}: {branch_length:.4f}"
            if isinstance(branch_length, float)
            else f"  {parent} -> {child}: {branch_length}"
        )

    # Test _get_sibling_data extraction
    print("\nTesting _get_sibling_data for each internal node:")
    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) == 2:
            left, right = children
            try:
                result = _get_sibling_data(tree, node, left, right)
                left_dist, right_dist, n_left, n_right, bl_left, bl_right = result
                print(f"\n  Parent: {node}")
                print(f"    Children: {left}, {right}")
                print(
                    f"    Branch lengths: left={bl_left:.4f}, right={bl_right:.4f}"
                    if bl_left
                    else f"    Branch lengths: left={bl_left}, right={bl_right}"
                )
                print(
                    f"    Sum: {bl_left + bl_right:.4f}"
                    if bl_left and bl_right
                    else "    Sum: N/A"
                )
            except Exception as e:
                print(f"  Error for {node}: {e}")

    print("\n✓ Branch length extraction test complete")


def test_variance_scaling():
    """Test that variance scaling works correctly with branch lengths."""
    print("\n" + "=" * 60)
    print("TEST 2: Variance Scaling with Branch Lengths")
    print("=" * 60)

    # Create simple distributions
    theta_1 = np.array([0.6, 0.3, 0.1])
    theta_2 = np.array([0.4, 0.4, 0.2])
    n_1, n_2 = 50, 50

    # Without branch length adjustment
    z_unadjusted, var_unadjusted = standardize_proportion_difference(
        theta_1, theta_2, n_1, n_2
    )

    # With branch length adjustment (sum = 2.0)
    branch_sum = 2.0
    z_adjusted, var_adjusted = standardize_proportion_difference(
        theta_1, theta_2, n_1, n_2, branch_length_sum=branch_sum
    )

    print(f"\nDistributions:")
    print(f"  θ₁ = {theta_1}")
    print(f"  θ₂ = {theta_2}")
    print(f"  n₁ = {n_1}, n₂ = {n_2}")

    print(f"\nUnadjusted (no branch lengths):")
    print(f"  Variance: {var_unadjusted}")
    print(f"  Z-scores: {z_unadjusted}")
    print(f"  ||z||²:   {np.sum(z_unadjusted**2):.4f}")

    print(f"\nAdjusted (branch_length_sum = {branch_sum}):")
    print(f"  Variance: {var_adjusted}")
    print(f"  Z-scores: {z_adjusted}")
    print(f"  ||z||²:   {np.sum(z_adjusted**2):.4f}")

    # Verify relationship
    expected_var_ratio = branch_sum
    actual_var_ratio = var_adjusted[0] / var_unadjusted[0]
    print(f"\nVariance ratio check:")
    print(f"  Expected: {expected_var_ratio}")
    print(f"  Actual:   {actual_var_ratio:.4f}")
    print(
        f"  Match:    {'✓' if np.isclose(expected_var_ratio, actual_var_ratio) else '✗'}"
    )

    # Z-scores should be smaller by sqrt(branch_sum)
    expected_z_ratio = 1 / np.sqrt(branch_sum)
    actual_z_ratio = z_adjusted[0] / z_unadjusted[0]
    print(f"\nZ-score ratio check:")
    print(f"  Expected: {expected_z_ratio:.4f}")
    print(f"  Actual:   {actual_z_ratio:.4f}")
    print(f"  Match:    {'✓' if np.isclose(expected_z_ratio, actual_z_ratio) else '✗'}")

    print("\n✓ Variance scaling test complete")


def test_sibling_divergence_with_branch_lengths():
    """Test sibling divergence test with and without branch length adjustment."""
    print("\n" + "=" * 60)
    print("TEST 3: Sibling Divergence Test with Branch Lengths")
    print("=" * 60)

    np.random.seed(123)

    # Create distributions with moderate difference
    d = 20  # features
    K = 4  # categories per feature

    # Base distribution
    base = np.random.dirichlet(np.ones(K), size=d)

    # Left sibling: slight perturbation
    left_dist = base + np.random.normal(0, 0.05, (d, K))
    left_dist = np.clip(left_dist, 0.01, 0.99)
    left_dist = left_dist / left_dist.sum(axis=1, keepdims=True)

    # Right sibling: different perturbation
    right_dist = base + np.random.normal(0, 0.05, (d, K))
    right_dist = np.clip(right_dist, 0.01, 0.99)
    right_dist = right_dist / right_dist.sum(axis=1, keepdims=True)

    n_left, n_right = 100, 100

    # Test without branch lengths
    stat_no_bl, df_no_bl, pval_no_bl = sibling_divergence_test(
        left_dist, right_dist, n_left, n_right
    )

    # Test with short branch lengths (expect similar result)
    stat_short, df_short, pval_short = sibling_divergence_test(
        left_dist,
        right_dist,
        n_left,
        n_right,
        branch_length_left=0.5,
        branch_length_right=0.5,
    )

    # Test with long branch lengths (expect larger p-value, less significant)
    stat_long, df_long, pval_long = sibling_divergence_test(
        left_dist,
        right_dist,
        n_left,
        n_right,
        branch_length_left=5.0,
        branch_length_right=5.0,
    )

    print(f"\nDistribution shape: {left_dist.shape}")
    print(f"Sample sizes: n_left={n_left}, n_right={n_right}")

    print(f"\nNo branch length adjustment:")
    print(f"  Statistic: {stat_no_bl:.4f}")
    print(f"  df:        {df_no_bl:.0f}")
    print(f"  p-value:   {pval_no_bl:.6f}")

    print(f"\nShort branches (b_L=0.5, b_R=0.5, sum=1.0):")
    print(f"  Statistic: {stat_short:.4f}")
    print(f"  df:        {df_short:.0f}")
    print(f"  p-value:   {pval_short:.6f}")

    print(f"\nLong branches (b_L=5.0, b_R=5.0, sum=10.0):")
    print(f"  Statistic: {stat_long:.4f}")
    print(f"  df:        {df_long:.0f}")
    print(f"  p-value:   {pval_long:.6f}")

    # Interpretation
    print("\nInterpretation:")
    print("  Longer branches -> more expected divergence -> larger variance")
    print("  -> smaller z-scores -> smaller test statistic -> larger p-value")
    print(f"  p_long > p_short: {'✓' if pval_long > pval_short else '✗'}")

    print("\n✓ Sibling divergence test complete")


def test_end_to_end_pipeline():
    """Test the full pipeline with real-ish data."""
    print("\n" + "=" * 60)
    print("TEST 4: End-to-End Pipeline")
    print("=" * 60)

    from benchmarks.shared.generators.generate_random_feature_matrix import (
        generate_random_feature_matrix,
    )
    from kl_clustering_analysis import config

    # Generate synthetic data
    np.random.seed(42)
    leaf_matrix_dict, _ = generate_random_feature_matrix(
        n_rows=50,
        n_cols=10,
        n_clusters=4,
        entropy_param=1.0,
        balanced_clusters=True,
        random_seed=42,
    )

    data_df = pd.DataFrame.from_dict(leaf_matrix_dict, orient="index")
    print(f"\nGenerated data: {data_df.shape[0]} samples, {data_df.shape[1]} features")

    # Build tree
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    print(f"Tree: {len(tree.nodes)} nodes, {len(tree.edges)} edges")

    # Count edges with branch lengths
    edges_with_bl = sum(1 for _, _, d in tree.edges(data=True) if "branch_length" in d)
    print(f"Edges with branch_length attribute: {edges_with_bl}/{len(tree.edges)}")

    # Run decomposition
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )

    stats_df = tree.stats_df
    print(f"\nStats DataFrame shape: {stats_df.shape}")

    # Check sibling test columns
    sibling_cols = [c for c in stats_df.columns if "Sibling" in c]
    print(f"Sibling columns: {sibling_cols}")

    # Summary of sibling tests
    if "Sibling_BH_Different" in stats_df.columns:
        sig_count = stats_df["Sibling_BH_Different"].sum()
        tested_count = stats_df["Sibling_Divergence_P_Value"].notna().sum()
        print(f"\nSibling tests: {tested_count} pairs tested, {sig_count} significant")

    print("\n✓ End-to-end pipeline test complete")


if __name__ == "__main__":
    test_branch_length_extraction()
    test_variance_scaling()
    test_sibling_divergence_with_branch_lengths()
    test_end_to_end_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
