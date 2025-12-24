"""Debug script for analyzing sibling divergence test p-values and projections.

This script investigates:
1. When random projection is triggered vs full-dimension test
2. How degrees of freedom are computed in each case
3. Whether the test statistics are calibrated correctly
4. Why Case 28 (36 features, low entropy) over-splits
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    should_use_projection,
    compute_projection_dimension,
)


def analyze_projection_thresholds():
    """Analyze when random projection is triggered."""
    print("=" * 70)
    print("RANDOM PROJECTION THRESHOLD ANALYSIS")
    print("=" * 70)
    print(f"\nConfig settings:")
    print(f"  USE_RANDOM_PROJECTION: {config.USE_RANDOM_PROJECTION}")
    print(
        "  PROJECTION_DECISION: JL-based (projection if n_features > n_samples and JL k < n_features)"
    )
    print(f"  PROJECTION_K_MULTIPLIER: {config.PROJECTION_K_MULTIPLIER}")
    print(f"  PROJECTION_MIN_K: {config.PROJECTION_MIN_K}")
    print()

    n_features = 36  # Case 28 has 36 features

    print(f"For n_features = {n_features}:")
    print("-" * 50)
    print(f"{'n_eff':>8} {'use_proj':>10} {'k (if proj)':>12}")
    print("-" * 50)

    for n_eff in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]:
        use_proj = should_use_projection(n_features, n_eff)
        if use_proj:
            k = compute_projection_dimension(n_eff, n_features)
            print(f"{n_eff:>8} {str(use_proj):>10} {k:>12}")
        else:
            print(f"{n_eff:>8} {str(use_proj):>10} {'N/A':>12}")

    print()
    print("Key insight:")
    print(
        "  Projection decision uses JL target dimension k; see table above for which n_eff cause projection."
    )


def analyze_case_28_sibling_tests():
    """Detailed analysis of sibling tests in Case 28."""
    print("\n" + "=" * 70)
    print("CASE 28 SIBLING TEST ANALYSIS")
    print("=" * 70)

    # Generate Case 28 data
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=96,
        n_cols=36,
        n_clusters=4,
        entropy_param=0.12,
        balanced_clusters=False,
        random_seed=2024,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)

    # Build tree
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Decompose
    decomp = tree.decompose(leaf_data=data_df)
    stats = tree.stats_df

    print(f"\nData: 96 samples × 36 features, 4 true clusters")
    print(f"Found: {decomp['num_clusters']} clusters")
    print()

    # Analyze sibling tests
    internal_nodes = [n for n in tree.nodes() if list(tree.successors(n))]

    # Collect test data
    test_data = []
    for node in internal_nodes:
        if node in stats.index:
            row = stats.loc[node]
            df = row.get("Sibling_Degrees_of_Freedom")
            ts = row.get("Sibling_Test_Statistic")
            pv = row.get("Sibling_Divergence_P_Value")
            sig = row.get("Sibling_BH_Different")
            n_leaves = row.get("leaf_count", 0)

            if pd.notna(df) and pd.notna(ts):
                # Infer if projection was used
                used_proj = df < 36

                # Compute ratio (should be ~1 under H0)
                ratio = ts / df if df > 0 else 0

                test_data.append(
                    {
                        "node": node,
                        "n_leaves": n_leaves,
                        "df": df,
                        "test_stat": ts,
                        "ratio": ratio,
                        "p_value": pv,
                        "significant": sig,
                        "used_projection": used_proj,
                    }
                )

    # Convert to DataFrame for analysis
    test_df = pd.DataFrame(test_data)

    print("PROJECTION USAGE SUMMARY:")
    print("-" * 50)
    n_with_proj = test_df["used_projection"].sum()
    n_without_proj = len(test_df) - n_with_proj
    print(f"  Tests with projection (df < 36): {n_with_proj}")
    print(f"  Tests without projection (df = 36): {n_without_proj}")

    print("\n" + "=" * 70)
    print("TESTS WITH PROJECTION (small sample sizes)")
    print("=" * 70)
    proj_df = test_df[test_df["used_projection"]].sort_values("n_leaves")
    print(
        f"{'node':>10} {'n_leaves':>8} {'df':>6} {'χ²':>10} {'ratio':>8} {'p-value':>12} {'sig':>5}"
    )
    print("-" * 70)
    for _, row in proj_df.head(20).iterrows():
        print(
            f"{row['node']:>10} {int(row['n_leaves']):>8} {row['df']:>6.1f} "
            f"{row['test_stat']:>10.2f} {row['ratio']:>8.2f} "
            f"{row['p_value']:>12.2e} {str(row['significant']):>5}"
        )

    print("\n" + "=" * 70)
    print("TESTS WITHOUT PROJECTION (full dimension)")
    print("=" * 70)
    noproj_df = test_df[~test_df["used_projection"]].sort_values("n_leaves")
    print(
        f"{'node':>10} {'n_leaves':>8} {'df':>6} {'χ²':>10} {'ratio':>8} {'p-value':>12} {'sig':>5}"
    )
    print("-" * 70)
    for _, row in noproj_df.head(20).iterrows():
        print(
            f"{row['node']:>10} {int(row['n_leaves']):>8} {row['df']:>6.1f} "
            f"{row['test_stat']:>10.2f} {row['ratio']:>8.2f} "
            f"{row['p_value']:>12.2e} {str(row['significant']):>5}"
        )

    print("\n" + "=" * 70)
    print("RATIO ANALYSIS (χ²/df - should be ~1.0 under H₀)")
    print("=" * 70)

    proj_ratios = proj_df["ratio"].values
    noproj_ratios = noproj_df["ratio"].values if len(noproj_df) > 0 else []

    if len(proj_ratios) > 0:
        print(f"\nWith projection:")
        print(f"  Mean ratio: {np.mean(proj_ratios):.2f}")
        print(f"  Median ratio: {np.median(proj_ratios):.2f}")
        print(f"  Range: [{np.min(proj_ratios):.2f}, {np.max(proj_ratios):.2f}]")
        print(f"  Nodes with ratio > 2.0: {(proj_ratios > 2.0).sum()}")
        print(f"  Nodes with ratio > 3.0: {(proj_ratios > 3.0).sum()}")

    if len(noproj_ratios) > 0:
        print(f"\nWithout projection:")
        print(f"  Mean ratio: {np.mean(noproj_ratios):.2f}")
        print(f"  Median ratio: {np.median(noproj_ratios):.2f}")
        print(f"  Range: [{np.min(noproj_ratios):.2f}, {np.max(noproj_ratios):.2f}]")

    # Check if ratios indicate real differences or calibration issues
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    avg_ratio = np.mean(test_df["ratio"].values)
    print(f"\nOverall mean ratio: {avg_ratio:.2f}")

    if avg_ratio > 2.0:
        print("""
DIAGNOSIS: Ratios >> 1.0 indicate REAL DIFFERENCES between siblings,
not test miscalibration. The algorithm is correctly detecting that
siblings have genuinely different distributions.

This happens because:
  1. Low entropy (0.12) = features are close to 0 or 1
  2. This creates distinct "patterns" within the 4 true clusters
  3. The hierarchical clustering groups similar patterns together
  4. These sub-groups ARE statistically different
  5. The algorithm correctly splits them

This is NOT a Type I error problem - it's the algorithm being
sensitive to real structure in the data.
""")
    else:
        print("""
DIAGNOSIS: Ratios near 1.0 suggest the test is well-calibrated
and the siblings are not significantly different under H₀.
""")


def test_chi_square_calibration():
    """Monte Carlo test of the chi-square approximation."""
    print("\n" + "=" * 70)
    print("CHI-SQUARE CALIBRATION TEST (Monte Carlo)")
    print("=" * 70)

    np.random.seed(42)
    n_simulations = 1000
    n_features = 36
    n_left, n_right = 10, 10  # Small sample sizes where projection is used

    # Simulate under H0: both siblings from same distribution
    theta_true = np.random.uniform(0.3, 0.7, n_features)

    test_stats = []
    for _ in range(n_simulations):
        # Sample from same distribution (H0)
        left_samples = np.random.binomial(1, theta_true, size=(n_left, n_features))
        right_samples = np.random.binomial(1, theta_true, size=(n_right, n_features))

        theta_left = left_samples.mean(axis=0)
        theta_right = right_samples.mean(axis=0)

        # Pooled estimate
        pooled = 0.5 * (theta_left + theta_right)
        pooled = np.clip(pooled, 1e-10, 1 - 1e-10)

        # Variance
        inverse_n_sum = 1.0 / n_left + 1.0 / n_right
        var_diff = pooled * (1.0 - pooled) * inverse_n_sum
        var_diff = np.maximum(var_diff, 1e-10)

        # Test statistic
        diff = theta_left - theta_right
        test_stat = np.sum(diff**2 / var_diff)
        test_stats.append(test_stat)

    test_stats = np.array(test_stats)

    print(f"\nSimulation: n_left={n_left}, n_right={n_right}, n_features={n_features}")
    print(f"H₀: Both siblings from same distribution")
    print()
    print(f"Test statistic distribution:")
    print(
        f"  Mean: {np.mean(test_stats):.2f} (expected for χ²({n_features}): {n_features})"
    )
    print(f"  Std:  {np.std(test_stats):.2f} (expected: {np.sqrt(2 * n_features):.2f})")
    print()

    # Type I error rates
    for alpha in [0.10, 0.05, 0.01]:
        p_values = chi2.sf(test_stats, df=n_features)
        type1_rate = (p_values < alpha).mean()
        print(f"  Type I error at α={alpha}: {type1_rate:.3f} (nominal: {alpha})")


if __name__ == "__main__":
    analyze_projection_thresholds()
    analyze_case_28_sibling_tests()
    test_chi_square_calibration()
