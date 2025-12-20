"""Debug script for Case 55: High-dimensional over-splitting.

Case 55 Parameters:
- n_samples: 360
- n_features: 3000
- n_clusters: 12 (true)
- entropy_param: 0.20
- balanced_clusters: False

Problem: Algorithm finds 27 clusters instead of 12 (over-splitting)

This is a HIGH-DIMENSIONAL case (3000 features >> 360 samples)
so random projection is definitely being used.
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


def analyze_case_55():
    """Detailed analysis of Case 55."""
    print("=" * 70)
    print("CASE 55 ANALYSIS: High-dimensional over-splitting")
    print("=" * 70)

    # Generate Case 55 data
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=360,
        n_cols=3000,
        n_clusters=12,
        entropy_param=0.20,
        balanced_clusters=False,
        random_seed=2024,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    print(f"\nData: {data_df.shape[0]} samples × {data_df.shape[1]} features")
    print(f"True clusters: 12")
    print(f"Entropy (noise): 0.20")

    # Feature statistics
    feature_means = data_df.mean(axis=0).values
    print(f"\nFeature statistics:")
    print(f"  Mean θ range: [{feature_means.min():.3f}, {feature_means.max():.3f}]")
    print(f"  Mean θ mean: {feature_means.mean():.3f}")

    # Variance-weighted df
    variance_weights = 4.0 * feature_means * (1.0 - feature_means)
    df_eff = variance_weights.sum()
    print(f"  Variance-weighted df: {df_eff:.1f} (vs raw df={len(feature_means)})")

    # Check projection settings
    print(f"\nProjection settings:")
    print(f"  USE_RANDOM_PROJECTION: {config.USE_RANDOM_PROJECTION}")
    print(f"  PROJECTION_THRESHOLD_RATIO: {config.PROJECTION_THRESHOLD_RATIO}")
    print(f"  PROJECTION_K_MULTIPLIER: {config.PROJECTION_K_MULTIPLIER}")
    print(f"  PROJECTION_MIN_K: {config.PROJECTION_MIN_K}")
    print(f"  PROJECTION_N_TRIALS: {config.PROJECTION_N_TRIALS}")
    print(f"  USE_MI_FEATURE_FILTER: {config.USE_MI_FEATURE_FILTER}")

    # Check projection at various sample sizes
    print(f"\nProjection dimension at different n_eff:")
    for n_eff in [10, 20, 30, 50, 100, 180]:
        if should_use_projection(3000, n_eff):
            k = compute_projection_dimension(n_eff, 3000)
            print(f"  n_eff={n_eff:3d}: k={k}")

    # Build tree
    print("\nBuilding hierarchical tree...")
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Decompose
    print("Running decomposition...")
    decomp = tree.decompose(leaf_data=data_df)
    stats = tree.stats_df

    print(f"\n{'=' * 70}")
    print(f"RESULT: Found {decomp['num_clusters']} clusters (expected: 12)")
    print(f"{'=' * 70}")

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
                    }
                )

    test_df = pd.DataFrame(test_data)

    print(f"\nSibling test summary:")
    print(f"  Total internal nodes: {len(test_df)}")
    print(f"  Significant (different siblings): {test_df['significant'].sum()}")
    print(f"  Not significant: {(~test_df['significant']).sum()}")

    print(f"\nTest statistic ratio (χ²/df) - should be ~1.0 under H₀:")
    print(f"  Mean ratio: {test_df['ratio'].mean():.2f}")
    print(f"  Median ratio: {test_df['ratio'].median():.2f}")
    print(f"  Min ratio: {test_df['ratio'].min():.2f}")
    print(f"  Max ratio: {test_df['ratio'].max():.2f}")

    # Breakdown by n_leaves
    print(f"\nRatio by subtree size:")
    for size_range, label in [
        ((0, 20), "n ≤ 20"),
        ((20, 50), "20 < n ≤ 50"),
        ((50, 100), "50 < n ≤ 100"),
        ((100, 500), "n > 100"),
    ]:
        mask = (test_df["n_leaves"] > size_range[0]) & (
            test_df["n_leaves"] <= size_range[1]
        )
        subset = test_df[mask]
        if len(subset) > 0:
            sig_rate = subset["significant"].mean()
            mean_ratio = subset["ratio"].mean()
            print(
                f"  {label:15s}: n={len(subset):3d}, sig_rate={sig_rate:.1%}, mean_ratio={mean_ratio:.2f}"
            )

    # Check the significant splits near the root (large subtrees)
    print(f"\n{'=' * 70}")
    print("SIGNIFICANT SPLITS AT LARGE SUBTREES (n > 50)")
    print("=" * 70)
    large_splits = test_df[
        (test_df["n_leaves"] > 50) & test_df["significant"]
    ].sort_values("n_leaves", ascending=False)
    print(
        f"{'node':>10} {'n_leaves':>8} {'df':>6} {'χ²':>10} {'ratio':>8} {'p-value':>12}"
    )
    print("-" * 60)
    for _, row in large_splits.head(15).iterrows():
        print(
            f"{row['node']:>10} {int(row['n_leaves']):>8} {row['df']:>6.1f} "
            f"{row['test_stat']:>10.2f} {row['ratio']:>8.2f} {row['p_value']:>12.2e}"
        )

    # Check small subtrees that are splitting
    print(f"\n{'=' * 70}")
    print("SIGNIFICANT SPLITS AT SMALL SUBTREES (n ≤ 20)")
    print("=" * 70)
    small_splits = test_df[
        (test_df["n_leaves"] <= 20) & test_df["significant"]
    ].sort_values("n_leaves")
    print(
        f"{'node':>10} {'n_leaves':>8} {'df':>6} {'χ²':>10} {'ratio':>8} {'p-value':>12}"
    )
    print("-" * 60)
    for _, row in small_splits.head(15).iterrows():
        print(
            f"{row['node']:>10} {int(row['n_leaves']):>8} {row['df']:>6.1f} "
            f"{row['test_stat']:>10.2f} {row['ratio']:>8.2f} {row['p_value']:>12.2e}"
        )

    # Check child-parent divergence
    print(f"\n{'=' * 70}")
    print("CHILD-PARENT DIVERGENCE ANALYSIS")
    print("=" * 70)

    cp_sig = (
        stats["Child_Parent_Divergence_Significant"].sum()
        if "Child_Parent_Divergence_Significant" in stats.columns
        else 0
    )
    cp_total = (
        stats["Child_Parent_Divergence_Significant"].notna().sum()
        if "Child_Parent_Divergence_Significant" in stats.columns
        else 0
    )
    print(f"  Child-parent tests significant: {cp_sig} / {cp_total}")

    if "Child_Parent_Divergence_df" in stats.columns:
        df_vals = stats["Child_Parent_Divergence_df"].dropna()
        print(f"  Child-parent df range: [{df_vals.min():.1f}, {df_vals.max():.1f}]")
        print(f"  Child-parent df mean: {df_vals.mean():.1f}")


def monte_carlo_projection_calibration():
    """Test if the projected chi-square is well-calibrated."""
    print("\n" + "=" * 70)
    print("MONTE CARLO: PROJECTED CHI-SQUARE CALIBRATION")
    print("=" * 70)

    np.random.seed(42)
    n_simulations = 500
    n_features = 3000
    n_left, n_right = 30, 30  # Typical small subtree sizes

    # Check if projection is used
    n_eff = (2 * n_left * n_right) / (n_left + n_right)
    use_proj = should_use_projection(n_features, int(n_eff))
    print(f"\nn_left={n_left}, n_right={n_right}, n_eff={n_eff:.1f}")
    print(f"n_features={n_features}")
    print(f"Use projection: {use_proj}")

    if use_proj:
        k = compute_projection_dimension(int(n_eff), n_features)
        print(f"Projection dimension k={k}")

    # Simulate under H0
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

        # Standardize
        diff = theta_left - theta_right
        z = diff / np.sqrt(var_diff)

        if use_proj:
            # Project
            from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
                generate_projection_matrix,
            )

            R = generate_projection_matrix(n_features, k, random_state=None)
            z_proj = R @ z
            test_stat = np.sum(z_proj**2)
            df = k
        else:
            test_stat = np.sum(z**2)
            df = n_features

        test_stats.append(test_stat)

    test_stats = np.array(test_stats)

    print(f"\nTest statistic distribution under H₀:")
    print(f"  Mean: {np.mean(test_stats):.2f} (expected for χ²({df}): {df})")
    print(f"  Std:  {np.std(test_stats):.2f} (expected: {np.sqrt(2 * df):.2f})")
    print()

    # Type I error rates
    for alpha in [0.10, 0.05, 0.01]:
        p_values = chi2.sf(test_stats, df=df)
        type1_rate = (p_values < alpha).mean()
        status = "✓" if abs(type1_rate - alpha) < 0.02 else "✗"
        print(
            f"  Type I error at α={alpha}: {type1_rate:.3f} (nominal: {alpha}) {status}"
        )


if __name__ == "__main__":
    analyze_case_55()
    monte_carlo_projection_calibration()
