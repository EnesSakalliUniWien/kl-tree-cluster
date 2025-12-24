"""Debug script for Case 28: Over-splitting in binary unbalanced data.

Case 28 Parameters:
- n_samples: 96
- n_features: 36
- n_clusters: 4 (true)
- entropy_param: 0.12 (low noise = good separation)
- balanced_clusters: False

Problem: Algorithm finds 13 clusters instead of 4 (over-splitting)
Compare to Case 27: Same but entropy=0.45 → finds exactly 4 clusters

Hypothesis: With low features (36) + low entropy (0.12), the sibling test
is TOO sensitive, detecting small sampling variations as significant.
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


def run_case_analysis(entropy_param, case_name):
    """Analyze a single case configuration."""
    print(f"\n{'=' * 70}")
    print(f"CASE: {case_name} (entropy={entropy_param})")
    print(f"{'=' * 70}")

    # Generate data
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=96,
        n_cols=36,
        n_clusters=4,
        entropy_param=entropy_param,
        balanced_clusters=False,
        random_seed=2024,
    )

    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    print(f"\nData: {data_df.shape[0]} samples × {data_df.shape[1]} features")
    print(f"True clusters: 4")

    # Feature statistics
    feature_means = data_df.mean(axis=0).values
    print(f"\nFeature statistics:")
    print(f"  Mean θ range: [{feature_means.min():.3f}, {feature_means.max():.3f}]")
    print(f"  Mean θ mean: {feature_means.mean():.3f}")

    # Variance-weighted df
    variance_weights = 4.0 * feature_means * (1.0 - feature_means)
    df_eff = variance_weights.sum()
    print(f"  Variance-weighted df: {df_eff:.1f} (vs raw df={len(feature_means)})")

    # Build tree
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Decompose
    decomp = tree.decompose(leaf_data=data_df)
    stats = tree.stats_df

    print(f"\nClustering result:")
    print(f"  Found: {decomp['num_clusters']} clusters (true: 4)")

    # Analyze root split
    root = tree.root()
    print(f"\n--- Root Analysis ({root}) ---")

    if root in stats.index:
        row = stats.loc[root]
        print(f"Sibling test at root:")
        print(f"  JSD: {row.get('Sibling_JSD', 'N/A')}")
        print(f"  Test statistic: {row.get('Sibling_Test_Statistic', 'N/A')}")
        print(f"  Degrees of freedom: {row.get('Sibling_Degrees_of_Freedom', 'N/A')}")
        print(f"  P-value: {row.get('Sibling_Divergence_P_Value', 'N/A')}")
        print(
            f"  P-value (BH corrected): {row.get('Sibling_Divergence_P_Value_Corrected', 'N/A')}"
        )
        print(f"  Siblings different? {row.get('Sibling_BH_Different', 'N/A')}")

    # Check children
    children = list(tree.successors(root))
    print(f"\nRoot children: {children}")

    for child in children:
        if child in stats.index:
            r = stats.loc[child]
            print(f"\n  Child {child}:")
            print(f"    Leaves: {r.get('leaf_count', 'N/A')}")
            print(
                f"    Child-Parent Sig: {r.get('Child_Parent_Divergence_Significant', 'N/A')}"
            )
            print(f"    Child-Parent df: {r.get('Child_Parent_Divergence_df', 'N/A')}")
            print(
                f"    Child-Parent P-value (BH): {r.get('Child_Parent_Divergence_P_Value_BH', 'N/A')}"
            )

    # Count significant sibling tests
    sibling_sig = (
        stats["Sibling_BH_Different"].sum()
        if "Sibling_BH_Different" in stats.columns
        else 0
    )
    total_internal = len([n for n in tree.nodes() if list(tree.successors(n))])
    print(f"\nSignificance summary:")
    print(f"  Internal nodes: {total_internal}")
    print(f"  Sibling tests significant: {sibling_sig}")
    print(f"  Ratio: {sibling_sig / total_internal:.1%}")

    return decomp["num_clusters"]


def analyze_projection_usage():
    """Check if random projection is being used for this case."""
    print("\n" + "=" * 70)
    print("PROJECTION ANALYSIS")
    print("=" * 70)

    n_features = 36
    n_samples = 96

    # For balanced binary split, n_eff ≈ n_samples / 2
    n_eff_approx = n_samples / 2

    use_proj = should_use_projection(n_features, n_eff_approx)

    print(f"\nProjection decision check:")
    print(f"  n_features: {n_features}")
    print(f"  n_eff (approx): {n_eff_approx}")
    print(f"  Projection used? {use_proj}")

    if use_proj:
        k = compute_projection_dimension(int(n_eff_approx), n_features)
        print(f"  Projection dimension k: {k}")
    else:
        print(f"  → Using full-dimension chi-square test (df={n_features})")


def compare_cases():
    """Compare Case 27 (high noise) vs Case 28 (low noise)."""
    print("\n" + "=" * 70)
    print("COMPARISON: High Noise vs Low Noise")
    print("=" * 70)

    n_case27 = run_case_analysis(0.45, "Case 27 (high noise)")
    n_case28 = run_case_analysis(0.12, "Case 28 (low noise)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Case 27 (entropy=0.45): {n_case27} clusters (expected: 4)")
    print(f"Case 28 (entropy=0.12): {n_case28} clusters (expected: 4)")

    if n_case28 > 4:
        print("\nDIAGNOSIS: Over-splitting at low entropy")
        print("Possible causes:")
        print("  1. Low entropy → features are more extreme (θ near 0 or 1)")
        print("  2. Variance-weighted df reduces degrees of freedom")
        print("  3. Lower df → test is more sensitive to small differences")
        print("  4. Small sampling fluctuations become 'significant'")


def investigate_test_sensitivity():
    """Investigate why the test is too sensitive."""
    print("\n" + "=" * 70)
    print("TEST SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Generate case 28 data
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
    decomp = tree.decompose(leaf_data=data_df)
    stats = tree.stats_df

    # Analyze sibling test statistics distribution
    print("\nSibling test statistics for all internal nodes:")
    print("-" * 60)

    internal_nodes = [n for n in tree.nodes() if list(tree.successors(n))]

    test_stats = []
    dfs = []
    pvals = []

    for node in sorted(internal_nodes):
        if node in stats.index:
            row = stats.loc[node]
            ts = row.get("Sibling_Test_Statistic")
            df = row.get("Sibling_Degrees_of_Freedom")
            pv = row.get("Sibling_Divergence_P_Value")
            sig = row.get("Sibling_BH_Different")
            n_leaves = row.get("leaf_count", 0)

            if pd.notna(ts) and pd.notna(df):
                test_stats.append(ts)
                dfs.append(df)
                pvals.append(pv if pd.notna(pv) else 1.0)

                # Print first 20
                if len(test_stats) <= 20:
                    ratio = ts / df if df > 0 else 0
                    print(
                        f"  {node:8s}: n={n_leaves:3}, df={df:5.1f}, χ²={ts:8.2f}, "
                        f"ratio={ratio:.2f}, p={pv:.4f}, sig={sig}"
                    )

    if test_stats:
        print(f"\n  ... ({len(test_stats)} total internal nodes)")
        print(f"\nStatistics summary:")
        print(f"  Mean test stat: {np.mean(test_stats):.2f}")
        print(f"  Mean df: {np.mean(dfs):.1f}")
        print(
            f"  Mean ratio (χ²/df): {np.mean(np.array(test_stats) / np.array(dfs)):.2f}"
        )
        print(f"  Expected ratio under H₀: 1.0")

        # Under H₀, χ²/df should be ~1
        # If ratio >> 1, there IS a real difference
        # If ratio ~1, the test is working correctly

        ratios = np.array(test_stats) / np.array(dfs)
        print(f"\n  Nodes with ratio > 2.0: {(ratios > 2.0).sum()}")
        print(f"  Nodes with ratio > 1.5: {(ratios > 1.5).sum()}")
        print(f"  Nodes with ratio < 1.2: {(ratios < 1.2).sum()}")


if __name__ == "__main__":
    analyze_projection_usage()
    compare_cases()
    investigate_test_sensitivity()
