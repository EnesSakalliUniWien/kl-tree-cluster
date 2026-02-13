"""Analyze Case 17: binary_balanced_low_noise with 72x160, 4 clusters, entropy=0.25."""

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    should_use_projection,
    compute_projection_dimension,
)


def analyze_case_17():
    """Analyze Case 17: why is it finding 1 cluster instead of 4?"""
    print("=" * 70)
    print("CASE 17: binary_balanced_low_noise (72x160, 4 clusters, entropy=0.25)")
    print("=" * 70)

    # Generate Case 17 data
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=72,
        n_cols=160,
        n_clusters=4,
        entropy_param=0.25,
        balanced_clusters=True,
        random_seed=314,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    n_samples, n_features = data_df.shape
    print(f"\nData: {n_samples} samples × {n_features} features")
    print(f"True clusters: 4")
    print(f"Entropy (noise): 0.25")

    # Get unique clusters
    unique_clusters = np.unique(true_labels)
    print(f"\nCluster sizes:")
    for c in unique_clusters:
        size = np.sum(true_labels == c)
        print(f"  Cluster {c}: {size} samples")

    # Check if projection is used
    n_eff = n_samples // 2  # approximate for balanced split
    print(f"\n{'=' * 70}")
    print("PROJECTION SETTINGS")
    print("=" * 70)
    print(f"n_features = {n_features}")
    print(f"n_eff (approx) = {n_eff}")
    print(f"Threshold: n_features > 2.0 * n_eff = {2.0 * n_eff}")
    print(f"Use projection? {should_use_projection(n_features, n_eff)}")

    if should_use_projection(n_features, n_eff):
        k = compute_projection_dimension(n_eff, n_features)
        print(f"Projection dimension k = {k}")
    else:
        print("No projection - using full dimensionality")

    # Compute mean distributions per cluster
    cluster_means = {}
    for c in unique_clusters:
        mask = true_labels == c
        cluster_means[c] = data_df.values[mask].mean(axis=0)

    # Feature statistics
    global_mean = data_df.values.mean(axis=0)
    print(f"\nFeature statistics (global):")
    print(f"  Mean θ range: [{global_mean.min():.3f}, {global_mean.max():.3f}]")
    print(f"  Mean θ mean: {global_mean.mean():.3f}")

    # Variance-weighted effective df
    variance_weights = 4.0 * global_mean * (1.0 - global_mean)
    df_eff = variance_weights.sum()
    print(f"  Variance-weighted df: {df_eff:.1f} (vs raw df={n_features})")

    # Compute pairwise signal strength
    print(f"\n{'=' * 70}")
    print("PAIRWISE SIGNAL STRENGTH BETWEEN TRUE CLUSTERS")
    print("=" * 70)

    d = n_features

    print(
        f"{'Pair':<8} {'n1':<5} {'n2':<5} {'||z||²':<10} {'df':<6} {'Ratio':<8} {'p-value'}"
    )
    print("-" * 60)

    signals = []
    for i, c1 in enumerate(unique_clusters):
        for c2 in unique_clusters[i + 1 :]:
            n1 = np.sum(true_labels == c1)
            n2 = np.sum(true_labels == c2)

            theta1 = cluster_means[c1]
            theta2 = cluster_means[c2]

            # Standardize difference
            pooled = 0.5 * (theta1 + theta2)
            pooled = np.clip(pooled, 1e-10, 1 - 1e-10)
            inv_n = 1 / n1 + 1 / n2
            var_diff = pooled * (1 - pooled) * inv_n
            var_diff = np.maximum(var_diff, 1e-10)

            diff = theta1 - theta2
            z = diff / np.sqrt(var_diff)
            z_sq = np.sum(z**2)

            p_full = chi2.sf(z_sq, df=d)
            ratio = z_sq / d

            signals.append((c1, c2, z_sq, ratio, p_full))
            print(
                f"{c1}-{c2:<6} {n1:<5} {n2:<5} {z_sq:<10.1f} {d:<6} {ratio:<8.2f} {p_full:.2e}"
            )

    avg_signal = np.mean([s[2] for s in signals])
    avg_ratio = np.mean([s[3] for s in signals])
    print(f"\nAverage ||z||²: {avg_signal:.1f}")
    print(f"Average ratio (||z||²/d): {avg_ratio:.2f}")

    # Check with projection if applicable
    if should_use_projection(n_features, n_eff):
        k = compute_projection_dimension(n_eff, n_features)
        print(f"\n{'=' * 70}")
        print(f"PROJECTED SIGNAL (k={k})")
        print("=" * 70)

        rng = np.random.default_rng(42)
        G = rng.standard_normal((k, d))
        Q, _ = qr(G.T, mode="economic")
        R = Q.T

        print(f"{'Pair':<8} {'||Rz||²':<12} {'df':<6} {'Ratio':<8} {'p-value'}")
        print("-" * 50)

        for c1, c2, z_sq_full, _, _ in signals:
            n1 = np.sum(true_labels == c1)
            n2 = np.sum(true_labels == c2)

            theta1 = cluster_means[c1]
            theta2 = cluster_means[c2]

            pooled = 0.5 * (theta1 + theta2)
            pooled = np.clip(pooled, 1e-10, 1 - 1e-10)
            inv_n = 1 / n1 + 1 / n2
            var_diff = pooled * (1 - pooled) * inv_n
            var_diff = np.maximum(var_diff, 1e-10)

            diff = theta1 - theta2
            z = diff / np.sqrt(var_diff)

            z_proj = R @ z
            z_sq_proj = np.sum(z_proj**2)

            p_proj = chi2.sf(z_sq_proj, df=k)
            ratio = z_sq_proj / k

            print(f"{c1}-{c2:<6} {z_sq_proj:<12.1f} {k:<6} {ratio:<8.2f} {p_proj:.2e}")

    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if avg_ratio < 1.5:
        print("⚠️  Weak signal: True clusters have limited separation")
        print(f"   Average ratio = {avg_ratio:.2f} (need >> 1.0 for detection)")
        print("   entropy=0.25 means 25% noise level")
    else:
        print("✓ Signal should be detectable")
        print(f"   Average ratio = {avg_ratio:.2f}")

    # Check the hierarchical structure
    print()
    print("=" * 70)
    print("RUNNING FULL DECOMPOSITION")
    print("=" * 70)

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    from kl_clustering_analysis.tree.poset_tree import PosetTree
    from kl_clustering_analysis import config

    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
        TreeDecomposition,
    )

    # 1. Update stats
    tree.populate_node_divergences(pd.DataFrame(data_df))
    from kl_clustering_analysis.hierarchy_analysis.statistics import (
        annotate_child_parent_divergence,
        annotate_sibling_divergence,
    )

    tree.stats_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    tree.stats_df = annotate_sibling_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )

    # 2. Run V2
    decomposer = TreeDecomposition(
        tree=tree,
        results_df=tree.stats_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
        use_signal_localization=True,
        localization_max_depth=5,
        localization_min_samples=2,
    )

    # Run V1 for comparison
    decomp_v1 = decomposer.decompose_tree()
    print(f"V1 Found {decomp_v1['num_clusters']} clusters (expected: 4)")

    # Run V2
    decomp = decomposer.decompose_tree_v2()
    print(f"V2 Found {decomp['num_clusters']} clusters (expected: 4)")

    # Check sibling test results
    internal_nodes = [n for n in tree.nodes() if list(tree.successors(n))]
    stats = tree.stats_df

    sig_count = 0
    not_sig_count = 0
    for node in internal_nodes:
        if node in stats.index:
            sig = stats.loc[node].get("Sibling_BH_Different", False)
            if sig:
                sig_count += 1
            else:
                not_sig_count += 1

    print(f"\nSibling tests:")
    print(f"  Significant (split): {sig_count}")
    print(f"  Not significant (merge): {not_sig_count}")

    # Look at the root split
    root = [n for n in tree.nodes() if tree.in_degree(n) == 0][0]
    print(f"\nRoot node: {root}")
    if root in stats.index:
        row = stats.loc[root]
        print(f"  Sibling_Test_Statistic: {row.get('Sibling_Test_Statistic', 'N/A')}")
        print(
            f"  Sibling_Degrees_of_Freedom: {row.get('Sibling_Degrees_of_Freedom', 'N/A')}"
        )
        print(
            f"  Sibling_Divergence_P_Value: {row.get('Sibling_Divergence_P_Value', 'N/A')}"
        )
        print(f"  Sibling_BH_Different: {row.get('Sibling_BH_Different', 'N/A')}")


if __name__ == "__main__":
    analyze_case_17()
