"""Debug script to analyze the projected chi-square scaling issue.

This script investigates why the projected local KL test is too sensitive
in high-noise cases, finding 42 clusters instead of 4.
"""

import numpy as np
from scipy.stats import chi2
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
    generate_projection_matrix,
)
from kl_clustering_analysis import config


def compute_projected_chi_square_debug(
    child_distribution: np.ndarray,
    parent_distribution: np.ndarray,
    child_leaf_count: int,
    n_features: int,
):
    """Debug version with verbose output."""
    eps = 1e-10
    n = float(child_leaf_count)

    k = compute_projection_dimension(int(n), n_features)
    print(f"n={n}, d={n_features}, k={k}")

    n_trials = config.PROJECTION_N_TRIALS
    base_seed = config.PROJECTION_RANDOM_SEED

    kl_projected_list = []
    for trial in range(n_trials):
        seed = base_seed + trial if base_seed is not None else None
        R = generate_projection_matrix(n_features, k, random_state=seed)

        child_proj = R @ child_distribution
        parent_proj = R @ parent_distribution

        diff_proj = child_proj - parent_proj

        # Current implementation: scale by average parent variance
        parent_arr = np.clip(parent_distribution, eps, 1.0 - eps)
        avg_var = float(np.mean(parent_arr * (1.0 - parent_arr)))

        # ||R·diff||² / avg_var
        norm_sq_dist = float(np.sum(diff_proj**2)) / avg_var
        kl_projected_list.append(norm_sq_dist)

        if trial == 0:
            print(f"  ||R·diff||² = {np.sum(diff_proj**2):.4f}")
            print(f"  avg_var = {avg_var:.4f}")
            print(f"  norm_sq_dist = {norm_sq_dist:.4f}")

    avg = float(np.mean(kl_projected_list))
    test_statistic = n * avg
    p_value = float(chi2.sf(test_statistic, df=k))

    print(f"  avg norm_sq_dist = {avg:.4f}")
    print(f"  test_stat (n * avg) = {test_statistic:.4f}")
    print(f"  p-value = {p_value:.6f}")

    return test_statistic, p_value, k


def analyze_critical_values():
    """Analyze critical values for different degrees of freedom."""
    print("=== Critical Values at α=0.05 ===")
    alpha = 0.05

    for df in [32, 50, 100, 200, 300]:
        crit_val = chi2.ppf(1 - alpha, df=df)
        norm_crit = crit_val / df
        print(
            f"df={df:3d}: critical χ² = {crit_val:.1f}, mean = {df}, ratio = {norm_crit:.3f}"
        )


def analyze_required_delta():
    """Analyze what delta is required to be significant."""
    print("\n=== Required Delta for Significance ===")
    n = 50  # sample size
    d = 300  # dimensions
    k = 32  # projection dimension
    var = 0.25  # Bernoulli variance at p=0.5

    # For full dimension test
    crit_full = chi2.ppf(0.95, d)
    required_scaled_full = crit_full / n
    required_delta_full = np.sqrt(required_scaled_full * var / d)
    print(f"Full dimension (d={d}): need delta > {required_delta_full:.4f}")

    # For projected test (with correct k/d scaling)
    crit_proj = chi2.ppf(0.95, k)
    required_scaled_proj = crit_proj / n / (k / d)
    required_delta_proj = np.sqrt(required_scaled_proj * var / d)
    print(
        f"Projected (k={k}, with k/d scaling): need delta > {required_delta_proj:.4f}"
    )


def test_with_known_differences():
    """Test the projection with known differences."""
    n = 50
    d = 300
    parent = np.full(d, 0.5)

    print("\n=== Testing with Known Differences ===")

    for delta in [0.01, 0.03, 0.05, 0.1]:
        print(f"\n--- Delta = {delta} ---")
        child = parent + delta
        compute_projected_chi_square_debug(child, parent, n, d)


def analyze_jl_lemma_scaling():
    """Analyze what the JL lemma tells us about scaling."""
    print("\n=== JL Lemma Scaling Analysis ===")
    d = 300
    k = 32
    n = 50
    delta = 0.03

    diff = np.full(d, delta)
    diff_sq = np.sum(diff**2)
    var = 0.25

    print(f"||diff||² = {diff_sq:.4f}")
    print(
        f"Expected ||R·diff||² = (k/d) * ||diff||² = ({k}/{d}) * {diff_sq:.4f} = {(k / d) * diff_sq:.4f}"
    )

    # Sampling variance of sample mean
    var_sample_mean = var / n
    print(f"\nSampling variance per feature = θ(1-θ)/n = {var_sample_mean:.6f}")
    print(f"d * sampling_variance = {d * var_sample_mean:.4f}")

    # Under H₀, the expected squared difference due to sampling noise
    expected_noise = d * var_sample_mean
    print(f"\nUnder H₀, E[||child - parent||²] = d * Var[θ̂] = {expected_noise:.4f}")

    # Our actual difference
    print(f"Our actual ||diff||² = {diff_sq:.4f}")
    print(f"Ratio (signal/noise) = {diff_sq / expected_noise:.2f}")


def test_with_real_clustering_data():
    """Test with the actual clustering data to see what differences we observe."""
    print("\n=== Testing with Real Clustering Data ===")

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    import pandas as pd
    from kl_clustering_analysis.benchmarking.generators import (
        generate_random_feature_matrix,
    )
    from kl_clustering_analysis.tree.poset_tree import PosetTree
    from kl_clustering_analysis import config

    # Generate high-noise data
    data_dict, true_labels = generate_random_feature_matrix(
        n_rows=200,
        n_cols=300,
        n_clusters=4,
        entropy_param=0.25,  # High noise
        random_seed=42,
    )
    data_df = pd.DataFrame(data_dict).T

    # Build tree
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(leaf_data=data_df)

    # Examine parent-child differences
    print("\nParent-child differences (first 20 edges with n > 5):")
    edge_list = list(tree.edges())

    count = 0
    for parent_id, child_id in edge_list:
        parent_dist = np.array(tree.nodes[parent_id].get("distribution", []))
        child_dist = np.array(tree.nodes[child_id].get("distribution", []))

        if len(parent_dist) == 0 or len(child_dist) == 0:
            continue

        diff = child_dist - parent_dist
        diff_sq = np.sum(diff**2)
        max_diff = np.max(np.abs(diff))
        mean_diff = np.mean(np.abs(diff))

        child_leaf_count = tree.nodes[child_id].get("leaf_count", 1)

        if child_leaf_count > 5:
            print(
                f"  {parent_id[:20]:20s} -> {child_id[:20]:20s}: "
                f"n={child_leaf_count:3d}, ||Δ||²={diff_sq:.4f}, "
                f"max|Δ|={max_diff:.4f}, mean|Δ|={mean_diff:.4f}"
            )
            count += 1
            if count >= 20:
                break


if __name__ == "__main__":
    analyze_critical_values()
    analyze_required_delta()
    analyze_jl_lemma_scaling()
    test_with_known_differences()
    test_with_real_clustering_data()
