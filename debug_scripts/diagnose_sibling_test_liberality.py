"""Diagnose why sibling test is too liberal (fails to detect differences).

We saw that with clearly different distributions (0.20 vs 0.05), the test
returned p > 0.05, merging everything into one cluster.

Potential causes:
1. Low sample size → high variance → low power
2. Branch length adjustment → inflated variance → even lower power
3. Projection dimension too small → information loss
4. Test statistic inappropriate for this data type
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import (
    compute_pooled_variance,
    standardize_proportion_difference,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
    generate_projection_matrix,
)
from kl_clustering_analysis import config


def diagnose_test_sensitivity():
    """Diagnose why the sibling test fails to detect differences."""
    print("=" * 70)
    print("DIAGNOSIS: Why is the sibling test too liberal?")
    print("=" * 70)

    # Create clearly different distributions
    np.random.seed(42)

    # Pattern A: high in first 4 features
    pattern_A = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
    # Pattern B: high in last 4 features
    pattern_B = np.array([0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20])

    print(f"\nPattern A: {pattern_A}")
    print(f"Pattern B: {pattern_B}")
    print(
        f"Absolute difference per feature: {np.abs(pattern_A - pattern_B).mean():.3f}"
    )

    # Test with different sample sizes
    print("\n" + "-" * 50)
    print("FACTOR 1: Sample Size Effect")
    print("-" * 50)

    print(
        f"\n{'n_samples':<12} {'z_mean':<10} {'sum_z²':<12} {'k':<6} {'stat':<10} {'p-value':<12} {'result'}"
    )
    print("-" * 75)

    for n in [2, 4, 8, 16, 32, 64]:
        z, var = standardize_proportion_difference(pattern_A, pattern_B, n, n)
        d = len(z)
        k = compute_projection_dimension(n, d)

        R = generate_projection_matrix(d, k, config.PROJECTION_RANDOM_SEED)
        projected = R @ z
        stat = float(np.sum(projected**2))
        p_val = chi2.sf(stat, df=k)

        result = "DIFFERENT" if p_val < 0.05 else "same (merge)"
        print(
            f"{n:<12} {np.mean(np.abs(z)):<10.2f} {np.sum(z**2):<12.2f} {k:<6} {stat:<10.2f} {p_val:<12.6f} {result}"
        )

    print("\n→ Need n≥32 to detect this difference!")

    # Test with branch length adjustment
    print("\n" + "-" * 50)
    print("FACTOR 2: Branch Length Adjustment Effect")
    print("-" * 50)

    n = 8  # Typical cluster size
    print(f"\nFixed n = {n} samples per group")
    print(
        f"\n{'branch_sum':<12} {'z_mean':<10} {'stat':<10} {'p-value':<12} {'result'}"
    )
    print("-" * 55)

    for branch_sum in [None, 0.1, 0.5, 1.0, 2.0]:
        z, var = standardize_proportion_difference(
            pattern_A, pattern_B, n, n, branch_length_sum=branch_sum
        )
        d = len(z)
        k = compute_projection_dimension(n, d)

        R = generate_projection_matrix(d, k, config.PROJECTION_RANDOM_SEED)
        projected = R @ z
        stat = float(np.sum(projected**2))
        p_val = chi2.sf(stat, df=k)

        result = "DIFFERENT" if p_val < 0.05 else "same (merge)"
        bl_str = str(branch_sum) if branch_sum else "None"
        print(
            f"{bl_str:<12} {np.mean(np.abs(z)):<10.2f} {stat:<10.2f} {p_val:<12.6f} {result}"
        )

    print("\n→ Branch lengths < 1.0 make test MORE sensitive (smaller variance)")
    print("→ Branch lengths > 1.0 make test LESS sensitive (larger variance)")

    # Test projection dimension
    print("\n" + "-" * 50)
    print("FACTOR 3: Projection Dimension Effect")
    print("-" * 50)

    n = 8
    z, _ = standardize_proportion_difference(pattern_A, pattern_B, n, n)
    d = len(z)
    sum_z_sq = np.sum(z**2)

    print(f"\nFixed n = {n}, d = {d}, sum(z²) = {sum_z_sq:.2f}")
    print(f"\n{'k':<6} {'stat':<10} {'critical':<10} {'p-value':<12} {'result'}")
    print("-" * 50)

    for k in [2, 4, 6, 8, 10, 16]:
        R = generate_projection_matrix(d, k, config.PROJECTION_RANDOM_SEED)
        projected = R @ z
        stat = float(np.sum(projected**2))
        critical = chi2.ppf(0.95, df=k)
        p_val = chi2.sf(stat, df=k)

        result = "DIFFERENT" if p_val < 0.05 else "same"
        print(f"{k:<6} {stat:<10.2f} {critical:<10.2f} {p_val:<12.6f} {result}")

    # What k does the JL lemma give us?
    k_jl = compute_projection_dimension(n, d)
    print(f"\n→ JL lemma gives k = {k_jl} for n = {n}")

    # Check the config
    print("\n" + "-" * 50)
    print("FACTOR 4: Current Configuration")
    print("-" * 50)
    print(f"\nPROJECTION_EPS = {config.PROJECTION_EPS}")
    print(f"PROJECTION_MIN_K = {config.PROJECTION_MIN_K}")
    print(f"PROJECTION_METHOD = {config.PROJECTION_METHOD}")


def analyze_real_scenario():
    """Analyze a realistic clustering scenario."""
    print("\n" + "=" * 70)
    print("REALISTIC SCENARIO: 20 samples, 2 clusters")
    print("=" * 70)

    np.random.seed(42)
    n_per_cluster = 10

    # Create data
    pattern_A = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
    pattern_B = np.array([0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20])

    cluster_A = [pattern_A + np.random.normal(0, 0.02, 8) for _ in range(n_per_cluster)]
    cluster_B = [pattern_B + np.random.normal(0, 0.02, 8) for _ in range(n_per_cluster)]

    data = np.vstack([cluster_A, cluster_B])
    data = np.clip(data, 0.01, 0.99)

    sample_names = [f"s{i}" for i in range(2 * n_per_cluster)]
    data_df = pd.DataFrame(data, index=sample_names)

    print(f"Data shape: {data_df.shape}")
    print(f"True clusters: 0-9 (A), 10-19 (B)")

    # Build tree
    Z = linkage(pdist(data_df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)
    tree.populate_node_divergences(leaf_data=data_df)

    # Check branch lengths
    branch_lengths = [d.get("branch_length", 0) for _, _, d in tree.edges(data=True)]
    print(
        f"\nBranch lengths: min={min(branch_lengths):.4f}, max={max(branch_lengths):.4f}, mean={np.mean(branch_lengths):.4f}"
    )

    # Find root and test siblings
    root = tree.root()
    children = list(tree.successors(root))

    if len(children) == 2:
        left, right = children

        left_dist = tree.nodes[left].get("distribution")
        right_dist = tree.nodes[right].get("distribution")
        n_left = tree.nodes[left].get("n_samples", 1)
        n_right = tree.nodes[right].get("n_samples", 1)

        left_bl = tree.edges[root, left].get("branch_length", 0)
        right_bl = tree.edges[root, right].get("branch_length", 0)
        branch_sum = left_bl + right_bl

        print(f"\nRoot children: {left} (n={n_left}), {right} (n={n_right})")
        print(f"Branch lengths: {left_bl:.4f} + {right_bl:.4f} = {branch_sum:.4f}")

        # Test WITHOUT branch adjustment
        z_no_bl, _ = standardize_proportion_difference(
            left_dist, right_dist, n_left, n_right
        )
        d = len(z_no_bl)
        k = compute_projection_dimension(
            int(2 * n_left * n_right / (n_left + n_right)), d
        )
        R = generate_projection_matrix(d, k, config.PROJECTION_RANDOM_SEED)
        proj_no_bl = R @ z_no_bl
        stat_no_bl = float(np.sum(proj_no_bl**2))
        p_no_bl = chi2.sf(stat_no_bl, df=k)

        # Test WITH branch adjustment
        z_with_bl, _ = standardize_proportion_difference(
            left_dist, right_dist, n_left, n_right, branch_length_sum=branch_sum
        )
        proj_with_bl = R @ z_with_bl
        stat_with_bl = float(np.sum(proj_with_bl**2))
        p_with_bl = chi2.sf(stat_with_bl, df=k)

        print(f"\nTest results (k={k}):")
        print(f"  Without branch adjustment: stat={stat_no_bl:.2f}, p={p_no_bl:.6f}")
        print(
            f"  With branch adjustment:    stat={stat_with_bl:.2f}, p={p_with_bl:.6f}"
        )

        if p_no_bl < 0.05:
            print("\n✓ Without branch adjustment: would SPLIT (correct!)")
        else:
            print("\n✗ Without branch adjustment: would MERGE (wrong!)")

        if p_with_bl < 0.05:
            print("✓ With branch adjustment: would SPLIT")
        else:
            print("✗ With branch adjustment: would MERGE")


def propose_solutions():
    """Propose solutions to make the test more sensitive."""
    print("\n" + "=" * 70)
    print("PROPOSED SOLUTIONS")
    print("=" * 70)

    print("""
    PROBLEM: The test is too liberal (merges too much) because:
    1. Low power with small sample sizes
    2. Branch length adjustment (if sum > 1) inflates variance further
    
    SOLUTIONS:
    
    1. DISABLE branch length adjustment
       - Set branch_length_sum = None in pooled_variance.py
       - Pro: Increases power
       - Con: Loses phylogenetic interpretation
    
    2. USE smaller significance threshold
       - Instead of α = 0.05, use α = 0.20 or α = 0.30
       - Pro: More likely to split
       - Con: More false positives
       - Note: This is opposite of usual "conservative" approach!
    
    3. USE one-sided test
       - Currently: testing "is difference non-zero?"
       - Alternative: test "is difference large enough?"
       - Pro: More focused hypothesis
       - Con: Need to define "large enough"
    
    4. USE distance-based criterion instead
       - Replace p-value test with distance threshold
       - E.g., Jensen-Shannon > 0.1 → split
       - Pro: Works regardless of sample size
       - Con: Need to tune threshold
    
    5. COMBINE p-value with distance
       - Split if: p < α OR distance > threshold
       - Pro: Gets both statistical and practical significance
       - Con: More parameters to tune
    
    6. ADJUST projection dimension
       - Use larger k to preserve more information
       - Pro: More accurate test
       - Con: Increases degrees of freedom, may reduce power
    
    RECOMMENDED: Start with option 4 (distance-based) or option 5 (combined).
    """)


if __name__ == "__main__":
    diagnose_test_sensitivity()
    analyze_real_scenario()
    propose_solutions()
