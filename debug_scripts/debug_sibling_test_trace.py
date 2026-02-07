"""Debug: Why is sibling test not detecting ANY differences?

All comparisons return p > 0.99, even for clearly different distributions.
Let's trace through the test step by step.
"""

import numpy as np
from scipy.stats import chi2

from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import (
    compute_pooled_proportion,
    compute_pooled_variance,
    standardize_proportion_difference,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
    generate_projection_matrix,
)
from kl_clustering_analysis import config


def manual_sibling_test():
    """Manually trace through sibling test to find the issue."""
    print("=" * 70)
    print("Manual Trace: Sibling Divergence Test")
    print("=" * 70)

    # Two clearly different probability distributions
    # Cluster 0: high in features 0-3
    # Cluster 1: high in features 4-7
    dist_A = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
    dist_B = np.array([0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20])

    n_A = 4  # 4 samples in cluster A
    n_B = 4  # 4 samples in cluster B

    print("\nInput:")
    print(f"  dist_A: {dist_A}")
    print(f"  dist_B: {dist_B}")
    print(f"  n_A: {n_A}, n_B: {n_B}")
    print(f"  Difference: {dist_A - dist_B}")

    # Step 1: Compute pooled proportion
    print("\n" + "-" * 50)
    print("Step 1: Pooled Proportion")
    print("-" * 50)

    pooled = compute_pooled_proportion(dist_A, dist_B, n_A, n_B)
    print(f"  Pooled = (n_A * dist_A + n_B * dist_B) / (n_A + n_B)")
    print(f"  Pooled: {pooled}")

    # Step 2: Compute variance
    print("\n" + "-" * 50)
    print("Step 2: Variance of Difference")
    print("-" * 50)

    variance = compute_pooled_variance(dist_A, dist_B, n_A, n_B)
    print(f"  Var = pooled * (1 - pooled) * (1/n_A + 1/n_B)")
    print(f"  Variance: {variance}")
    print(f"  Std Dev: {np.sqrt(variance)}")

    # Step 3: Compute Z-scores
    print("\n" + "-" * 50)
    print("Step 3: Z-scores (standardized difference)")
    print("-" * 50)

    z, var = standardize_proportion_difference(dist_A, dist_B, n_A, n_B)
    print(f"  z = (dist_A - dist_B) / sqrt(variance)")
    print(f"  z: {z}")
    print(f"  |z|: {np.abs(z)}")
    print(f"  Sum of z²: {np.sum(z**2):.4f}")

    # Step 4: Projection dimension
    print("\n" + "-" * 50)
    print("Step 4: Projection Dimension (JL lemma)")
    print("-" * 50)

    n_eff = 2 * n_A * n_B / (n_A + n_B)  # harmonic mean
    d = len(z)
    k = compute_projection_dimension(int(n_eff), d)
    print(f"  n_eff (harmonic mean): {n_eff:.2f}")
    print(f"  d (dimension): {d}")
    print(f"  k (projection dim): {k}")

    # Step 5: Random projection
    print("\n" + "-" * 50)
    print("Step 5: Random Projection")
    print("-" * 50)

    R = generate_projection_matrix(d, k, config.PROJECTION_RANDOM_SEED)
    print(f"  R shape: {R.shape}")
    print(f"  R @ R.T (should be I):\n{R @ R.T}")

    projected = R @ z
    print(f"  Projected z: {projected}")

    # Step 6: Test statistic
    print("\n" + "-" * 50)
    print("Step 6: Test Statistic")
    print("-" * 50)

    stat = float(np.sum(projected**2))
    print(f"  Statistic = sum(projected²) = {stat:.4f}")
    print(f"  Compare to sum(z²) = {np.sum(z**2):.4f}")
    print(f"  Ratio (should be ~k/d = {k / d:.3f}): {stat / np.sum(z**2):.4f}")

    # Step 7: P-value
    print("\n" + "-" * 50)
    print("Step 7: P-value from χ²(k)")
    print("-" * 50)

    p_val = chi2.sf(stat, df=k)
    print(f"  P-value: {p_val:.6f}")
    print(f"  Critical value χ²(k={k}, α=0.05): {chi2.ppf(0.95, df=k):.4f}")

    if p_val < 0.05:
        print(f"\n  ✓ SIGNIFICANT at α=0.05: Distributions are DIFFERENT")
    else:
        print(f"\n  ✗ NOT significant: Test says distributions are SAME")
        print(f"    But they are clearly different!")

    # Diagnose the issue
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    print("""
    The issue is likely the projection dimension k.
    
    If k is too small, we lose information and can't detect differences.
    
    Let's check with different k values:
    """)

    for k_test in [1, 2, 4, 6, 8]:
        R_test = generate_projection_matrix(d, k_test, config.PROJECTION_RANDOM_SEED)
        proj_test = R_test @ z
        stat_test = float(np.sum(proj_test**2))
        p_test = chi2.sf(stat_test, df=k_test)
        sig = "***" if p_test < 0.05 else ""
        print(f"    k={k_test}: stat={stat_test:.2f}, p={p_test:.6f} {sig}")

    print("\n    Full dimension (no projection):")
    stat_full = float(np.sum(z**2))
    p_full = chi2.sf(stat_full, df=d)
    print(f"    k={d} (full): stat={stat_full:.2f}, p={p_full:.6f}")


def test_without_projection():
    """Test without random projection - just sum of z²."""
    print("\n" + "=" * 70)
    print("Test WITHOUT Random Projection")
    print("=" * 70)

    # Same clear difference
    dist_A = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
    dist_B = np.array([0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20])

    for n in [1, 2, 4, 8, 16, 32]:
        z, _ = standardize_proportion_difference(dist_A, dist_B, n, n)
        stat = float(np.sum(z**2))
        d = len(z)
        p_val = chi2.sf(stat, df=d)
        sig = "***" if p_val < 0.05 else ""
        print(f"  n={n:2d}: stat={stat:8.2f}, df={d}, p={p_val:.6f} {sig}")


if __name__ == "__main__":
    manual_sibling_test()
    test_without_projection()
