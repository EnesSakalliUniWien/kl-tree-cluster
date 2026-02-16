"""
Purpose: Debug script to investigate mathematical inconsistency in branch length usage.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_math_consistency__branch_length__diagnostic.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import (
    compute_pooled_variance,
    standardize_proportion_difference,
)
from kl_clustering_analysis import config


def analyze_branch_length_scale():
    """Analyze what scale/units branch lengths have."""
    print("=" * 70)
    print("ANALYSIS 1: Branch Length Scale from Linkage")
    print("=" * 70)

    np.random.seed(42)
    n_samples, n_features = 20, 10

    # Generate binary data
    data = np.random.binomial(1, 0.5, (n_samples, n_features))
    data_df = pd.DataFrame(data, index=[f"s{i}" for i in range(n_samples)])

    # Build tree with Hamming distance
    Z = linkage(pdist(data_df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Collect branch lengths
    branch_lengths = []
    for parent, child, edge_data in tree.edges(data=True):
        bl = edge_data.get("branch_length", 0)
        branch_lengths.append(bl)

    print(f"\nUsing Hamming distance + average linkage:")
    print(f"  Data: {n_samples} samples × {n_features} features (binary)")
    print(f"\nBranch length statistics:")
    print(f"  Min:    {min(branch_lengths):.4f}")
    print(f"  Max:    {max(branch_lengths):.4f}")
    print(f"  Mean:   {np.mean(branch_lengths):.4f}")
    print(f"  Median: {np.median(branch_lengths):.4f}")
    print(f"  Std:    {np.std(branch_lengths):.4f}")

    # Hamming distance is in [0, 1], so branch lengths should be small
    print(f"\nNote: Hamming distance is in [0, 1]")
    print(f"      Branch lengths are merge_dist(parent) - merge_dist(child)")

    return branch_lengths, tree


def analyze_variance_scaling():
    """Show how current variance scaling works."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Current Variance Scaling Logic")
    print("=" * 70)

    # Example distributions
    theta_1 = np.array([0.6, 0.4])  # Left sibling
    theta_2 = np.array([0.4, 0.6])  # Right sibling
    n_1, n_2 = 10, 10  # Sample sizes

    print(f"\nExample sibling pair:")
    print(f"  θ₁ = {theta_1}")
    print(f"  θ₂ = {theta_2}")
    print(f"  n₁ = {n_1}, n₂ = {n_2}")

    # Base variance (no branch adjustment)
    base_var = compute_pooled_variance(theta_1, theta_2, n_1, n_2)
    print(f"\nBase variance (sampling only):")
    print(f"  Var = p(1-p) × (1/n₁ + 1/n₂)")
    print(f"  Var = {base_var}")

    # With different branch lengths
    mean_branch_length = 1.0
    print(f"\nWith Felsenstein adjustment (variance × branch_sum / mean_branch_length):")
    for branch_sum in [0.1, 0.5, 1.0, 2.0, 5.0]:
        z_adj, var_adj = standardize_proportion_difference(
            theta_1,
            theta_2,
            n_1,
            n_2,
            branch_length_sum=branch_sum,
            mean_branch_length=mean_branch_length,
        )
        z_base, _ = standardize_proportion_difference(
            theta_1, theta_2, n_1, n_2, branch_length_sum=None
        )

        print(f"\n  branch_sum = {branch_sum}:")
        print(f"    adjusted_var = {var_adj}")
        print(f"    |z_adjusted| = {np.abs(z_adj)}")
        print(f"    |z_base|     = {np.abs(z_base)}")
        print(f"    ratio        = {np.abs(z_adj[0]) / np.abs(z_base[0]):.4f}")


def analyze_mathematical_inconsistency():
    """Explain the mathematical inconsistency."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: The Mathematical Inconsistency")
    print("=" * 70)

    print("""
PROBLEM: The current implementation has a conceptual mismatch.

FELSENSTEIN'S MODEL (1985):
---------------------------
- Branch length = expected evolutionary time × mutation rate
- Variance of trait difference INCREASES with branch length
- Formula: Var(X_L - X_R) = σ² × (b_L + b_R)
- Intuition: More time to diverge → more expected difference → larger variance

CURRENT IMPLEMENTATION:
-----------------------
- Branch length = linkage merge distance (Hamming, Euclidean, etc.)
- These are in range [0, 1] for Hamming distance
- We compute: adjusted_var = base_var × branch_sum

THE ISSUE:
----------
When branch_sum < 1 (common with Hamming):
  → adjusted_var < base_var
  → |z| increases (since z = diff / sqrt(var))
  → p-value DECREASES
  → Test becomes MORE significant

This is BACKWARDS from Felsenstein's intent!
  - Felsenstein: longer branch → MORE expected divergence → LESS significant
  - Current: smaller branch_sum → smaller variance → MORE significant

WHAT BRANCH LENGTHS REPRESENT:
------------------------------
In linkage clustering:
  - Branch length = merge_dist(parent) - merge_dist(child)
  - This is the OBSERVED distance increment, not expected variance
  
In Felsenstein's PIC:
  - Branch length = time × rate = expected variance contribution
  - Estimated from molecular clock or fossil calibration
""")


def propose_solutions():
    """Propose solutions to fix the inconsistency."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Possible Solutions")
    print("=" * 70)

    print("""
OPTION 1: DISABLE BRANCH ADJUSTMENT (Simplest)
----------------------------------------------
- Remove the branch_length_sum scaling entirely
- Use only sampling variance: Var = p(1-p) × (1/n₁ + 1/n₂)
- This is what we had before and it works well

OPTION 2: NORMALIZE BRANCH LENGTHS
-----------------------------------
- Scale branch lengths so tree height = 1
- This makes branch_sum ≈ 1 on average
- But still doesn't give proper evolutionary interpretation

OPTION 3: USE INVERSE RELATIONSHIP
-----------------------------------
- Instead of: var × branch_sum
- Use: var / branch_sum (with safeguards)
- Intuition: Observed distance IS the divergence we're testing
- If branches are long (high observed distance), the difference is "explained"

OPTION 4: MODEL-BASED BRANCH LENGTHS (Complex)
----------------------------------------------
- Fit a Brownian motion or mutation model to the data
- Estimate proper evolutionary branch lengths
- Use those for Felsenstein adjustment
- This requires significant new machinery

RECOMMENDATION:
---------------
Option 1 (disable) is safest - the current test works well without it.
The Felsenstein adjustment assumes a generative model (traits evolving on tree)
that doesn't match our inferential setting (tree inferred from trait distances).
""")


def run_comparison_test():
    """Compare test behavior with and without branch adjustment."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Empirical Comparison")
    print("=" * 70)

    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
        sibling_divergence_test,
    )

    np.random.seed(123)

    # Create two slightly different distributions
    d = 10  # features
    base = np.random.uniform(0.3, 0.7, d)
    left_dist = base + np.random.normal(0, 0.05, d)
    right_dist = base + np.random.normal(0, 0.05, d)
    left_dist = np.clip(left_dist, 0.01, 0.99)
    right_dist = np.clip(right_dist, 0.01, 0.99)

    n_left, n_right = 20, 20

    print(f"\nTest case: {d} features, n={n_left} per sibling")
    print(f"Mean |diff|: {np.mean(np.abs(left_dist - right_dist)):.4f}")

    # Test without branch adjustment
    stat_none, df_none, p_none = sibling_divergence_test(
        left_dist, right_dist, n_left, n_right
    )

    mean_branch_length = 1.0

    # Test with typical Hamming-based branch lengths
    stat_small, df_small, p_small = sibling_divergence_test(
        left_dist,
        right_dist,
        n_left,
        n_right,
        branch_length_left=0.1,
        branch_length_right=0.1,  # sum=0.2
        mean_branch_length=mean_branch_length,
    )

    # Test with larger branch lengths
    stat_large, df_large, p_large = sibling_divergence_test(
        left_dist,
        right_dist,
        n_left,
        n_right,
        branch_length_left=1.0,
        branch_length_right=1.0,  # sum=2.0
        mean_branch_length=mean_branch_length,
    )

    print(f"\nResults:")
    print(f"{'Condition':<25} {'Statistic':<12} {'df':<6} {'p-value':<12}")
    print("-" * 55)
    print(f"{'No adjustment':<25} {stat_none:<12.2f} {df_none:<6.0f} {p_none:<12.6f}")
    print(
        f"{'branch_sum=0.2 (small)':<25} {stat_small:<12.2f} {df_small:<6.0f} {p_small:<12.6f}"
    )
    print(
        f"{'branch_sum=2.0 (large)':<25} {stat_large:<12.2f} {df_large:<6.0f} {p_large:<12.6f}"
    )

    print(f"\nObservation:")
    if p_small < p_none:
        print(
            f"  ⚠ Small branch_sum makes test MORE significant (p={p_small:.4f} < p={p_none:.4f})"
        )
        print(f"  This is opposite to Felsenstein's intention!")
    if p_large > p_none:
        print(
            f"  ✓ Large branch_sum makes test LESS significant (as Felsenstein intended)"
        )


if __name__ == "__main__":
    branch_lengths, tree = analyze_branch_length_scale()
    analyze_variance_scaling()
    analyze_mathematical_inconsistency()
    propose_solutions()
    run_comparison_test()

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
