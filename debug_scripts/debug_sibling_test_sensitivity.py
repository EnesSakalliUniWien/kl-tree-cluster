"""Debug: Why is the sibling test not finding differences?

From the previous run:
- Tree correctly grouped: N4 contains s0,s2 (both pattern A), N5 contains s1,s3 (both pattern B)
- But sibling test N4 vs N5 returned p=0.5733 (NOT different!)

This is wrong. Let's investigate what distributions are being compared.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)


def investigate_sibling_test_failure():
    """Investigate why sibling test failed to detect difference."""
    print("=" * 70)
    print("Investigating Sibling Test Failure")
    print("=" * 70)

    # Create the same data
    np.random.seed(42)
    pattern_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pattern_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    data = np.array(
        [
            pattern_a + np.array([0, 0, 0, 0, 0.2, 0.2, 0, 0]),  # s0
            pattern_b + np.array([0.2, 0.2, 0, 0, 0, 0, 0, 0]),  # s1
            pattern_a + np.array([0, 0, 0.2, 0.2, 0, 0, 0, 0]),  # s2
            pattern_b + np.array([0, 0, 0, 0, 0, 0, 0.2, 0.2]),  # s3
        ]
    )

    data_df = pd.DataFrame(data, index=["s0", "s1", "s2", "s3"])

    print("\nInput Data:")
    print(data_df)

    # Build tree
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Populate distributions
    tree.populate_node_divergences(leaf_data=data_df)

    print("\n" + "-" * 50)
    print("Node Distributions after populate_node_divergences:")
    print("-" * 50)

    for node in tree.nodes:
        dist = tree.nodes[node].get("distribution")
        n_samples = tree.nodes[node].get("n_samples")
        label = tree.nodes[node].get("label", node)
        is_leaf = tree.nodes[node].get("is_leaf", False)

        if dist is not None:
            print(
                f"\n{node} (label={label}, is_leaf={is_leaf}, n_samples={n_samples}):"
            )
            print(f"  distribution: {dist}")

    print("\n" + "-" * 50)
    print("Direct Sibling Test: N4 vs N5")
    print("-" * 50)

    # Get distributions for N4 and N5
    dist_n4 = tree.nodes["N4"].get("distribution")
    dist_n5 = tree.nodes["N5"].get("distribution")
    n_n4 = tree.nodes["N4"].get("n_samples")
    n_n5 = tree.nodes["N5"].get("n_samples")

    print(f"\nN4 (s0, s2) distribution: {dist_n4}")
    print(f"N4 n_samples: {n_n4}")
    print(f"\nN5 (s1, s3) distribution: {dist_n5}")
    print(f"N5 n_samples: {n_n5}")

    # Run sibling test
    stat, df, p_val = sibling_divergence_test(
        dist_n4, dist_n5, n_left=n_n4, n_right=n_n5
    )

    print(f"\nSibling test result:")
    print(f"  statistic = {stat:.4f}")
    print(f"  df = {df}")
    print(f"  p-value = {p_val:.6f}")

    if p_val > 0.05:
        print("\n⚠️ TEST SAYS: NOT DIFFERENT (p > 0.05)")
        print("   But the distributions are clearly different!")

    # Now let's understand WHY
    print("\n" + "-" * 50)
    print("Understanding WHY the test failed")
    print("-" * 50)

    print("\n1. What should the pooled distributions look like?")
    print(f"   N4 aggregates s0 and s2:")
    print(f"     s0 = {data_df.loc['s0'].values}")
    print(f"     s2 = {data_df.loc['s2'].values}")
    print(
        f"     Expected average: {(data_df.loc['s0'].values + data_df.loc['s2'].values) / 2}"
    )

    print(f"\n   N5 aggregates s1 and s3:")
    print(f"     s1 = {data_df.loc['s1'].values}")
    print(f"     s3 = {data_df.loc['s3'].values}")
    print(
        f"     Expected average: {(data_df.loc['s1'].values + data_df.loc['s3'].values) / 2}"
    )

    print("\n2. How are distributions populated for internal nodes?")

    # Check the populate_node_divergences logic
    from kl_clustering_analysis.tree.poset_tree import PosetTree
    import inspect

    # Let's look at what populate_node_divergences does
    print("\n   Reading populate_node_divergences source...")

    # Manually compute what should happen
    print("\n3. Computing expected distributions manually:")

    # For leaf nodes, distribution should be the raw data
    # For internal nodes, it should be the average of children? Or sum?

    # Let's check by reading the actual implementation
    print("\n   The stored distribution for N4:", dist_n4)
    print("   Sum of s0 and s2:", data_df.loc["s0"].values + data_df.loc["s2"].values)
    print(
        "   Mean of s0 and s2:",
        (data_df.loc["s0"].values + data_df.loc["s2"].values) / 2,
    )

    # Check if it matches sum or mean
    if np.allclose(dist_n4, data_df.loc["s0"].values + data_df.loc["s2"].values):
        print("   -> N4 distribution = SUM of children")
    elif np.allclose(
        dist_n4, (data_df.loc["s0"].values + data_df.loc["s2"].values) / 2
    ):
        print("   -> N4 distribution = MEAN of children")
    else:
        print("   -> N4 distribution = SOMETHING ELSE")
        print(
            f"   Difference from sum: {np.abs(dist_n4 - (data_df.loc['s0'].values + data_df.loc['s2'].values))}"
        )
        print(
            f"   Difference from mean: {np.abs(dist_n4 - (data_df.loc['s0'].values + data_df.loc['s2'].values) / 2)}"
        )


def test_with_raw_leaf_data():
    """Test sibling divergence directly with leaf-level data."""
    print("\n" + "=" * 70)
    print("Testing with Raw Leaf-Level Data")
    print("=" * 70)

    np.random.seed(42)
    pattern_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pattern_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # s0 and s2 (pattern A)
    s0 = pattern_a + np.array([0, 0, 0, 0, 0.2, 0.2, 0, 0])
    s2 = pattern_a + np.array([0, 0, 0.2, 0.2, 0, 0, 0, 0])

    # s1 and s3 (pattern B)
    s1 = pattern_b + np.array([0.2, 0.2, 0, 0, 0, 0, 0, 0])
    s3 = pattern_b + np.array([0, 0, 0, 0, 0, 0, 0.2, 0.2])

    print("\nRaw data:")
    print(f"  s0 = {s0}")
    print(f"  s2 = {s2}")
    print(f"  s1 = {s1}")
    print(f"  s3 = {s3}")

    # Compare s0 (single sample) vs s1 (single sample)
    print("\n--- Test: s0 vs s1 (single samples) ---")
    stat, df, p_val = sibling_divergence_test(s0, s1, n_left=1.0, n_right=1.0)
    print(f"  stat={stat:.2f}, df={df}, p={p_val:.6f}")

    # Compare (s0+s2)/2 vs (s1+s3)/2 with n=2 each
    print("\n--- Test: mean(s0,s2) vs mean(s1,s3) with n=2 each ---")
    mean_a = (s0 + s2) / 2
    mean_b = (s1 + s3) / 2
    print(f"  mean(s0,s2) = {mean_a}")
    print(f"  mean(s1,s3) = {mean_b}")
    stat, df, p_val = sibling_divergence_test(mean_a, mean_b, n_left=2.0, n_right=2.0)
    print(f"  stat={stat:.2f}, df={df}, p={p_val:.6f}")

    # Compare sum(s0,s2) vs sum(s1,s3) with n=2 each
    print("\n--- Test: sum(s0,s2) vs sum(s1,s3) with n=2 each ---")
    sum_a = s0 + s2
    sum_b = s1 + s3
    print(f"  sum(s0,s2) = {sum_a}")
    print(f"  sum(s1,s3) = {sum_b}")
    stat, df, p_val = sibling_divergence_test(sum_a, sum_b, n_left=2.0, n_right=2.0)
    print(f"  stat={stat:.2f}, df={df}, p={p_val:.6f}")

    # The issue might be that the test expects PROBABILITY distributions
    # but we're passing raw counts/values
    print("\n" + "-" * 50)
    print("CHECKING: Does sibling test expect probabilities?")
    print("-" * 50)

    # Normalize to probabilities
    print("\n--- Test with NORMALIZED distributions (sum to 1) ---")
    prob_a = mean_a / mean_a.sum()
    prob_b = mean_b / mean_b.sum()
    print(f"  prob(s0,s2) = {prob_a} (sum={prob_a.sum():.4f})")
    print(f"  prob(s1,s3) = {prob_b} (sum={prob_b.sum():.4f})")
    stat, df, p_val = sibling_divergence_test(prob_a, prob_b, n_left=2.0, n_right=2.0)
    print(f"  stat={stat:.2f}, df={df}, p={p_val:.6f}")

    if p_val < 0.05:
        print("  -> SIGNIFICANT! Normalization matters!")


def check_sibling_test_implementation():
    """Check what sibling_divergence_test actually does."""
    print("\n" + "=" * 70)
    print("Checking sibling_divergence_test Implementation")
    print("=" * 70)

    from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
        sibling_divergence_test,
    )

    # Read the source code
    import inspect

    source = inspect.getsource(sibling_divergence_test)

    # Print first 100 lines
    lines = source.split("\n")[:100]
    print("\nFirst 100 lines of sibling_divergence_test:")
    print("-" * 50)
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")


if __name__ == "__main__":
    investigate_sibling_test_failure()
    test_with_raw_leaf_data()
    check_sibling_test_implementation()
