"""Clean test: Asymmetric comparison with proper probability data.

The issue in previous tests:
1. Data values > 1.0 (not probabilities)
2. n=1 gives very high variance, hard to detect differences

Let's use proper probability data and larger samples.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)


def create_proper_probability_data():
    """Create data with proper probabilities in [0,1]."""
    print("=" * 70)
    print("Proper Probability Data Test")
    print("=" * 70)

    np.random.seed(42)

    # Each sample is a probability distribution over 8 categories (sums to 1)
    # Cluster 1: strong in features 0-3
    # Cluster 2: strong in features 4-7

    n_per_cluster = 10
    n_features = 8

    # Create cluster 1: 10 samples
    cluster1 = []
    for i in range(n_per_cluster):
        # High prob in first 4 features
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05])
        noise = np.random.dirichlet(np.ones(n_features) * 10)  # Add noise
        sample = 0.8 * probs + 0.2 * noise  # Mix: mostly pattern, some noise
        sample = sample / sample.sum()  # Normalize to sum to 1
        cluster1.append(sample)

    # Create cluster 2: 10 samples
    cluster2 = []
    for i in range(n_per_cluster):
        # High prob in last 4 features
        probs = np.array([0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2])
        noise = np.random.dirichlet(np.ones(n_features) * 10)
        sample = 0.8 * probs + 0.2 * noise
        sample = sample / sample.sum()
        cluster2.append(sample)

    data = np.vstack([cluster1, cluster2])
    labels = [0] * n_per_cluster + [1] * n_per_cluster

    data_df = pd.DataFrame(data, index=[f"s{i}" for i in range(2 * n_per_cluster)])

    print(f"\nData shape: {data_df.shape}")
    print(f"True clusters: 0-9 (cluster 0), 10-19 (cluster 1)")
    print(f"\nSample row sums (should be ~1): {data_df.sum(axis=1).values[:5]}")

    return data_df, labels


def test_with_proper_data():
    """Test with proper probability data."""
    data_df, true_labels = create_proper_probability_data()

    # Build tree
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Run decomposition
    print("\n" + "-" * 50)
    print("Running decomposition...")
    print("-" * 50)

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    cluster_assignments = decomp.get("cluster_assignments", {})
    print(f"\nFound {len(cluster_assignments)} clusters:")

    pred_labels = {}
    for cid, info in cluster_assignments.items():
        leaves = sorted(info["leaves"], key=lambda x: int(x[1:]))
        print(f"  Cluster {cid}: {leaves}")
        for leaf in info["leaves"]:
            pred_labels[leaf] = cid

    # Compute ARI
    gt = true_labels
    pred = [pred_labels.get(f"s{i}", -1) for i in range(len(true_labels))]

    ari = adjusted_rand_score(gt, pred)
    print(f"\nARI vs ground truth: {ari:.4f}")

    return tree, data_df, ari


def test_asymmetric_scenario():
    """Create asymmetric scenario where A's child belongs with B."""
    print("\n" + "=" * 70)
    print("ASYMMETRIC SCENARIO: Some of A's descendants belong with B")
    print("=" * 70)

    np.random.seed(123)
    n_features = 8

    # True structure: 3 clusters
    # - Cluster 0: s0, s1, s4, s5 (strong in features 0-3)
    # - Cluster 1: s2, s3, s6, s7 (strong in features 4-7)
    # - But hierarchical clustering might put s4,s5 under different parent

    # Create samples with clear patterns
    pattern_A = np.array([0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05])
    pattern_B = np.array([0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2])

    data = []
    true_labels = []

    for i in range(8):
        if i in [0, 1, 4, 5]:  # Cluster 0
            base = pattern_A.copy()
            true_labels.append(0)
        else:  # Cluster 1
            base = pattern_B.copy()
            true_labels.append(1)

        # Add noise
        noise = np.random.dirichlet(np.ones(n_features) * 20)
        sample = 0.85 * base + 0.15 * noise
        sample = sample / sample.sum()
        data.append(sample)

    data_df = pd.DataFrame(np.array(data), index=[f"s{i}" for i in range(8)])

    print("\nData (probability distributions):")
    print(data_df.round(3))
    print(f"\nGround truth: Cluster 0 = [s0,s1,s4,s5], Cluster 1 = [s2,s3,s6,s7]")

    # Build tree
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(leaf_data=data_df)

    print("\n" + "-" * 50)
    print("Tree structure:")
    print("-" * 50)

    for parent, child in tree.edges():
        label = tree.nodes[child].get("label", child)
        print(f"  {parent} -> {child} ({label})")

    # Find root and its children
    root = tree.root()
    root_children = list(tree.successors(root))

    print(f"\nRoot: {root}")
    print(f"Root's children: {root_children}")

    # For each child of root, show its descendants
    for child in root_children:
        descendants = list(tree.successors(child))
        if descendants:
            print(f"  {child} -> {descendants}")
            for d in descendants:
                dd = list(tree.successors(d))
                if dd:
                    print(f"    {d} -> {dd}")

    print("\n" + "-" * 50)
    print("PROPOSED: Compare A with children of B")
    print("-" * 50)

    if len(root_children) == 2:
        A, B = root_children

        dist_A = tree.nodes[A].get("distribution")
        n_A = tree.nodes[A].get("n_samples", 1)

        dist_B = tree.nodes[B].get("distribution")
        n_B = tree.nodes[B].get("n_samples", 1)

        # Standard sibling test
        stat, df, p_val = sibling_divergence_test(
            dist_A, dist_B, n_left=n_A, n_right=n_B
        )
        print(
            f"\n{A} vs {B}: p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
        )

        # Compare A with children of B
        B_children = list(tree.successors(B))
        if B_children:
            print(f"\nComparing {A} with each child of {B}:")
            for child in B_children:
                dist_child = tree.nodes[child].get("distribution")
                n_child = tree.nodes[child].get("n_samples", 1)
                label = tree.nodes[child].get("label", child)

                if dist_child is not None:
                    stat, df, p_val = sibling_divergence_test(
                        dist_A, dist_child, n_left=n_A, n_right=n_child
                    )
                    result = (
                        "DIFFERENT" if p_val < 0.05 else "SAME (could merge with A!)"
                    )
                    print(
                        f"  {A} vs {child} ({label}): n_child={n_child}, p={p_val:.4f} -> {result}"
                    )

        # Compare B with children of A
        A_children = list(tree.successors(A))
        if A_children:
            print(f"\nComparing {B} with each child of {A}:")
            for child in A_children:
                dist_child = tree.nodes[child].get("distribution")
                n_child = tree.nodes[child].get("n_samples", 1)
                label = tree.nodes[child].get("label", child)

                if dist_child is not None:
                    stat, df, p_val = sibling_divergence_test(
                        dist_B, dist_child, n_left=n_B, n_right=n_child
                    )
                    result = (
                        "DIFFERENT" if p_val < 0.05 else "SAME (could merge with B!)"
                    )
                    print(
                        f"  {B} vs {child} ({label}): n_child={n_child}, p={p_val:.4f} -> {result}"
                    )

    # Run decomposition
    print("\n" + "-" * 50)
    print("Current decomposition result:")
    print("-" * 50)

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    cluster_assignments = decomp.get("cluster_assignments", {})
    print(f"\nFound {len(cluster_assignments)} clusters:")

    pred_labels = {}
    for cid, info in cluster_assignments.items():
        leaves = sorted(info["leaves"], key=lambda x: int(x[1:]))
        print(f"  Cluster {cid}: {leaves}")
        for leaf in info["leaves"]:
            pred_labels[leaf] = cid

    gt = true_labels
    pred = [pred_labels.get(f"s{i}", -1) for i in range(8)]
    ari = adjusted_rand_score(gt, pred)

    print(f"\nARI vs ground truth: {ari:.4f}")

    if ari < 0.9:
        print("⚠️ Low ARI - asymmetric comparison might help!")
    else:
        print("✓ Good clustering")


def summary():
    print("\n" + "=" * 70)
    print("SUMMARY: Comparing A with Children of B")
    print("=" * 70)

    print("""
    Your insight is correct:
    
    When we have:
              P
             / \\
            A   B
               / \\
              C   D
    
    And C belongs with A but D is different:
    
    CURRENT ALGORITHM:
    - Tests A vs B
    - If different: A stays separate from {C, D}
    - Result: {A} and {C, D} → WRONG if C belongs with A!
    
    PROPOSED (A vs children of B):
    - Test A vs C → if SAME, C should join A
    - Test A vs D → if DIFFERENT, D stays separate
    - Result: {A, C} and {D} → CORRECT!
    
    IMPLEMENTATION:
    ---------------
    In _should_split() or the sibling test phase:
    
    1. After testing A vs B and finding they're DIFFERENT:
       - If B has children {C, D}:
         - Test A vs C: if SAME → mark C for merge with A
         - Test A vs D: if SAME → mark D for merge with A
       - If A has children {E, F}:
         - Test B vs E: if SAME → mark E for merge with B
         - Test B vs F: if SAME → mark F for merge with B
    
    2. Build clusters based on these cross-boundary tests
    
    This is more efficient than testing all pairs - just O(k) extra tests
    where k is the number of children, not O(n²) for all leaves.
    """)


if __name__ == "__main__":
    test_with_proper_data()
    test_asymmetric_scenario()
    summary()
