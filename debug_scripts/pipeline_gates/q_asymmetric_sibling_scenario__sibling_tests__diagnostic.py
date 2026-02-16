"""
Purpose: Debug script to analyze asymmetric sibling scenarios.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/pipeline_gates/q_asymmetric_sibling_scenario__sibling_tests__diagnostic.py
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


def create_asymmetric_scenario():
    """Create data where the natural clusters cross sibling boundaries."""
    print("=" * 70)
    print("SCENARIO 1: Cross-Boundary Clusters")
    print("=" * 70)

    print("""
    We create 4 samples where the TRUE clusters are:
    - Cluster A: samples 0, 2 (similar pattern)
    - Cluster B: samples 1, 3 (different pattern)
    
    But hierarchical clustering might group them as:
    - (0, 1) vs (2, 3) based on some distance metric
    
    This creates a cross-boundary situation.
    """)

    # Create data where samples 0,2 are similar and 1,3 are similar
    # But the tree might group (0,1) and (2,3) as siblings
    np.random.seed(42)

    # Pattern A: strong in first 4 features
    pattern_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Pattern B: strong in last 4 features
    pattern_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Samples: 0 and 2 have pattern A, 1 and 3 have pattern B
    # But we'll add noise so that (0,1) might cluster together
    data = np.array(
        [
            pattern_a + np.array([0, 0, 0, 0, 0.2, 0.2, 0, 0]),  # sample 0: mostly A
            pattern_b + np.array([0.2, 0.2, 0, 0, 0, 0, 0, 0]),  # sample 1: mostly B
            pattern_a + np.array([0, 0, 0.2, 0.2, 0, 0, 0, 0]),  # sample 2: mostly A
            pattern_b + np.array([0, 0, 0, 0, 0, 0, 0.2, 0.2]),  # sample 3: mostly B
        ]
    )

    data_df = pd.DataFrame(data, index=["s0", "s1", "s2", "s3"])
    true_labels = [0, 1, 0, 1]  # Ground truth: 0,2 together and 1,3 together

    print("Data:")
    print(data_df)
    print(f"\nGround truth clusters: s0,s2 together; s1,s3 together")

    return data_df, true_labels


def analyze_tree_structure(data_df, true_labels):
    """Analyze how the tree groups the samples."""
    print("\n" + "-" * 50)
    print("Tree Structure Analysis")
    print("-" * 50)

    # Build tree
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    print(f"\nTree nodes: {len(tree.nodes)}")

    # Print tree structure
    print("\nTree edges (parent -> child):")
    for parent, child in tree.edges():
        print(f"  {parent} -> {child}")

    # Find sibling pairs at each internal node
    print("\nSibling pairs at each internal node:")
    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) == 2:
            print(f"  {node}: {children[0]} vs {children[1]}")

    return tree, Z


def test_all_pairwise_comparisons(tree, data_df):
    """Test all pairwise comparisons to see what the sibling test would say."""
    print("\n" + "-" * 50)
    print("Pairwise Sibling Test Results")
    print("-" * 50)

    # Populate distributions
    tree.populate_node_divergences(leaf_data=data_df)

    # Get all leaf nodes
    leaves = [n for n in tree.nodes if tree.nodes[n].get("is_leaf", False)]

    print("\nLeaf distributions:")
    for leaf in leaves:
        dist = tree.nodes[leaf].get("distribution")
        label = tree.nodes[leaf].get("label", leaf)
        print(f"  {label}: {dist[:4]}... (first 4)")

    print("\nPairwise sibling tests (all leaf pairs):")
    print(f"{'Pair':<15} {'Stat':<10} {'df':<5} {'p-value':<12} {'Interpretation'}")
    print("-" * 55)

    for i, leaf_i in enumerate(leaves):
        for leaf_j in leaves[i + 1 :]:
            dist_i = tree.nodes[leaf_i].get("distribution")
            dist_j = tree.nodes[leaf_j].get("distribution")
            label_i = tree.nodes[leaf_i].get("label", leaf_i)
            label_j = tree.nodes[leaf_j].get("label", leaf_j)

            # Run sibling test with n=1 each (leaves are single samples)
            stat, df, p_val = sibling_divergence_test(
                dist_i, dist_j, n_left=1.0, n_right=1.0
            )

            interp = "SAME" if p_val > 0.05 else "DIFFERENT"
            print(
                f"{label_i} vs {label_j:<7} {stat:<10.2f} {df:<5.0f} {p_val:<12.4f} {interp}"
            )


def run_decomposition_and_compare(tree, data_df, true_labels):
    """Run decomposition and compare to ground truth."""
    print("\n" + "-" * 50)
    print("Decomposition Results")
    print("-" * 50)

    # Run decomposition
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    stats_df = tree.stats_df

    # Show sibling test results
    print("\nSibling tests performed during decomposition:")
    if "Sibling_BH_Different" in stats_df.columns:
        tested = stats_df[stats_df["Sibling_Divergence_P_Value"].notna()]
        for node_id, row in tested.iterrows():
            children = list(tree.successors(node_id))
            p_val = row["Sibling_Divergence_P_Value"]
            is_diff = row["Sibling_BH_Different"]
            print(f"  {node_id}: {children}, p={p_val:.4f}, different={is_diff}")

    # Show clusters
    cluster_assignments = decomp.get("cluster_assignments", {})
    print(f"\nFound {len(cluster_assignments)} clusters:")

    pred_labels = {}
    for cid, info in cluster_assignments.items():
        print(f"  Cluster {cid}: {info['leaves']}")
        for leaf in info["leaves"]:
            pred_labels[leaf] = cid

    # Compute ARI
    gt = [true_labels[int(s[1])] for s in ["s0", "s1", "s2", "s3"]]
    pred = [pred_labels.get(s, -1) for s in ["s0", "s1", "s2", "s3"]]

    ari = adjusted_rand_score(gt, pred)
    print(f"\nARI vs ground truth: {ari:.4f}")

    if ari < 1.0:
        print("\n⚠️ MISMATCH: Algorithm didn't find the correct clusters!")
        print("   This demonstrates the cross-boundary issue.")
    else:
        print("\n✓ Algorithm found correct clusters")

    return ari


def propose_solution():
    """Propose a solution for the asymmetric case."""
    print("\n" + "=" * 70)
    print("PROPOSED SOLUTION: Extended Sibling Comparison")
    print("=" * 70)

    print("""
CURRENT ALGORITHM:
------------------
At each internal node, compare the TWO direct children.
Decision: SPLIT or MERGE based on sibling test.

THE PROBLEM:
------------
When we SPLIT at a node, we assume the two subtrees are homogeneous.
But a child of the LEFT subtree might belong with a child of the RIGHT subtree.

PROPOSED EXTENSION:
-------------------
Instead of just comparing A vs B, also compare:
- Children of A vs B (if A has children)
- A vs Children of B (if B has children)
- Children of A vs Children of B (all pairs)

Then decide based on the pattern of significant differences.

ALGORITHM SKETCH:
-----------------
At node P with children A and B:

1. Get all "atomic units" under A and B:
   - If A is a leaf: atomic_A = {A}
   - If A has children C, D: atomic_A = {C, D}
   
2. Test all pairs across the boundary:
   - For each a in atomic_A, b in atomic_B: test a vs b
   
3. Build a similarity graph and find connected components:
   - Nodes that are NOT significantly different are connected
   - Each connected component becomes a cluster

This allows {C, E} and {D, F} to emerge as clusters even if C and D
are under A, and E and F are under B.

SIMPLER ALTERNATIVE:
--------------------
Just extend the post-hoc merge to run more aggressively:
- After initial decomposition, compare ALL pairs of clusters
- Not just cross-boundary pairs
- Merge any that are not significantly different

This is less elegant but easier to implement.
""")


def demonstrate_with_larger_example():
    """Demonstrate the issue with a larger, more realistic example."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Larger Cross-Boundary Example")
    print("=" * 70)

    np.random.seed(123)

    # 8 samples in 2 TRUE clusters, but tree might group differently
    # Cluster 1 (samples 0,1,4,5): high in features 0-3
    # Cluster 2 (samples 2,3,6,7): high in features 4-7

    cluster1_pattern = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2])
    cluster2_pattern = np.array([0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8])

    data = []
    true_labels = []
    for i in range(8):
        if i in [0, 1, 4, 5]:
            base = cluster1_pattern.copy()
            true_labels.append(0)
        else:
            base = cluster2_pattern.copy()
            true_labels.append(1)

        # Add noise
        noise = np.random.normal(0, 0.1, 8)
        sample = np.clip(base + noise, 0, 1)
        data.append(sample)

    data_df = pd.DataFrame(data, index=[f"s{i}" for i in range(8)])

    print("Data shape:", data_df.shape)
    print("Ground truth: Cluster 0 = [s0,s1,s4,s5], Cluster 1 = [s2,s3,s6,s7]")

    # Build tree
    Z = linkage(pdist(data_df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    print("\nTree structure (top levels):")
    root = tree.root()
    children_root = list(tree.successors(root))
    print(f"  ROOT: {root}")
    for child in children_root:
        grandchildren = list(tree.successors(child))
        print(f"    {child}: {grandchildren}")

    # Run decomposition
    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    cluster_assignments = decomp.get("cluster_assignments", {})
    print(f"\nFound {len(cluster_assignments)} clusters:")

    pred_labels = {}
    for cid, info in cluster_assignments.items():
        print(f"  Cluster {cid}: {sorted(info['leaves'])}")
        for leaf in info["leaves"]:
            pred_labels[leaf] = cid

    # Compute ARI
    gt = true_labels
    pred = [pred_labels.get(f"s{i}", -1) for i in range(8)]

    ari = adjusted_rand_score(gt, pred)
    print(f"\nARI vs ground truth: {ari:.4f}")

    if ari < 0.9:
        print("\n⚠️ Cross-boundary issue may be affecting results")
    else:
        print("\n✓ Algorithm handled cross-boundary case well")


if __name__ == "__main__":
    # Scenario 1: Small example
    data_df, true_labels = create_asymmetric_scenario()
    tree, Z = analyze_tree_structure(data_df, true_labels)
    test_all_pairwise_comparisons(tree, data_df)
    ari1 = run_decomposition_and_compare(tree, data_df, true_labels)

    # Scenario 2: Larger example
    demonstrate_with_larger_example()

    # Proposed solution
    propose_solution()

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
