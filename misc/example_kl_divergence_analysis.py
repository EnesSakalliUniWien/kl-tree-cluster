"""
Example: Calculate KL divergence for a PosetTree from binary data.

This script demonstrates how to:
1. Create a PosetTree from a linkage matrix
2. Calculate distributions and KL divergence for all nodes
3. Analyze which internal nodes are most different from the overall mean
"""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence


def main():
    print("=" * 80)
    print("Example: PosetTree KL Divergence Analysis")
    print("=" * 80)

    # Create some example binary data with clear clusters
    # Cluster 1: has features 0-1
    # Cluster 2: has features 2-3
    data = {
        "Sample_A": np.array([1, 1, 0, 0]),
        "Sample_B": np.array([1, 1, 0, 0]),
        "Sample_C": np.array([0, 0, 1, 1]),
        "Sample_D": np.array([0, 0, 1, 1]),
    }

    print("\nInput Data:")
    for name, features in data.items():
        print(f"  {name}: {features}")

    # Create linkage matrix from the data
    leaf_names = list(data.keys())
    X = np.array([data[name] for name in leaf_names])
    distances = pdist(X, metric="euclidean")
    Z = linkage(distances, method="complete")

    print("\nLinkage Matrix:")
    print("  [left_cluster, right_cluster, distance, n_samples]")
    for i, row in enumerate(Z):
        print(f"  Merge {i}: {row}")

    # Create PosetTree from linkage matrix
    tree = PosetTree.from_linkage(Z, leaf_names=leaf_names)

    print(f"\nPosetTree Structure:")
    print(f"  Nodes: {tree.number_of_nodes()}")
    print(f"  Edges: {tree.number_of_edges()}")

    # Calculate distributions and KL divergence
    stats = calculate_hierarchy_kl_divergence(tree, data)

    print("\n" + "=" * 80)
    print("KL Divergence Analysis Results")
    print("=" * 80)

    # Sort internal nodes by KL divergence
    internal_stats = {
        node_id: node_stats
        for node_id, node_stats in stats.items()
        if not node_stats["is_leaf"]
    }

    sorted_nodes = sorted(
        internal_stats.items(), key=lambda x: x[1]["kl_divergence"], reverse=True
    )

    print("\nInternal Nodes (sorted by KL divergence):")
    print(f"{'Node ID':<10} {'KL Div':<12} {'Leaf Count':<12} {'Distribution'}")
    print("-" * 70)

    for node_id, node_stats in sorted_nodes:
        kl_div = node_stats["kl_divergence"]
        leaf_count = node_stats["leaf_count"]
        dist = node_stats["distribution"]
        print(f"{node_id:<10} {kl_div:<12.6f} {leaf_count:<12} {dist}")

    print("\nLeaf Nodes:")
    print(f"{'Node ID':<10} {'Label':<12} {'Distribution'}")
    print("-" * 70)

    for node_id, node_stats in stats.items():
        if node_stats["is_leaf"]:
            label = tree.nodes[node_id].get("label", node_id)
            dist = node_stats["distribution"]
            print(f"{node_id:<10} {label:<12} {dist}")

    # Find the root node and show its distribution (= overall mean)
    roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
    root = roots[0]
    overall_mean = stats[root]["distribution"]

    print("\n" + "=" * 80)
    print("Overall Mean Distribution (Root):")
    print("=" * 80)
    print(f"  {overall_mean}")
    print(f"  Root node: {root}")

    # Show per-column KL divergence for the most divergent node
    if sorted_nodes:
        most_divergent_id, most_divergent_stats = sorted_nodes[0]
        print("\n" + "=" * 80)
        print(f"Most Divergent Node: {most_divergent_id}")
        print("=" * 80)
        print(f"  Total KL Divergence: {most_divergent_stats['kl_divergence']:.6f}")
        print(f"  Per-column KL Divergence:")
        for i, kl_val in enumerate(most_divergent_stats["kl_divergence_per_column"]):
            print(f"    Feature {i}: {kl_val:.6f}")
        print(f"  Node Distribution: {most_divergent_stats['distribution']}")
        print(f"  Overall Mean:      {overall_mean}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    return tree, stats


if __name__ == "__main__":
    tree, stats = main()
    print("\nObjects available for inspection:")
    print("  - tree: PosetTree with KL divergence attributes")
    print("  - stats: Dictionary of node statistics")
