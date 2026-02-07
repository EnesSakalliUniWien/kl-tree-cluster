#!/usr/bin/env python3
"""
Explore tree height values in benchmark data.

This script generates synthetic data and examines:
1. What height values look like at different tree levels
2. How branch length relates to divergence
3. The relationship between merge height and statistical test outcomes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def generate_two_cluster_data(
    n_samples_per_cluster: int = 100,
    n_features: int = 200,
    n_categories: int = 4,
    divergence: float = 0.3,
    seed: int = 42,
):
    """Generate simple two-cluster data using Jukes-Cantor evolution."""
    rng = np.random.RandomState(seed)

    # Ancestral sequence
    ancestor = rng.randint(0, n_categories, size=n_features)

    # Jukes-Cantor transition matrix
    k = n_categories
    p_same = (1.0 / k) + ((k - 1.0) / k) * np.exp(-k * divergence / (k - 1))
    p_diff = (1.0 / k) * (1 - np.exp(-k * divergence / (k - 1)))
    P = np.full((k, k), p_diff)
    np.fill_diagonal(P, p_same)

    def evolve(seq, branch_len):
        P_b = np.full((k, k), (1.0 / k) * (1 - np.exp(-k * branch_len / (k - 1))))
        np.fill_diagonal(
            P_b, (1.0 / k) + ((k - 1.0) / k) * np.exp(-k * branch_len / (k - 1))
        )
        evolved = np.zeros_like(seq)
        for i, state in enumerate(seq):
            evolved[i] = rng.choice(k, p=P_b[state])
        return evolved

    # Create two cluster ancestors
    cluster_a_ancestor = evolve(ancestor, divergence)
    cluster_b_ancestor = evolve(ancestor, divergence)

    # Sample from each cluster
    samples = []
    labels = []

    terminal_branch = 0.05  # Small within-cluster variation

    for i in range(n_samples_per_cluster):
        samples.append(evolve(cluster_a_ancestor, terminal_branch))
        labels.append(0)

    for i in range(n_samples_per_cluster):
        samples.append(evolve(cluster_b_ancestor, terminal_branch))
        labels.append(1)

    X = np.array(samples)
    y = np.array(labels)

    return X, y


def analyze_tree_heights(tree: PosetTree):
    """Extract and analyze height information from tree."""
    heights = []
    depths = []
    leaf_counts = []
    node_ids = []
    is_leaf_list = []

    # Compute depths via BFS from root
    root = tree.root()
    depth_map = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for child in tree.successors(node):
            depth_map[child] = depth_map[node] + 1
            queue.append(child)

    # Compute leaf counts
    leaf_count_map = {}
    for node in tree.nodes:
        if tree.nodes[node].get("is_leaf", False):
            leaf_count_map[node] = 1

    # Bottom-up to compute leaf counts for internal nodes
    for node in reversed(list(nx.topological_sort(tree))):
        if node not in leaf_count_map:
            leaf_count_map[node] = sum(
                leaf_count_map.get(child, 0) for child in tree.successors(node)
            )

    for node in tree.nodes:
        node_data = tree.nodes[node]
        is_leaf = node_data.get("is_leaf", False)
        height = node_data.get("height", 0.0)

        heights.append(height)
        depths.append(depth_map[node])
        leaf_counts.append(leaf_count_map[node])
        node_ids.append(node)
        is_leaf_list.append(is_leaf)

    df = pd.DataFrame(
        {
            "node_id": node_ids,
            "height": heights,
            "depth": depths,
            "leaf_count": leaf_counts,
            "is_leaf": is_leaf_list,
        }
    )

    return df


def compute_branch_lengths(tree: PosetTree, df: pd.DataFrame):
    """Compute branch length = parent_height - child_height."""
    branch_lengths = []
    parent_heights = []

    for node in df["node_id"]:
        parents = list(tree.predecessors(node))
        if parents:
            parent = parents[0]
            parent_height = tree.nodes[parent].get("height", 0.0)
            child_height = tree.nodes[node].get("height", 0.0)
            bl = parent_height - child_height
        else:
            # Root has no parent
            bl = 0.0
            parent_height = tree.nodes[node].get("height", 0.0)

        branch_lengths.append(bl)
        parent_heights.append(parent_height)

    df["branch_length"] = branch_lengths
    df["parent_height"] = parent_heights

    return df


def main():
    import networkx as nx

    print("=" * 70)
    print("TREE HEIGHT EXPLORATION")
    print("=" * 70)

    # Test different divergence levels
    divergences = [0.1, 0.3, 0.5, 1.0, 2.0]

    for div in divergences:
        print(f"\n{'=' * 70}")
        print(f"DIVERGENCE = {div}")
        print("=" * 70)

        # Generate data
        X, y = generate_two_cluster_data(
            n_samples_per_cluster=50,
            n_features=100,
            divergence=div,
            seed=42,
        )

        print(f"Data shape: {X.shape}")
        print(f"True clusters: {np.bincount(y)}")

        # Build tree
        sample_names = [f"S{i}" for i in range(len(X))]
        Z = linkage(pdist(X, metric="hamming"), method="weighted")
        tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

        # Analyze heights
        df = analyze_tree_heights(tree)
        df = compute_branch_lengths(tree, df)

        # Summary statistics
        internal = df[~df["is_leaf"]]

        print(f"\n--- Internal Nodes ({len(internal)}) ---")
        print(
            f"Height range: [{internal['height'].min():.4f}, {internal['height'].max():.4f}]"
        )
        print(f"Height mean: {internal['height'].mean():.4f}")
        print(f"Height std: {internal['height'].std():.4f}")

        print(f"\n--- Branch Lengths ---")
        print(
            f"Range: [{df['branch_length'].min():.4f}, {df['branch_length'].max():.4f}]"
        )
        print(f"Mean: {df['branch_length'].mean():.4f}")
        print(f"Median: {df['branch_length'].median():.4f}")

        # Show top-level nodes (near root)
        print(f"\n--- Top 10 nodes by height ---")
        top_nodes = internal.nlargest(10, "height")[
            ["node_id", "height", "depth", "leaf_count", "branch_length"]
        ]
        print(top_nodes.to_string(index=False))

        # Root info
        root = tree.root()
        root_height = tree.nodes[root].get("height", 0.0)
        print(f"\n--- Root Node ---")
        print(f"Root: {root}, Height: {root_height:.4f}")

        # Children of root
        root_children = list(tree.successors(root))
        print(f"Root children: {root_children}")
        for child in root_children:
            child_height = tree.nodes[child].get("height", 0.0)
            child_leaves = df[df["node_id"] == child]["leaf_count"].values[0]
            bl = root_height - child_height
            print(
                f"  {child}: height={child_height:.4f}, leaves={child_leaves}, branch_length={bl:.4f}"
            )

    print("\n" + "=" * 70)
    print("ANALYSIS: Height vs True Cluster Separation")
    print("=" * 70)

    # Check if root's children separate the true clusters
    X, y = generate_two_cluster_data(
        n_samples_per_cluster=50,
        n_features=100,
        divergence=0.5,
        seed=42,
    )

    sample_names = [f"S{i}" for i in range(len(X))]
    Z = linkage(pdist(X, metric="hamming"), method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    root = tree.root()
    root_children = list(tree.successors(root))

    print(f"\nRoot children: {root_children}")

    for child in root_children:
        leaves = tree.get_leaves(child, return_labels=True)
        leaf_indices = [int(s[1:]) for s in leaves]  # Extract index from "S{i}"
        labels_in_child = y[leaf_indices]

        print(f"\n{child}:")
        print(f"  Leaves: {len(leaves)}")
        print(f"  Label distribution: {np.bincount(labels_in_child, minlength=2)}")
        purity = max(np.bincount(labels_in_child, minlength=2)) / len(labels_in_child)
        print(f"  Purity: {purity:.2%}")


if __name__ == "__main__":
    import networkx as nx

    main()
