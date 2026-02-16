"""
Purpose: Debug script to test branch length integration in distribution calculation.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_distribution_calculation_test__branch_length__validation.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.distributions import populate_distributions


def create_simple_two_taxa_tree():
    """Create a simple 2-leaf tree with asymmetric branch lengths."""
    # Two leaves that merge at different distances
    # Leaf A is close (bl=0.2), Leaf B is far (bl=0.8)
    X = np.array(
        [
            [0.0, 0.0],  # Leaf A
            [1.0, 1.0],  # Leaf B
        ]
    )
    Z = linkage(X, method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=["A", "B"])
    return tree


def create_four_taxa_tree():
    """Create a 4-leaf tree with varied branch lengths."""
    X = np.array(
        [
            [0.0, 0.0],  # Close pair
            [0.1, 0.1],
            [5.0, 5.0],  # Far pair
            [5.2, 5.2],
        ]
    )
    Z = linkage(X, method="ward")
    tree = PosetTree.from_linkage(Z, leaf_names=["A", "B", "C", "D"])
    return tree


def test_two_taxa_binary():
    """Test with binary categorical values (0 or 1)."""
    print("=" * 60)
    print("TWO TAXA - BINARY CATEGORIES")
    print("=" * 60)

    tree = create_simple_two_taxa_tree()

    # Print branch lengths
    print("\nBranch lengths:")
    for u, v, data in tree.edges(data=True):
        print(f"  {u} -> {v}: {data.get('branch_length', 'N/A'):.4f}")

    # Binary leaf data: A=0, B=1 (one-hot encoded)
    leaf_data = pd.DataFrame(
        {
            "feature": [1.0, 0.0]  # A has value 0 → P(1)=1.0, B has value 1 → P(1)=0.0
            # Wait, let's be explicit: use probabilities directly
        },
        index=["A", "B"],
    )

    # Actually let's use: A is category 0, B is category 1
    # As probabilities: P(category=1) for each leaf
    leaf_data = pd.DataFrame(
        {
            "prob_cat1": [0.0, 1.0]  # A: P(1)=0, B: P(1)=1
        },
        index=["A", "B"],
    )

    print(f"\nLeaf distributions:")
    print(leaf_data)

    # Test current method (leaf_count only)
    tree_current = PosetTree.from_linkage(
        linkage(np.array([[0.0], [1.0]]), method="average"), leaf_names=["A", "B"]
    )
    populate_distributions(tree_current, leaf_data, use_branch_length=False)

    # Test harmonic method
    tree_harmonic = PosetTree.from_linkage(
        linkage(np.array([[0.0], [1.0]]), method="average"), leaf_names=["A", "B"]
    )
    populate_distributions(tree_harmonic, leaf_data, use_branch_length=True)

    print("\n--- Results ---")
    root_current = tree_current.graph.get("root") or next(
        n for n, d in tree_current.in_degree() if d == 0
    )
    root_harmonic = tree_harmonic.graph.get("root") or next(
        n for n, d in tree_harmonic.in_degree() if d == 0
    )

    print(f"\nCurrent method (leaf_count only):")
    print(f"  Parent distribution: {tree_current.nodes[root_current]['distribution']}")

    print(f"\nHarmonic method (1/branch_length):")
    print(
        f"  Parent distribution: {tree_harmonic.nodes[root_harmonic]['distribution']}"
    )

    # Show weights
    print("\n--- Weight Analysis ---")
    for child_id in tree_harmonic.successors(root_harmonic):
        bl = tree_harmonic.edges[root_harmonic, child_id].get("branch_length", 1.0)
        lc = tree_harmonic.nodes[child_id].get("leaf_count", 1)
        print(
            f"  {child_id}: leaf_count={lc}, branch_length={bl:.4f}, conductance={1 / bl:.4f}"
        )


def test_four_taxa_categorical():
    """Test with 4 taxa and multiple categories."""
    print("\n" + "=" * 60)
    print("FOUR TAXA - CATEGORICAL")
    print("=" * 60)

    tree = create_four_taxa_tree()

    # Print tree structure with branch lengths
    print("\nTree structure (branch lengths):")
    for u, v, data in tree.edges(data=True):
        print(f"  {u} -> {v}: bl={data.get('branch_length', 'N/A'):.4f}")

    # Leaf data: categorical distributions
    # 3 categories
    leaf_data = pd.DataFrame(
        {
            "cat1": [0.9, 0.8, 0.1, 0.0],  # A,B have high cat1; C,D have low cat1
            "cat2": [0.05, 0.1, 0.8, 0.85],  # opposite
            "cat3": [0.05, 0.1, 0.1, 0.15],
        },
        index=["A", "B", "C", "D"],
    )

    print(f"\nLeaf distributions:")
    print(leaf_data)

    # Compare methods
    tree_current = create_four_taxa_tree()
    populate_distributions(tree_current, leaf_data, use_branch_length=False)

    tree_harmonic = create_four_taxa_tree()
    populate_distributions(tree_harmonic, leaf_data, use_branch_length=True)

    print("\n--- Internal Node Distributions ---")
    for node_id in tree_current.nodes():
        if not tree_current.nodes[node_id].get("is_leaf", False):
            dist_current = tree_current.nodes[node_id]["distribution"]
            dist_harmonic = tree_harmonic.nodes[node_id]["distribution"]
            print(f"\n{node_id}:")
            print(f"  Current:  {np.round(dist_current, 3)}")
            print(f"  Harmonic: {np.round(dist_harmonic, 3)}")


def test_asymmetric_branch_lengths():
    """Test where branch lengths are very asymmetric."""
    print("\n" + "=" * 60)
    print("ASYMMETRIC BRANCH LENGTHS (MANUAL)")
    print("=" * 60)

    # Manually create a tree with asymmetric branch lengths
    from kl_clustering_analysis.tree.poset_tree import PosetTree

    tree = PosetTree()
    tree.add_node("L0", is_leaf=True, label="A")
    tree.add_node("L1", is_leaf=True, label="B")
    tree.add_node("N2", is_leaf=False)
    tree.add_edge("N2", "L0", branch_length=0.2)  # A is CLOSE
    tree.add_edge("N2", "L1", branch_length=0.8)  # B is FAR
    tree.graph["root"] = "N2"

    print("\nBranch lengths (manually set):")
    for u, v, data in tree.edges(data=True):
        print(f"  {u} -> {v}: bl={data.get('branch_length', 'N/A'):.4f}")

    # A and B have opposite distributions
    leaf_data = pd.DataFrame(
        {
            "feature": [1.0, 0.0]  # A=1, B=0
        },
        index=["A", "B"],
    )

    print(f"\nLeaf distributions: A=1.0, B=0.0")

    # Compare methods
    tree_current = PosetTree()
    tree_current.add_node("L0", is_leaf=True, label="A")
    tree_current.add_node("L1", is_leaf=True, label="B")
    tree_current.add_node("N2", is_leaf=False)
    tree_current.add_edge("N2", "L0", branch_length=0.2)
    tree_current.add_edge("N2", "L1", branch_length=0.8)
    tree_current.graph["root"] = "N2"

    tree_harmonic = PosetTree()
    tree_harmonic.add_node("L0", is_leaf=True, label="A")
    tree_harmonic.add_node("L1", is_leaf=True, label="B")
    tree_harmonic.add_node("N2", is_leaf=False)
    tree_harmonic.add_edge("N2", "L0", branch_length=0.2)
    tree_harmonic.add_edge("N2", "L1", branch_length=0.8)
    tree_harmonic.graph["root"] = "N2"

    populate_distributions(tree_current, leaf_data, use_branch_length=False)
    populate_distributions(tree_harmonic, leaf_data, use_branch_length=True)

    print(f"\nParent distribution:")
    print(f"  Current (leaf_count only): {tree_current.nodes['N2']['distribution']}")
    print(f"  Harmonic (1/branch_len):   {tree_harmonic.nodes['N2']['distribution']}")

    # Calculate expected
    w_A = 1 / 0.2  # 5.0
    w_B = 1 / 0.8  # 1.25
    expected = (w_A * 1.0 + w_B * 0.0) / (w_A + w_B)
    print(f"\n  Expected harmonic: {expected:.4f}")
    print(
        f"  (A gets {w_A / (w_A + w_B) * 100:.1f}% weight, B gets {w_B / (w_A + w_B) * 100:.1f}% weight)"
    )


def test_epsilon_handling():
    """Test that zero/small branch lengths don't cause problems."""
    print("\n" + "=" * 60)
    print("EPSILON HANDLING")
    print("=" * 60)

    # Create identical points (zero distance)
    X = np.array(
        [
            [0.0],
            [0.0],  # Identical to A
            [1.0],
        ]
    )
    Z = linkage(X, method="average")

    tree = PosetTree.from_linkage(Z, leaf_names=["A", "B", "C"])

    print("\nBranch lengths (including zero):")
    for u, v, data in tree.edges(data=True):
        print(f"  {u} -> {v}: bl={data.get('branch_length', 'N/A'):.6f}")

    leaf_data = pd.DataFrame({"feature": [1.0, 0.5, 0.0]}, index=["A", "B", "C"])

    try:
        populate_distributions(tree, leaf_data, use_branch_length=True)
        root = tree.graph.get("root") or next(n for n, d in tree.in_degree() if d == 0)
        print(f"\nSuccessfully computed (no div by zero):")
        print(f"  Root distribution: {tree.nodes[root]['distribution']}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    test_two_taxa_binary()
    test_four_taxa_categorical()
    test_asymmetric_branch_lengths()
    test_epsilon_handling()
