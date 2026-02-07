"""
Explore the relationship between branch lengths and distribution weighting.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.distributions import populate_distributions


def create_two_taxa_tree(bl_A, bl_B):
    """Create a 2-leaf tree with specified branch lengths."""
    tree = PosetTree()
    tree.add_node("L0", is_leaf=True, label="A")
    tree.add_node("L1", is_leaf=True, label="B")
    tree.add_node("P", is_leaf=False)
    tree.add_edge("P", "L0", branch_length=bl_A)
    tree.add_edge("P", "L1", branch_length=bl_B)
    tree.graph["root"] = "P"
    return tree


def test_branch_length_ratios():
    """How does the ratio of branch lengths affect weighting?"""
    print("=" * 60)
    print("EFFECT OF BRANCH LENGTH RATIO")
    print("=" * 60)
    print("\nLeaf A = 1.0, Leaf B = 0.0")
    print("Parent distribution = weighted average")
    print()

    # A=1, B=0
    leaf_data = pd.DataFrame({"f": [1.0, 0.0]}, index=["A", "B"])

    ratios = [1, 2, 4, 10, 100]

    print(
        f"{'bl_A':>8} {'bl_B':>8} {'ratio':>8} | {'Current':>10} {'Harmonic':>10} {'A_weight%':>10}"
    )
    print("-" * 70)

    for ratio in ratios:
        bl_A, bl_B = 1.0, float(ratio)

        tree_curr = create_two_taxa_tree(bl_A, bl_B)
        tree_harm = create_two_taxa_tree(bl_A, bl_B)

        populate_distributions(tree_curr, leaf_data, use_branch_length=False)
        populate_distributions(tree_harm, leaf_data, use_branch_length=True)

        curr_val = tree_curr.nodes["P"]["distribution"][0]
        harm_val = tree_harm.nodes["P"]["distribution"][0]

        # Calculate A's weight in harmonic
        w_A = 1 / bl_A
        w_B = 1 / bl_B
        a_weight = w_A / (w_A + w_B) * 100

        print(
            f"{bl_A:>8.1f} {bl_B:>8.1f} {ratio:>8} | {curr_val:>10.3f} {harm_val:>10.3f} {a_weight:>9.1f}%"
        )


def test_symmetric_vs_asymmetric():
    """Compare symmetric and asymmetric branch lengths."""
    print("\n" + "=" * 60)
    print("SYMMETRIC vs ASYMMETRIC")
    print("=" * 60)
    print("\nLeaf A = 1.0, Leaf B = 0.0")
    print()

    leaf_data = pd.DataFrame({"f": [1.0, 0.0]}, index=["A", "B"])

    cases = [
        ("Symmetric (1:1)", 1.0, 1.0),
        ("A closer (1:2)", 1.0, 2.0),
        ("A closer (1:4)", 1.0, 4.0),
        ("B closer (2:1)", 2.0, 1.0),
        ("B closer (4:1)", 4.0, 1.0),
        ("Very asymm (1:10)", 1.0, 10.0),
        ("Very asymm (10:1)", 10.0, 1.0),
    ]

    print(f"{'Case':<20} {'bl_A':>6} {'bl_B':>6} | {'Current':>8} {'Harmonic':>8}")
    print("-" * 60)

    for name, bl_A, bl_B in cases:
        tree_curr = create_two_taxa_tree(bl_A, bl_B)
        tree_harm = create_two_taxa_tree(bl_A, bl_B)

        populate_distributions(tree_curr, leaf_data, use_branch_length=False)
        populate_distributions(tree_harm, leaf_data, use_branch_length=True)

        curr_val = tree_curr.nodes["P"]["distribution"][0]
        harm_val = tree_harm.nodes["P"]["distribution"][0]

        print(
            f"{name:<20} {bl_A:>6.1f} {bl_B:>6.1f} | {curr_val:>8.3f} {harm_val:>8.3f}"
        )


def test_with_different_leaf_values():
    """Test with various leaf distribution values."""
    print("\n" + "=" * 60)
    print("DIFFERENT LEAF VALUES (bl_A=1, bl_B=4)")
    print("=" * 60)
    print("\nA is 4x closer than B")
    print()

    bl_A, bl_B = 1.0, 4.0

    cases = [
        (0.0, 1.0),
        (0.2, 0.8),
        (0.5, 0.5),
        (0.8, 0.2),
        (1.0, 0.0),
    ]

    print(f"{'A_val':>6} {'B_val':>6} | {'Current':>8} {'Harmonic':>8} {'Diff':>8}")
    print("-" * 50)

    for a_val, b_val in cases:
        leaf_data = pd.DataFrame({"f": [a_val, b_val]}, index=["A", "B"])

        tree_curr = create_two_taxa_tree(bl_A, bl_B)
        tree_harm = create_two_taxa_tree(bl_A, bl_B)

        populate_distributions(tree_curr, leaf_data, use_branch_length=False)
        populate_distributions(tree_harm, leaf_data, use_branch_length=True)

        curr_val = tree_curr.nodes["P"]["distribution"][0]
        harm_val = tree_harm.nodes["P"]["distribution"][0]
        diff = harm_val - curr_val

        print(
            f"{a_val:>6.1f} {b_val:>6.1f} | {curr_val:>8.3f} {harm_val:>8.3f} {diff:>+8.3f}"
        )


def test_three_taxa():
    """Test with 3 children (non-binary)."""
    print("\n" + "=" * 60)
    print("THREE TAXA (non-binary tree)")
    print("=" * 60)

    tree = PosetTree()
    tree.add_node("L0", is_leaf=True, label="A")
    tree.add_node("L1", is_leaf=True, label="B")
    tree.add_node("L2", is_leaf=True, label="C")
    tree.add_node("P", is_leaf=False)
    tree.add_edge("P", "L0", branch_length=1.0)  # A: close
    tree.add_edge("P", "L1", branch_length=2.0)  # B: medium
    tree.add_edge("P", "L2", branch_length=4.0)  # C: far
    tree.graph["root"] = "P"

    print("\nBranch lengths: A=1.0, B=2.0, C=4.0")
    print("Leaf values: A=1.0, B=0.5, C=0.0")

    leaf_data = pd.DataFrame({"f": [1.0, 0.5, 0.0]}, index=["A", "B", "C"])

    tree_curr = PosetTree()
    tree_curr.add_node("L0", is_leaf=True, label="A")
    tree_curr.add_node("L1", is_leaf=True, label="B")
    tree_curr.add_node("L2", is_leaf=True, label="C")
    tree_curr.add_node("P", is_leaf=False)
    tree_curr.add_edge("P", "L0", branch_length=1.0)
    tree_curr.add_edge("P", "L1", branch_length=2.0)
    tree_curr.add_edge("P", "L2", branch_length=4.0)
    tree_curr.graph["root"] = "P"

    tree_harm = PosetTree()
    tree_harm.add_node("L0", is_leaf=True, label="A")
    tree_harm.add_node("L1", is_leaf=True, label="B")
    tree_harm.add_node("L2", is_leaf=True, label="C")
    tree_harm.add_node("P", is_leaf=False)
    tree_harm.add_edge("P", "L0", branch_length=1.0)
    tree_harm.add_edge("P", "L1", branch_length=2.0)
    tree_harm.add_edge("P", "L2", branch_length=4.0)
    tree_harm.graph["root"] = "P"

    populate_distributions(tree_curr, leaf_data, use_branch_length=False)
    populate_distributions(tree_harm, leaf_data, use_branch_length=True)

    print(f"\nCurrent: {tree_curr.nodes['P']['distribution'][0]:.3f}")
    print(f"Harmonic: {tree_harm.nodes['P']['distribution'][0]:.3f}")

    # Manual calculation
    w_A, w_B, w_C = 1 / 1.0, 1 / 2.0, 1 / 4.0
    total_w = w_A + w_B + w_C
    expected = (w_A * 1.0 + w_B * 0.5 + w_C * 0.0) / total_w
    print(f"\nExpected harmonic: {expected:.3f}")
    print(
        f"Weights: A={w_A / total_w * 100:.1f}%, B={w_B / total_w * 100:.1f}%, C={w_C / total_w * 100:.1f}%"
    )


def test_formula():
    """Show the formula clearly."""
    print("\n" + "=" * 60)
    print("THE HARMONIC FORMULA")
    print("=" * 60)

    print("""
    Current method:
        weight_child = leaf_count
        P_parent = Σ(weight × P_child) / Σ(weight)
    
    Harmonic method:
        conductance = 1 / branch_length
        weight_child = leaf_count × conductance
        P_parent = Σ(weight × P_child) / Σ(weight)
    
    For binary tree (2 children with leaf_count=1 each):
        w_A = 1 / bl_A
        w_B = 1 / bl_B
        
        P_parent = (w_A × P_A + w_B × P_B) / (w_A + w_B)
                 = (P_A/bl_A + P_B/bl_B) / (1/bl_A + 1/bl_B)
    
    Key insight:
        - Shorter branch → higher conductance → more influence
        - The ratio bl_B/bl_A determines relative weighting
    """)


if __name__ == "__main__":
    test_formula()
    test_branch_length_ratios()
    test_symmetric_vs_asymmetric()
    test_with_different_leaf_values()
    test_three_taxa()
