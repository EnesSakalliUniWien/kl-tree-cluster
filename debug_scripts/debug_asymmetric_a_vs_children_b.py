"""Debug: Compare A with children of B (asymmetric sibling comparison).

User's insight: Instead of comparing all atomic units, just compare:
- A with each child of B (if B has children)
- B with each child of A (if A has children)

This catches the asymmetric case more simply.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)


def demonstrate_asymmetric_comparison():
    """Demonstrate comparing A with children of B."""
    print("=" * 70)
    print("Asymmetric Sibling Comparison: A vs Children(B)")
    print("=" * 70)

    print("""
    Tree structure:
              ROOT (N6)
             /        \\
           A (N4)      B (N5)
          /    \\      /    \\
         s0    s2    s1    s3

    Current: Compare N4 vs N5
    Proposed: Also compare N4 vs s1, N4 vs s3, N5 vs s0, N5 vs s2
    
    If N4 is similar to s1 but different from s3:
    → Maybe s1 belongs with {s0, s2}
    """)

    # Create data
    np.random.seed(42)
    pattern_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pattern_b = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    data = np.array(
        [
            pattern_a + np.array([0, 0, 0, 0, 0.2, 0.2, 0, 0]),  # s0: pattern A
            pattern_b + np.array([0.2, 0.2, 0, 0, 0, 0, 0, 0]),  # s1: pattern B
            pattern_a + np.array([0, 0, 0.2, 0.2, 0, 0, 0, 0]),  # s2: pattern A
            pattern_b + np.array([0, 0, 0, 0, 0, 0, 0.2, 0.2]),  # s3: pattern B
        ]
    )

    data_df = pd.DataFrame(data, index=["s0", "s1", "s2", "s3"])

    # Build tree
    Z = linkage(pdist(data_df.values, metric="euclidean"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(leaf_data=data_df)

    print("\nTree structure:")
    for parent, child in tree.edges():
        print(f"  {parent} -> {child}")

    # Get node info
    print("\n" + "-" * 50)
    print("Node distributions:")
    print("-" * 50)
    for node in ["N4", "N5", "L0", "L1", "L2", "L3"]:
        if node in tree.nodes:
            dist = tree.nodes[node].get("distribution")
            n = tree.nodes[node].get("n_samples", 1)
            label = tree.nodes[node].get("label", node)
            print(f"  {node} ({label}): n={n}, dist={dist[:4]}...")

    print("\n" + "-" * 50)
    print("CURRENT: Sibling test at root (N4 vs N5)")
    print("-" * 50)

    dist_n4 = tree.nodes["N4"].get("distribution")
    dist_n5 = tree.nodes["N5"].get("distribution")
    n_n4 = tree.nodes["N4"].get("n_samples", 2)
    n_n5 = tree.nodes["N5"].get("n_samples", 2)

    stat, df, p_val = sibling_divergence_test(
        dist_n4, dist_n5, n_left=n_n4, n_right=n_n5
    )
    print(
        f"  N4 vs N5: stat={stat:.2f}, p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
    )

    print("\n" + "-" * 50)
    print("PROPOSED: Compare A (N4) with children of B (s1, s3)")
    print("-" * 50)

    # N4 vs s1
    dist_s1 = tree.nodes["L1"].get("distribution")
    stat, df, p_val = sibling_divergence_test(dist_n4, dist_s1, n_left=n_n4, n_right=1)
    print(
        f"  N4 vs s1: stat={stat:.2f}, p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
    )

    # N4 vs s3
    dist_s3 = tree.nodes["L3"].get("distribution")
    stat, df, p_val = sibling_divergence_test(dist_n4, dist_s3, n_left=n_n4, n_right=1)
    print(
        f"  N4 vs s3: stat={stat:.2f}, p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
    )

    print("\n" + "-" * 50)
    print("PROPOSED: Compare B (N5) with children of A (s0, s2)")
    print("-" * 50)

    # N5 vs s0
    dist_s0 = tree.nodes["L0"].get("distribution")
    stat, df, p_val = sibling_divergence_test(dist_n5, dist_s0, n_left=n_n5, n_right=1)
    print(
        f"  N5 vs s0: stat={stat:.2f}, p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
    )

    # N5 vs s2
    dist_s2 = tree.nodes["L2"].get("distribution")
    stat, df, p_val = sibling_divergence_test(dist_n5, dist_s2, n_left=n_n5, n_right=1)
    print(
        f"  N5 vs s2: stat={stat:.2f}, p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
    )

    print("\n" + "-" * 50)
    print("ALSO: Direct leaf comparisons")
    print("-" * 50)

    leaves = [("L0", "s0"), ("L1", "s1"), ("L2", "s2"), ("L3", "s3")]
    for i, (node_i, label_i) in enumerate(leaves):
        for node_j, label_j in leaves[i + 1 :]:
            dist_i = tree.nodes[node_i].get("distribution")
            dist_j = tree.nodes[node_j].get("distribution")
            stat, df, p_val = sibling_divergence_test(
                dist_i, dist_j, n_left=1, n_right=1
            )
            print(
                f"  {label_i} vs {label_j}: stat={stat:.2f}, p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
            )


def propose_algorithm_modification():
    """Propose how to modify the algorithm."""
    print("\n" + "=" * 70)
    print("ALGORITHM MODIFICATION PROPOSAL")
    print("=" * 70)

    print("""
    CURRENT ALGORITHM (at internal node P with children A, B):
    ----------------------------------------------------------
    1. Test A vs B
    2. If different → split (process A and B separately)
    3. If same → merge (all leaves under P become one cluster)
    
    PROPOSED MODIFICATION:
    ----------------------
    At internal node P with children A, B:
    
    1. Test A vs B (as before)
    
    2. If SAME → merge all (as before)
    
    3. If DIFFERENT → additional checks:
       a. If B has children {C, D}:
          - Test A vs C, A vs D
          - If A ~ C (same): A should merge with C, not with D
          - If A ~ D (same): A should merge with D, not with C
          
       b. If A has children {E, F}:
          - Test B vs E, B vs F
          - Similarly, check for asymmetric merges
    
    4. Build a similarity graph from all comparisons:
       - Nodes = leaves under P
       - Edges = pairs that are NOT significantly different
       - Find connected components → these are the clusters
    
    SIMPLER VERSION (just one level deeper):
    ----------------------------------------
    At node P with children A, B where both are internal:
    
    1. Get immediate children of A: {C, D}
    2. Get immediate children of B: {E, F}
    3. Test all cross pairs: C vs E, C vs F, D vs E, D vs F
    4. Build similarity graph on {C, D, E, F}
    5. Connected components become clusters
    
    This catches cases where C ~ E and D ~ F even though
    they're on opposite sides of the A-B split.
    """)


def demonstrate_with_clear_asymmetric_case():
    """Create a case where asymmetric comparison clearly helps."""
    print("\n" + "=" * 70)
    print("SCENARIO: Clear Asymmetric Case")
    print("=" * 70)

    print("""
    True clusters: {s0, s1, s2} and {s3}
    
    But tree might be:
              ROOT
             /    \\
           A        B
          / \\      / \\
         s0  s1   s2  s3
    
    Here s2 is under B but belongs with A!
    
    Current algorithm at ROOT:
    - Test A vs B → might say "different" → split
    - Result: {s0, s1} and {s2, s3} → WRONG!
    
    With asymmetric comparison:
    - A vs s2: should be SAME
    - A vs s3: should be DIFFERENT
    - This tells us s2 should join A, not stay with s3
    """)

    np.random.seed(42)

    # s0, s1, s2 are similar (cluster 1)
    # s3 is different (cluster 2)
    pattern_cluster1 = np.array([0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2])
    pattern_cluster2 = np.array([0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8])

    data = np.array(
        [
            pattern_cluster1 + np.random.normal(0, 0.05, 8),  # s0
            pattern_cluster1 + np.random.normal(0, 0.05, 8),  # s1
            pattern_cluster1
            + np.random.normal(0, 0.05, 8),  # s2 - belongs to cluster 1!
            pattern_cluster2 + np.random.normal(0, 0.05, 8),  # s3 - different
        ]
    )
    data = np.clip(data, 0, 1)

    data_df = pd.DataFrame(data, index=["s0", "s1", "s2", "s3"])

    print("\nData:")
    print(data_df.round(2))
    print("\nGround truth: {s0, s1, s2} and {s3}")

    # Force a specific tree structure for demonstration
    # We'll use a custom distance to get the "wrong" tree
    from scipy.cluster.hierarchy import linkage

    # Create custom distances that force (s0,s1) and (s2,s3) grouping
    # even though s2 is similar to s0,s1
    custom_distances = np.array(
        [
            0.1,  # s0-s1 (close)
            0.5,  # s0-s2 (artificially far)
            0.9,  # s0-s3 (far)
            0.5,  # s1-s2 (artificially far)
            0.9,  # s1-s3 (far)
            0.2,  # s2-s3 (artificially close)
        ]
    )

    Z = linkage(custom_distances, method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(leaf_data=data_df)

    print("\nTree structure (forced to be 'wrong'):")
    for parent, child in tree.edges():
        label = tree.nodes[child].get("label", child)
        print(f"  {parent} -> {child} ({label})")

    # Show which nodes are siblings
    print("\nSibling pairs:")
    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) == 2:
            labels = [tree.nodes[c].get("label", c) for c in children]
            print(
                f"  At {node}: {children[0]} ({labels[0]}) vs {children[1]} ({labels[1]})"
            )

    print("\n" + "-" * 50)
    print("Running comparisons:")
    print("-" * 50)

    # Find the root
    root = tree.root()
    root_children = list(tree.successors(root))

    print(f"\nRoot: {root}, children: {root_children}")

    # Compare root's children
    if len(root_children) == 2:
        A, B = root_children
        dist_A = tree.nodes[A].get("distribution")
        dist_B = tree.nodes[B].get("distribution")
        n_A = tree.nodes[A].get("n_samples", 1)
        n_B = tree.nodes[B].get("n_samples", 1)

        stat, df, p_val = sibling_divergence_test(
            dist_A, dist_B, n_left=n_A, n_right=n_B
        )
        print(
            f"\n{A} vs {B}: p={p_val:.4f} -> {'DIFFERENT' if p_val < 0.05 else 'SAME'}"
        )

        # Now compare A with children of B
        B_children = list(tree.successors(B))
        if B_children:
            print(f"\nComparing {A} with children of {B}: {B_children}")
            for child in B_children:
                dist_child = tree.nodes[child].get("distribution")
                label = tree.nodes[child].get("label", child)
                n_child = tree.nodes[child].get("n_samples", 1)
                stat, df, p_val = sibling_divergence_test(
                    dist_A, dist_child, n_left=n_A, n_right=n_child
                )
                result = "DIFFERENT" if p_val < 0.05 else "SAME (should merge with A!)"
                print(f"  {A} vs {child} ({label}): p={p_val:.4f} -> {result}")

        # And B with children of A
        A_children = list(tree.successors(A))
        if A_children:
            print(f"\nComparing {B} with children of {A}: {A_children}")
            for child in A_children:
                dist_child = tree.nodes[child].get("distribution")
                label = tree.nodes[child].get("label", child)
                n_child = tree.nodes[child].get("n_samples", 1)
                stat, df, p_val = sibling_divergence_test(
                    dist_B, dist_child, n_left=n_B, n_right=n_child
                )
                result = "DIFFERENT" if p_val < 0.05 else "SAME"
                print(f"  {B} vs {child} ({label}): p={p_val:.4f} -> {result}")


if __name__ == "__main__":
    demonstrate_asymmetric_comparison()
    demonstrate_with_clear_asymmetric_case()
    propose_algorithm_modification()
