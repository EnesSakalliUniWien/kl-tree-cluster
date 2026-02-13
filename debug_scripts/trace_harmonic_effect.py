#!/usr/bin/env python3
"""
Trace why harmonic weighting decreases boundary detection AUC.

Let's dig into a specific example to understand the mechanism.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics import (
    compute_node_divergences,
)
from benchmarks.shared.generators.generate_phylogenetic import (
    generate_phylogenetic_data,
)


def generate_simple_2_cluster_data(seed=42):
    """Generate a simple 2-cluster dataset."""
    sample_dict, cluster_assignments, distributions, metadata = (
        generate_phylogenetic_data(
            n_taxa=2,
            n_features=20,
            n_categories=4,
            samples_per_taxon=10,
            mutation_rate=0.1,
            random_seed=seed,
        )
    )
    sample_names = list(sample_dict.keys())
    data = np.array([sample_dict[s] for s in sample_names])
    labels = np.array([cluster_assignments[s] for s in sample_names])
    return data, labels, sample_names


def data_to_probability_df(data, sample_names):
    K = int(data.max()) + 1
    n_samples, n_features = data.shape
    prob_data = np.zeros((n_samples, n_features * K))
    for i in range(n_samples):
        for j in range(n_features):
            prob_data[i, j * K + data[i, j]] = 1.0
    return pd.DataFrame(prob_data, index=sample_names)


def get_leaves_under(tree, node_id):
    """Get all leaf labels under a node."""
    if tree.nodes[node_id].get("is_leaf", False):
        return [tree.nodes[node_id].get("label", node_id)]
    leaves = []
    for child in tree.successors(node_id):
        leaves.extend(get_leaves_under(tree, child))
    return leaves


def trace_single_example():
    """Trace through a single example in detail."""
    print("=" * 80)
    print("TRACING HARMONIC WEIGHTING EFFECT")
    print("=" * 80)

    # Generate data
    data, labels, sample_names = generate_simple_2_cluster_data(seed=42)
    prob_df = data_to_probability_df(data, sample_names)
    name_to_label = dict(zip(sample_names, labels))

    print(f"\nData: {len(sample_names)} samples, 2 clusters")
    print(f"Cluster 0: {sum(labels == 0)} samples")
    print(f"Cluster 1: {sum(labels == 1)} samples")

    # Build tree
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="ward")

    # Create trees for both methods
    tree_curr = PosetTree.from_linkage(Z, leaf_names=sample_names)
    tree_harm = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Get root
    root = tree_curr.graph.get("root") or next(
        n for n, d in tree_curr.in_degree() if d == 0
    )

    # Find the TRUE BOUNDARY node - the node where children split cluster 0 from cluster 1
    print("\n" + "-" * 40)
    print("FINDING TRUE BOUNDARY NODES")
    print("-" * 40)

    for node_id in tree_curr.nodes():
        if tree_curr.nodes[node_id].get("is_leaf", False):
            continue

        children = list(tree_curr.successors(node_id))
        if len(children) != 2:
            continue

        left_leaves = get_leaves_under(tree_curr, children[0])
        right_leaves = get_leaves_under(tree_curr, children[1])

        left_labels = set(name_to_label.get(l, -1) for l in left_leaves)
        right_labels = set(name_to_label.get(l, -1) for l in right_leaves)

        is_pure_boundary = (
            len(left_labels) == 1
            and len(right_labels) == 1
            and left_labels != right_labels
        )

        if is_pure_boundary:
            print(f"\n*** TRUE BOUNDARY: {node_id} ***")
            print(
                f"  Left child {children[0]}: {len(left_leaves)} leaves, labels={left_labels}"
            )
            print(
                f"  Right child {children[1]}: {len(right_leaves)} leaves, labels={right_labels}"
            )

            # Get branch lengths
            bl_left = tree_curr.edges[node_id, children[0]].get("branch_length", 1.0)
            bl_right = tree_curr.edges[node_id, children[1]].get("branch_length", 1.0)
            print(f"  Branch lengths: left={bl_left:.4f}, right={bl_right:.4f}")
            print(f"  Ratio: {bl_left / bl_right:.2f}")

    # Now compute distributions and compare
    print("\n" + "-" * 40)
    print("COMPARING DISTRIBUTIONS AT KEY NODES")
    print("-" * 40)

    populate_distributions(tree_curr, prob_df, use_branch_length=False)
    populate_distributions(tree_harm, prob_df, use_branch_length=True)

    # Compare at a few key internal nodes
    internal_nodes = [
        n for n in tree_curr.nodes() if not tree_curr.nodes[n].get("is_leaf", False)
    ]

    print(f"\nNumber of internal nodes: {len(internal_nodes)}")

    # Focus on nodes with interesting structure
    for node_id in sorted(internal_nodes)[:5]:  # First 5 internal nodes
        children = list(tree_curr.successors(node_id))
        if len(children) != 2:
            continue

        bl_left = tree_curr.edges[node_id, children[0]].get("branch_length", 1.0)
        bl_right = tree_curr.edges[node_id, children[1]].get("branch_length", 1.0)

        # Get leaves to determine if this is a boundary
        left_leaves = get_leaves_under(tree_curr, children[0])
        right_leaves = get_leaves_under(tree_curr, children[1])
        left_labels = set(name_to_label.get(l, -1) for l in left_leaves)
        right_labels = set(name_to_label.get(l, -1) for l in right_leaves)

        is_boundary = len(left_labels & right_labels) == 0

        dist_curr = tree_curr.nodes[node_id]["distribution"]
        dist_harm = tree_harm.nodes[node_id]["distribution"]

        # How different are the distributions?
        diff = np.abs(dist_harm - dist_curr).sum()

        print(f"\n{node_id} (boundary={is_boundary}):")
        print(
            f"  Branch lengths: L={bl_left:.4f}, R={bl_right:.4f}, ratio={bl_left / bl_right:.2f}"
        )
        print(f"  Left: {len(left_leaves)} leaves, labels={left_labels}")
        print(f"  Right: {len(right_leaves)} leaves, labels={right_labels}")
        print(f"  Distribution diff (L1): {diff:.4f}")


def trace_kl_divergence_changes():
    """Trace how KL divergences change with harmonic weighting."""
    print("\n" + "=" * 80)
    print("TRACING KL DIVERGENCE CHANGES")
    print("=" * 80)

    # Generate data
    data, labels, sample_names = generate_simple_2_cluster_data(seed=42)
    prob_df = data_to_probability_df(data, sample_names)
    name_to_label = dict(zip(sample_names, labels))

    # Build tree
    distances = pdist(data, metric="hamming")
    Z = linkage(distances, method="ward")

    # Compute with both methods
    tree_curr = PosetTree.from_linkage(Z, leaf_names=sample_names)
    tree_harm = PosetTree.from_linkage(Z, leaf_names=sample_names)

    stats_curr = compute_node_divergences(tree_curr, prob_df, use_branch_length=False)
    stats_harm = compute_node_divergences(tree_harm, prob_df, use_branch_length=True)

    # Compare internal nodes
    internal_curr = stats_curr[~stats_curr["is_leaf"]].copy()
    internal_harm = stats_harm[~stats_harm["is_leaf"]].copy()

    # Add boundary information
    def is_true_boundary(tree, node_id):
        children = list(tree.successors(node_id))
        if len(children) != 2:
            return False
        left_leaves = get_leaves_under(tree, children[0])
        right_leaves = get_leaves_under(tree, children[1])
        left_labels = set(name_to_label.get(l, -1) for l in left_leaves)
        right_labels = set(name_to_label.get(l, -1) for l in right_leaves)
        return (
            len(left_labels & right_labels) == 0
            and len(left_labels) == 1
            and len(right_labels) == 1
        )

    internal_curr["is_boundary"] = [
        is_true_boundary(tree_curr, n) for n in internal_curr.index
    ]
    internal_harm["is_boundary"] = [
        is_true_boundary(tree_harm, n) for n in internal_harm.index
    ]

    print("\n--- BOUNDARY NODES ---")
    boundary_nodes = internal_curr[internal_curr["is_boundary"]].index
    for node_id in boundary_nodes:
        kl_curr = internal_curr.loc[node_id, "kl_divergence_local"]
        kl_harm = internal_harm.loc[node_id, "kl_divergence_local"]
        print(
            f"{node_id}: current={kl_curr:.4f}, harmonic={kl_harm:.4f}, diff={kl_harm - kl_curr:+.4f}"
        )

    print("\n--- NON-BOUNDARY NODES ---")
    non_boundary_nodes = internal_curr[~internal_curr["is_boundary"]].index
    for node_id in list(non_boundary_nodes)[:5]:  # First 5
        kl_curr = internal_curr.loc[node_id, "kl_divergence_local"]
        kl_harm = internal_harm.loc[node_id, "kl_divergence_local"]
        print(
            f"{node_id}: current={kl_curr:.4f}, harmonic={kl_harm:.4f}, diff={kl_harm - kl_curr:+.4f}"
        )

    # Key question: Does harmonic increase KL MORE at boundaries or non-boundaries?
    print("\n--- SUMMARY ---")

    boundary_kl_curr = internal_curr[internal_curr["is_boundary"]][
        "kl_divergence_local"
    ].mean()
    boundary_kl_harm = internal_harm[internal_harm["is_boundary"]][
        "kl_divergence_local"
    ].mean()

    non_boundary_kl_curr = internal_curr[~internal_curr["is_boundary"]][
        "kl_divergence_local"
    ].mean()
    non_boundary_kl_harm = internal_harm[~internal_harm["is_boundary"]][
        "kl_divergence_local"
    ].mean()

    print(f"Boundary nodes:")
    print(f"  Current: {boundary_kl_curr:.4f}")
    print(f"  Harmonic: {boundary_kl_harm:.4f}")
    print(
        f"  Change: {boundary_kl_harm - boundary_kl_curr:+.4f} ({(boundary_kl_harm / boundary_kl_curr - 1) * 100:+.1f}%)"
    )

    print(f"\nNon-boundary nodes:")
    print(f"  Current: {non_boundary_kl_curr:.4f}")
    print(f"  Harmonic: {non_boundary_kl_harm:.4f}")
    print(
        f"  Change: {non_boundary_kl_harm - non_boundary_kl_curr:+.4f} ({(non_boundary_kl_harm / non_boundary_kl_curr - 1) * 100:+.1f}%)"
    )

    # The ratio of boundary to non-boundary KL is what matters for detection
    ratio_curr = boundary_kl_curr / non_boundary_kl_curr
    ratio_harm = boundary_kl_harm / non_boundary_kl_harm

    print(f"\nBoundary/Non-boundary KL ratio:")
    print(f"  Current: {ratio_curr:.4f}")
    print(f"  Harmonic: {ratio_harm:.4f}")
    print(f"  Change: {ratio_harm - ratio_curr:+.4f}")

    if ratio_harm < ratio_curr:
        print("\n>>> HARMONIC REDUCES THE SIGNAL-TO-NOISE RATIO!")
        print(">>> This is why boundary detection gets worse.")


def trace_branch_length_correlation():
    """Check if branch length correlates with cluster boundaries."""
    print("\n" + "=" * 80)
    print("BRANCH LENGTH vs CLUSTER BOUNDARY CORRELATION")
    print("=" * 80)

    # Run multiple seeds
    boundary_bls = []
    non_boundary_bls = []

    for seed in range(10):
        data, labels, sample_names = generate_simple_2_cluster_data(seed=seed)
        name_to_label = dict(zip(sample_names, labels))

        distances = pdist(data, metric="hamming")
        Z = linkage(distances, method="ward")
        tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

        for node_id in tree.nodes():
            if tree.nodes[node_id].get("is_leaf", False):
                continue

            children = list(tree.successors(node_id))
            if len(children) != 2:
                continue

            bl_sum = sum(
                tree.edges[node_id, c].get("branch_length", 1.0) for c in children
            )

            left_leaves = get_leaves_under(tree, children[0])
            right_leaves = get_leaves_under(tree, children[1])
            left_labels = set(name_to_label.get(l, -1) for l in left_leaves)
            right_labels = set(name_to_label.get(l, -1) for l in right_leaves)

            is_boundary = (
                len(left_labels & right_labels) == 0
                and len(left_labels) == 1
                and len(right_labels) == 1
            )

            if is_boundary:
                boundary_bls.append(bl_sum)
            else:
                non_boundary_bls.append(bl_sum)

    print(f"\nBranch length sum at BOUNDARY nodes:")
    print(f"  Mean: {np.mean(boundary_bls):.4f}")
    print(f"  Std: {np.std(boundary_bls):.4f}")
    print(f"  N: {len(boundary_bls)}")

    print(f"\nBranch length sum at NON-BOUNDARY nodes:")
    print(f"  Mean: {np.mean(non_boundary_bls):.4f}")
    print(f"  Std: {np.std(non_boundary_bls):.4f}")
    print(f"  N: {len(non_boundary_bls)}")

    # t-test
    from scipy import stats

    t_stat, p_val = stats.ttest_ind(boundary_bls, non_boundary_bls)
    print(f"\nt-test: t={t_stat:.4f}, p={p_val:.4f}")

    if p_val > 0.05:
        print(
            "\n>>> Branch lengths do NOT significantly differ between boundary and non-boundary!"
        )
        print(">>> This means harmonic weighting adds NOISE, not signal.")


if __name__ == "__main__":
    trace_single_example()
    trace_kl_divergence_changes()
    trace_branch_length_correlation()
