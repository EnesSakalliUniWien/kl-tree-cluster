"""
Purpose: Explore relationship between branch lengths and statistical test results.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_vs_statistical_tests__branch_length__analysis.py
"""

#!/usr/bin/env python3
"""
Explore relationship between branch lengths and statistical test results.

This script:
1. Runs the full clustering pipeline on benchmark data
2. Extracts branch lengths and test statistics at each node
3. Analyzes the correlation between branch length and test outcomes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import networkx as nx

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree


def generate_two_cluster_data(
    n_samples_per_cluster: int = 100,
    n_features: int = 200,
    n_categories: int = 4,
    divergence: float = 0.3,
    seed: int = 42,
):
    """Generate simple two-cluster data using Jukes-Cantor evolution."""
    rng = np.random.RandomState(seed)

    ancestor = rng.randint(0, n_categories, size=n_features)
    k = n_categories

    def evolve(seq, branch_len):
        P_b = np.full((k, k), (1.0 / k) * (1 - np.exp(-k * branch_len / (k - 1))))
        np.fill_diagonal(
            P_b, (1.0 / k) + ((k - 1.0) / k) * np.exp(-k * branch_len / (k - 1))
        )
        evolved = np.zeros_like(seq)
        for i, state in enumerate(seq):
            evolved[i] = rng.choice(k, p=P_b[state])
        return evolved

    cluster_a_ancestor = evolve(ancestor, divergence)
    cluster_b_ancestor = evolve(ancestor, divergence)

    samples = []
    labels = []
    terminal_branch = 0.05

    for i in range(n_samples_per_cluster):
        samples.append(evolve(cluster_a_ancestor, terminal_branch))
        labels.append(0)

    for i in range(n_samples_per_cluster):
        samples.append(evolve(cluster_b_ancestor, terminal_branch))
        labels.append(1)

    X = np.array(samples)
    y = np.array(labels)
    sample_names = [f"S{i}" for i in range(len(X))]

    return X, y, sample_names


def compute_branch_lengths(tree: PosetTree) -> dict:
    """Compute branch length for each node (parent_height - node_height)."""
    branch_lengths = {}

    for node in tree.nodes:
        parents = list(tree.predecessors(node))
        if parents:
            parent = parents[0]
            parent_height = tree.nodes[parent].get("height", 0.0)
            node_height = tree.nodes[node].get("height", 0.0)
            branch_lengths[node] = parent_height - node_height
        else:
            branch_lengths[node] = 0.0  # Root

    return branch_lengths


def run_full_pipeline(X, sample_names, n_features):
    """Run the full annotation pipeline and return results."""
    # Build tree
    Z = linkage(pdist(X, metric="hamming"), method="weighted")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Build data DataFrame
    feature_names = [f"F{j}" for j in range(n_features)]
    data_df = pd.DataFrame(X, index=sample_names, columns=feature_names)

    # Use PosetTree's decompose method which handles all annotations
    results = tree.decompose(leaf_data=data_df)

    # Get the stats DataFrame created during decomposition
    results_df = tree.stats_df.copy() if hasattr(tree, "stats_df") else pd.DataFrame()

    # Compute branch lengths
    branch_lengths = compute_branch_lengths(tree)
    if not results_df.empty:
        results_df["branch_length"] = results_df.index.map(branch_lengths)

        # Add node metadata
        results_df["is_leaf"] = results_df.index.map(
            lambda n: tree.nodes[n].get("is_leaf", False)
        )
        results_df["height"] = results_df.index.map(
            lambda n: tree.nodes[n].get("height", 0.0)
        )
        results_df["leaf_count"] = results_df.index.map(
            lambda n: tree.nodes[n].get("leaf_count", 1)
        )

    return tree, results_df, results


def analyze_branch_length_vs_tests(results_df: pd.DataFrame):
    """Analyze relationship between branch length and test outcomes."""

    # Focus on internal nodes only
    internal = results_df[~results_df["is_leaf"]].copy()

    print("\n" + "=" * 70)
    print("BRANCH LENGTH vs CHILD-PARENT DIVERGENCE")
    print("=" * 70)

    if "Child_Parent_Divergence_Significant" in internal.columns:
        sig = internal[internal["Child_Parent_Divergence_Significant"] == True]
        non_sig = internal[internal["Child_Parent_Divergence_Significant"] == False]

        print(f"\nSignificant child-parent divergence: {len(sig)} nodes")
        if len(sig) > 0:
            print(
                f"  Branch length range: [{sig['branch_length'].min():.4f}, {sig['branch_length'].max():.4f}]"
            )
            print(f"  Branch length mean: {sig['branch_length'].mean():.4f}")

        print(f"\nNon-significant child-parent divergence: {len(non_sig)} nodes")
        if len(non_sig) > 0:
            print(
                f"  Branch length range: [{non_sig['branch_length'].min():.4f}, {non_sig['branch_length'].max():.4f}]"
            )
            print(f"  Branch length mean: {non_sig['branch_length'].mean():.4f}")

    print("\n" + "=" * 70)
    print("BRANCH LENGTH vs SIBLING DIVERGENCE")
    print("=" * 70)

    if "Sibling_BH_Different" in internal.columns:
        # Filter to nodes where sibling test was actually run
        tested = internal[internal.get("Sibling_Divergence_Skipped", False) == False]

        diff = tested[tested["Sibling_BH_Different"] == True]
        same = tested[tested["Sibling_BH_Different"] == False]

        print(f"\nSiblings significantly different: {len(diff)} nodes")
        if len(diff) > 0:
            print(
                f"  Branch length range: [{diff['branch_length'].min():.4f}, {diff['branch_length'].max():.4f}]"
            )
            print(f"  Branch length mean: {diff['branch_length'].mean():.4f}")
            print(
                f"  Height range: [{diff['height'].min():.4f}, {diff['height'].max():.4f}]"
            )

        print(f"\nSiblings NOT significantly different: {len(same)} nodes")
        if len(same) > 0:
            print(
                f"  Branch length range: [{same['branch_length'].min():.4f}, {same['branch_length'].max():.4f}]"
            )
            print(f"  Branch length mean: {same['branch_length'].mean():.4f}")
            print(
                f"  Height range: [{same['height'].min():.4f}, {same['height'].max():.4f}]"
            )

    print("\n" + "=" * 70)
    print("TOP NODES BY BRANCH LENGTH")
    print("=" * 70)

    cols_to_show = ["height", "branch_length", "leaf_count"]
    if "Child_Parent_Divergence_Significant" in internal.columns:
        cols_to_show.append("Child_Parent_Divergence_Significant")
    if "Sibling_BH_Different" in internal.columns:
        cols_to_show.append("Sibling_BH_Different")
    if "Sibling_Pvalue" in internal.columns:
        cols_to_show.append("Sibling_Pvalue")

    top_bl = internal.nlargest(15, "branch_length")[cols_to_show]
    print("\nTop 15 by branch length:")
    print(top_bl.to_string())

    return internal


def main():
    print("=" * 70)
    print("BRANCH LENGTH vs STATISTICAL TESTS ANALYSIS")
    print("=" * 70)

    divergences = [0.1, 0.3, 0.5, 1.0]

    for div in divergences:
        print(f"\n\n{'#' * 70}")
        print(f"# DIVERGENCE = {div}")
        print(f"{'#' * 70}")

        X, y, sample_names = generate_two_cluster_data(
            n_samples_per_cluster=50,
            n_features=100,
            divergence=div,
            seed=42,
        )

        tree, results_df, decomp_results = run_full_pipeline(
            X, sample_names, n_features=100
        )

        if results_df.empty:
            print("WARNING: No stats_df available")
            continue

        internal_df = analyze_branch_length_vs_tests(results_df)

        # Check root split
        root = tree.root()
        print(f"\n--- ROOT NODE ANALYSIS ---")
        print(f"Root: {root}")

        if root in results_df.index:
            root_row = results_df.loc[root]
            print(f"Height: {root_row.get('height', 'N/A'):.4f}")
            print(f"Branch length: {root_row.get('branch_length', 'N/A'):.4f}")
            print(
                f"Child-Parent Significant: {root_row.get('Child_Parent_Divergence_Significant', 'N/A')}"
            )
            print(f"Sibling Different: {root_row.get('Sibling_BH_Different', 'N/A')}")
            print(f"Sibling P-value: {root_row.get('Sibling_Pvalue', 'N/A')}")

        # Check cluster purity at root split
        root_children = list(tree.successors(root))
        print(f"\nRoot children: {root_children}")

        for child in root_children:
            leaves = tree.get_leaves(child, return_labels=True)
            leaf_indices = [int(s[1:]) for s in leaves]
            labels_in_child = y[leaf_indices]
            purity = max(np.bincount(labels_in_child, minlength=2)) / len(
                labels_in_child
            )

            child_row = results_df.loc[child] if child in results_df.index else None
            bl = (
                child_row.get("branch_length", "N/A")
                if child_row is not None
                else "N/A"
            )

            print(
                f"  {child}: {len(leaves)} leaves, purity={purity:.1%}, branch_length={bl:.4f}"
            )


if __name__ == "__main__":
    main()
