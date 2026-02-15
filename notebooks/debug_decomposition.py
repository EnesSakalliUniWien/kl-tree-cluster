"""
Minimal Decomposition Debug Script.
Uses a small synthetic dataset to trace the decomposition logic.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
    TreeDecomposition,
)
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_divergence,
)


def create_synthetic_data():
    # 2 distinct clusters: 10 samples each
    # Cluster A: First 5 features = 1
    # Cluster B: Last 5 features = 1
    n_features = 10
    n_samples = 20
    X = np.zeros((n_samples, n_features))

    # Cluster A (Samples 0-9)
    X[:10, :5] = 1
    # Cluster B (Samples 10-19)
    X[10:, 5:] = 1

    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.choice([0, 1], size=X.shape, p=[0.9, 0.1])
    X = np.abs(X - noise)  # Flip bits

    sample_names = [f"S{i}" for i in range(n_samples)]
    return X, sample_names


def main():
    X, sample_names = create_synthetic_data()
    data = pd.DataFrame(X, index=sample_names)

    print("Building tree...")
    Z = linkage(pdist(data.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    print("Populating distributions...")
    tree.populate_node_divergences(data)

    print("Annotating statistics...")
    tree.stats_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    tree.stats_df = annotate_sibling_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )

    # Print root stats
    root = tree.root()
    print(f"\nRoot: {root}")
    if root in tree.stats_df.index:
        row = tree.stats_df.loc[root]
        print(f"Child-Parent Significant: {row['Child_Parent_Divergence_Significant']}")
        print(f"Sibling Different: {row['Sibling_BH_Different']}")
        print(f"Leaf Count: {row['leaf_count']}")

    decomposer = TreeDecomposition(
        tree=tree,
        results_df=tree.stats_df,
        alpha_local=0.05,
        sibling_alpha=0.05,
        use_signal_localization=True,
        localization_max_depth=3,
    )

    print("\nRunning decomposition...")
    results = decomposer.decompose_tree_v2()

    n_clusters = results["num_clusters"]
    print(f"\nFound {n_clusters} clusters.")

    if n_clusters == 2:
        print("SUCCESS: Correctly found 2 clusters.")
    else:
        print(f"FAILURE: Expected 2 clusters, found {n_clusters}.")


if __name__ == "__main__":
    main()
