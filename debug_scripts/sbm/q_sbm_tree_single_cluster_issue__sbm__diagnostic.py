"""
Purpose: Debug why SBM hierarchical clustering finds only 1 cluster.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/sbm/q_sbm_tree_single_cluster_issue__sbm__diagnostic.py
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from benchmarks.shared.generators.generate_sbm import generate_sbm
from kl_clustering_analysis.tree.poset_tree import PosetTree


def debug_sbm_tree():
    """Debug why our statistical tests reject all splits for SBM data."""

    # Generate the clearest SBM case
    sizes = [30, 30]
    p_intra = 0.12
    p_inter = 0.005
    seed = 123

    print("=" * 70)
    print("DEBUGGING SBM HIERARCHICAL TREE")
    print("=" * 70)

    G, ground_truth, A, meta = generate_sbm(
        sizes=sizes, p_intra=p_intra, p_inter=p_inter, seed=seed
    )

    print(
        f"Generated SBM graph: {meta['n_nodes']} nodes, {meta['n_blocks']} communities"
    )

    # Compute modularity distance
    degrees = A.sum(axis=1)
    m = A.sum() / 2
    expected = np.outer(degrees, degrees) / (2 * m)
    B = A - expected
    B_shifted = B - B.min()
    B_norm = B_shifted / (B_shifted.max() + 1e-10)
    distance_matrix = 1.0 - B_norm
    np.fill_diagonal(distance_matrix, 0.0)

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")

    # Create condensed distance
    distance_condensed = squareform(distance_matrix)

    # Build linkage
    Z = linkage(distance_condensed, method="average")

    # First, test raw hierarchical cut
    labels_k2 = fcluster(Z, t=2, criterion="maxclust") - 1
    print(f"\nRaw hierarchical cut at k=2: {np.bincount(labels_k2)}")

    from sklearn.metrics import adjusted_rand_score

    ari_raw = adjusted_rand_score(ground_truth, labels_k2)
    print(f"Raw ARI: {ari_raw:.4f}")

    # Now create PosetTree and analyze
    sample_ids = [f"S{i}" for i in range(len(ground_truth))]

    print("\n" + "-" * 70)
    print("Creating PosetTree...")
    print("-" * 70)

    tree = PosetTree.from_linkage(linkage_matrix=Z, leaf_names=sample_ids)

    print(f"Tree created with {len(tree.nodes)} nodes")

    # What does the data look like?
    # For SBM, the "data" is the modularity-based distance, not binary features
    # This is a problem! Our tests expect binary/categorical data

    print("\n" + "-" * 70)
    print("THE PROBLEM:")
    print("-" * 70)
    print("""
Our statistical tests (child-parent divergence, sibling divergence) are designed
for BINARY/CATEGORICAL data where we compute entropy distributions.

For SBM data:
- The original data is a binary adjacency matrix (0/1)
- But the 'features' are other nodes (columns = nodes)
- This doesn't represent categorical features in the traditional sense

When we compute KL divergence on adjacency rows:
- Each sample (node) has a feature vector of connections to other nodes
- The 'distribution' is over connection patterns, not meaningful categories

This breaks the assumptions of our statistical tests!
""")

    # Let's check what happens with the decomposition
    print("\n" + "-" * 70)
    print("Checking tree decomposition...")
    print("-" * 70)

    # Try to decompose with very permissive alpha
    try:
        leaf_data = pd.DataFrame(A, index=sample_ids)
        decomp = tree.decompose(
            leaf_data=leaf_data,
            alpha_local=0.99,
            sibling_alpha=0.99,
        )  # Very permissive
        n_clusters = int(decomp.get("num_clusters", 0))
        print(f"With alpha=0.99: {n_clusters} clusters")
    except Exception as e:
        print(f"Decomposition failed: {e}")

    # The issue is that data_df passed to the tree contains adjacency matrix rows
    # But our tests expect binary/categorical feature distributions
    print("\n" + "-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    print("""
The SBM data format is fundamentally incompatible with our entropy-based tests:

1. Our tests compute feature distributions within each cluster
2. For SBM, 'features' are connections to other nodes
3. These don't form meaningful categorical distributions

Options:
A) Accept that SBM is out-of-scope for our method
B) Use a different test for graph data (e.g., modularity-based splitting)
C) Convert graph to node features (e.g., spectral embedding) before clustering

For now, the modularity distance helps with tree construction, but the 
statistical tests for tree pruning don't work with graph data.
""")


if __name__ == "__main__":
    debug_sbm_tree()
