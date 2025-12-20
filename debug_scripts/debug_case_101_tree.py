"""
Debug why test case 101 (overlap_hd_6c_2k) finds only 1 cluster
despite UMAP showing separation.

Traces through the actual tree decomposition to see where splits fail.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_cases_config import get_default_test_cases
from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def main():
    # Get test case 101
    cases = get_default_test_cases()
    tc = cases[100]

    print("=" * 80)
    print(f"Debugging Test Case 101: {tc['name']}")
    print("=" * 80)
    print(
        f"  n_rows={tc['n_rows']}, n_cols={tc['n_cols']}, n_clusters={tc['n_clusters']}"
    )
    print(f"  entropy_param={tc['entropy_param']}")
    print()

    # Generate data
    leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=tc["n_rows"],
        n_cols=tc["n_cols"],
        n_clusters=tc["n_clusters"],
        entropy_param=tc["entropy_param"],
        balanced_clusters=tc["balanced_clusters"],
        random_seed=tc["seed"],
    )

    # Convert to DataFrame
    sample_names = sorted(leaf_matrix_dict.keys(), key=lambda x: int(x[1:]))
    X = np.array([leaf_matrix_dict[name] for name in sample_names])
    y = np.array([cluster_assignments[name] for name in sample_names])
    data_df = pd.DataFrame(X, index=sample_names)

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print()

    # Build tree
    print("Building hierarchical tree...")
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    print(f"Tree built with {len(tree.get_leaves())} leaves")
    print()

    # Fix random seed for reproducible permutation tests
    np.random.seed(42)

    # Decompose with verbose output
    print("=" * 80)
    print("DECOMPOSITION (sibling_alpha=0.01)")
    print("=" * 80)

    decomp = tree.decompose(
        leaf_data=data_df,
        sibling_alpha=0.01,
    )

    print()
    print("=" * 80)
    print(f"RESULT: num_clusters = {decomp['num_clusters']}")
    print("=" * 80)

    # Parse cluster assignments from the decomposition result
    cluster_info = decomp["cluster_assignments"]

    print(f"\nFound {len(cluster_info)} clusters:")

    # Show cluster composition
    for cluster_id, info in sorted(cluster_info.items()):
        leaves = info["leaves"]
        true_labels = [
            cluster_assignments[leaf] for leaf in leaves if leaf in cluster_assignments
        ]
        if true_labels:
            unique, counts = np.unique(true_labels, return_counts=True)
            print(
                f"\n  Cluster {cluster_id} (root={info['root_node']}, size={info['size']}):"
            )
            for u, c in zip(unique, counts):
                pct = 100 * c / len(true_labels)
                print(f"    - True cluster {u}: {c} samples ({pct:.1f}%)")

    # Calculate ARI
    from sklearn.metrics import adjusted_rand_score

    # Create predicted labels
    pred_labels = {}
    for cluster_id, info in cluster_info.items():
        for leaf in info["leaves"]:
            pred_labels[leaf] = cluster_id

    # Build arrays for ARI
    y_true = []
    y_pred = []
    for name in sample_names:
        if name in pred_labels:
            y_true.append(cluster_assignments[name])
            y_pred.append(pred_labels[name])

    ari = adjusted_rand_score(y_true, y_pred)
    print(f"\nAdjusted Rand Index: {ari:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"True clusters: 6")
    print(f"Found clusters: {len(cluster_info)}")
    print(f"ARI: {ari:.4f}")

    if ari > 0.8:
        print("\n✓ Algorithm correctly separates the overlapping clusters!")
        print("  Even though individual features are noisy (entropy=0.33),")
        print("  the high-dimensional structure (2000 features) preserves signal.")
    else:
        print("\n✗ Algorithm fails to separate overlapping clusters.")
        print("  The feature-level noise overwhelms the signal.")


if __name__ == "__main__":
    main()
