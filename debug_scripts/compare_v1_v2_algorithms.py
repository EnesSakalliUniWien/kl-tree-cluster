"""Compare decompose_tree (v1) vs decompose_tree_v2 (signal localization).

This script runs both algorithms on synthetic datasets designed to show
the difference between hard and soft boundary detection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import warnings

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
    TreeDecomposition,
)
from kl_clustering_analysis import config


def create_partial_overlap_dataset(n_samples=400, random_state=42, n_noise_features=18):
    """Create a dataset where two sub-clusters overlap.

    Creates 4 clusters + noise features.
    """
    np.random.seed(random_state)

    # Centers: 0 and 2 are close, 1 and 3 are far apart
    centers = [
        [0.0, 0.0],  # Cluster 0 (A left)
        [5.0, 0.0],  # Cluster 1 (A right)
        [0.0, 0.2],  # Cluster 2 (B left) - close to 0!
        [5.0, 8.0],  # Cluster 3 (B right) - far from 1
    ]

    X_core, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=0.5,
        random_state=random_state,
    )

    # Add noise features if requested (to match debug script success settings)
    if n_noise_features > 0:
        X_noise = (
            np.random.randn(n_samples, n_noise_features) * 2.0
        )  # Higher variance noise
        X = np.hstack([X_core, X_noise])
    else:
        X = X_core

    # Ground truth: 0 and 2 should be merged (they overlap)
    y_optimal = y_true.copy()
    y_optimal[y_optimal == 2] = 0  # Merge cluster 2 into 0

    return X, y_true, y_optimal


def create_clean_separation_dataset(
    n_samples=400, random_state=42, n_noise_features=18
):
    """Create a dataset with clean cluster separation."""
    centers = [
        [0.0, 0.0],
        [6.0, 0.0],
        [0.0, 6.0],
        [6.0, 6.0],
    ]

    X_core, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=0.5,
        random_state=random_state,
    )

    if n_noise_features > 0:
        X_noise = np.random.randn(n_samples, n_noise_features) * 2.0
        X = np.hstack([X_core, X_noise])
    else:
        X = X_core

    return X, y_true, y_true  # Optimal = true labels


def run_comparison(X, y_true, y_optimal, dataset_name):
    """Run both v1 and v2, compare results."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(
        f"n_samples: {len(X)}, true_k: {len(np.unique(y_true))}, optimal_k: {len(np.unique(y_optimal))}"
    )
    print(f"{'=' * 60}")

    # Binarize data (simple threshold)
    X_binary = (X > np.median(X, axis=0)).astype(int)
    sample_names = [f"Sample_{j}" for j in range(X.shape[0])]
    data = pd.DataFrame(X_binary, index=sample_names)

    # Build linkage and tree
    Z = linkage(pdist(data.values, metric="rogerstanimoto"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # V1: Original algorithm
    results_v1 = tree.decompose(
        leaf_data=data,
        alpha_local=0.05,
        sibling_alpha=0.05,
    )

    # Debug info
    if hasattr(tree, "stats_df") and tree.stats_df is not None:
        n_sig = tree.stats_df["Child_Parent_Divergence_Significant"].sum()
        print(f"  [DEBUG] Significant nodes found: {n_sig} / {len(tree.stats_df)}")

        # Check root children
        root = [n for n, d in tree.in_degree() if d == 0][0]
        children = list(tree.successors(root))
        print(f"  [DEBUG] Root: {root} ({tree.nodes[root].get('leaf_count')})")
        for child in children:
            if child in tree.stats_df.index:
                sig = tree.stats_df.loc[child, "Child_Parent_Divergence_Significant"]
                pval = tree.stats_df.loc[child, "Child_Parent_Divergence_P_Value"]
                print(
                    f"    Child {child} ({tree.nodes[child].get('leaf_count')}): Sig={sig}, p={pval:.4f}"
                )
            else:
                print(f"    Child {child}: Not in stats_df")

        if n_sig > 0:
            # print(f"  [DEBUG] Sample sizes of sig nodes: {tree.stats_df[tree.stats_df['Child_Parent_Divergence_Significant']]['leaf_count'].unique()}")
            pass

    # Extract labels from cluster_assignments
    cluster_assignments_v1 = results_v1.get("cluster_assignments", {})
    labels_v1 = np.zeros(len(X), dtype=int)
    for cluster_id, info in cluster_assignments_v1.items():
        # 'info' is a dictionary containing 'leaves'
        leaf_list = info.get("leaves", [])
        for sample in leaf_list:
            if sample in sample_names:  # Skip non-leaf nodes
                idx = sample_names.index(sample)
                labels_v1[idx] = cluster_id

    k_v1 = len(np.unique(labels_v1))
    ari_v1_true = adjusted_rand_score(y_true, labels_v1)
    ari_v1_optimal = adjusted_rand_score(y_optimal, labels_v1)

    print(f"\nV1 (original):")
    print(f"  k found: {k_v1}")
    print(f"  ARI vs true labels: {ari_v1_true:.4f}")
    print(f"  ARI vs optimal labels: {ari_v1_optimal:.4f}")

    # V2: Signal localization
    print(f"\nV2 (signal localization):")
    try:
        # Use existing stats_df from the tree (populated by V1 run)
        decomposer = TreeDecomposition(
            tree=tree,
            results_df=tree.stats_df,
            alpha_local=0.05,
            sibling_alpha=0.05,
            use_signal_localization=True,
        )

        # Call decompose_tree_v2 explicitly
        results_v2 = decomposer.decompose_tree_v2()

        cluster_assignments_v2 = results_v2.get("cluster_assignments", {})
        labels_v2 = np.zeros(len(X), dtype=int)
        for cluster_id, info in cluster_assignments_v2.items():
            leaf_list = info.get("leaves", [])
            for sample in leaf_list:
                if sample in sample_names:  # Skip non-leaf nodes
                    idx = sample_names.index(sample)
                    labels_v2[idx] = cluster_id

        k_v2 = len(np.unique(labels_v2))
        ari_v2_true = adjusted_rand_score(y_true, labels_v2)
        ari_v2_optimal = adjusted_rand_score(y_optimal, labels_v2)

        print(f"  k found: {k_v2}")
        print(f"  ARI vs true labels: {ari_v2_true:.4f}")
        print(f"  ARI vs optimal labels: {ari_v2_optimal:.4f}")

        print(f"\nDifference (V2 - V1):")
        print(f"  Delta ARI (vs optimal): {ari_v2_optimal - ari_v1_optimal:+.4f}")

        return {
            "dataset": dataset_name,
            "k_v1": k_v1,
            "k_v2": k_v2,
            "ari_v1_true": ari_v1_true,
            "ari_v2_true": ari_v2_true,
            "ari_v1_optimal": ari_v1_optimal,
            "ari_v2_optimal": ari_v2_optimal,
            "delta_ari": ari_v2_optimal - ari_v1_optimal,
        }
    except Exception as e:
        print(f"V2 failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "k_v1": k_v1,
            "k_v2": None,
            "ari_v1_true": ari_v1_true,
            "ari_v2_true": None,
            "ari_v1_optimal": ari_v1_optimal,
            "ari_v2_optimal": None,
            "delta_ari": None,
        }

    except Exception as e:
        print(f"\nV2 failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "k_v1": k_v1,
            "k_v2": None,
            "ari_v1_true": ari_v1_true,
            "ari_v2_true": None,
            "ari_v1_optimal": ari_v1_optimal,
            "ari_v2_optimal": None,
            "delta_ari": None,
        }


def main():
    """Run comparison on multiple datasets."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    results = []

    # Test 1: Partial overlap (should benefit from v2)
    X, y_true, y_optimal = create_partial_overlap_dataset()
    result = run_comparison(X, y_true, y_optimal, "partial_overlap")
    if result:
        results.append(result)

    # Test 2: Clean separation (v1 and v2 should be similar)
    X, y_true, y_optimal = create_clean_separation_dataset()
    result = run_comparison(X, y_true, y_optimal, "clean_separation")
    if result:
        results.append(result)

    # Test 3: Partial overlap with different random seeds
    for seed in [123, 456, 789]:
        X, y_true, y_optimal = create_partial_overlap_dataset(random_state=seed)
        result = run_comparison(X, y_true, y_optimal, f"partial_overlap_seed{seed}")
        if result:
            results.append(result)

    # Summary table
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\nMean delta ARI: {df['delta_ari'].mean():+.4f}")
        print(f"Datasets where V2 > V1: {(df['delta_ari'] > 0).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
