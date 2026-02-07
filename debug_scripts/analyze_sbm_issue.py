#!/usr/bin/env python
"""Debug script to understand why SBM test cases get ARI = 0."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.benchmarking.generators.generate_sbm import generate_sbm


def analyze_sbm_case():
    """Analyze the SBM data generation and clustering pipeline."""

    print("=" * 70)
    print("ANALYZING SBM TEST CASE")
    print("=" * 70)

    # Generate the "sbm_clear_small" case
    sizes = [30, 30]
    p_intra = 0.12
    p_inter = 0.005
    seed = 123

    print(f"\nGenerating SBM graph:")
    print(f"  Sizes: {sizes}")
    print(f"  p_intra: {p_intra} (prob. of edge within community)")
    print(f"  p_inter: {p_inter} (prob. of edge between communities)")
    print(f"  seed: {seed}")

    G, ground_truth, A, sbm_meta = generate_sbm(
        sizes=sizes,
        p_intra=p_intra,
        p_inter=p_inter,
        seed=seed,
    )

    print(f"\nGenerated graph:")
    print(f"  Number of nodes: {sbm_meta['n_nodes']}")
    print(f"  Number of blocks (true clusters): {sbm_meta['n_blocks']}")
    print(f"  Adjacency matrix shape: {A.shape}")
    print(
        f"  Ground truth labels: {np.unique(ground_truth)} with counts {np.bincount(ground_truth)}"
    )

    # Show what the adjacency matrix looks like
    print(f"\nAdjacency matrix statistics:")
    print(f"  Min: {A.min()}, Max: {A.max()}")
    print(f"  Mean: {A.mean():.4f}")
    print(f"  Sparsity: {(A == 0).sum() / A.size:.2%} zeros")

    # Block structure analysis
    n_block_1 = sizes[0]
    block_1_adj = A[:n_block_1, :n_block_1]
    block_2_adj = A[n_block_1:, n_block_1:]
    inter_block_adj = A[:n_block_1, n_block_1:]

    print(
        f"\n  Block 1 internal density: {block_1_adj.sum() / (n_block_1 * n_block_1):.4f}"
    )
    print(
        f"  Block 2 internal density: {block_2_adj.sum() / (sizes[1] * sizes[1]):.4f}"
    )
    print(
        f"  Inter-block density: {inter_block_adj.sum() / (n_block_1 * sizes[1]):.4f}"
    )

    # Now simulate what the pipeline does:
    # Convert adjacency to distance: distance = 1 - adjacency
    print("\n" + "=" * 70)
    print("PIPELINE CONVERSION: adjacency -> distance")
    print("=" * 70)

    distance_matrix = 1.0 - A
    np.fill_diagonal(distance_matrix, 0.0)

    print(f"\nDistance matrix (1 - adjacency):")
    print(f"  Min: {distance_matrix.min()}, Max: {distance_matrix.max()}")
    print(f"  Mean: {distance_matrix.mean():.4f}")

    # The problem: most edges are 0 (no connection), so distance = 1 - 0 = 1
    # Only connected nodes have distance = 1 - 1 = 0
    print(f"\n  Distance = 0 (connected nodes): {(distance_matrix == 0).sum()} pairs")
    print(f"  Distance = 1 (not connected): {(distance_matrix == 1).sum()} pairs")

    # Hierarchical clustering on this distance matrix
    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 70)

    distance_condensed = squareform(distance_matrix)
    Z = linkage(distance_condensed, method="average")

    # Cut at k=2 clusters
    k = 2
    predicted_labels = fcluster(Z, t=k, criterion="maxclust")
    predicted_labels = predicted_labels - 1  # 0-indexed

    ari = adjusted_rand_score(ground_truth, predicted_labels)

    print(f"\nHierarchical clustering with average linkage, k={k}:")
    print(f"  Predicted cluster sizes: {np.bincount(predicted_labels)}")
    print(f"  Ground truth sizes: {np.bincount(ground_truth)}")
    print(f"  ARI: {ari:.4f}")

    # What's happening? Let's look at the dendrogram structure
    print("\n" + "=" * 70)
    print("UNDERSTANDING THE ISSUE")
    print("=" * 70)

    print("""
THE PROBLEM:
============
The adjacency matrix A has entries:
  - A[i,j] = 1 if edge exists between i and j
  - A[i,j] = 0 if no edge exists

With p_intra=0.12, only ~12% of within-community pairs are connected.
With p_inter=0.005, only ~0.5% of between-community pairs are connected.

The pipeline converts this to distance = 1 - adjacency:
  - Distance = 0 for connected nodes (rare: ~12% within, ~0.5% between)
  - Distance = 1 for not-connected nodes (common: ~88% within, ~99.5% between)

THIS IS THE WRONG REPRESENTATION!

For hierarchical clustering:
  - We need: similar nodes to have SMALL distance
  - We have: connected nodes have distance 0, but most nodes aren't directly connected!
  
A node in cluster 1 might not be directly connected to another node in cluster 1,
so their distance is 1. But this same distance (1) applies to most cross-cluster pairs too.

The correct approach for graph data:
  - Use graph-based distances (shortest path, resistance distance, etc.)
  - Or use graph clustering algorithms (spectral, Louvain, Leiden, etc.)
""")

    # Try with a proper graph distance
    print("\n" + "=" * 70)
    print("ALTERNATIVE: SHORTEST PATH DISTANCE")
    print("=" * 70)

    try:
        import networkx as nx

        # Compute shortest path distances
        sp_dist = dict(nx.all_pairs_shortest_path_length(G))

        # Create distance matrix
        n = len(ground_truth)
        sp_matrix = np.full((n, n), np.inf)
        for i in range(n):
            for j, d in sp_dist[i].items():
                sp_matrix[i, j] = d

        # Handle disconnected nodes (set to max distance + 1)
        max_finite = sp_matrix[np.isfinite(sp_matrix)].max()
        sp_matrix[np.isinf(sp_matrix)] = max_finite + 1
        np.fill_diagonal(sp_matrix, 0)

        print(f"Shortest path distance matrix:")
        print(f"  Min: {sp_matrix.min()}, Max: {sp_matrix.max()}")
        print(f"  Mean: {sp_matrix.mean():.4f}")

        # Cluster with shortest path distance
        sp_condensed = squareform(sp_matrix)
        Z_sp = linkage(sp_condensed, method="average")
        pred_sp = fcluster(Z_sp, t=k, criterion="maxclust") - 1

        ari_sp = adjusted_rand_score(ground_truth, pred_sp)
        print(f"\nHierarchical clustering with shortest path distance:")
        print(f"  Predicted cluster sizes: {np.bincount(pred_sp)}")
        print(f"  ARI: {ari_sp:.4f}")

    except Exception as e:
        print(f"Error computing shortest path: {e}")

    # Try spectral clustering
    print("\n" + "=" * 70)
    print("ALTERNATIVE: SPECTRAL CLUSTERING")
    print("=" * 70)

    try:
        from sklearn.cluster import SpectralClustering

        # Use adjacency as affinity
        spectral = SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=42
        )
        pred_spectral = spectral.fit_predict(A)

        ari_spectral = adjusted_rand_score(ground_truth, pred_spectral)
        print(f"Spectral clustering with adjacency affinity:")
        print(f"  Predicted cluster sizes: {np.bincount(pred_spectral)}")
        print(f"  ARI: {ari_spectral:.4f}")

    except Exception as e:
        print(f"Error with spectral clustering: {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The ARI = 0 for SBM data is expected because:

1. The pipeline uses distance = 1 - adjacency
2. In sparse graphs, most pairs have adjacency = 0, so distance = 1
3. This makes most within-cluster AND between-cluster pairs equidistant
4. Hierarchical clustering can't distinguish communities with this representation

Solutions:
- Use proper graph distances (shortest path, random walk, etc.)
- Use graph-specific algorithms (spectral, Louvain, Leiden)
- This is NOT a bug - it's a fundamental limitation of applying 
  hierarchical clustering to graph/network data
""")


if __name__ == "__main__":
    analyze_sbm_case()
