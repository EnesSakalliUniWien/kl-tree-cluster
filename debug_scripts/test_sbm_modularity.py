#!/usr/bin/env python
"""Test modularity matrix transformation on all SBM test cases."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.benchmarking.generators.generate_sbm import generate_sbm
from kl_clustering_analysis.benchmarking.test_cases.sbm import SBM_CASES


def compute_modularity_distance(A):
    """Convert adjacency to distance using modularity matrix.

    Modularity matrix: B = A - k_i*k_j / 2m

    This captures community structure because:
    - B[i,j] > 0 if there are MORE edges between i,j than expected by chance
    - B[i,j] < 0 if there are FEWER edges between i,j than expected

    Nodes in the same community have more edges than expected, so B[i,j] > 0.
    """
    degrees = A.sum(axis=1)
    m = A.sum() / 2  # Number of edges

    if m == 0:
        return np.ones_like(A)

    # B[i,j] = A[i,j] - k_i * k_j / (2m)
    expected = np.outer(degrees, degrees) / (2 * m)
    B = A - expected

    # Shift to make it a proper similarity (non-negative)
    B_shifted = B - B.min()
    B_norm = B_shifted / (B_shifted.max() + 1e-10)

    # Convert to distance
    distance = 1.0 - B_norm
    np.fill_diagonal(distance, 0)

    return distance


def test_sbm_with_modularity():
    """Test modularity transformation on all SBM cases."""

    print("=" * 70)
    print("TESTING MODULARITY MATRIX TRANSFORMATION ON ALL SBM CASES")
    print("=" * 70)
    print()

    sbm_cases = SBM_CASES.get("sbm_graphs", [])

    results = []

    for case in sbm_cases:
        name = case.get("name")
        sizes = case.get("sizes")
        p_intra = case.get("p_intra", 0.1)
        p_inter = case.get("p_inter", 0.01)
        seed = case.get("seed")
        k = len(sizes)

        print(f"Case: {name}")
        print(f"  Communities: {k}, sizes: {sizes}")
        print(f"  p_intra: {p_intra}, p_inter: {p_inter}")

        # Generate graph
        G, ground_truth, A, meta = generate_sbm(
            sizes=sizes, p_intra=p_intra, p_inter=p_inter, seed=seed
        )

        # Current approach: 1 - adjacency
        dist_current = 1.0 - A
        np.fill_diagonal(dist_current, 0)
        ari_current = cluster_and_score(dist_current, ground_truth, k)

        # Modularity approach
        dist_modularity = compute_modularity_distance(A)
        ari_modularity = cluster_and_score(dist_modularity, ground_truth, k)

        print(f"  Current (1-adj):  ARI = {ari_current:.4f}")
        print(f"  Modularity:       ARI = {ari_modularity:.4f}")
        print(f"  Improvement:      +{ari_modularity - ari_current:.4f}")
        print()

        results.append(
            {
                "name": name,
                "k": k,
                "p_intra": p_intra,
                "p_inter": p_inter,
                "ari_current": ari_current,
                "ari_modularity": ari_modularity,
            }
        )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_current = np.mean([r["ari_current"] for r in results])
    mean_modularity = np.mean([r["ari_modularity"] for r in results])

    print(f"Mean ARI (current):    {mean_current:.4f}")
    print(f"Mean ARI (modularity): {mean_modularity:.4f}")
    print(f"Improvement:           +{mean_modularity - mean_current:.4f}")

    return results


def cluster_and_score(distance_matrix, ground_truth, k):
    """Perform hierarchical clustering and return ARI."""
    try:
        distance_condensed = squareform(distance_matrix)
        Z = linkage(distance_condensed, method="average")
        predicted = fcluster(Z, t=k, criterion="maxclust") - 1
        return adjusted_rand_score(ground_truth, predicted)
    except Exception as e:
        print(f"     Clustering error: {e}")
        return 0.0


if __name__ == "__main__":
    test_sbm_with_modularity()
