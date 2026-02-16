"""
Purpose: Test different data transformations for SBM graphs to make them work with hierarchical clustering.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/sbm/q_sbm_transformation_comparison__sbm__validation.py
"""

#!/usr/bin/env python
"""Test different data transformations for SBM graphs to make them work with hierarchical clustering."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding

from benchmarks.shared.generators.generate_sbm import generate_sbm


def test_transformations():
    """Test different ways to transform SBM data for hierarchical clustering."""

    # Generate SBM graph
    sizes = [30, 30]
    p_intra = 0.12
    p_inter = 0.005
    seed = 123
    k = 2

    G, ground_truth, A, sbm_meta = generate_sbm(
        sizes=sizes, p_intra=p_intra, p_inter=p_inter, seed=seed
    )

    print("=" * 70)
    print("TESTING DATA TRANSFORMATIONS FOR SBM GRAPHS")
    print("=" * 70)
    print(f"Graph: {sbm_meta['n_nodes']} nodes, {k} communities")
    print(f"Sparsity: {(A == 0).sum() / A.size:.1%} zeros")
    print()

    results = []

    # 1. Original: distance = 1 - adjacency (current approach)
    print("1. CURRENT: distance = 1 - adjacency")
    dist1 = 1.0 - A
    np.fill_diagonal(dist1, 0)
    ari1 = cluster_and_score(dist1, ground_truth, k)
    results.append(("1-adjacency (current)", ari1))
    print(f"   ARI: {ari1:.4f}")
    print()

    # 2. Jaccard similarity of neighborhoods
    print(
        "2. JACCARD SIMILARITY: sim(i,j) = |neighbors(i) ∩ neighbors(j)| / |neighbors(i) ∪ neighbors(j)|"
    )
    jaccard_sim = compute_jaccard_similarity(A)
    dist2 = 1.0 - jaccard_sim
    np.fill_diagonal(dist2, 0)
    ari2 = cluster_and_score(dist2, ground_truth, k)
    results.append(("Jaccard similarity", ari2))
    print(f"   ARI: {ari2:.4f}")
    print()

    # 3. Common neighbors count
    print("3. COMMON NEIGHBORS: count of shared neighbors between i and j")
    common_neighbors = A @ A  # A^2[i,j] = number of common neighbors
    max_cn = common_neighbors.max()
    if max_cn > 0:
        cn_sim = common_neighbors / max_cn
    else:
        cn_sim = common_neighbors
    dist3 = 1.0 - cn_sim
    np.fill_diagonal(dist3, 0)
    ari3 = cluster_and_score(dist3, ground_truth, k)
    results.append(("Common neighbors", ari3))
    print(f"   ARI: {ari3:.4f}")
    print()

    # 4. Cosine similarity of adjacency rows
    print("4. COSINE SIMILARITY: treat each row as a feature vector")
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim = cosine_similarity(A)
    dist4 = 1.0 - cosine_sim
    np.fill_diagonal(dist4, 0)
    dist4 = np.clip(dist4, 0, None)  # Ensure non-negative
    ari4 = cluster_and_score(dist4, ground_truth, k)
    results.append(("Cosine similarity", ari4))
    print(f"   ARI: {ari4:.4f}")
    print()

    # 5. Spectral embedding -> Euclidean distance
    print("5. SPECTRAL EMBEDDING: embed in k-dim space using Laplacian eigenvectors")
    try:
        embedding = SpectralEmbedding(
            n_components=k, affinity="precomputed", random_state=42
        )
        X_spectral = embedding.fit_transform(A)
        dist5 = squareform(pdist(X_spectral, metric="euclidean"))
        ari5 = cluster_and_score(dist5, ground_truth, k)
        results.append(("Spectral embedding", ari5))
        print(f"   ARI: {ari5:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        results.append(("Spectral embedding", 0.0))
    print()

    # 6. Adamic-Adar index
    print("6. ADAMIC-ADAR: sum of 1/log(degree) for common neighbors")
    adamic_adar = compute_adamic_adar(A)
    max_aa = adamic_adar.max()
    if max_aa > 0:
        aa_sim = adamic_adar / max_aa
    else:
        aa_sim = adamic_adar
    dist6 = 1.0 - aa_sim
    np.fill_diagonal(dist6, 0)
    ari6 = cluster_and_score(dist6, ground_truth, k)
    results.append(("Adamic-Adar", ari6))
    print(f"   ARI: {ari6:.4f}")
    print()

    # 7. Personalized PageRank similarity
    print("7. PERSONALIZED PAGERANK: random walk based similarity")
    try:
        ppr_sim = compute_ppr_similarity(A, alpha=0.85)
        dist7 = 1.0 - ppr_sim
        np.fill_diagonal(dist7, 0)
        dist7 = np.clip(dist7, 0, None)
        ari7 = cluster_and_score(dist7, ground_truth, k)
        results.append(("PageRank similarity", ari7))
        print(f"   ARI: {ari7:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        results.append(("PageRank similarity", 0.0))
    print()

    # 8. Heat kernel similarity
    print("8. HEAT KERNEL: exp(-t * L) where L is the Laplacian")
    try:
        heat_sim = compute_heat_kernel(A, t=1.0)
        dist8 = 1.0 - heat_sim
        np.fill_diagonal(dist8, 0)
        dist8 = np.clip(dist8, 0, None)
        ari8 = cluster_and_score(dist8, ground_truth, k)
        results.append(("Heat kernel", ari8))
        print(f"   ARI: {ari8:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        results.append(("Heat kernel", 0.0))
    print()

    # 9. Modularity-based distance
    print("9. MODULARITY MATRIX: B = A - k_i*k_j / 2m")
    try:
        mod_sim = compute_modularity_matrix(A)
        # Shift to make it a proper similarity (non-negative)
        mod_sim_shifted = mod_sim - mod_sim.min()
        mod_sim_norm = mod_sim_shifted / mod_sim_shifted.max()
        dist9 = 1.0 - mod_sim_norm
        np.fill_diagonal(dist9, 0)
        ari9 = cluster_and_score(dist9, ground_truth, k)
        results.append(("Modularity matrix", ari9))
        print(f"   ARI: {ari9:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        results.append(("Modularity matrix", 0.0))
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results.sort(key=lambda x: x[1], reverse=True)
    for name, ari in results:
        bar = "█" * int(ari * 40)
        print(f"  {name:25s}: ARI = {ari:.4f} {bar}")

    print()
    best = results[0]
    print(f"BEST TRANSFORMATION: {best[0]} with ARI = {best[1]:.4f}")

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


def compute_jaccard_similarity(A):
    """Compute Jaccard similarity between all node pairs based on neighborhoods."""
    n = A.shape[0]
    jaccard = np.zeros((n, n))

    for i in range(n):
        neighbors_i = set(np.where(A[i] > 0)[0])
        for j in range(i, n):
            neighbors_j = set(np.where(A[j] > 0)[0])
            intersection = len(neighbors_i & neighbors_j)
            union = len(neighbors_i | neighbors_j)
            if union > 0:
                sim = intersection / union
            else:
                sim = 0.0
            jaccard[i, j] = jaccard[j, i] = sim

    return jaccard


def compute_adamic_adar(A):
    """Compute Adamic-Adar index for all node pairs."""
    n = A.shape[0]
    degrees = A.sum(axis=1)
    aa = np.zeros((n, n))

    for i in range(n):
        neighbors_i = np.where(A[i] > 0)[0]
        for j in range(i, n):
            neighbors_j = np.where(A[j] > 0)[0]
            common = np.intersect1d(neighbors_i, neighbors_j)
            score = 0.0
            for c in common:
                if degrees[c] > 1:
                    score += 1.0 / np.log(degrees[c])
            aa[i, j] = aa[j, i] = score

    return aa


def compute_ppr_similarity(A, alpha=0.85, max_iter=100):
    """Compute personalized PageRank similarity matrix."""
    n = A.shape[0]

    # Normalize adjacency to get transition matrix
    degrees = A.sum(axis=1)
    degrees[degrees == 0] = 1  # Avoid division by zero
    T = A / degrees[:, np.newaxis]

    # Compute PPR for each node
    I = np.eye(n)
    ppr_matrix = np.zeros((n, n))

    for i in range(n):
        # Personalization vector
        p = np.zeros(n)
        p[i] = 1.0

        # Power iteration
        ppr = p.copy()
        for _ in range(max_iter):
            ppr = alpha * (T.T @ ppr) + (1 - alpha) * p

        ppr_matrix[i] = ppr

    # Make symmetric
    ppr_sim = (ppr_matrix + ppr_matrix.T) / 2

    # Normalize
    ppr_sim = ppr_sim / ppr_sim.max()

    return ppr_sim


def compute_heat_kernel(A, t=1.0):
    """Compute heat kernel similarity: exp(-t * L)."""
    n = A.shape[0]

    # Laplacian: L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Heat kernel: H = exp(-t * L)
    # Use eigendecomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    H = eigenvectors @ np.diag(np.exp(-t * eigenvalues)) @ eigenvectors.T

    # Normalize
    H = (H - H.min()) / (H.max() - H.min() + 1e-10)

    return H


def compute_modularity_matrix(A):
    """Compute the modularity matrix B = A - k_i*k_j / 2m."""
    degrees = A.sum(axis=1)
    m = A.sum() / 2  # Number of edges

    if m == 0:
        return np.zeros_like(A)

    # B[i,j] = A[i,j] - k_i * k_j / (2m)
    expected = np.outer(degrees, degrees) / (2 * m)
    B = A - expected

    return B


if __name__ == "__main__":
    test_transformations()
