"""
Purpose: Debug file: Understanding why SBM test cases fail with our statistical tests.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/sbm/q_sbm_statistical_test_failure__sbm__diagnostic.py
"""

#!/usr/bin/env python
"""Debug file: Understanding why SBM test cases fail with our statistical tests.

This script demonstrates:
1. The raw hierarchical clustering works well on SBM data (with modularity distance)
2. Our statistical tests (designed for feature data) fail on graph/relational data
3. Potential solutions: spectral embedding or accepting the limitation
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding

from benchmarks.shared.generators.generate_sbm import generate_sbm


def main():
    print("=" * 70)
    print("SBM DEBUG: Why Statistical Tests Fail on Graph Data")
    print("=" * 70)

    # Generate SBM graph (the clearest case)
    sizes = [30, 30]
    p_intra = 0.12
    p_inter = 0.005
    seed = 123
    k = 2

    G, ground_truth, A, meta = generate_sbm(
        sizes=sizes, p_intra=p_intra, p_inter=p_inter, seed=seed
    )

    print(f"\n1. GENERATED SBM GRAPH")
    print("-" * 70)
    print(f"   Nodes: {meta['n_nodes']}")
    print(f"   Communities: {k} (sizes: {sizes})")
    print(f"   p_intra: {p_intra} (edge prob within community)")
    print(f"   p_inter: {p_inter} (edge prob between communities)")
    print(f"   Sparsity: {(A == 0).sum() / A.size:.1%} zeros in adjacency")

    # ==========================================================================
    # STEP 2: Show raw hierarchical clustering works
    # ==========================================================================
    print(f"\n2. RAW HIERARCHICAL CLUSTERING (with modularity distance)")
    print("-" * 70)

    # Compute modularity-based distance
    degrees = A.sum(axis=1)
    m = A.sum() / 2
    expected = np.outer(degrees, degrees) / (2 * m)
    B = A - expected  # Modularity matrix
    B_shifted = B - B.min()
    B_norm = B_shifted / (B_shifted.max() + 1e-10)
    dist_modularity = 1.0 - B_norm
    np.fill_diagonal(dist_modularity, 0.0)

    # Hierarchical clustering
    Z = linkage(squareform(dist_modularity), method="average")
    labels_raw = fcluster(Z, t=k, criterion="maxclust") - 1
    ari_raw = adjusted_rand_score(ground_truth, labels_raw)

    print(f"   Distance: Modularity matrix -> distance")
    print(f"   Predicted clusters: {np.bincount(labels_raw)}")
    print(f"   True clusters:      {np.bincount(ground_truth)}")
    print(f"   ARI: {ari_raw:.4f}  ✅ WORKS!")

    # ==========================================================================
    # STEP 3: Show why our statistical tests fail
    # ==========================================================================
    print(f"\n3. WHY STATISTICAL TESTS FAIL")
    print("-" * 70)
    print("""
   Our tests (child-parent divergence, sibling divergence) are designed for
   BINARY/CATEGORICAL FEATURE data:
   
   Expected data format:
   ┌─────────┬──────┬──────┬──────┬──────┐
   │ Sample  │ Gene1│ Gene2│ Gene3│ Gene4│  <- Categorical features
   ├─────────┼──────┼──────┼──────┼──────┤
   │ Cell_1  │  0   │  1   │  1   │  0   │
   │ Cell_2  │  0   │  1   │  0   │  0   │
   │ Cell_3  │  1   │  0   │  1   │  1   │
   └─────────┴──────┴──────┴──────┴──────┘
   
   SBM data format:
   ┌─────────┬──────┬──────┬──────┬──────┐
   │ Node    │ Node0│ Node1│ Node2│ Node3│  <- Other nodes (relational!)
   ├─────────┼──────┼──────┼──────┼──────┤
   │ Node_0  │  0   │  1   │  0   │  0   │  (edges to other nodes)
   │ Node_1  │  1   │  0   │  1   │  0   │
   │ Node_2  │  0   │  1   │  0   │  1   │
   └─────────┴──────┴──────┴──────┴──────┘
   
   The columns in SBM are NOT independent categorical features!
   They represent RELATIONSHIPS between samples.
   
   Our entropy-based tests compute feature distributions within clusters:
   - For each feature column, count 0s and 1s
   - Compare parent vs child distributions
   - Use KL divergence / chi-squared tests
   
   For SBM adjacency rows:
   - Each column is a specific node's connection status
   - The "distribution" over columns is meaningless
   - No significant difference detected -> single cluster returned
""")

    # ==========================================================================
    # STEP 4: Demonstrate with feature distributions
    # ==========================================================================
    print(f"\n4. FEATURE DISTRIBUTION COMPARISON")
    print("-" * 70)

    # Split data by ground truth
    cluster_0_samples = np.where(ground_truth == 0)[0]
    cluster_1_samples = np.where(ground_truth == 1)[0]

    # Compute "feature distribution" for each cluster
    # (proportion of 1s in each column)
    dist_c0 = A[cluster_0_samples].mean(axis=0)  # Mean per column for cluster 0
    dist_c1 = A[cluster_1_samples].mean(axis=0)  # Mean per column for cluster 1
    dist_all = A.mean(axis=0)  # Mean per column for all samples

    print(f"   Cluster 0 (nodes 0-29): {len(cluster_0_samples)} nodes")
    print(f"   Cluster 1 (nodes 30-59): {len(cluster_1_samples)} nodes")
    print()
    print(f"   Feature distribution (mean edge probability per column):")
    print(f"   - Cluster 0: mean={dist_c0.mean():.4f}, std={dist_c0.std():.4f}")
    print(f"   - Cluster 1: mean={dist_c1.mean():.4f}, std={dist_c1.std():.4f}")
    print(f"   - All data:  mean={dist_all.mean():.4f}, std={dist_all.std():.4f}")
    print()
    print("   The distributions look similar because:")
    print("   - Columns 0-29 (cluster 0 nodes) have higher values for cluster 0 rows")
    print("   - Columns 30-59 (cluster 1 nodes) have higher values for cluster 1 rows")
    print("   - But the MARGINAL distributions (mean over columns) are similar!")

    # ==========================================================================
    # STEP 5: Solution - Spectral Embedding
    # ==========================================================================
    print(f"\n5. SOLUTION: SPECTRAL EMBEDDING")
    print("-" * 70)

    # Embed nodes into k-dimensional space
    embedding = SpectralEmbedding(
        n_components=k, affinity="precomputed", random_state=42
    )
    X_embedded = embedding.fit_transform(A)

    print(f"   Embed nodes into {k}D space using Laplacian eigenvectors")
    print(f"   Embedded shape: {X_embedded.shape}")

    # Now cluster using standard distance
    dist_embedded = squareform(pdist(X_embedded, metric="euclidean"))
    Z_embedded = linkage(squareform(dist_embedded), method="average")
    labels_embedded = fcluster(Z_embedded, t=k, criterion="maxclust") - 1
    ari_embedded = adjusted_rand_score(ground_truth, labels_embedded)

    print(f"   Hierarchical clustering on embeddings:")
    print(f"   Predicted: {np.bincount(labels_embedded)}")
    print(f"   ARI: {ari_embedded:.4f}")

    # The embedded data has meaningful feature distributions!
    print()
    print("   Why this works for our tests:")
    print("   - Embedded coordinates ARE meaningful features")
    print("   - Dimension 1: captures community 0 vs community 1")
    print("   - Feature distributions within clusters are now distinct")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
   PROBLEM: Our statistical tests assume FEATURE data, not RELATIONAL data.
   
   SBM adjacency matrix is relational - columns are other samples, not features.
   The tests can't detect community structure in this format.
   
   OPTIONS:
   
   A) ACCEPT LIMITATION
      - Document that SBM/graph data is out of scope
      - Recommend Louvain/Leiden for graph clustering (already in pipeline)
   
   B) SPECTRAL EMBEDDING PREPROCESSING  
      - For SBM data, first embed nodes in k-dim space
      - Then run our method on the embedded features
      - This converts relational -> feature data
   
   C) GRAPH-SPECIFIC TEST
      - Add a modularity-based pruning criterion for graph data
      - Instead of entropy tests, check if split increases modularity
   
   RECOMMENDATION: Option A (accept limitation) is simplest.
   Our method excels at binary/categorical feature data.
   For graphs, use graph-specific methods.
""")


if __name__ == "__main__":
    main()
