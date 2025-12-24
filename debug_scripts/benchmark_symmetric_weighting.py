#!/usr/bin/env python3
"""
Benchmark: Symmetric Global Weighting with Data-Driven Neutral Point

This script tests the new symmetric weighting approach that gives:
- Bonus (weight < 1.0) for edges with strong local signal (global structures)
- Penalty (weight > 1.0) for edges with weak local signal (noise)

The neutral point is computed as the median contribution fraction from the data.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
import pandas as pd

print("=" * 80)
print("BENCHMARK: Symmetric Global Weighting (Data-Driven Neutral Point)")
print("=" * 80)

# Check config
print(f"\nConfiguration:")
print(f"  USE_GLOBAL_DIVERGENCE_WEIGHTING: {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}")
print(f"  GLOBAL_WEIGHT_METHOD: {config.GLOBAL_WEIGHT_METHOD}")
print(f"  GLOBAL_WEIGHT_STRENGTH: {config.GLOBAL_WEIGHT_STRENGTH}")

results = []

# Test datasets - use the same format as quick_start.py
datasets = [
    ("blobs_4", 200, 30, 4, 1.0),
    ("blobs_6", 300, 30, 6, 1.0),
    ("blobs_noisy", 200, 30, 4, 2.0),
    ("blobs_easy", 100, 30, 3, 0.8),
]

for name, n_samples, n_features, n_clusters, cluster_std in datasets:
    print(f"\nDataset: {name}")
    print("-" * 40)

    # Generate blobs
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=42,
    )
    n_true = len(np.unique(y_true))

    # Create binary data like quick_start.py and pipeline.py
    X_binary = (X > np.median(X, axis=0)).astype(int)
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    data_df = pd.DataFrame(
        X_binary, index=sample_names, columns=[f"F{j}" for j in range(n_features)]
    )

    # Build tree using linkage like pipeline.py
    distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    # Run decomposition like pipeline.py
    decomp_results = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )
    clusters = decomp_results.get("cluster_assignments", {})
    n_found = len(clusters)

    # Get cluster labels - use the same logic as pipeline.py
    assignments = {sample: -1 for sample in sample_names}
    for cluster_id, info in clusters.items():
        for leaf in info["leaves"]:
            assignments[leaf] = cluster_id

    y_pred = np.array([assignments[sample] for sample in sample_names], dtype=int)

    # Calculate metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print(f"  True clusters: {n_true}")
    print(f"  Found clusters: {n_found}")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")

    results.append(
        {
            "dataset": name,
            "true_k": n_true,
            "found_k": n_found,
            "ARI": ari,
            "NMI": nmi,
        }
    )

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print(f"\nMean ARI: {df['ARI'].mean():.4f}")
print(f"Mean NMI: {df['NMI'].mean():.4f}")
