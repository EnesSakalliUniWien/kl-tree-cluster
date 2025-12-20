"""Test the generate_random_feature_matrix function for data separability."""

import numpy as np
import sys
from pathlib import Path

# Ensure project root on path when executed from this subdirectory
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from kl_clustering_analysis.benchmarking.generators import generate_random_feature_matrix
from sklearn.metrics import silhouette_score

print("Testing generate_random_feature_matrix data separability")
print("=" * 60)

for entropy in [0.05, 0.10, 0.15, 0.20, 0.25]:
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=50,
        n_cols=30,
        n_clusters=3,
        entropy_param=entropy,
        balanced_clusters=True,
        random_seed=42,
    )

    X = np.array([data_dict[name] for name in data_dict.keys()])
    labels = np.array([cluster_dict[name] for name in data_dict.keys()])

    # Check data
    sil = silhouette_score(X, labels, metric="hamming")

    # Check within vs between cluster distances
    within = []
    between = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            dist = np.sum(X[i] != X[j]) / len(X[i])
            if labels[i] == labels[j]:
                within.append(dist)
            else:
                between.append(dist)

    print(
        f"entropy={entropy:.2f}: silhouette={sil:.3f}, within={np.mean(within):.3f}, between={np.mean(between):.3f}"
    )
