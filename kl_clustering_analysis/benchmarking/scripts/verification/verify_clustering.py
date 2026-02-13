"""Verify clustering works correctly with right key."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Generate data
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=50, n_cols=30, n_clusters=3, entropy_param=0.05,
    balanced_clusters=True, random_seed=42,
)
data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
true_labels = np.array([cluster_dict[name] for name in data_df.index])

# Build tree and decompose
Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomposition = tree.decompose(leaf_data=data_df)

# Get clusters using correct key
cluster_assignments = decomposition.get("cluster_assignments", {})

print("="*70)
print("CLUSTERING VERIFICATION")
print("="*70)

print(f"\nData: {data_df.shape}")
print(f"True clusters: {len(np.unique(true_labels))}")
print(f"Found clusters: {len(cluster_assignments)}")

# Build label map from cluster_assignments
label_map = {}
for cl_id, info in cluster_assignments.items():
    for leaf in info["leaves"]:
        label_map[leaf] = cl_id

pred_labels = np.array([label_map.get(name, -1) for name in data_df.index])

# Calculate ARI
ari = adjusted_rand_score(true_labels, pred_labels)
print(f"\nAdjusted Rand Index: {ari:.4f}")

if ari == 1.0:
    print("\n✅ PERFECT CLUSTERING!")
else:
    print(f"\nClustering not perfect. Details:")
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        if true != pred:
            print(f"  Sample {data_df.index[i]}: true={true}, pred={pred}")
