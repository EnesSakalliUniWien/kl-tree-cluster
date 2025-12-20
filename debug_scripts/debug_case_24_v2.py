"""
Debug Case 24: Run the actual clustering pipeline with verbose output
to understand why it finds only 1 cluster when the data has clear structure.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from tests.test_cases_config import DEFAULT_TEST_CASES_CONFIG
from kl_clustering_analysis.benchmarking.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Get Case 24 config
sparse_cases = DEFAULT_TEST_CASES_CONFIG.get("binary_sparse_features", [])
case_config = None
for c in sparse_cases:
    if c.get("name") == "sparse_features_moderate":
        case_config = c
        break

print("=" * 70)
print("CASE 24: sparse_features_moderate - CLUSTERING PIPELINE DEBUG")
print("=" * 70)
print(f"Config: {case_config}")
print(f"\nCurrent config settings:")
print(f"  SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"  ALPHA_LOCAL: {config.ALPHA_LOCAL}")
print(f"  TREE_DISTANCE_METRIC: {config.TREE_DISTANCE_METRIC}")
print(f"  TREE_LINKAGE_METHOD: {config.TREE_LINKAGE_METHOD}")

# Generate the data
np.random.seed(case_config["seed"])
leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
    n_rows=case_config["n_rows"],
    n_cols=case_config["n_cols"],
    n_clusters=case_config["n_clusters"],
    entropy_param=case_config["entropy_param"],
    feature_sparsity=case_config.get("feature_sparsity"),
    balanced_clusters=case_config.get("balanced_clusters", True),
    random_seed=case_config["seed"],
)

# Convert to DataFrame
sample_names = list(leaf_matrix_dict.keys())
matrix = np.array([leaf_matrix_dict[name] for name in sample_names], dtype=int)
feature_names = [f"F{j}" for j in range(matrix.shape[1])]
data_df = pd.DataFrame(matrix, index=sample_names, columns=feature_names)

print(f"\nGenerated data: {data_df.shape[0]} samples, {data_df.shape[1]} features")

# True labels
true_labels = np.array([cluster_assignments[name] for name in sample_names])
print(f"True clusters: {np.bincount(true_labels)}")

# Build tree
print("\n" + "=" * 70)
print("STEP 1: BUILD TREE")
print("=" * 70)

Z = linkage(
    pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
    method=config.TREE_LINKAGE_METHOD,
)
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
print(f"Tree built successfully")

# Decompose
print("\n" + "=" * 70)
print("STEP 2: DECOMPOSE")
print("=" * 70)

decomp = tree.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=0.01,  # Using 0.01 as in notebook
)

print(f"\nDecomposition result:")
print(f"  Number of clusters: {decomp['num_clusters']}")
print(f"  Keys: {decomp.keys()}")

# Get stats_df for inspection
stats_df = tree.stats_df
print(f"\nStats dataframe has {len(stats_df)} rows")
print(f"Columns: {stats_df.columns.tolist()}")

# Extract predicted labels
pred_labels = {}
for cluster_id, info in decomp.get("cluster_assignments", {}).items():
    for leaf in info["leaves"]:
        pred_labels[leaf] = cluster_id

# Calculate ARI
if pred_labels:
    pred_array = np.array([pred_labels.get(name, -1) for name in sample_names])
    ari = adjusted_rand_score(true_labels, pred_array)
    print(f"\n" + "=" * 70)
    print(f"FINAL RESULT: ARI = {ari:.4f}")
    print(f"  True clusters: {case_config['n_clusters']}")
    print(f"  Found clusters: {decomp['num_clusters']}")
    print("=" * 70)
else:
    print("\nNo clusters found!")
