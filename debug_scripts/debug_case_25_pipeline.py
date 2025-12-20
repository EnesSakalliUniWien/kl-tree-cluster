"""
Debug Case 25: sparse_features_moderate
- 100 samples, 200 features, 4 clusters
- entropy_param=0.15, feature_sparsity=0.1

Goal: Understand why the algorithm over-splits (finds more clusters than expected)
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

# Get Case 25 config (sparse_features_moderate)
sparse_cases = DEFAULT_TEST_CASES_CONFIG.get("binary_sparse_features", [])
case_config = None
for c in sparse_cases:
    if c.get("name") == "sparse_features_moderate":
        case_config = c
        break

if case_config is None:
    print("Case not found!")
    exit(1)

print("=" * 70)
print("CASE 25: sparse_features_moderate - PIPELINE DEBUG")
print("=" * 70)
print(f"Config: {case_config}")

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
sample_names = sorted(leaf_matrix_dict.keys(), key=lambda x: int(x[1:]))
X = np.array([leaf_matrix_dict[name] for name in sample_names])
y = np.array([cluster_assignments[name] for name in sample_names])

data_df = pd.DataFrame(
    X, index=sample_names, columns=[f"F{j}" for j in range(X.shape[1])]
)

print(f"\nGenerated data: {len(sample_names)} samples, {X.shape[1]} features")
print(f"True cluster distribution: {np.bincount(y)}")

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

# Run decomposition
print("\n" + "=" * 70)
print("STEP 2: DECOMPOSE")
print("=" * 70)

decomp = tree.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=0.01,
)

print(f"\nDecomposition result:")
print(f"  Number of clusters found: {decomp['num_clusters']}")
print(f"  Threshold: {decomp.get('threshold', 'N/A')}")

# Get stats dataframe
stats_df = tree.stats_df

print("\n" + "=" * 70)
print("STEP 3: ANALYZE STATS")
print("=" * 70)

# Show columns available
print(f"Stats columns: {stats_df.columns.tolist()}")

# Check edge significance
if "Child_Parent_Divergence_Significant" in stats_df.columns:
    edge_sig = stats_df[stats_df["Child_Parent_Divergence_Significant"] == True]
    edge_not_sig = stats_df[stats_df["Child_Parent_Divergence_Significant"] == False]
    print(f"\nEdge significant: {len(edge_sig)}")
    print(f"Edge not significant: {len(edge_not_sig)}")

# Check sibling significance
if "Sibling_BH_Different" in stats_df.columns:
    sib_diff = stats_df[stats_df["Sibling_BH_Different"] == True]
    sib_same = stats_df[stats_df["Sibling_BH_Same"] == True]
    print(f"\nSibling different (split): {len(sib_diff)}")
    print(f"Sibling same (merge): {len(sib_same)}")

# Show nodes with their p-values (excluding leaves)
non_leaf = stats_df[stats_df["is_leaf"] == False].copy()
if "Child_Parent_Divergence_P_Value" in non_leaf.columns:
    non_leaf_sorted = non_leaf.sort_values("Child_Parent_Divergence_P_Value")
    print("\nInternal nodes (sorted by edge p-value):")
    cols_to_show = [
        "Child_Parent_Divergence_P_Value",
        "Child_Parent_Divergence_Significant",
    ]
    if "Sibling_Divergence_P_Value" in non_leaf.columns:
        cols_to_show.extend(["Sibling_Divergence_P_Value", "Sibling_BH_Different"])
    print(non_leaf_sorted[cols_to_show].head(20).to_string())

# Calculate ARI
print("\n" + "=" * 70)
print("STEP 4: EVALUATE CLUSTERING")
print("=" * 70)

# Extract predicted labels from decomposition
pred_labels = {}
for cluster_id, info in decomp.get("cluster_assignments", {}).items():
    for leaf in info["leaves"]:
        pred_labels[leaf] = cluster_id

# Calculate ARI
true_labels_list = [cluster_assignments[name] for name in sample_names]
pred_labels_list = [pred_labels.get(name, -1) for name in sample_names]

ari = adjusted_rand_score(true_labels_list, pred_labels_list)
print(f"\nAdjusted Rand Index: {ari:.4f}")
print(f"True clusters: {case_config['n_clusters']}")
print(f"Found clusters: {decomp['num_clusters']}")

# Show cluster sizes
cluster_sizes = {}
for sample, cluster in pred_labels.items():
    cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
print(f"\nPredicted cluster sizes: {dict(sorted(cluster_sizes.items()))}")

# Analyze over-splitting
print("\n" + "=" * 70)
print("ANALYSIS: WHY OVER-SPLITTING?")
print("=" * 70)

# Check sibling p-values for nodes that split same-cluster samples
if "Sibling_Divergence_P_Value" in stats_df.columns:
    # Get sibling test results
    sib_tests = stats_df[stats_df["Sibling_Divergence_P_Value"].notna()].copy()
    print(f"\nSibling tests performed: {len(sib_tests)}")

    # Show distribution of sibling p-values
    print(f"\nSibling p-value distribution:")
    print(f"  Min: {sib_tests['Sibling_Divergence_P_Value'].min():.6f}")
    print(f"  Max: {sib_tests['Sibling_Divergence_P_Value'].max():.6f}")
    print(f"  Mean: {sib_tests['Sibling_Divergence_P_Value'].mean():.6f}")
    print(f"  Median: {sib_tests['Sibling_Divergence_P_Value'].median():.6f}")

    # Count significant vs not at different thresholds
    for alpha in [0.01, 0.05, 0.10]:
        n_sig = (sib_tests["Sibling_Divergence_P_Value"] < alpha).sum()
        print(f"  Significant at α={alpha}: {n_sig}/{len(sib_tests)}")
