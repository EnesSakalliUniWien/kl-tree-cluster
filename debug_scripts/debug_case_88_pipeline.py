"""
Debug Case 88: Run the actual clustering pipeline with verbose output
to understand clustering behavior on overlapping binary heavy case.

Case 88: overlap_heavy_4c_med_feat
  Category: overlapping_binary_heavy
  Config: n_rows=500, n_cols=200, n_clusters=4, entropy_param=0.4, seed=8001
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Case 88 config
case_config = {
    "name": "overlap_heavy_4c_med_feat",
    "generator": "binary",
    "n_rows": 500,
    "n_cols": 200,
    "n_clusters": 4,
    "entropy_param": 0.4,
    "balanced_clusters": True,
    "seed": 8001,
}

print("=" * 70)
print("CASE 88: overlap_heavy_4c_med_feat - CLUSTERING PIPELINE DEBUG")
print("=" * 70)
print(f"Config: {case_config}")
print(f"\nCurrent config settings:")
print(f"  PROJECTION_EPS: {config.PROJECTION_EPS}")
print(f"  SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"  ALPHA_LOCAL: {config.ALPHA_LOCAL}")

# Generate the data
np.random.seed(case_config["seed"])
leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
    n_rows=case_config["n_rows"],
    n_cols=case_config["n_cols"],
    n_clusters=case_config["n_clusters"],
    entropy_param=case_config["entropy_param"],
    balanced_clusters=case_config.get("balanced_clusters", True),
    random_seed=case_config["seed"],
)

print(
    f"\nGenerated data: {len(leaf_matrix_dict)} samples, {len(list(leaf_matrix_dict.values())[0])} features"
)

# Convert to DataFrame
data_df = pd.DataFrame.from_dict(leaf_matrix_dict, orient="index")

# Step 1: Build tree using PosetTree API
print("\n" + "=" * 70)
print("STEP 1: BUILD TREE")
print("=" * 70)

distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

n_nodes = (
    len(tree.graph) if hasattr(tree.graph, "__len__") else tree.graph.number_of_nodes()
)
print(f"Tree built with {n_nodes} nodes")

# Step 2: Run decomposition
print("\n" + "=" * 70)
print("STEP 2: RUN DECOMPOSITION")
print("=" * 70)

decomp = tree.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=config.SIBLING_ALPHA,
)

n_clusters_found = decomp.get("num_clusters", 0)
cluster_assignments_found = decomp.get("cluster_assignments", {})

print(f"Clusters found: {n_clusters_found}")

# Check cluster sizes from decomposition output
# Format: {cluster_id: {'root_node': ..., 'leaves': [...], 'size': ...}}
print("\nCluster sizes from decomposition:")
for cluster_id, info in cluster_assignments_found.items():
    print(f"  Cluster {cluster_id}: {info['size']} samples (root: {info['root_node']})")

# Convert to sample->cluster mapping for comparison
sample_to_cluster = {}
for cluster_id, info in cluster_assignments_found.items():
    for leaf in info["leaves"]:
        sample_to_cluster[leaf] = cluster_id

# Step 3: Examine stats_df for edge/sibling significance
print("\n" + "=" * 70)
print("STEP 3: SIGNIFICANCE DETAILS")
print("=" * 70)

stats_df = tree.stats_df
if stats_df is not None and not stats_df.empty:
    print(f"Stats DataFrame shape: {stats_df.shape}")
    print(f"Columns: {list(stats_df.columns)}")

    # Edge significance stats
    if "edge_significant" in stats_df.columns:
        sig_edges = stats_df["edge_significant"].sum()
        nonsig_edges = (~stats_df["edge_significant"]).sum()
        print(f"Significant edges: {sig_edges}")
        print(f"Non-significant edges: {nonsig_edges}")

    # Sibling significance stats
    if "sibling_significant" in stats_df.columns:
        sibling_sig = stats_df["sibling_significant"].sum()
        sibling_not_sig = (~stats_df["sibling_significant"]).sum()
        print(f"Sibling significant: {sibling_sig}")
        print(f"Sibling not significant: {sibling_not_sig}")

    # Show top edge p-values
    if "edge_p_value" in stats_df.columns and "node_type" in stats_df.columns:
        internal = stats_df[stats_df["node_type"] == "internal"].sort_values(
            "edge_p_value"
        )
        print("\nTop 10 internal nodes by edge p-value:")
        for _, row in internal.head(10).iterrows():
            sig = "SIG" if row.get("edge_significant", False) else "not sig"
            print(f"  {row['node_id']}: p={row['edge_p_value']:.6f} ({sig})")
else:
    print("No stats_df available")

# Compare to ground truth
print("\n" + "=" * 70)
print("STEP 4: COMPARISON TO GROUND TRUTH")
print("=" * 70)

true_labels = cluster_assignments
pred_labels = sample_to_cluster

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Make sure we have matching keys
common_keys = sorted(set(true_labels.keys()) & set(pred_labels.keys()))
if common_keys:
    ari = adjusted_rand_score(
        [true_labels[k] for k in common_keys],
        [pred_labels[k] for k in common_keys],
    )
    nmi = normalized_mutual_info_score(
        [true_labels[k] for k in common_keys],
        [pred_labels[k] for k in common_keys],
    )
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Info: {nmi:.4f}")
else:
    print("No common keys between ground truth and predictions")

# Show ground truth cluster sizes
print("\nGround truth cluster distribution:")
gt_sizes = {}
for sample, cluster in true_labels.items():
    gt_sizes[cluster] = gt_sizes.get(cluster, 0) + 1
print(f"  {dict(sorted(gt_sizes.items()))}")
