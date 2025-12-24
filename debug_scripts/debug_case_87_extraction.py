"""
Debug Case 87: Why is the algorithm returning 1 cluster when TreeBH correctly identifies 4 main splits?

The edge significance test correctly identifies:
- N996, N997: Significant splits from root
- N991, N995, N982, N994: The 4 main cluster subtrees

But somehow the final result is 1 cluster. Let's investigate the cluster extraction logic.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Case 87 configuration
case = {
    "n_rows": 500,
    "n_cols": 50,
    "n_clusters": 4,
    "entropy_param": 0.4,
    "seed": 8000,
}

print("=" * 70)
print("DEBUG: Cluster Extraction Logic")
print("=" * 70)

# Generate data
np.random.seed(case["seed"])
leaf_matrix_dict, _ = generate_random_feature_matrix(
    n_rows=case["n_rows"],
    n_cols=case["n_cols"],
    n_clusters=case["n_clusters"],
    entropy_param=case["entropy_param"],
    balanced_clusters=True,
    random_seed=case["seed"],
)

data_df = pd.DataFrame.from_dict(leaf_matrix_dict, orient="index")
distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

# Run decomposition
decomp = tree.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=config.SIBLING_ALPHA,
)

stats_df = tree.stats_df

print("\nDecomposition result:")
print(f"  num_clusters: {decomp.get('num_clusters')}")

# Check the sibling test results
print("\n" + "=" * 70)
print("SIBLING TEST RESULTS")
print("=" * 70)

sibling_cols = [c for c in stats_df.columns if "Sibling" in c]
print(f"Sibling columns: {sibling_cols}")

# Look at the critical nodes: N996, N997, N991, N995, N982, N994
critical_nodes = ["N998", "N996", "N997", "N991", "N995", "N982", "N994"]

print("\nCritical node sibling test results:")
for node in critical_nodes:
    if node in stats_df.index:
        row = stats_df.loc[node]
        print(f"\n{node}:")
        for col in sibling_cols:
            if col in row.index:
                val = row[col]
                print(f"  {col}: {val}")

        # Also show edge significance
        edge_sig = row.get("Child_Parent_Divergence_Significant", "N/A")
        edge_p = row.get("Child_Parent_Divergence_P_Value", "N/A")
        print(f"  Edge Significant: {edge_sig}")
        print(
            f"  Edge raw p-value: {edge_p:.2e}"
            if isinstance(edge_p, float)
            else f"  Edge raw p-value: {edge_p}"
        )

# Check what determines cluster boundaries
print("\n" + "=" * 70)
print("CLUSTER BOUNDARY ANALYSIS")
print("=" * 70)

# Look at the independence_analysis from decomposition
if "independence_analysis" in decomp:
    print("\nIndependence analysis:")
    for key, val in decomp["independence_analysis"].items():
        if isinstance(val, (int, float, str, bool)):
            print(f"  {key}: {val}")
        elif isinstance(val, dict) and len(val) < 10:
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: (complex data, type={type(val).__name__})")

# Now trace the _should_split logic manually
print("\n" + "=" * 70)
print("MANUAL _should_split TRACE FOR ROOT")
print("=" * 70)

from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict

# Get the data structures
_local_significant = extract_bool_column_dict(
    stats_df, "Child_Parent_Divergence_Significant"
)
_sibling_different = extract_bool_column_dict(stats_df, "Sibling_BH_Different")

root = "N998"
children = list(tree.successors(root))
print(f"\nRoot: {root}")
print(f"Children: {children}")

if len(children) == 2:
    left_child, right_child = children

    # Gate 2: Sibling divergence
    is_different = _sibling_different.get(root)
    print(f"\nGate 2 (Sibling different): {is_different}")

    # Gate 3: Local KL divergence
    left_diverges = _local_significant.get(left_child)
    right_diverges = _local_significant.get(right_child)
    print(f"Gate 3 (Left {left_child} diverges): {left_diverges}")
    print(f"Gate 3 (Right {right_child} diverges): {right_diverges}")

    should_split = is_different and (left_diverges or right_diverges)
    print(f"\nFinal _should_split result: {should_split}")

    # Check what's in the DataFrame
    print(f"\nDataFrame values for {left_child}:")
    if left_child in stats_df.index:
        row = stats_df.loc[left_child]
        print(
            f"  Child_Parent_Divergence_Significant: {row.get('Child_Parent_Divergence_Significant', 'MISSING')}"
        )
        print(
            f"  Edge p-value: {row.get('Child_Parent_Divergence_P_Value', 'MISSING')}"
        )
        print(
            f"  Edge BH p-value: {row.get('Child_Parent_Divergence_P_Value_BH', 'MISSING')}"
        )

    print(f"\nDataFrame values for {right_child}:")
    if right_child in stats_df.index:
        row = stats_df.loc[right_child]
        print(
            f"  Child_Parent_Divergence_Significant: {row.get('Child_Parent_Divergence_Significant', 'MISSING')}"
        )
        print(
            f"  Edge p-value: {row.get('Child_Parent_Divergence_P_Value', 'MISSING')}"
        )
        print(
            f"  Edge BH p-value: {row.get('Child_Parent_Divergence_P_Value_BH', 'MISSING')}"
        )

# Let's run decomposition WITHOUT posthoc merge
print("\n" + "=" * 70)
print("DECOMPOSITION WITHOUT POSTHOC MERGE")
print("=" * 70)

# Recreate tree
tree2 = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomp2 = tree2.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=config.SIBLING_ALPHA,
    posthoc_merge=False,  # Disable posthoc merge
)

print(f"Without posthoc_merge: {decomp2.get('num_clusters')} clusters")
print(f"With posthoc_merge: {decomp.get('num_clusters')} clusters")

if decomp2.get("num_clusters", 0) != decomp.get("num_clusters", 0):
    print("\n*** POSTHOC MERGE IS CAUSING THE ISSUE! ***")

    # Show cluster sizes without posthoc merge
    assignments = decomp2.get("cluster_assignments", {})
    print(f"\nCluster sizes without posthoc_merge:")
    for cid, info in assignments.items():
        print(f"  Cluster {cid}: {info['size']} samples (root: {info['root_node']})")
# Debug the posthoc merge process
print("\n" + "=" * 70)
print("POSTHOC MERGE DEBUGGING")
print("=" * 70)

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    compute_sibling_divergence,
)

# Get the cluster roots from decomp2 (without posthoc merge)
cluster_roots_before = {
    info["root_node"] for info in decomp2.get("cluster_assignments", {}).values()
}
print(f"\nCluster roots before posthoc merge: {cluster_roots_before}")

# Now trace what the posthoc merge test_divergence function returns
# Looking at key cluster pairs
print("\nTesting divergence between cluster pairs:")


# Get leaf distributions
def get_cluster_distribution(root_node, tree, data_df):
    """Get aggregated distribution for a cluster."""
    # Get all leaves under this node
    descendants = list(tree.leaves_under(root_node))
    if not descendants:
        # The node itself might be a leaf
        if root_node.startswith("L"):
            descendants = [root_node]
        else:
            return None

    # Convert leaf names to original sample names
    sample_names = []
    for d in descendants:
        # Leaf names might be like 'L1', 'L2', etc.
        if d in data_df.index:
            sample_names.append(d)
        else:
            # Try to find matching index
            for idx in data_df.index:
                if str(idx) == d or f"L{idx}" == d:
                    sample_names.append(idx)
                    break

    if not sample_names:
        return None

    return data_df.loc[sample_names].mean(axis=0).values


# Compare pairs of clusters
cluster_list = list(cluster_roots_before)
print(f"\nComparing cluster pairs (using JSD):")
for i in range(min(3, len(cluster_list))):
    for j in range(i + 1, min(4, len(cluster_list))):
        c1, c2 = cluster_list[i], cluster_list[j]
        print(f"\n  {c1} vs {c2}:")

        # Get sizes
        size1 = tree2.leaf_count(c1)
        size2 = tree2.leaf_count(c2)
        print(f"    Sizes: {size1} vs {size2}")
