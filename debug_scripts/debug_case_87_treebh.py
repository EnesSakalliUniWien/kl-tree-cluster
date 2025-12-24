"""
Debug Case 87: Investigate why TreeBH correction is too aggressive.

This script analyzes the tree structure and shows why only 6 nodes
are marked significant despite many having tiny raw p-values.
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
    "name": "overlap_heavy_4c_small_feat",
    "n_rows": 500,
    "n_cols": 50,
    "n_clusters": 4,
    "entropy_param": 0.4,
    "seed": 8000,
}

print("=" * 70)
print("DEBUG CASE 87: TreeBH Correction Analysis")
print("=" * 70)
print(
    f"Config: {case['n_rows']} samples, {case['n_cols']} features, {case['n_clusters']} clusters"
)
print()

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

# PosetTree inherits from nx.DiGraph, so tree itself is the graph
import networkx as nx

g = tree  # tree IS the networkx graph

print("=" * 70)
print("SIGNIFICANT NODES ANALYSIS")
print("=" * 70)

internal = stats_df[stats_df["is_leaf"] == False].copy()
sig_nodes = internal[internal["Child_Parent_Divergence_Significant"] == True]

print(f"\nTotal internal nodes: {len(internal)}")
print(f"Significant nodes: {len(sig_nodes)}")

print("\nSignificant nodes details:")
for node_id, row in sig_nodes.iterrows():
    parents = list(g.predecessors(node_id))
    children = list(g.successors(node_id))
    print(f"  {node_id}:")
    print(f"    raw_p={row['Child_Parent_Divergence_P_Value']:.2e}")
    print(f"    bh_p={row['Child_Parent_Divergence_P_Value_BH']:.4f}")
    print(f"    leaf_count={row['leaf_count']}")
    print(f"    parent={parents[0] if parents else 'ROOT'}")
    print(f"    children={children}")

print("\n" + "=" * 70)
print("TREE STRUCTURE NEAR ROOT")
print("=" * 70)

# Find root
roots = [n for n in g.nodes() if g.in_degree(n) == 0]
print(f"\nRoot nodes: {roots}")


# Show hierarchy from root down a few levels
def show_tree_level(node, level=0, max_level=4):
    if level > max_level:
        return
    indent = "  " * level

    if node in stats_df.index:
        row = stats_df.loc[node]
        is_leaf = row.get("is_leaf", True)
        if not is_leaf:
            raw_p = row.get("Child_Parent_Divergence_P_Value", float("nan"))
            bh_p = row.get("Child_Parent_Divergence_P_Value_BH", float("nan"))
            sig = row.get("Child_Parent_Divergence_Significant", False)
            leaf_count = row.get("leaf_count", "?")
            sig_str = "SIG" if sig else "not sig"
            print(
                f"{indent}{node}: leaves={leaf_count}, raw_p={raw_p:.2e}, bh_p={bh_p:.4f} ({sig_str})"
            )
        else:
            print(f"{indent}{node}: LEAF")
    else:
        print(f"{indent}{node}: (no stats)")

    children = list(g.successors(node))
    for child in children[:5]:  # Limit to first 5 children
        show_tree_level(child, level + 1, max_level)
    if len(children) > 5:
        print(f"{indent}  ... and {len(children) - 5} more children")


for root in roots:
    print(f"\nTree from {root}:")
    show_tree_level(root)

print("\n" + "=" * 70)
print("P-VALUE COMPARISON: RAW vs BH-CORRECTED")
print("=" * 70)

# Show nodes where raw p-value is tiny but BH says not significant
internal_sorted = internal.sort_values("Child_Parent_Divergence_P_Value")
print("\nNodes with tiny raw p-value but NOT significant after BH:")
count = 0
for node_id, row in internal_sorted.iterrows():
    raw_p = row["Child_Parent_Divergence_P_Value"]
    bh_p = row["Child_Parent_Divergence_P_Value_BH"]
    sig = row["Child_Parent_Divergence_Significant"]

    if raw_p < 1e-10 and not sig:
        parent = list(g.predecessors(node_id))
        parent_str = parent[0] if parent else "ROOT"
        print(f"  {node_id}: raw_p={raw_p:.2e}, bh_p={bh_p:.4f}, parent={parent_str}")
        count += 1
        if count >= 20:
            print("  ... (showing first 20)")
            break

# Count how many have raw_p < 0.05
n_raw_sig = (internal["Child_Parent_Divergence_P_Value"] < 0.05).sum()
print(f"\nNodes with raw p < 0.05: {n_raw_sig}")
print(f"Nodes with BH-corrected significant: {len(sig_nodes)}")
