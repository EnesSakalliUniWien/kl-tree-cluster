#!/usr/bin/env python3
"""Diagnose why feature_matrix_go_terms finds K=1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# Print current config
print(f"SIBLING_TEST_METHOD: {config.SIBLING_TEST_METHOD}")
print(f"EDGE_ALPHA: {config.EDGE_ALPHA}")
print(f"SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"SPECTRAL_METHOD: {config.SPECTRAL_METHOD}")
print(f"PROJECTION_MINIMUM_DIMENSION: {config.PROJECTION_MINIMUM_DIMENSION}")
print(f"POSTHOC_MERGE: {config.POSTHOC_MERGE}")
print(f"FELSENSTEIN_SCALING: {config.FELSENSTEIN_SCALING}")
print()

# Load data
data = pd.read_csv("feature_matrix.tsv", sep="\t", index_col=0)
print(f"Data shape: {data.shape}")
print(f"Sparsity: {(data.values == 0).mean():.3f}")
print(
    f"Row sums (min/med/max): {data.sum(axis=1).min()} / {data.sum(axis=1).median():.0f} / {data.sum(axis=1).max()}"
)
print(
    f"Col sums (min/med/max): {data.sum(axis=0).min()} / {data.sum(axis=0).median():.0f} / {data.sum(axis=0).max()}"
)
print()

# Build tree
D = pdist(data.values, metric=config.TREE_DISTANCE_METRIC)
Z = linkage(D, method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
root = [n for n in tree.nodes() if tree.in_degree(n) == 0][0]
print(f"Tree: {tree.number_of_nodes()} nodes, root={root}")

# Decompose
results = tree.decompose(
    leaf_data=data,
    alpha_local=config.EDGE_ALPHA,
    sibling_alpha=config.SIBLING_ALPHA,
)
K = results["num_clusters"]
print(f"\n*** Clusters found: {K} ***\n")

# Examine annotations
annotations = tree.annotations_df
root_children = list(tree.successors(root))
print(f"Root children: {root_children}")

# Gate columns
gate_cols = [
    "Child_Parent_Divergence_Significant",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_df",
    "Sibling_BH_Different",
    "Sibling_BH_Same",
    "Sibling_Divergence_Skipped",
    "Sibling_Test_Statistic",
    "Sibling_Degrees_of_Freedom",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_Test_Method",
]

for node in root_children + [root]:
    if node in annotations.index:
        print(f"\n--- {node} ---")
        for col in gate_cols:
            if col in annotations.columns:
                print(f"  {col}: {annotations.loc[node, col]}")

# Summary counts
if "Child_Parent_Divergence_Significant" in annotations.columns:
    print(
        f"\nEdge-significant nodes: {annotations['Child_Parent_Divergence_Significant'].sum()} / {len(annotations)}"
    )
if "Sibling_BH_Different" in annotations.columns:
    print(f"Sibling different: {annotations['Sibling_BH_Different'].sum()}")
    if "Sibling_BH_Same" in annotations.columns:
        print(f"Sibling same: {annotations['Sibling_BH_Same'].sum()}")
    if "Sibling_Divergence_Skipped" in annotations.columns:
        print(f"Sibling skipped: {annotations['Sibling_Divergence_Skipped'].sum()}")

# Calibration audit
audit = annotations.attrs.get("sibling_divergence_audit", {})
if audit:
    print("\nCalibration audit:")
    for k, v in audit.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
else:
    print("\nNo calibration audit found in annotations_df.attrs")

# Top sibling tests
print("\n--- Top 20 sibling tests by raw p-value ---")
sib = annotations.dropna(subset=["Sibling_Test_Statistic"]).sort_values("Sibling_Divergence_P_Value")
cols = [
    c
    for c in [
        "Sibling_Test_Statistic",
        "Sibling_Degrees_of_Freedom",
        "Sibling_Divergence_P_Value",
        "Sibling_Divergence_P_Value_Corrected",
        "Sibling_BH_Different",
        "Sibling_Test_Method",
    ]
    if c in sib.columns
]
print(sib[cols].head(20).to_string())
