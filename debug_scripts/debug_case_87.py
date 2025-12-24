"""
Debug Case 87: Why does the algorithm find only 1 cluster instead of 4?

Case 87: 500 samples, 50 features (more samples than features)
This is the problematic case where everything collapses into 1 cluster.
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
print("DEBUG CASE 87: overlap_heavy_4c_small_feat")
print("=" * 70)
print(
    f"Config: {case['n_rows']} samples, {case['n_cols']} features, {case['n_clusters']} clusters"
)
print(f"Ratio n_samples/n_features = {case['n_rows'] / case['n_cols']:.1f}")
print()

print("Current config settings:")
print(f"  PROJECTION_EPS: {config.PROJECTION_EPS}")
print(f"  SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"  ALPHA_LOCAL: {config.ALPHA_LOCAL}")
print(f"  USE_RANDOM_PROJECTION: {config.USE_RANDOM_PROJECTION}")
print(
    "  PROJECTION_DECISION: JL-based (projection if n_features > n_samples and JL k < n_features)"
)
print()

# Generate data
np.random.seed(case["seed"])
leaf_matrix_dict, true_labels = generate_random_feature_matrix(
    n_rows=case["n_rows"],
    n_cols=case["n_cols"],
    n_clusters=case["n_clusters"],
    entropy_param=case["entropy_param"],
    balanced_clusters=True,
    random_seed=case["seed"],
)

data_df = pd.DataFrame.from_dict(leaf_matrix_dict, orient="index")
print(f"Data shape: {data_df.shape}")

# Build tree
distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

# Run decomposition
decomp = tree.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=config.SIBLING_ALPHA,
)

n_found = decomp.get("num_clusters", 0)
print(f"\nResult: Found {n_found} clusters (expected {case['n_clusters']})")

# Analyze stats_df
stats_df = tree.stats_df

print("\n" + "=" * 70)
print("EDGE SIGNIFICANCE ANALYSIS")
print("=" * 70)

if "Child_Parent_Divergence_P_Value" in stats_df.columns:
    internal = stats_df[stats_df["is_leaf"] == False].copy()
    internal = internal.sort_values("Child_Parent_Divergence_P_Value")

    n_sig = internal["Child_Parent_Divergence_Significant"].sum()
    n_total = len(internal)
    print(f"Significant edges: {n_sig} / {n_total} ({100 * n_sig / n_total:.1f}%)")

    # Check BH-corrected p-values
    if "Child_Parent_Divergence_P_Value_BH" in stats_df.columns:
        print("\nTop 15 internal nodes by edge p-value (with BH-corrected):")
        for i, (node_id, row) in enumerate(internal.head(15).iterrows()):
            sig = (
                "SIG"
                if row.get("Child_Parent_Divergence_Significant", False)
                else "not sig"
            )
            df = row.get("Child_Parent_Divergence_df", "?")
            raw_p = row["Child_Parent_Divergence_P_Value"]
            bh_p = row["Child_Parent_Divergence_P_Value_BH"]
            print(f"  {node_id}: raw_p={raw_p:.2e}, bh_p={bh_p:.4f}, df={df} ({sig})")
    else:
        print("\nTop 15 internal nodes by edge p-value:")
        for i, (node_id, row) in enumerate(internal.head(15).iterrows()):
            sig = (
                "SIG"
                if row.get("Child_Parent_Divergence_Significant", False)
                else "not sig"
            )
            df = row.get("Child_Parent_Divergence_df", "?")
            print(
                f"  {node_id}: p={row['Child_Parent_Divergence_P_Value']:.6f}, df={df} ({sig})"
            )

    print("\nP-value distribution:")
    pvals = internal["Child_Parent_Divergence_P_Value"].dropna()
    print(f"  min: {pvals.min():.6f}")
    print(f"  25%: {pvals.quantile(0.25):.6f}")
    print(f"  50%: {pvals.quantile(0.50):.6f}")
    print(f"  75%: {pvals.quantile(0.75):.6f}")
    print(f"  max: {pvals.max():.6f}")

    # Show BH p-value distribution too
    if "Child_Parent_Divergence_P_Value_BH" in stats_df.columns:
        bh_pvals = internal["Child_Parent_Divergence_P_Value_BH"].dropna()
        print("\nBH-corrected p-value distribution:")
        print(f"  min: {bh_pvals.min():.6f}")
        print(f"  25%: {bh_pvals.quantile(0.25):.6f}")
        print(f"  50%: {bh_pvals.quantile(0.50):.6f}")
        print(f"  75%: {bh_pvals.quantile(0.75):.6f}")
        print(f"  max: {bh_pvals.max():.6f}")
else:
    print("No Child_Parent_Divergence_P_Value column found")

print("\n" + "=" * 70)
print("SIBLING DIVERGENCE ANALYSIS")
print("=" * 70)

if "Sibling_Divergence_P_Value" in stats_df.columns:
    has_sibling = stats_df[stats_df["Sibling_Divergence_P_Value"].notna()].copy()
    has_sibling = has_sibling.sort_values("Sibling_Divergence_P_Value")

    if "Sibling_BH_Different" in stats_df.columns:
        n_sig = has_sibling["Sibling_BH_Different"].sum()
        n_total = len(has_sibling)
        print(f"Sibling significant (different): {n_sig} / {n_total}")

    print("\nTop 15 nodes by sibling p-value:")
    for i, (node_id, row) in enumerate(has_sibling.head(15).iterrows()):
        sig = "DIFF" if row.get("Sibling_BH_Different", False) else "same"
        df = row.get("Sibling_Degrees_of_Freedom", "?")
        stat = row.get("Sibling_Test_Statistic", "?")
        print(
            f"  {node_id}: p={row['Sibling_Divergence_P_Value']:.6f}, stat={stat:.2f}, df={df} ({sig})"
        )

    print("\nSibling p-value distribution:")
    pvals = has_sibling["Sibling_Divergence_P_Value"].dropna()
    if len(pvals) > 0:
        print(f"  min: {pvals.min():.6f}")
        print(f"  25%: {pvals.quantile(0.25):.6f}")
        print(f"  50%: {pvals.quantile(0.50):.6f}")
        print(f"  75%: {pvals.quantile(0.75):.6f}")
        print(f"  max: {pvals.max():.6f}")
else:
    print("No Sibling_Divergence_P_Value column found")

# Show ground truth distribution
print("\n" + "=" * 70)
print("GROUND TRUTH")
print("=" * 70)
gt_sizes = {}
for sample, cluster in true_labels.items():
    gt_sizes[cluster] = gt_sizes.get(cluster, 0) + 1
print(f"Ground truth cluster sizes: {dict(sorted(gt_sizes.items()))}")
