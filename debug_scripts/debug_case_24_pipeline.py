"""
Debug Case 24: Run the actual clustering pipeline with verbose output
to understand why it finds only 1 cluster when the data has clear structure.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_cases_config import DEFAULT_TEST_CASES_CONFIG
from kl_clustering_analysis.benchmarking.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.build_tree import build_kl_tree

# Threshold utilities were removed from the package; provide a light fallback for debug scripts
try:
    from kl_clustering_analysis.threshold.thresholding import calculate_threshold  # type: ignore
except Exception:  # pragma: no cover - fallback when threshold package is absent

    def calculate_threshold(
        node_df, significance_level: float = 0.01, verbose: bool = False
    ):
        """Fallback threshold calculator for debugging when threshold utils are removed.

        This simple heuristic returns the quantile of sibling-corrected p-values
        (or 0.5 if unavailable). It's intentionally conservative and suitable for
        debugging workflows; production-grade thresholding has been removed.
        """
        # Prefer sibling-corrected p-values when available, else use edge p-values.
        if "Sibling_Divergence_P_Value_Corrected" in node_df.columns:
            vals = node_df["Sibling_Divergence_P_Value_Corrected"].dropna().values
        elif "edge_p_value" in node_df.columns:
            vals = node_df["edge_p_value"].dropna().values
        else:
            if verbose:
                print("No p-value columns available; returning default threshold 0.5")
            return 0.5

        if len(vals) == 0:
            return 0.5

        # Use the provided significance level as a quantile threshold
        q = float(significance_level)
        thr = float(np.quantile(vals, q))
        if verbose:
            print(f"Fallback threshold (quantile {q}) = {thr}")
        return thr


from kl_clustering_analysis.hierarchy_analysis.edge_significance import (
    annotate_child_parent_divergence,
)

from kl_clustering_analysis.hierarchy_analysis.sibling_divergence_test import (
    annotate_sibling_divergence,
)

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

print(
    f"\nGenerated data: {len(leaf_matrix_dict)} samples, {len(list(leaf_matrix_dict.values())[0])} features"
)

# Step 1: Build tree
print("\n" + "=" * 70)
print("STEP 1: BUILD TREE")
print("=" * 70)

node_df, linkage_df = build_kl_tree(leaf_matrix_dict)
print(f"Tree built with {len(node_df)} nodes")
print(f"Node types: {node_df['node_type'].value_counts().to_dict()}")

# Step 2: Annotate edge significance
print("\n" + "=" * 70)
print("STEP 2: EDGE SIGNIFICANCE (child-parent divergence)")
print("=" * 70)

node_df = annotate_child_parent_divergence(node_df, verbose=True)

# Check which edges are significant
sig_edges = node_df[node_df["edge_significant"] == True]
nonsig_edges = node_df[
    (node_df["edge_significant"] == False) & (node_df["parent_node"].notna())
]

print(f"\nSignificant edges: {len(sig_edges)}")
print(f"Non-significant edges: {len(nonsig_edges)}")

# Show p-values for internal nodes
internal_nodes = node_df[node_df["node_type"] == "internal"].sort_values("edge_p_value")
print("\nInternal nodes edge p-values (sorted):")
for _, row in internal_nodes.head(20).iterrows():
    sig = "SIG" if row["edge_significant"] else "not sig"
    print(f"  {row['node_id']}: p={row['edge_p_value']:.6f} ({sig})")

# Step 3: Sibling divergence test
print("\n" + "=" * 70)
print("STEP 3: SIBLING DIVERGENCE TEST")
print("=" * 70)

node_df = annotate_sibling_divergence(node_df, significance_level=0.01, verbose=True)

# Check sibling significance
sibling_sig = node_df[node_df["sibling_significant"] == True]
sibling_not_sig = node_df[node_df["sibling_significant"] == False]
print(f"\nSibling significant: {len(sibling_sig)}")
print(f"Sibling not significant: {len(sibling_not_sig)}")

# Step 4: Calculate threshold
print("\n" + "=" * 70)
print("STEP 4: THRESHOLD CALCULATION")
print("=" * 70)

# Check raw thresholding decisions
edge_sig_nodes = set(node_df[node_df["edge_significant"] == True]["node_id"].tolist())
sibling_sig_nodes = set(
    node_df[node_df["sibling_significant"] == True]["node_id"].tolist()
)
print(f"\nEdge significant nodes: {len(edge_sig_nodes)}")
print(f"Sibling significant nodes: {len(sibling_sig_nodes)}")

# Find nodes where both tests pass
both_sig = edge_sig_nodes & sibling_sig_nodes
print(f"Nodes with BOTH edge AND sibling significant: {len(both_sig)}")
if both_sig:
    print(f"  Nodes: {sorted(both_sig)[:20]}...")

# Identify cluster-forming nodes
# A node is cluster-forming if it is NOT significant in EITHER test
non_sig_edge = set(
    node_df[(node_df["edge_significant"] == False) & (node_df["node_type"] != "leaf")][
        "node_id"
    ].tolist()
)
non_sig_sibling = set(
    node_df[
        (node_df["sibling_significant"] == False) & (node_df["node_type"] != "leaf")
    ]["node_id"].tolist()
)
print(f"\nNodes with edge NOT significant: {len(non_sig_edge)}")
print(f"Nodes with sibling NOT significant: {len(non_sig_sibling)}")

# Calculate actual threshold
threshold = calculate_threshold(node_df, significance_level=0.01, verbose=True)
print(f"\nFinal threshold: {threshold}")

# Get clusters
from kl_clustering_analysis.hierarchy_analysis.cluster_extraction import (
    extract_clusters_from_tree,
)

clusters_dict = extract_clusters_from_tree(node_df, threshold)
n_clusters_found = len(set(clusters_dict.values()))
print(f"\nClusters found: {n_clusters_found}")

# Check cluster sizes
cluster_sizes = {}
for sample, cluster in clusters_dict.items():
    cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
print(f"Cluster sizes: {dict(sorted(cluster_sizes.items()))}")

# Compare to ground truth
true_labels = cluster_assignments
pred_labels = clusters_dict

from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(
    [true_labels[k] for k in sorted(true_labels.keys())],
    [pred_labels[k] for k in sorted(pred_labels.keys())],
)
print(f"\nAdjusted Rand Index: {ari:.4f}")
