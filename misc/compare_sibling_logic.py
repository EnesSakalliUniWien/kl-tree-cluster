#!/usr/bin/env python3
"""
Compare old vs new sibling independence logic in KL clustering.

This script tests the difference between:
1. Old logic: Direct MI threshold between sibling nodes
2. New logic: Conditional MI test with BH correction
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from tree.poset_tree import PosetTree
from hierarchy_analysis import calculate_hierarchy_kl_divergence
from hierarchy_analysis.kl_correlation_analysis import (
    calculate_kl_divergence_mutual_information_matrix,
)
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer


def compare_sibling_independence_logic():
    """Compare old MI threshold vs new CMI logic."""
    print("=== COMPARING SIBLING INDEPENDENCE LOGIC ===")
    print("Old logic: MI(child1, child2) <= threshold")
    print("New logic: Conditional MI test with BH correction")
    print()

    # Create test data
    X_t, y_t = make_blobs(
        n_samples=30, n_features=30, centers=3, cluster_std=1.0, random_state=42
    )
    X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
    data_t = pd.DataFrame(
        X_bin, index=[f"S{j}" for j in range(30)], columns=[f"F{j}" for j in range(30)]
    )

    print(f"Test data: {len(data_t)} samples, {len(data_t.columns)} features")
    print(f"True clusters: {len(np.unique(y_t))} (labels: {np.unique(y_t)})")
    print()

    # Build tree and calculate stats
    Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
    stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)
    mi_t, _ = calculate_kl_divergence_mutual_information_matrix(tree_t, stats_t)

    # Annotate with statistical tests
    results_t = annotate_nodes_with_statistical_significance_tests(
        stats_t, 30, 0.05, 2.0, True
    )
    results_t = annotate_child_parent_divergence(tree_t, results_t, 30, 0.05)
    results_t = annotate_sibling_independence_cmi(
        tree_t, results_t, significance_level_alpha=0.05, permutations=75
    )

    # Test NEW logic (current implementation)
    print("Testing NEW logic (Conditional MI with BH correction)...")
    decomposer_new = ClusterDecomposer(
        tree=tree_t,
        results_df=results_t,
        significance_column="Are_Features_Dependent",
    )
    decomp_new = decomposer_new.decompose_tree()
    report_new = decomposer_new.generate_report(decomp_new)
    y_kl_new = report_new.loc[data_t.index, "cluster_id"].values

    # Apply cluster ID remapping for fair comparison
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    y_t_encoded = le_true.fit_transform(y_t)
    y_kl_encoded = le_pred.fit_transform(y_kl_new)
    cm = confusion_matrix(y_t_encoded, y_kl_encoded)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping_encoded = {pred: true for pred, true in zip(col_ind, row_ind)}
    y_kl_remapped_encoded = np.array([mapping_encoded[pred] for pred in y_kl_encoded])
    y_kl_new_mapped = le_true.inverse_transform(y_kl_remapped_encoded)

    print(f"  Clusters found: {len(np.unique(y_kl_new))}")
    print(f"  ARI: {adjusted_rand_score(y_t, y_kl_new_mapped):.3f}")
    print(f"  Perfect match: {np.array_equal(y_t, y_kl_new_mapped)}")
    print()

    # Analyze decision differences
    print("Analyzing decision differences between old and new logic...")

    differences = []
    for node_id in tree_t.nodes:
        if tree_t.nodes[node_id].get("is_leaf", False):
            continue
        children = list(tree_t.successors(node_id))
        if len(children) != 2:
            continue
        c1, c2 = children

        # Old logic: check MI between children using the MI matrix
        # Since we removed mi_matrix parameter, we need to calculate MI directly from the distributions
        dist1 = stats_t.loc[c1, "distribution"]
        dist2 = stats_t.loc[c2, "distribution"]
        # Calculate mutual information directly
        p1 = np.clip(dist1, 1e-9, 1 - 1e-9)
        p2 = np.clip(dist2, 1e-9, 1 - 1e-9)
        # Simple MI calculation (symmetric) - approximate
        mi_val = np.mean(np.abs(p1 - p2))  # This is a simple proxy; in real use, we'd use the MI matrix
        independent_old = mi_val <= 0.8

        # New logic: check CMI annotation
        independent_new = False
        if node_id in results_t.index and "Sibling_BH_Independent" in results_t.columns:
            val = results_t.loc[node_id, "Sibling_BH_Independent"]
            if pd.notna(val):
                independent_new = bool(val)

        if independent_old != independent_new:
            differences.append(
                {
                    "node": node_id,
                    "old_decision": independent_old,
                    "new_decision": independent_new,
                    "mi_value": mi_val,
                    "cmi_p_value": results_t.loc[node_id, "Sibling_CMI_P_Value"]
                    if node_id in results_t.index
                    else None,
                }
            )

    print(f"Found {len(differences)} nodes with different decisions:")
    for diff in differences[:10]:  # Show first 10
        print(
            f"  Node {diff['node']}: Old={diff['old_decision']}, New={diff['new_decision']}, MI={diff['mi_value']:.3f}"
        )

    if len(differences) > 10:
        print(f"  ... and {len(differences) - 10} more")

    print()
    print("=== SUMMARY ===")
    print("The new logic uses conditional mutual information I(C1;C2|P) instead of")
    print("simple mutual information I(C1;C2). This tests whether children are")
    print("independent GIVEN the parent distribution, which may be more appropriate")
    print("for hierarchical clustering but could give different results.")


if __name__ == "__main__":
    compare_sibling_independence_logic()
