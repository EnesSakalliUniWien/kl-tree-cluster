"""Debug script for low entropy test cases."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from kl_clustering_analysis.benchmarking.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _jensen_shannon_divergence,
    _sibling_divergence_chi_square_test,
)
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# Generate the failing test case (low entropy)
data_dict, cluster_assignments = generate_random_feature_matrix(
    n_rows=72,
    n_cols=72,
    entropy_param=0.25,
    n_clusters=4,
    random_seed=314,
    balanced_clusters=True,
)

# Convert to matrix
original_names = list(data_dict.keys())
X = np.array([data_dict[name] for name in original_names], dtype=float)
y = np.array([cluster_assignments[name] for name in original_names], dtype=int)

print(f"Data shape: {X.shape}")
print(f"Unique labels: {np.unique(y)}")
print(f"Feature means (first 10): {X.mean(axis=0)[:10]}")
print(f"Feature variance (first 10): {(X.mean(axis=0) * (1 - X.mean(axis=0)))[:10]}")
print()

# Build tree
sample_names = [f"S{i}" for i in range(len(original_names))]
feature_names = [f"F{j}" for j in range(X.shape[1])]
data_df = pd.DataFrame(X, index=sample_names, columns=feature_names)

Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

# Run decompose to populate distributions
from kl_clustering_analysis import config

decomposition = tree.decompose(
    leaf_data=data_df,
    alpha_local=config.ALPHA_LOCAL,
    sibling_alpha=0.05,
)

# Check the root's children distributions
root = tree.root()  # root is a method
children = list(tree.successors(root))
print(f"Root: {root}")
print(f"Children: {children}")

if len(children) == 2:
    left, right = children
    left_dist = tree.nodes[left].get("distribution")
    right_dist = tree.nodes[right].get("distribution")
    parent_dist = tree.nodes[root].get("distribution")

    if left_dist is not None and right_dist is not None:
        print(f"Left dist (first 10): {left_dist[:10]}")
        print(f"Right dist (first 10): {right_dist[:10]}")
        print(f"Diff (first 10): {np.abs(left_dist[:10] - right_dist[:10])}")

        # Calculate JSD
        jsd = _jensen_shannon_divergence(left_dist, right_dist)
        print(f"\nJSD: {jsd:.6f}")

        # Get sample sizes
        left_n = tree.nodes[left].get("leaf_count", 1)
        right_n = tree.nodes[right].get("leaf_count", 1)
        print(f"Sample sizes: left={left_n}, right={right_n}")

        # Run the chi-square test
        jsd, test_stat, df, p_val = _sibling_divergence_chi_square_test(
            left_dist, right_dist, left_n, right_n, parent_dist
        )
        print(f"\nChi-square test:")
        print(f"  JSD: {jsd:.6f}")
        print(f"  Test statistic: {test_stat:.4f}")
        print(f"  Degrees of freedom: {df:.4f}")
        print(f"  P-value: {p_val:.6f}")
        print(f"  Significant at α=0.05? {p_val < 0.05}")

        # Also check what happens with theoretical df
        from scipy.stats import chi2

        p_val_theoretical = chi2.sf(test_stat, df=len(left_dist))
        print(f"\nWith theoretical df={len(left_dist)}:")
        print(f"  P-value: {p_val_theoretical:.6f}")
        print(f"  Significant at α=0.05? {p_val_theoretical < 0.05}")
