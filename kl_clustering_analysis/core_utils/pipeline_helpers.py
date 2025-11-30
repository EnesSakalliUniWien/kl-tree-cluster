#!/usr/bin/env python3
"""
Utility functions for building and analyzing hierarchical trees for KL divergence clustering.

This module provides functions to generate synthetic data, build hierarchical trees, and run
statistical analysis. It is used by tests and notebooks.
"""

import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_root_node_significance,
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)
from kl_clustering_analysis import config


def create_test_case_data(
    n_samples=50, n_features=20, n_clusters=3, noise_level=1.0, seed=42
):
    """Create synthetic test data using the same method as the validation suite."""
    from sklearn.datasets import make_blobs
    import numpy as np

    print(
        f"Generating test data: {n_samples} samples, {n_features} features, {n_clusters} clusters"
    )

    # Use make_blobs like the validation suite, then binarize
    X_continuous, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=noise_level,
        random_state=seed,
    )

    # Binarize by median threshold (same as validation suite)
    X_binary = (X_continuous > np.median(X_continuous, axis=0)).astype(int)

    # Convert to DataFrame
    X = pd.DataFrame(
        X_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    y_true = pd.Series(y_true)

    print(f"Generated data shape: {X.shape}")
    print(f"True cluster sizes: {y_true.value_counts().sort_index()}")

    return X, y_true


def build_hierarchical_tree(X, linkage_method="complete", distance_metric="hamming"):
    """Build hierarchical tree from feature matrix using scipy linkage."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    print("Building hierarchical tree...")

    # Compute pairwise distances
    distance_matrix = pdist(X.values, metric=distance_metric)

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=linkage_method)

    # Create PosetTree from linkage matrix
    tree = PosetTree.from_linkage(linkage_matrix, X.index.tolist())

    print(f"Tree built with {len(tree.nodes())} nodes")

    return tree, linkage_matrix


def run_statistical_analysis(tree, X):
    """Run statistical tests on the hierarchical tree."""
    print("Running statistical analysis...")

    # Calculate KL divergence statistics
    tree.populate_node_divergences(X)

    # Run statistical tests (skip root-level test for speed/consistency)
    results_df = tree.stats_df.copy()

    results_df = annotate_child_parent_divergence(
        tree,
        results_df,
        total_number_of_features=X.shape[1],
        significance_level_alpha=config.SIGNIFICANCE_ALPHA,
    )
    results_df = annotate_sibling_independence_cmi(
        tree,
        results_df,
        significance_level_alpha=config.SIGNIFICANCE_ALPHA,
        n_permutations=config.N_PERMUTATIONS,
    )

    print("Statistical analysis complete.")

    return results_df
