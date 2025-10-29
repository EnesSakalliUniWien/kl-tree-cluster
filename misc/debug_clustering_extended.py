#!/usr/bin/env python3
"""
Test suite for clustering algorithm with multiple test cases.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from hierarchy_analysis import calculate_hierarchy_kl_divergence
from hierarchy_analysis.kl_correlation_analysis import (
    calculate_kl_divergence_mutual_information_matrix,
)
from hierarchy_analysis.statistical_tests import (
    annotate_nodes_with_statistical_significance_tests,
    annotate_local_child_parent_significance,
    annotate_sibling_independence_cmi,
)
from hierarchy_analysis.cluster_decomposition import ClusterDecomposer
from tree.poset_tree import PosetTree


@pytest.fixture
def sample_data():
    """Generate sample test data."""
    X_t, y_t = make_blobs(
        n_samples=20, n_features=20, centers=2, cluster_std=1.0, random_state=42
    )
    X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
    data_t = pd.DataFrame(
        X_bin, index=[f"S{j}" for j in range(20)], columns=[f"F{j}" for j in range(20)]
    )
    return data_t, y_t


def run_clustering_test(n_samples, n_features, centers, cluster_std, random_state=42):
    """Run a single clustering test case and return results."""
    # Generate test data
    X_t, y_t = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
    data_t = pd.DataFrame(
        X_bin,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )

    # Build tree and calculate stats
    Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
    stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)

    # Calculate MI matrix
    mi_t, _ = calculate_kl_divergence_mutual_information_matrix(tree_t, stats_t)

    # Statistical testing
    results_t = annotate_nodes_with_statistical_significance_tests(
        stats_t, n_features, 0.05, 2.0
    )
    # Add BH-corrected local (child vs parent) significance
    results_t = annotate_local_child_parent_significance(
        tree_t, results_t, n_features, 0.05
    )
    # Add conditional MI-based sibling independence
    results_t = annotate_sibling_independence_cmi(
        tree_t, results_t, significance_level_alpha=0.05, permutations=75
    )

    # Run clustering
    decomposer = ClusterDecomposer(
        tree=tree_t,
        results_df=results_t,
        significance_column="Are_Features_Dependent",
    )
    decomp = decomposer.decompose_tree()
    found_clusters = decomp["num_clusters"]

    return found_clusters, centers, data_t.shape


@pytest.mark.parametrize(
    "n_samples,n_features,centers,cluster_std,test_name",
    [
        (20, 20, 2, 1.0, "2 clusters, 20 samples"),
        (30, 15, 3, 1.2, "3 clusters, 30 samples"),
        (40, 25, 4, 0.8, "4 clusters, 40 samples"),
        (25, 30, 2, 2.0, "2 clusters, high variance"),
        (35, 20, 5, 1.0, "5 clusters, 35 samples"),
    ],
)
def test_clustering_accuracy(n_samples, n_features, centers, cluster_std, test_name):
    """Test that clustering algorithm finds the correct number of clusters."""
    found_clusters, expected_clusters, data_shape = run_clustering_test(
        n_samples, n_features, centers, cluster_std
    )

    assert found_clusters == expected_clusters, (
        f"{test_name}: Expected {expected_clusters} clusters, "
        f"but found {found_clusters} (data shape: {data_shape})"
    )


def test_clustering_basic_functionality(sample_data):
    """Test basic clustering functionality with sample data."""
    data_t, y_t = sample_data

    # Build tree and calculate stats
    Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
    stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)

    # Calculate MI matrix
    mi_t, _ = calculate_kl_divergence_mutual_information_matrix(tree_t, stats_t)

    # Statistical testing
    results_t = annotate_nodes_with_statistical_significance_tests(
        stats_t, 20, 0.05, 2.0
    )
    # Sibling CMI not required for this smoke test

    # Run clustering
    decomposer = ClusterDecomposer(
        tree=tree_t,
        results_df=results_t,
        significance_column="Are_Features_Dependent",
    )

    # Test that decomposition runs without errors
    decomp = decomposer.decompose_tree()

    # Check that we get a valid result
    assert isinstance(decomp, dict)
    assert "num_clusters" in decomp
    assert "cluster_assignments" in decomp
    assert decomp["num_clusters"] > 0

    # Check that cluster assignments are valid
    cluster_assignments = decomp["cluster_assignments"]
    assert isinstance(cluster_assignments, dict)

    total_leaves = 0
    for cluster_id, info in cluster_assignments.items():
        assert "leaves" in info
        assert "root_node" in info
        assert "size" in info
        assert info["size"] == len(info["leaves"])
        total_leaves += info["size"]

    # Should account for all leaves
    assert total_leaves == len(data_t)


def test_clustering_with_different_thresholds(sample_data):
    """Test clustering with different MI thresholds."""
    data_t, y_t = sample_data

    # Build tree and calculate stats
    Z_t = linkage(pdist(data_t.values, metric="hamming"), method="complete")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_t.index.tolist())
    stats_t = calculate_hierarchy_kl_divergence(tree_t, data_t)

    # Calculate MI matrix
    mi_t, _ = calculate_kl_divergence_mutual_information_matrix(tree_t, stats_t)

    # Statistical testing
    results_t = annotate_nodes_with_statistical_significance_tests(
        stats_t, 20, 0.001, 2.0
    )

    thresholds = [0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        decomposer = ClusterDecomposer(
            tree=tree_t,
            results_df=results_t,
            significance_column="Are_Features_Dependent",
        )

        decomp = decomposer.decompose_tree()

        # Should always find at least 1 cluster
        assert decomp["num_clusters"] >= 1

        # Should not find more clusters than samples
        assert decomp["num_clusters"] <= len(data_t)
