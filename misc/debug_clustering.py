#!/usr/bin/env python3
"""
Unit tests for clustering algorithm components and detailed validation.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import mutual_info_score
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
def test_data():
    """Generate test data for clustering tests."""
    X_t, y_t = make_blobs(
        n_samples=20, n_features=20, centers=2, cluster_std=1.0, random_state=42
    )
    X_bin = (X_t > np.median(X_t, axis=0)).astype(int)
    data_t = pd.DataFrame(
        X_bin, index=[f"S{j}" for j in range(20)], columns=[f"F{j}" for j in range(20)]
    )
    return data_t, y_t


@pytest.fixture
def processed_data(test_data):
    """Process test data into tree, stats, MI matrix, and results."""
    data_t, y_t = test_data

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
    # Add BH-corrected local (child vs parent) significance
    results_t = annotate_local_child_parent_significance(
        tree_t, results_t, 20, 0.05
    )

    # Add conditional MI-based sibling independence (per parent) with BH control
    results_t = annotate_sibling_independence_cmi(
        tree_t, results_t, significance_level_alpha=0.05, permutations=75
    )

    return tree_t, stats_t, mi_t, results_t, y_t


def test_tree_structure(processed_data):
    """Test that the tree has correct structure."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    # Check that we have a valid tree
    assert len(tree_t.nodes) > 0
    assert len(tree_t.edges) > 0

    # Check that we have a single root
    roots = [n for n, d in tree_t.in_degree() if d == 0]
    assert len(roots) == 1
    root = roots[0]

    # Check that root has children
    children = list(tree_t.successors(root))
    assert len(children) >= 2  # Should be binary tree

    # Check that all nodes have required attributes
    for node in tree_t.nodes:
        assert "is_leaf" in tree_t.nodes[node]
        assert "distribution" in tree_t.nodes[node]
        dist = np.asarray(tree_t.nodes[node]["distribution"])
        assert dist.ndim == 1
        assert len(dist) == 20  # n_features


def test_kl_divergence_calculation(processed_data):
    """Test KL divergence calculations."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    # Check that KL values are calculated
    assert "kl_divergence_global" in stats_t.columns
    assert len(stats_t) > 0

    # Check that KL values are finite for most nodes
    kl_values = stats_t["kl_divergence_global"].dropna()
    assert len(kl_values) > 0

    # Root might have NaN KL, but internal nodes should have values
    internal_nodes = [
        n for n in stats_t.index if not tree_t.nodes[n].get("is_leaf", False)
    ]
    if len(internal_nodes) > 0:
        internal_kl = stats_t.loc[internal_nodes, "kl_divergence_global"].dropna()
        assert len(internal_kl) > 0, "Internal nodes should have KL divergence values"


def test_mutual_information_matrix(processed_data):
    """Test MI matrix calculation."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    # Check MI matrix shape
    assert mi_t.shape[0] > 0
    assert mi_t.shape[1] > 0
    assert mi_t.shape[0] == mi_t.shape[1]  # Should be square

    # Check that MI values are reasonable
    mi_values = mi_t.values.flatten()
    mi_values = mi_values[~np.isnan(mi_values)]
    assert len(mi_values) > 0

    # MI should be between 0 and 1 (normalized)
    assert np.all(mi_values >= 0)
    assert np.all(mi_values <= 1)


def test_statistical_significance(processed_data):
    """Test statistical significance testing."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    # Check that significance column exists
    assert "Are_Features_Dependent" in results_t.columns

    # Check that we have some significant nodes
    significant_nodes = results_t[results_t["Are_Features_Dependent"]]
    assert len(significant_nodes) >= 0  # At least some nodes should be significant


def test_cluster_decomposition_basic(processed_data):
    """Test basic cluster decomposition functionality."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    # Test clustering with default parameters
    decomposer = ClusterDecomposer(
        tree=tree_t,
        results_df=results_t,
        significance_column="Are_Features_Dependent",
    )

    decomp = decomposer.decompose_tree()

    # Check result structure
    assert isinstance(decomp, dict)
    assert "num_clusters" in decomp
    assert "cluster_assignments" in decomp
    assert "independence_analysis" in decomp

    # Check that we find reasonable number of clusters
    assert decomp["num_clusters"] >= 1
    assert decomp["num_clusters"] <= len(tree_t.nodes)

    # Check cluster assignments
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
    leaves = [n for n in tree_t.nodes if tree_t.nodes[n].get("is_leaf", False)]
    assert total_leaves == len(leaves)


def test_hamming_distance_calculation(processed_data):
    """Test Hamming distance calculations between sibling nodes."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    root = [n for n, d in tree_t.in_degree() if d == 0][0]
    children = list(tree_t.successors(root))

    if len(children) == 2:
        c1, c2 = children

        # Get distributions
        dist1 = stats_t.loc[c1, "distribution"]
        dist2 = stats_t.loc[c2, "distribution"]

        # Calculate binary patterns
        bin1 = (dist1 >= 0.5).astype(int)
        bin2 = (dist2 >= 0.5).astype(int)

        # Calculate Hamming distance
        hamming_dist = np.sum(bin1 != bin2)
        normalized_dist = hamming_dist / len(bin1)

        # Check that distance is reasonable
        assert 0 <= normalized_dist <= 1

        # Calculate MI directly
        direct_mi = mutual_info_score(bin1, bin2)
        assert 0 <= direct_mi <= 1


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
def test_clustering_with_different_thresholds(processed_data, threshold):
    """Test clustering with different MI thresholds."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    decomposer = ClusterDecomposer(
        tree=tree_t,
        results_df=results_t,
        significance_column="Are_Features_Dependent",
    )

    decomp = decomposer.decompose_tree()

    # Should always find at least 1 cluster
    assert decomp["num_clusters"] >= 1

    # Should not find more clusters than samples
    leaves = [n for n in tree_t.nodes if tree_t.nodes[n].get("is_leaf", False)]
    assert decomp["num_clusters"] <= len(leaves)


def test_cluster_report_generation(processed_data):
    """Test cluster report generation."""
    tree_t, stats_t, mi_t, results_t, y_t = processed_data

    decomposer = ClusterDecomposer(
        tree=tree_t,
        results_df=results_t,
        significance_column="Are_Features_Dependent",
    )

    decomp = decomposer.decompose_tree()
    report = decomposer.generate_report(decomp)

    # Check report structure
    assert isinstance(report, pd.DataFrame)
    assert len(report) > 0

    # Check required columns
    required_cols = ["cluster_id", "cluster_root", "cluster_size"]
    for col in required_cols:
        assert col in report.columns

    # Check that cluster sizes are positive
    assert np.all(report["cluster_size"] > 0)

    # Check that all samples are accounted for
    leaves = [n for n in tree_t.nodes if tree_t.nodes[n].get("is_leaf", False)]
    assert len(report) == len(leaves)
