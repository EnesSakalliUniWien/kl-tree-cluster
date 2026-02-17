"""Tests for cluster tree visualization with significance styling."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from kl_clustering_analysis.plot.cluster_tree_visualization import (
    _group_edges_for_sibling_style,
    _group_internal_nodes_for_halo,
    plot_tree_with_clusters,
)


def test_plot_tree_with_significance_legend():
    """Ensure significance-aware tree plotting produces legends without error."""
    G = nx.DiGraph()
    G.add_edges_from([("root", "a"), ("root", "b")])

    cluster_assignments = {
        0: {"leaves": ["a"], "root_node": "a", "size": 1},
        1: {"leaves": ["b"], "root_node": "b", "size": 1},
    }
    decomposition = {"cluster_assignments": cluster_assignments, "num_clusters": 2}

    stats_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, True, False],
            "Sibling_BH_Different": [True, False, False],
            "Sibling_Divergence_Skipped": [False, False, False],
        },
        index=["root", "a", "b"],
    )

    fig, ax = plot_tree_with_clusters(
        tree=G,
        decomposition_results=decomposition,
        results_df=stats_df,
        show=False,
    )

    # Basic sanity check - figure should have been created
    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_tree_style_grouping_uses_significance_and_skips():
    """Edge and halo grouping follows sibling + child-parent test outcomes."""
    G = nx.DiGraph()
    G.add_edges_from([("root", "a"), ("root", "b"), ("a", "c"), ("a", "d")])

    leaves = {n for n in G.nodes() if G.out_degree(n) == 0}

    stats_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, True, False, False, False],
            "Sibling_BH_Different": [True, False, False, False, False],
            "Sibling_Divergence_Skipped": [False, True, False, False, False],
        },
        index=["root", "a", "b", "c", "d"],
    )

    sig_nodes, nonsig_nodes = _group_internal_nodes_for_halo(G, leaves, stats_df)
    assert set(sig_nodes) == {"a"}
    assert set(nonsig_nodes) == set()

    edge_groups = _group_edges_for_sibling_style(G, stats_df)
    assert set(edge_groups["different"]) == {("root", "a"), ("root", "b")}
    assert set(edge_groups["missing"]) == {("a", "c"), ("a", "d")}
    assert edge_groups["not_different"] == []
