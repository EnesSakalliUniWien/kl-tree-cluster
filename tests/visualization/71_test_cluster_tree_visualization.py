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
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, "label")

    cluster_assignments = {
        0: {"leaves": ["a"], "root_node": "a", "size": 1},
        1: {"leaves": ["b"], "root_node": "b", "size": 1},
    }
    decomposition = {"cluster_assignments": cluster_assignments, "num_clusters": 2}

    annotations_df = pd.DataFrame(
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
        annotations_df=annotations_df,
    )

    # Basic sanity check - figure should have been created
    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_plot_tree_summarizes_many_cluster_legend_entries():
    """Dense tree legends summarize clusters instead of listing every cluster."""
    G = nx.DiGraph()
    leaves = [f"leaf_{i}" for i in range(24)]
    G.add_edges_from([("root", leaf) for leaf in leaves])
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, "label")

    cluster_assignments = {
        i: {"leaves": [leaf], "root_node": leaf, "size": 1}
        for i, leaf in enumerate(leaves)
    }
    decomposition = {
        "cluster_assignments": cluster_assignments,
        "num_clusters": len(leaves),
    }
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False] * len(G.nodes()),
            "Sibling_BH_Different": [False] * len(G.nodes()),
            "Sibling_Divergence_Skipped": [False] * len(G.nodes()),
        },
        index=list(G.nodes()),
    )

    fig, ax = plot_tree_with_clusters(
        tree=G,
        decomposition_results=decomposition,
        annotations_df=annotations_df,
        max_cluster_legend_entries=10,
    )

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert "24 clusters" in labels
    assert "0" not in labels

    plt.close(fig)


def test_plot_tree_can_suppress_embedded_panel_legend():
    """Embedded tree panels can disable legends so nodes are not obscured."""
    G = nx.DiGraph()
    G.add_edges_from([("root", "a"), ("root", "b")])
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, "label")

    decomposition = {
        "cluster_assignments": {
            0: {"leaves": ["a"], "root_node": "a", "size": 1},
            1: {"leaves": ["b"], "root_node": "b", "size": 1},
        },
        "num_clusters": 2,
    }
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, False, False],
            "Sibling_BH_Different": [False, False, False],
            "Sibling_Divergence_Skipped": [False, False, False],
        },
        index=["root", "a", "b"],
    )

    fig, ax = plot_tree_with_clusters(
        tree=G,
        decomposition_results=decomposition,
        annotations_df=annotations_df,
        show_legend=False,
    )

    assert ax.get_legend() is None
    plt.close(fig)


def test_rectangular_tree_edges_are_routed_as_elbows():
    """Rectangular tree plots avoid long diagonal edges through subtrees."""
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("root", "left"),
            ("root", "right"),
            ("left", "a"),
            ("left", "b"),
            ("right", "deep"),
            ("deep", "c"),
            ("deep", "d"),
        ]
    )
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, "label")

    decomposition = {
        "cluster_assignments": {
            0: {"leaves": ["a", "b"], "root_node": "left", "size": 2},
            1: {"leaves": ["c", "d"], "root_node": "deep", "size": 2},
        },
        "num_clusters": 2,
    }
    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False] * len(G.nodes()),
            "Sibling_BH_Different": [True] * len(G.nodes()),
            "Sibling_Divergence_Skipped": [False] * len(G.nodes()),
        },
        index=list(G.nodes()),
    )

    fig, ax = plot_tree_with_clusters(
        tree=G,
        decomposition_results=decomposition,
        annotations_df=annotations_df,
        show_legend=False,
        layout="rectangular",
    )

    assert ax.lines
    assert all(len(line.get_xdata()) == 4 for line in ax.lines)
    assert all(line.get_xdata()[0] == line.get_xdata()[1] for line in ax.lines)
    assert all(line.get_ydata()[1] == line.get_ydata()[2] for line in ax.lines)
    plt.close(fig)


def test_tree_style_grouping_uses_significance_and_skips():
    """Edge and halo grouping follows sibling + child-parent test outcomes."""
    G = nx.DiGraph()
    G.add_edges_from([("root", "a"), ("root", "b"), ("a", "c"), ("a", "d")])
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, "label")

    leaves = {n for n in G.nodes() if G.out_degree(n) == 0}

    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, True, False, False, False],
            "Sibling_BH_Different": [True, False, False, False, False],
            "Sibling_Divergence_Skipped": [False, True, False, False, False],
        },
        index=["root", "a", "b", "c", "d"],
    )

    sig_nodes, nonsig_nodes = _group_internal_nodes_for_halo(G, leaves, annotations_df)
    assert set(sig_nodes) == {"a"}
    assert set(nonsig_nodes) == set()

    edge_groups = _group_edges_for_sibling_style(G, annotations_df)
    assert set(edge_groups["different"]) == {("root", "a"), ("root", "b")}
    assert set(edge_groups["missing"]) == {("a", "c"), ("a", "d")}
    assert edge_groups["not_different"] == []
