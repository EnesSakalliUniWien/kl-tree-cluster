"""Tests for cluster tree visualization with significance styling."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from kl_clustering_analysis.plot.cluster_tree_visualization import (
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
            "Child_Parent_Divergence_P_Value_Corrected": [0.001, 0.02, 0.2],
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
