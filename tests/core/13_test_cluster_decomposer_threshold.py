"""Targeted tests for TreeDecomposition's near-threshold logic."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _simple_tree() -> tuple[PosetTree, pd.DataFrame]:
    """Build a minimal binary tree with one internal node."""

    G = PosetTree()
    G.add_node("R", is_leaf=False, distribution=np.array([0.5, 0.5]))
    G.add_node("L", is_leaf=True, label="L", distribution=np.array([0.52, 0.48]))
    G.add_node("S", is_leaf=True, label="S", distribution=np.array([0.48, 0.52]))
    G.add_edge("R", "L")
    G.add_edge("R", "S")

    data = {
        "R": {
            "distribution": G.nodes["R"]["distribution"],
            "is_leaf": False,
            "Child_Parent_Divergence_Significant": True,
            "Sibling_BH_Different": True,
            "Sibling_Divergence_Skipped": False,
            "Sibling_Divergence_P_Value_Corrected": 0.01,
        },
        "L": {
            "distribution": G.nodes["L"]["distribution"],
            "is_leaf": True,
            "Child_Parent_Divergence_Significant": True,
            "Sibling_BH_Different": False,
            "Sibling_Divergence_Skipped": False,
        },
        "S": {
            "distribution": G.nodes["S"]["distribution"],
            "is_leaf": True,
            "Child_Parent_Divergence_Significant": True,
            "Sibling_BH_Different": False,
            "Sibling_Divergence_Skipped": False,
        },
    }
    df = pd.DataFrame.from_dict(data, orient="index")
    return G, df


def test_near_threshold_override_merges_borderline_siblings():
    """Test that siblings with low divergence p-value near alpha are NOT split when Sibling_BH_Different=False."""
    tree, df = _simple_tree()

    # Bypass annotation pipeline — this test controls gate columns directly.
    with patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df):
        baseline_result = tree.decompose(
            annotations_df=df
        )
        assert baseline_result["num_clusters"] == 2

        # Set siblings as NOT significantly different - should merge
        df.loc["R", "Sibling_BH_Different"] = False
        merged_result = tree.decompose(annotations_df=df)
        assert merged_result["num_clusters"] == 1


def test_tree_decomposition_preserves_non_string_node_ids_in_annotations():
    tree = PosetTree()
    tree.add_node(0, is_leaf=False)
    tree.add_node(1, is_leaf=True, label="left")
    tree.add_node(2, is_leaf=True, label="right")
    tree.add_edge(0, 1)
    tree.add_edge(0, 2)

    annotations = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, True, True],
            "Sibling_BH_Different": [True, False, False],
            "Sibling_Divergence_Skipped": [False, False, False],
        },
        index=[0, 1, 2],
    )

    with patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df):
        result = tree.decompose(annotations_df=annotations)

    assert result["num_clusters"] == 2
    assert [cluster["root_node"] for cluster in result["cluster_assignments"].values()] == [1, 2]
