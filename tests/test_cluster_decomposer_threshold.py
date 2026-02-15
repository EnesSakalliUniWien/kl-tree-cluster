"""Targeted tests for TreeDecomposition's near-threshold logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    baseline_result = tree.decompose(results_df=df, posthoc_merge=False)
    assert baseline_result["num_clusters"] == 2

    # Set siblings as NOT significantly different - should merge
    df.loc["R", "Sibling_BH_Different"] = False
    merged_result = tree.decompose(results_df=df)
    assert merged_result["num_clusters"] == 1
