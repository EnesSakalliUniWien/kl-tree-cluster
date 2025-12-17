"""Targeted tests for ClusterDecomposer's shortlist and near-threshold logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposer import (
    ClusterDecomposer,
)


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
            "Local_BH_Significant": True,
            "Sibling_BH_Independent": True,
            "Sibling_CMI_P_Value_Corrected": 0.052,
        },
        "L": {
            "distribution": G.nodes["L"]["distribution"],
            "is_leaf": True,
            "Local_BH_Significant": True,
        },
        "S": {
            "distribution": G.nodes["S"]["distribution"],
            "is_leaf": True,
            "Local_BH_Significant": True,
        },
    }
    df = pd.DataFrame.from_dict(data, orient="index")
    return G, df


def _two_level_tree() -> tuple[PosetTree, pd.DataFrame]:
    """Construct a two-level binary tree with two internal nodes."""

    G = PosetTree()
    G.add_node("P", is_leaf=False, distribution=np.array([0.5, 0.5]))
    G.add_node("A", is_leaf=False, distribution=np.array([0.55, 0.45]))
    G.add_node("B", is_leaf=False, distribution=np.array([0.45, 0.55]))
    G.add_node("A1", is_leaf=True, label="A1", distribution=np.array([0.58, 0.42]))
    G.add_node("A2", is_leaf=True, label="A2", distribution=np.array([0.52, 0.48]))
    G.add_node("B1", is_leaf=True, label="B1", distribution=np.array([0.47, 0.53]))
    G.add_node("B2", is_leaf=True, label="B2", distribution=np.array([0.43, 0.57]))

    G.add_edge("P", "A")
    G.add_edge("P", "B")
    G.add_edge("A", "A1")
    G.add_edge("A", "A2")
    G.add_edge("B", "B1")
    G.add_edge("B", "B2")

    data = {
        "P": {
            "distribution": G.nodes["P"]["distribution"],
            "is_leaf": False,
            "Local_BH_Significant": True,
            "Sibling_BH_Independent": True,
            "Sibling_CMI_P_Value_Corrected": 0.02,
        },
        "A": {
            "distribution": G.nodes["A"]["distribution"],
            "is_leaf": False,
            "Local_BH_Significant": True,
            "Sibling_BH_Independent": True,
            "Sibling_CMI_P_Value_Corrected": 0.03,
        },
        "B": {
            "distribution": G.nodes["B"]["distribution"],
            "is_leaf": False,
            "Local_BH_Significant": True,
            "Sibling_BH_Independent": True,
            "Sibling_CMI_P_Value_Corrected": 0.025,
        },
        "A1": {
            "distribution": G.nodes["A1"]["distribution"],
            "is_leaf": True,
            "Local_BH_Significant": True,
        },
        "A2": {
            "distribution": G.nodes["A2"]["distribution"],
            "is_leaf": True,
            "Local_BH_Significant": True,
        },
        "B1": {
            "distribution": G.nodes["B1"]["distribution"],
            "is_leaf": True,
            "Local_BH_Significant": True,
        },
        "B2": {
            "distribution": G.nodes["B2"]["distribution"],
            "is_leaf": True,
            "Local_BH_Significant": True,
        },
    }
    df = pd.DataFrame.from_dict(data, orient="index")
    return G, df


def _sorted_leaf_sets(result: dict[str, object]) -> list[list[str]]:
    assignments = result["cluster_assignments"]
    return sorted([sorted(cluster["leaves"]) for cluster in assignments.values()])


def test_near_threshold_override_merges_borderline_siblings():
    tree, df = _simple_tree()

    baseline_result = tree.decompose(results_df=df)
    assert baseline_result["num_clusters"] == 2

    relaxed_result = tree.decompose(
        results_df=df,
        near_independence_alpha_buffer=0.01,
        near_independence_kl_gap=1.0,
    )
    assert relaxed_result["num_clusters"] == 1


def test_shortlist_processing_matches_default_clusters():
    tree, df = _two_level_tree()

    baseline = tree.decompose(results_df=df)
    shortlist = tree.decompose(
        results_df=df,
        sibling_shortlist_size=1,
    )

    base_leaf_sets = _sorted_leaf_sets(baseline)
    shortlist_leaf_sets = _sorted_leaf_sets(shortlist)
    assert base_leaf_sets == shortlist_leaf_sets


def test_shortlist_heap_and_stack_paths(monkeypatch):
    tree, df = _two_level_tree()
    baseline = tree.decompose(results_df=df)
    expected = _sorted_leaf_sets(baseline)

    heap_stats = {"non_none": 0, "calls": 0}
    original_pop = ClusterDecomposer._pop_candidate

    def tracking_pop(self, heap, queued, processed):
        heap_stats["calls"] += 1
        node = original_pop(self, heap, queued, processed)
        if node is not None:
            heap_stats["non_none"] += 1
        return node

    monkeypatch.setattr(ClusterDecomposer, "_pop_candidate", tracking_pop)

    shortlist = tree.decompose(
        results_df=df,
        sibling_shortlist_size=2,
    )
    heap_leaf_sets = _sorted_leaf_sets(shortlist)

    assert heap_leaf_sets == expected
    assert heap_stats["calls"] > 0
    assert heap_stats["non_none"] > 0
