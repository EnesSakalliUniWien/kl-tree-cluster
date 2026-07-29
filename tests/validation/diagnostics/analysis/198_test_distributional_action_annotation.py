from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.diagnostics.analysis.distributional_action import (
    annotate_distributional_action_diagnostics,
)
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns
from tree_break_selection.tree.poset_tree import PosetTree


def _unequal_mass_tree() -> tuple[PosetTree, pd.DataFrame]:
    tree = PosetTree()
    for node_id, is_leaf in [
        ("root", False),
        ("small", False),
        ("large", False),
        ("S0", True),
        ("L0", True),
        ("L1", True),
        ("L2", True),
    ]:
        tree.add_node(node_id, is_leaf=is_leaf, label=node_id)
    for parent, child in [
        ("root", "small"),
        ("root", "large"),
        ("small", "S0"),
        ("large", "L0"),
        ("large", "L1"),
        ("large", "L2"),
    ]:
        tree.add_edge(parent, child, branch_length=1.0)
    tree.graph["root"] = "root"

    leaf_data = pd.DataFrame(
        {
            "x": [0.0, 10.0, 10.0, 10.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
        },
        index=["S0", "L0", "L1", "L2"],
    )
    tree.populate_node_divergences(
        leaf_data,
        feature_space=continuous_feature_space_from_columns(tuple(leaf_data.columns)),
    )
    return tree, leaf_data


def _child_parent_annotations(tree: PosetTree) -> pd.DataFrame:
    annotations = tree.annotations_df.copy()
    annotations["Child_Parent_Divergence_Significant"] = False
    annotations.loc[["small", "large"], "Child_Parent_Divergence_Significant"] = True
    return annotations


def _two_split_action_tree() -> tuple[PosetTree, pd.DataFrame]:
    tree = PosetTree()
    for node_id, is_leaf in [
        ("root", False),
        ("high", False),
        ("low", False),
        ("H0", True),
        ("H1", True),
        ("L0", True),
        ("L1", True),
    ]:
        tree.add_node(node_id, is_leaf=is_leaf, label=node_id)
    for parent, child in [
        ("root", "high"),
        ("root", "low"),
        ("high", "H0"),
        ("high", "H1"),
        ("low", "L0"),
        ("low", "L1"),
    ]:
        tree.add_edge(parent, child, branch_length=1.0)
    tree.graph["root"] = "root"
    leaf_data = pd.DataFrame(
        {"x": [0.0, 10.0, 5.0, 6.0]},
        index=["H0", "H1", "L0", "L1"],
    )
    tree.populate_node_divergences(
        leaf_data,
        feature_space=continuous_feature_space_from_columns(tuple(leaf_data.columns)),
    )
    return tree, leaf_data


def _two_split_annotations(tree: PosetTree) -> pd.DataFrame:
    annotations = tree.annotations_df.copy()
    annotations["Child_Parent_Divergence_Significant"] = False
    annotations.loc[["H0", "H1", "L0", "L1"], "Child_Parent_Divergence_Significant"] = True
    return annotations


def test_distributional_action_diagnostics_do_not_change_edge_gate() -> None:
    tree, leaf_data = _unequal_mass_tree()
    annotations = _child_parent_annotations(tree)

    annotated = annotate_distributional_action_diagnostics(
        tree,
        annotations,
        leaf_data,
    )

    assert annotated["Child_Parent_Divergence_Significant"].equals(
        annotations["Child_Parent_Divergence_Significant"]
    )
    assert (
        annotated.loc["small", "Distributional_Action"]
        > annotated.loc["large", "Distributional_Action"]
    )


def test_distributional_action_diagnostics_preserve_all_open_splits() -> None:
    tree, leaf_data = _two_split_action_tree()
    annotations = _two_split_annotations(tree)

    annotated = annotate_distributional_action_diagnostics(
        tree,
        annotations,
        leaf_data,
    )

    assert annotated["Child_Parent_Divergence_Significant"].equals(
        annotations["Child_Parent_Divergence_Significant"]
    )
    assert np.isfinite(annotated.loc["H0", "Distributional_Split_Action"])
    assert np.isfinite(annotated.loc["L0", "Distributional_Split_Action"])
