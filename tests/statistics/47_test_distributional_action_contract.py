from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tree_break_selection.hierarchy_analysis.statistics.distributional_action import (
    DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
    DISTRIBUTIONAL_ACTION_SPLIT_FILTER_OPEN_SPLIT_QUANTILE,
    annotate_distributional_action_split_filter,
    binary_split_distributional_action,
    binary_split_distributional_action_summary,
    edge_distributional_action,
    edge_distributional_action_summary,
    split_distributional_action_summary,
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


def test_binary_split_action_matches_child_parent_variance_identity() -> None:
    left_mean = np.array([0.0, 2.0])
    right_mean = np.array([4.0, 2.0])
    left_count = 2
    right_count = 6

    action = binary_split_distributional_action(
        left_mean,
        right_mean,
        left_count,
        right_count,
    )

    parent_mean = (left_count * left_mean + right_count * right_mean) / (left_count + right_count)
    child_parent_action = left_count * np.sum((left_mean - parent_mean) ** 2)
    child_parent_action += right_count * np.sum((right_mean - parent_mean) ** 2)
    pairwise_action = (left_count * right_count / (left_count + right_count)) * np.sum(
        (left_mean - right_mean) ** 2
    )

    assert np.isclose(action, child_parent_action)
    assert np.isclose(action, pairwise_action)


def test_binary_split_edges_are_mass_tied_to_one_sibling_contrast() -> None:
    left_mean = np.array([0.0])
    right_mean = np.array([10.0])
    left_count = 1
    right_count = 99

    summary = binary_split_distributional_action_summary(
        left_mean,
        right_mean,
        left_count,
        right_count,
    )

    assert np.allclose(summary.parent_mean, np.array([9.9]))
    assert np.isclose(summary.left_edge_action, 98.01)
    assert np.isclose(summary.right_edge_action, 0.99)
    assert np.isclose(summary.action, 99.0)
    assert np.isclose(summary.left_edge_action / summary.action, 0.99)
    assert np.isclose(summary.right_edge_action / summary.action, 0.01)


def test_small_edge_contribution_does_not_mean_small_barycentric_split() -> None:
    summary = binary_split_distributional_action_summary(
        left_mean=np.array([0.0]),
        right_mean=np.array([10.0]),
        left_leaf_count=1,
        right_leaf_count=99,
    )

    assert summary.right_edge_action < 1.0
    assert np.isclose(summary.action, 100.0 * summary.right_edge_action)


def test_split_action_decomposes_total_inertia_recursively_across_topology() -> None:
    leaf_values = np.array([[0.0], [2.0], [10.0], [14.0]])
    left = split_distributional_action_summary(
        child_means=[leaf_values[0], leaf_values[1]],
        child_masses=[1, 1],
    )
    right = split_distributional_action_summary(
        child_means=[leaf_values[2], leaf_values[3]],
        child_masses=[1, 1],
    )
    root = split_distributional_action_summary(
        child_means=[left.parent_mean, right.parent_mean],
        child_masses=[left.parent_mass, right.parent_mass],
    )

    root_mean = leaf_values.mean(axis=0)
    total_inertia = float(np.sum((leaf_values - root_mean) ** 2))

    assert np.isclose(left.action, 2.0)
    assert np.isclose(right.action, 8.0)
    assert np.isclose(root.action, 121.0)
    assert np.isclose(total_inertia, left.action + right.action + root.action)


def test_edge_action_is_independent_of_branch_length() -> None:
    parent_mean = np.array([0.0, 0.0])
    child_mean = np.array([3.0, 4.0])
    child_count = 5
    short_branch_length = 0.1
    long_branch_length = 10.0

    short_branch_action = edge_distributional_action(
        parent_mean,
        child_mean,
        child_count,
    )
    long_branch_action = edge_distributional_action(
        parent_mean,
        child_mean,
        child_count,
    )

    assert short_branch_length != long_branch_length
    assert short_branch_action == long_branch_action
    assert np.isclose(short_branch_action, 125.0)


def test_edge_action_changes_when_distribution_moves_with_same_branch_length() -> None:
    parent_mean = np.array([0.0, 0.0])
    child_count = 5
    branch_length = 1.0

    near_action = edge_distributional_action(
        parent_mean,
        np.array([1.0, 0.0]),
        child_count,
    )
    far_action = edge_distributional_action(
        parent_mean,
        np.array([2.0, 0.0]),
        child_count,
    )

    assert branch_length == 1.0
    assert far_action == 4.0 * near_action


def test_edge_action_summary_exposes_distribution_masses() -> None:
    summary = edge_distributional_action_summary(
        parent_mean=np.array([0.0, 0.0]),
        child_mean=np.array([3.0, 4.0]),
        parent_leaf_count=20,
        child_leaf_count=5,
    )

    assert summary.parent_mass == 20.0
    assert summary.child_mass == 5.0
    assert summary.child_parent_mass_fraction == 0.25
    assert summary.squared_displacement == 25.0
    assert summary.action == edge_distributional_action(
        np.array([0.0, 0.0]),
        np.array([3.0, 4.0]),
        5,
    )


def test_binary_split_summary_exposes_child_and_parent_masses() -> None:
    summary = binary_split_distributional_action_summary(
        left_mean=np.array([0.0, 0.0]),
        right_mean=np.array([4.0, 0.0]),
        left_leaf_count=2,
        right_leaf_count=6,
    )

    assert summary.left_mass == 2.0
    assert summary.right_mass == 6.0
    assert summary.parent_mass == 8.0
    assert summary.left_parent_mass_fraction == 0.25
    assert summary.right_parent_mass_fraction == 0.75
    assert summary.squared_child_displacement == 16.0
    assert summary.action == binary_split_distributional_action(
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        2,
        6,
    )


def test_distributional_action_split_filter_none_annotates_without_changing_edge_gate() -> None:
    tree, leaf_data = _unequal_mass_tree()
    annotations = _child_parent_annotations(tree)

    filtered, metadata = annotate_distributional_action_split_filter(
        tree,
        annotations,
        leaf_data,
        policy=DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
        quantile=0.5,
    )

    assert filtered["Child_Parent_Divergence_Significant"].equals(
        annotations["Child_Parent_Divergence_Significant"]
    )
    assert (
        filtered.loc["small", "Distributional_Action"]
        > filtered.loc["large", "Distributional_Action"]
    )
    assert bool(filtered.loc["large", "Distributional_Split_Action_Filter_Passes"])
    assert not bool(filtered.loc["large", "Distributional_Split_Action_Filtered"])
    assert metadata["distributional_action_split_filter_policy"] == (
        DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE
    )
    assert metadata["distributional_action_split_filter_filtered_parent_count"] == 0
    assert metadata["distributional_action_split_filter_open_parent_count_before"] == 1
    assert metadata["distributional_action_split_filter_open_parent_count_after"] == 1


def test_uncalibrated_split_filter_policy_is_rejected() -> None:
    tree, leaf_data = _unequal_mass_tree()
    annotations = _child_parent_annotations(tree)

    with pytest.raises(ValueError, match="diagnostic-only"):
        annotate_distributional_action_split_filter(
            tree,
            annotations,
            leaf_data,
            policy=DISTRIBUTIONAL_ACTION_SPLIT_FILTER_OPEN_SPLIT_QUANTILE,
            quantile=0.5,
        )


def test_distributional_action_annotation_does_not_close_low_action_splits() -> None:
    tree, leaf_data = _two_split_action_tree()
    annotations = _two_split_annotations(tree)

    filtered, metadata = annotate_distributional_action_split_filter(
        tree,
        annotations,
        leaf_data,
        policy=DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
        quantile=0.5,
    )

    assert bool(filtered.loc["H0", "Child_Parent_Divergence_Significant"])
    assert bool(filtered.loc["H1", "Child_Parent_Divergence_Significant"])
    assert bool(filtered.loc["L0", "Child_Parent_Divergence_Significant"])
    assert bool(filtered.loc["L1", "Child_Parent_Divergence_Significant"])
    assert bool(filtered.loc["H0", "Distributional_Split_Action_Filter_Passes"])
    assert bool(filtered.loc["L0", "Distributional_Split_Action_Filter_Passes"])
    assert not bool(filtered.loc["L0", "Distributional_Split_Action_Filtered"])
    assert metadata["distributional_action_split_filter_filtered_parent_count"] == 0
    assert metadata["distributional_action_split_filter_filtered_edge_count"] == 0
    assert metadata["distributional_action_split_filter_open_parent_count_before"] == 2
    assert metadata["distributional_action_split_filter_open_parent_count_after"] == 2
