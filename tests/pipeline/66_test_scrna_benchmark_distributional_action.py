from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns
from tree_break_selection.tree.poset_tree import PosetTree

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kl_te_cluster_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/kl_te_cluster_numba")


def _small_benchmark_tree() -> PosetTree:
    tree = PosetTree()
    for node_id, is_leaf in [
        ("root", False),
        ("left", False),
        ("right", False),
        ("L0", True),
        ("L1", True),
        ("L2", True),
        ("L3", True),
    ]:
        tree.add_node(node_id, is_leaf=is_leaf, label=node_id)
    for parent, child, length in [
        ("root", "left", 10.0),
        ("root", "right", 0.1),
        ("left", "L0", 0.5),
        ("left", "L1", 0.5),
        ("right", "L2", 0.5),
        ("right", "L3", 0.5),
    ]:
        tree.add_edge(parent, child, branch_length=length)
    tree.graph["root"] = "root"
    return tree


def test_scrna_benchmark_tree_edges_include_distributional_action_and_mass(
    tmp_path: Path,
) -> None:
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    pancreas = importlib.import_module("applications.scrna.pancreas_benchmark")

    leaf_data = pd.DataFrame(
        {
            "PC1": [0.0, 0.0, 4.0, 4.0],
            "PC2": [0.0, 0.0, 0.0, 0.0],
        },
        index=["L0", "L1", "L2", "L3"],
    )
    feature_space = continuous_feature_space_from_columns(tuple(leaf_data.columns))
    tree = _small_benchmark_tree()
    tree.populate_node_divergences(leaf_data, feature_space=feature_space)
    tree.annotations_df["Distributional_Action"] = np.nan
    tree.annotations_df["Distributional_Split_Action"] = np.nan
    tree.annotations_df["Distributional_Split_Action_Filtered"] = False
    tree.annotations_df.loc["left", "Distributional_Action"] = 123.0
    tree.annotations_df.loc["left", "Distributional_Split_Action"] = 456.0
    tree.annotations_df.loc["left", "Distributional_Split_Action_Filtered"] = True

    config = pancreas.MethodConfig(
        method_id="tbs",
        label="TBS action benchmark contract",
        params={},
    )
    pancreas._write_tbs_tree_diagnostics(
        config=config,
        result_extra={
            "tree": tree,
            "annotations": tree.annotations_df,
            "edge_branch_length_variance_policy": "none",
        },
        leaf_data=leaf_data,
        feature_space=feature_space,
        sample_ids=leaf_data.index.to_numpy(),
        y_true=np.array(["a", "a", "b", "b"]),
        output_dir=tmp_path,
        length_sensitivity_rows=[],
    )

    edge_path = tmp_path / f"{config.assignment_key}_tree_edges.csv"
    edges = pd.read_csv(edge_path)
    expected_columns = {
        "distributional_action_parent_mass",
        "distributional_action_child_mass",
        "distributional_action_child_parent_mass_fraction",
        "distributional_action_squared_displacement",
        "distributional_action",
        "distributional_action_per_branch_length",
        "Distributional_Action",
        "Distributional_Split_Action",
        "Distributional_Split_Action_Filtered",
    }
    assert expected_columns.issubset(edges.columns)

    left_edge = edges[(edges["parent"] == "root") & (edges["child"] == "left")].iloc[0]
    right_edge = edges[(edges["parent"] == "root") & (edges["child"] == "right")].iloc[0]
    assert left_edge["distributional_action_parent_mass"] == 4.0
    assert left_edge["distributional_action_child_mass"] == 2.0
    assert left_edge["distributional_action_child_parent_mass_fraction"] == 0.5
    assert left_edge["distributional_action_squared_displacement"] == 4.0
    assert left_edge["distributional_action"] == 8.0
    assert left_edge["Distributional_Action"] == 123.0
    assert left_edge["Distributional_Split_Action"] == 456.0
    assert bool(left_edge["Distributional_Split_Action_Filtered"])
    assert right_edge["distributional_action"] == 8.0
    assert left_edge["branch_length"] != right_edge["branch_length"]
    assert left_edge["distributional_action"] == right_edge["distributional_action"]
