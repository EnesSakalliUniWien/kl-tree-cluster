from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.diagnostics.oracle.gate_path_trace import build_gate_path_trace_dataframe
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.column_contracts import (
    EDGE_GATE_COLUMNS,
    SIBLING_GATE_COLUMNS,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _tree() -> PosetTree:
    linkage_matrix = np.asarray(
        [
            [0, 1, 1.0, 2],
            [2, 3, 1.0, 2],
            [4, 5, 2.0, 4],
        ]
    )
    tree = PosetTree.from_linkage(
        linkage_matrix,
        leaf_names=["A", "B", "C", "D"],
    )
    for node in tree.nodes:
        tree.nodes[node]["leaf_count"] = len(tree.get_leaves(node=node))
    return tree


def _annotations(tree: PosetTree) -> pd.DataFrame:
    annotations = pd.DataFrame(index=list(tree.nodes))
    for column in EDGE_GATE_COLUMNS:
        if column.endswith(("Invalid", "Tested", "Ancestor_Blocked", "Significant")):
            annotations[column] = False
        else:
            annotations[column] = np.nan
    for column in SIBLING_GATE_COLUMNS:
        if column in {
            "Sibling_Divergence_Skipped",
            "Sibling_Divergence_Invalid",
            "Sibling_BH_Different",
            "Sibling_BH_Same",
        }:
            annotations[column] = False
        elif column == "Sibling_Test_Method":
            annotations[column] = ""
        else:
            annotations[column] = np.nan
    return annotations


def test_gate_path_trace_marks_oracle_boundary_split_by_actual_traversal() -> None:
    tree = _tree()
    annotations = _annotations(tree)
    root = tree.root()
    left_internal, right_internal = list(tree.successors(root))

    annotations.loc[[left_internal, right_internal], "Child_Parent_Divergence_Tested"] = True
    annotations.loc[
        [left_internal, right_internal],
        "Child_Parent_Divergence_Significant",
    ] = True
    annotations.loc[[left_internal, right_internal], "Child_Parent_Divergence_P_Value"] = 0.001
    annotations.loc[
        [left_internal, right_internal],
        "Child_Parent_Divergence_P_Value_BH",
    ] = 0.002
    annotations.loc[root, "Sibling_Test_Statistic"] = 10.0
    annotations.loc[root, "Sibling_Degrees_of_Freedom"] = 2.0
    annotations.loc[root, "Sibling_Divergence_P_Value"] = 0.001
    annotations.loc[root, "Sibling_Divergence_P_Value_Corrected"] = 0.002
    annotations.loc[root, "Sibling_BH_Different"] = True

    decomposition = {
        "cluster_assignments": {
            0: {"root_node": left_internal, "leaves": ["A", "B"], "size": 2},
            1: {"root_node": right_internal, "leaves": ["C", "D"], "size": 2},
        }
    }

    trace = build_gate_path_trace_dataframe(
        tree=tree,
        annotations_df=annotations,
        decomposition=decomposition,
        oracle_true_k_boundary_nodes=(root,),
        oracle_any_k_boundary_nodes=(root,),
        sibling_inflation_trace_by_parent={},
        case_id="synthetic",
        failure_class="gate_over_split",
        kl_ari=0.0,
        oracle_true_k_ari=1.0,
        oracle_any_k_ari=1.0,
    )

    root_row = trace.loc[trace["node_id"] == root].iloc[0]
    assert root_row["actual_decision"] == "split"
    assert root_row["trace_relation"] == "actual_splits_oracle_boundary"
    assert bool(root_row["oracle_true_k_boundary"])
