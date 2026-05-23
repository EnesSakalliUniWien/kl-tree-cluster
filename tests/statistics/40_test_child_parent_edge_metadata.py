from __future__ import annotations

import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.child_parent_edge_metadata import (
    determine_whether_sibling_pair_is_gate2_blocked,
    determine_whether_sibling_pair_is_null_like,
    extract_child_parent_edge_significance_by_node,
    extract_child_parent_edge_testing_status_by_node,
)


def test_child_parent_edge_metadata_preserves_tree_node_id_identity() -> None:
    annotations = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": [False, True, False],
            "Child_Parent_Divergence_P_Value_BH": [1.0, 0.03, 0.8],
            "Child_Parent_Divergence_Tested": [True, False, True],
            "Child_Parent_Divergence_Ancestor_Blocked": [False, True, False],
        },
        index=[0, 1, 2],
    )

    significance_by_node = extract_child_parent_edge_significance_by_node(annotations)
    tested_by_node, blocked_by_node = extract_child_parent_edge_testing_status_by_node(
        annotations
    )

    assert significance_by_node == {0: False, 1: True, 2: False}
    assert tested_by_node == {0: True, 1: False, 2: True}
    assert blocked_by_node == {0: False, 1: True, 2: False}

    assert (
        determine_whether_sibling_pair_is_gate2_blocked(
            1,
            2,
            child_parent_edge_tested_by_node=tested_by_node,
            child_parent_edge_ancestor_blocked_by_node=blocked_by_node,
        )
        is True
    )
    assert (
        determine_whether_sibling_pair_is_null_like(
            1,
            2,
            child_parent_edge_significance_by_node=significance_by_node,
        )
        is False
    )
