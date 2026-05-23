from __future__ import annotations

import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.child_parent_edge_metadata import (
    determine_whether_sibling_pair_is_gate2_blocked,
    determine_whether_sibling_pair_is_null_like,
    estimate_sibling_null_prior_from_child_parent_edges,
    extract_child_parent_edge_pvalues_by_node,
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
    pvalues_by_node = extract_child_parent_edge_pvalues_by_node(annotations)
    tested_by_node, blocked_by_node = extract_child_parent_edge_testing_status_by_node(
        annotations
    )

    assert significance_by_node == {0: False, 1: True, 2: False}
    assert pvalues_by_node == {0: 1.0, 1: 0.03, 2: 0.8}
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
    assert estimate_sibling_null_prior_from_child_parent_edges(
        1,
        2,
        child_parent_edge_pvalues_by_node=pvalues_by_node,
        child_parent_edge_tested_by_node=tested_by_node,
        child_parent_edge_ancestor_blocked_by_node=blocked_by_node,
    ) == pytest.approx(0.03)


def test_child_parent_edge_prior_allows_missing_pvalue_only_for_untested_edge() -> None:
    pvalues_by_node = {"L": float("nan"), "R": 0.25}
    tested_by_node = {"L": False, "R": True}
    blocked_by_node = {"L": True, "R": False}

    assert estimate_sibling_null_prior_from_child_parent_edges(
        "L",
        "R",
        child_parent_edge_pvalues_by_node=pvalues_by_node,
        child_parent_edge_tested_by_node=tested_by_node,
        child_parent_edge_ancestor_blocked_by_node=blocked_by_node,
    ) == pytest.approx(0.25)

    tested_by_node["L"] = True
    blocked_by_node["L"] = False
    with pytest.raises(ValueError, match="unless the child edge was not tested"):
        estimate_sibling_null_prior_from_child_parent_edges(
            "L",
            "R",
            child_parent_edge_pvalues_by_node=pvalues_by_node,
            child_parent_edge_tested_by_node=tested_by_node,
            child_parent_edge_ancestor_blocked_by_node=blocked_by_node,
        )
