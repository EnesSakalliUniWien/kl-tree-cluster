from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from tree_break_selection.core_utils.tree_utils import bottom_up_nodes
from tree_break_selection.tree.construction import SUPPORTED_TREE_BUILDERS, build_tree


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.0], [0.1], [2.0], [2.1]],
        index=[10, 11, 20, 21],
        columns=["x"],
    )


def test_supported_tree_builder_order_is_explicit() -> None:
    assert SUPPORTED_TREE_BUILDERS == ("linkage", "neighbor_joining", "iqtree3")


def test_linkage_builder_preserves_input_index_labels() -> None:
    data = _data()

    result = build_tree(
        data,
        pdist(data.to_numpy()),
        builder="linkage",
        rooting="linkage_root",
        linkage_method="average",
    )

    assert result.builder == "linkage"
    assert result.linkage_matrix is not None
    assert result.tree.get_leaves() == [10, 11, 20, 21]


@pytest.mark.parametrize(
    ("builder", "rooting", "message"),
    [
        ("linkage", "mad", "Linkage trees require"),
        ("neighbor_joining", "linkage_root", "require rooting='mad'"),
        ("iqtree3", "linkage_root", "require rooting='mad'"),
    ],
)
def test_builder_rejects_incompatible_rooting(
    builder: str,
    rooting: str,
    message: str,
) -> None:
    data = _data()

    with pytest.raises(ValueError, match=message):
        build_tree(
            data,
            pdist(data.to_numpy()),
            builder=builder,
            rooting=rooting,
            linkage_method="average",
        )


def test_bottom_up_order_does_not_depend_on_node_insertion_order() -> None:
    first = nx.DiGraph()
    first.add_edges_from([("root", "right"), ("root", "left")])
    second = nx.DiGraph()
    second.add_edges_from([("root", "left"), ("root", "right")])

    first_order = list(bottom_up_nodes(first))
    second_order = list(bottom_up_nodes(second))

    assert first_order == second_order
    assert first_order[-1] == "root"
