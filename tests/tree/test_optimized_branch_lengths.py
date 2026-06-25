from __future__ import annotations

import numpy as np
import pandas as pd
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    fit_fixed_topology_nnls_branch_lengths,
)
from tree_break_selection.tree.poset_tree import PosetTree


def _small_binary_tree() -> PosetTree:
    tree = PosetTree()
    for node_id, is_leaf in [
        ("root", False),
        ("A", False),
        ("B", False),
        ("a", True),
        ("b", True),
        ("c", True),
        ("d", True),
    ]:
        tree.add_node(node_id, is_leaf=is_leaf, label=node_id)
    for parent, child in [
        ("root", "A"),
        ("root", "B"),
        ("A", "a"),
        ("A", "b"),
        ("B", "c"),
        ("B", "d"),
    ]:
        tree.add_edge(parent, child, branch_length=0.5)
    tree.graph["root"] = "root"
    return tree


def test_fixed_topology_nnls_recovers_additive_tree_distances() -> None:
    tree = _small_binary_tree()
    # A continuous path-incidence embedding: each coordinate is active for
    # leaves descending one tree edge. Squared standardized distances are then
    # exactly additive over the path separating two leaves.
    data = pd.DataFrame(
        {
            "root_A": [1.0, 1.0, 0.0, 0.0],
            "root_B": [0.0, 0.0, 1.0, 1.0],
            "A_a": [1.0, 0.0, 0.0, 0.0],
            "A_b": [0.0, 1.0, 0.0, 0.0],
            "B_c": [0.0, 0.0, 1.0, 0.0],
            "B_d": [0.0, 0.0, 0.0, 1.0],
        },
        index=["a", "b", "c", "d"],
        dtype=float,
    )

    result = fit_fixed_topology_nnls_branch_lengths(
        tree,
        data,
        pair_sample_size=None,
        random_state=0,
        solver_tolerance=1e-10,
    )

    assert result.status == "ok"
    assert result.n_pairs_used == 6
    assert result.residual_rmse < 1e-8
    assert result.branch_length_min >= -1e-10
    assert {
        tree.edges[parent, child]["branch_length_source"]
        for parent, child in tree.edges()
    } == {BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS}
    assert all("linkage_branch_length" in tree.edges[parent, child] for parent, child in tree.edges)
    assert np.isclose(
        tree.edges["root", "A"]["branch_length"],
        tree.edges["root", "B"]["branch_length"],
    )
    assert np.isclose(tree.edges["A", "a"]["branch_length"], tree.edges["A", "b"]["branch_length"])
    assert np.isclose(tree.edges["B", "c"]["branch_length"], tree.edges["B", "d"]["branch_length"])
