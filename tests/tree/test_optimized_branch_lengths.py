from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
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


def _prepared_geometry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "left": [1.0, 1.0, 0.0, 0.0],
            "right": [0.0, 0.0, 1.0, 1.0],
            "leaf_a": [1.0, 0.0, 0.0, 0.0],
            "leaf_b": [0.0, 1.0, 0.0, 0.0],
            "leaf_c": [0.0, 0.0, 1.0, 0.0],
            "leaf_d": [0.0, 0.0, 0.0, 1.0],
        },
        index=["a", "b", "c", "d"],
    )


def _dense_path_design(tree: PosetTree, leaf_labels: list[str]) -> np.ndarray:
    edge_index_by_pair = {edge: index for index, edge in enumerate(tree.edges())}
    rows: list[np.ndarray] = []
    undirected_tree = tree.to_undirected()
    for left_position, left_label in enumerate(leaf_labels):
        for right_label in leaf_labels[left_position + 1 :]:
            row = np.zeros(len(edge_index_by_pair), dtype=float)
            path = nx.shortest_path(undirected_tree, left_label, right_label)
            for parent, child in zip(path[:-1], path[1:], strict=True):
                directed_edge = (
                    (parent, child)
                    if (parent, child) in edge_index_by_pair
                    else (child, parent)
                )
                row[edge_index_by_pair[directed_edge]] = 1.0
            rows.append(row)
    return np.vstack(rows)


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
    assert result.design_nnz == 20
    assert np.isclose(result.design_density, 20.0 / 36.0)
    assert result.n_zero_design_columns == 0
    assert result.zero_design_column_fraction == 0.0
    assert result.applied_to_tree is True
    assert result.apply_nonconverged is False
    assert result.residual_rmse < 1e-8
    assert result.residual_rmse_to_target_mean < 1e-8
    assert result.branch_length_min >= -1e-10
    assert {
        tree.edges[parent, child]["branch_length_source"] for parent, child in tree.edges()
    } == {BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS}
    assert all("linkage_branch_length" in tree.edges[parent, child] for parent, child in tree.edges)
    assert np.isclose(
        tree.edges["root", "A"]["branch_length"],
        tree.edges["root", "B"]["branch_length"],
    )
    assert np.isclose(tree.edges["A", "a"]["branch_length"], tree.edges["A", "b"]["branch_length"])
    assert np.isclose(tree.edges["B", "c"]["branch_length"], tree.edges["B", "d"]["branch_length"])


def test_fixed_topology_nnls_uses_prepared_geometry_without_restandardizing() -> None:
    tree = _small_binary_tree()
    geometry = _prepared_geometry()

    result = fit_fixed_topology_nnls_branch_lengths(
        tree,
        geometry,
        target_metric=BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
        pair_sample_size=None,
        solver_tolerance=1e-10,
    )

    assert result.status == "ok"
    assert result.target_metric == BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN
    assert result.residual_rmse < 1e-8
    assert np.isclose(result.target_mean, 10.0 / 3.0)


def test_fixed_topology_nnls_matches_dense_scipy_nnls_on_tiny_tree() -> None:
    tree = _small_binary_tree()
    geometry = _prepared_geometry()
    design = _dense_path_design(tree, list(geometry.index))
    values = geometry.to_numpy(dtype=float)
    pair_targets = []
    for left_position in range(len(values)):
        for right_position in range(left_position + 1, len(values)):
            diff = values[left_position] - values[right_position]
            pair_targets.append(float(np.sum(diff * diff)))
    dense_solution, _ = nnls(design, np.asarray(pair_targets, dtype=float))

    result = fit_fixed_topology_nnls_branch_lengths(
        tree,
        geometry,
        target_metric=BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
        pair_sample_size=None,
        solver_tolerance=1e-10,
    )
    fitted_lengths = np.asarray(
        [tree.edges[parent, child]["branch_length"] for parent, child in tree.edges()],
        dtype=float,
    )

    assert result.status == "ok"
    assert np.allclose(design @ fitted_lengths, design @ dense_solution, atol=1e-8)
    assert np.isclose(
        np.linalg.norm((design @ fitted_lengths) - np.asarray(pair_targets, dtype=float)),
        np.linalg.norm((design @ dense_solution) - np.asarray(pair_targets, dtype=float)),
        atol=1e-8,
    )


def test_fixed_topology_nnls_does_not_mutate_tree_when_solver_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _small_binary_tree()
    geometry = _prepared_geometry()

    def fake_lsq_linear(design: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.full(design.shape[1], 9.0, dtype=float),
            success=False,
            cost=123.0,
            optimality=456.0,
            nit=7,
            message="forced non-convergence",
        )

    monkeypatch.setattr(
        "tree_break_selection.tree.optimized_branch_lengths.lsq_linear",
        fake_lsq_linear,
    )

    result = fit_fixed_topology_nnls_branch_lengths(
        tree,
        geometry,
        target_metric=BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
        pair_sample_size=None,
    )

    assert result.status == "solver_not_converged"
    assert result.applied_to_tree is False
    assert result.apply_nonconverged is False
    assert {attrs["branch_length"] for _, _, attrs in tree.edges(data=True)} == {0.5}
    assert all("linkage_branch_length" not in attrs for _, _, attrs in tree.edges(data=True))
    assert all("branch_length_source" not in attrs for _, _, attrs in tree.edges(data=True))
    assert tree.graph["branch_length_optimization"]["applied_to_tree"] is False


def test_fixed_topology_nnls_can_explicitly_apply_nonconverged_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _small_binary_tree()
    geometry = _prepared_geometry()

    def fake_lsq_linear(design: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.full(design.shape[1], 9.0, dtype=float),
            success=False,
            cost=123.0,
            optimality=456.0,
            nit=7,
            message="forced non-convergence",
        )

    monkeypatch.setattr(
        "tree_break_selection.tree.optimized_branch_lengths.lsq_linear",
        fake_lsq_linear,
    )

    result = fit_fixed_topology_nnls_branch_lengths(
        tree,
        geometry,
        target_metric=BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
        pair_sample_size=None,
        apply_nonconverged=True,
    )

    assert result.status == "solver_not_converged"
    assert result.applied_to_tree is True
    assert result.apply_nonconverged is True
    assert {attrs["branch_length"] for _, _, attrs in tree.edges(data=True)} == {9.0}
    assert {
        tree.edges[parent, child]["branch_length_source"] for parent, child in tree.edges()
    } == {BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS}
