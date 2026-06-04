from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    run_child_parent_tests_across_tree,
)


def _make_two_edge_tree(
    left_branch_length: float | None,
    right_branch_length: float | None,
) -> nx.DiGraph:
    tree = nx.DiGraph()

    left_edge_attrs = {}
    if left_branch_length is not None:
        left_edge_attrs["branch_length"] = left_branch_length
    tree.add_edge("root", "A", **left_edge_attrs)

    right_edge_attrs = {}
    if right_branch_length is not None:
        right_edge_attrs["branch_length"] = right_branch_length
    tree.add_edge("root", "B", **right_edge_attrs)

    tree.nodes["root"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["A"]["distribution"] = np.array([0.3, 0.7], dtype=float)
    tree.nodes["B"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    return tree


def _run_edge_projection_with_capture(
    tree: nx.DiGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    captured: list[tuple[int, int]] = []

    def _fake_projected_test(
        child_dist: np.ndarray,
        parent_dist: np.ndarray,
        n_child: int,
        n_parent: int,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
        feature_space: object | None = None,
        continuous_covariance_by_block: object | None = None,
    ) -> tuple[float, float, float, bool]:
        captured.append((n_child, n_parent))
        return 0.0, 1.0, 1.0, False

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.tree_testing.run_child_parent_projected_wald_test",
        _fake_projected_test,
    )

    run_child_parent_tests_across_tree(
        tree=tree,
        child_ids=["A", "B"],
        parent_ids=["root", "root"],
        child_leaf_counts=np.array([5, 5], dtype=float),
        parent_leaf_counts=np.array([10, 10], dtype=float),
        spectral_dims={"root": 1},
        pca_projections={"root": np.array([[1.0, 0.0]], dtype=float)},
        pca_eigenvalues={"root": np.array([1.0], dtype=float)},
    )
    return captured


def test_branch_length_utility_uses_only_positive_observations() -> None:
    tree = _make_two_edge_tree(left_branch_length=1.0, right_branch_length=3.0)

    assert compute_mean_branch_length(tree) == 2.0


def test_branch_length_utility_returns_none_without_positive_observations() -> None:
    tree = _make_two_edge_tree(left_branch_length=0.0, right_branch_length=None)

    assert compute_mean_branch_length(tree) is None


def test_branch_length_utility_rejects_malformed_observations() -> None:
    tree = _make_two_edge_tree(left_branch_length=float("nan"), right_branch_length=4.0)

    with pytest.raises(ValueError, match="finite non-negative branch length"):
        compute_mean_branch_length(tree)


def test_edge_projection_does_not_use_branch_length_variance_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _make_two_edge_tree(left_branch_length=1.0, right_branch_length=3.0)

    captured = _run_edge_projection_with_capture(tree, monkeypatch)

    assert captured == [(5, 10), (5, 10)]


def test_edge_projection_does_not_validate_unused_branch_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _make_two_edge_tree(left_branch_length=float("nan"), right_branch_length=-1.0)

    captured = _run_edge_projection_with_capture(tree, monkeypatch)

    assert captured == [(5, 10), (5, 10)]
