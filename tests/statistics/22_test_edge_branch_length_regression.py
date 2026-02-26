from __future__ import annotations

import networkx as nx
import numpy as np

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_mean_branch_length,
    _compute_p_values_via_projection,
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
    monkeypatch,
) -> list[tuple[float | None, float | None]]:
    captured: list[tuple[float | None, float | None]] = []

    # Enable Felsenstein scaling so mean_branch_length is computed
    monkeypatch.setattr(config, "FELSENSTEIN_SCALING", True)

    def _fake_projected_test(
        child_dist: np.ndarray,
        parent_dist: np.ndarray,
        n_child: int,
        n_parent: int,
        seed: int,
        branch_length: float | None = None,
        mean_branch_length: float | None = None,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
    ) -> tuple[float, float, float, bool]:
        captured.append((branch_length, mean_branch_length))
        return 0.0, 1.0, 1.0, False

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance._compute_projected_test",
        _fake_projected_test,
    )

    _compute_p_values_via_projection(
        tree=tree,
        child_ids=["A", "B"],
        parent_ids=["root", "root"],
        child_leaf_counts=np.array([5, 5], dtype=float),
        parent_leaf_counts=np.array([10, 10], dtype=float),
    )
    return captured


def test_no_branch_lengths_have_no_arbitrary_mean_normalization(monkeypatch) -> None:
    tree = _make_two_edge_tree(left_branch_length=None, right_branch_length=None)

    # Regression contract: no branch lengths => no normalization constant fallback.
    assert _compute_mean_branch_length(tree) is None

    captured = _run_edge_projection_with_capture(tree, monkeypatch)
    assert captured == [(None, None), (None, None)]


def test_mixed_missing_and_present_branch_lengths_use_only_valid_edges(monkeypatch) -> None:
    tree = _make_two_edge_tree(left_branch_length=2.0, right_branch_length=None)

    assert _compute_mean_branch_length(tree) == 2.0

    captured = _run_edge_projection_with_capture(tree, monkeypatch)
    assert captured == [(2.0, 2.0), (None, 2.0)]


def test_positive_branch_lengths_apply_tree_mean_normalization(monkeypatch) -> None:
    tree = _make_two_edge_tree(left_branch_length=1.0, right_branch_length=3.0)

    assert _compute_mean_branch_length(tree) == 2.0

    captured = _run_edge_projection_with_capture(tree, monkeypatch)
    assert captured == [(1.0, 2.0), (3.0, 2.0)]


def test_non_positive_branch_lengths_disable_adjustment(monkeypatch) -> None:
    """Non-positive branch lengths are invalid for Felsenstein scaling.

    Complex real-world failure mode:
    - tree has edge attributes, but values are 0 or negative
    - fallback mean=1.0 makes these values "look valid" and can shrink variance
      (for negatives), inflating z-scores.
    """
    tree = _make_two_edge_tree(left_branch_length=0.0, right_branch_length=-2.0)

    # Regression contract: no positive branch lengths => no BL normalization.
    assert _compute_mean_branch_length(tree) is None

    captured = _run_edge_projection_with_capture(tree, monkeypatch)
    assert captured == [(None, None), (None, None)]


def test_mixed_positive_and_invalid_branch_lengths_use_only_positive(monkeypatch) -> None:
    """Only strictly positive finite branch lengths should drive normalization."""
    tree = _make_two_edge_tree(left_branch_length=-1.0, right_branch_length=3.0)

    # Regression contract: mean is computed from valid positive edges only.
    assert _compute_mean_branch_length(tree) == 3.0

    captured = _run_edge_projection_with_capture(tree, monkeypatch)
    assert captured == [(None, 3.0), (3.0, 3.0)]


def test_non_finite_branch_length_values_are_ignored_for_adjustment(monkeypatch) -> None:
    """NaN branch length metadata should not activate BL scaling."""
    tree = _make_two_edge_tree(left_branch_length=float("nan"), right_branch_length=4.0)

    # Regression contract: non-finite values do not contribute to mean BL.
    assert _compute_mean_branch_length(tree) == 4.0

    captured = _run_edge_projection_with_capture(tree, monkeypatch)
    assert captured == [(None, 4.0), (4.0, 4.0)]
