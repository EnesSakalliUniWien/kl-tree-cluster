"""Tests for gate evaluation and the live TreeDecomposition traversal path."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    ClusterBoundary,
    build_cluster_assignments,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    GateEvaluator,
    TraversalDecision,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _annotate_tree_structure(tree: nx.DiGraph, leaves: set[str]) -> None:
    for node in tree.nodes:
        tree.nodes[node]["is_leaf"] = node in leaves
        tree.nodes[node]["label"] = node
        tree.nodes[node]["distribution"] = np.array([0.5], dtype=float)


def _make_binary_tree() -> PosetTree:
    tree = PosetTree()
    tree.add_edges_from(
        [
            ("root", "L"),
            ("root", "R"),
            ("L", "L1"),
            ("L", "L2"),
            ("R", "R1"),
            ("R", "R2"),
        ]
    )
    _annotate_tree_structure(tree, {"L1", "L2", "R1", "R2"})
    return tree


def _make_deep_tree() -> PosetTree:
    tree = PosetTree()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "A1"),
            ("A", "A2"),
            ("B", "C"),
            ("B", "D"),
            ("C", "C1"),
            ("C", "C2"),
            ("D", "D1"),
            ("D", "D2"),
        ]
    )
    _annotate_tree_structure(tree, {"A1", "A2", "C1", "C2", "D1", "D2"})
    return tree


def _make_gate(
    tree: nx.DiGraph | None = None,
    *,
    edge_divergent: dict[str, bool] | None = None,
    sibling_different: dict[str, bool] | None = None,
    sibling_skipped: dict[str, bool] | None = None,
    children_map: dict[str, list[str]] | None = None,
    passthrough: bool = False,
) -> GateEvaluator:
    if tree is None:
        tree = _make_binary_tree()

    if children_map is None:
        children_map = {node: list(tree.successors(node)) for node in tree.nodes}

    if edge_divergent is None:
        edge_divergent = {node: True for node in tree.nodes}

    if sibling_different is None:
        sibling_different = {node: True for node in tree.nodes}

    if sibling_skipped is None:
        sibling_skipped = {node: False for node in tree.nodes}

    return GateEvaluator(
        tree=tree,
        edge_divergent=edge_divergent,
        sibling_different=sibling_different,
        sibling_skipped=sibling_skipped,
        children_map=children_map,
        passthrough=passthrough,
    )


def _make_annotations(
    tree: nx.DiGraph,
    *,
    edge_divergent: dict[str, bool],
    sibling_different: dict[str, bool],
    sibling_skipped: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if sibling_skipped is None:
        sibling_skipped = {node: False for node in tree.nodes}

    return pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": pd.Series(edge_divergent, dtype=bool),
            "Sibling_BH_Different": pd.Series(sibling_different, dtype=bool),
            "Sibling_Divergence_Skipped": pd.Series(sibling_skipped, dtype=bool),
        }
    ).reindex(list(tree.nodes))


def _decompose_with_annotations(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passthrough: bool,
) -> dict[str, object]:
    monkeypatch.setattr(TreeDecomposition, "_prepare_annotations", lambda self, df: df)
    decomposer = TreeDecomposition(
        tree=tree,
        annotations_df=annotations_df,
        passthrough=passthrough,
    )
    return decomposer.decompose_tree()


def _cluster_leaf_sets(decomposition_results: dict[str, object]) -> list[set[str]]:
    cluster_assignments = decomposition_results["cluster_assignments"]
    return [set(cluster["leaves"]) for cluster in cluster_assignments.values()]


def test_cluster_assignments_use_explicit_boundary_root() -> None:
    assignments = build_cluster_assignments(
        [
            ClusterBoundary(root_node="left_subtree", leaves=frozenset({"L1", "L2"})),
            ClusterBoundary(root_node="right_leaf", leaves=frozenset({"R"})),
        ]
    )

    assert assignments == {
        0: {"root_node": "left_subtree", "leaves": ["L1", "L2"], "size": 2},
        1: {"root_node": "right_leaf", "leaves": ["R"], "size": 1},
    }


class TestGateEvaluator:
    def test_gate1_nonbinary_node_returns_false(self) -> None:
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A"), ("root", "B"), ("root", "C")])
        _annotate_tree_structure(tree, {"A", "B", "C"})

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A", "B", "C"], "A": [], "B": [], "C": []},
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_gate1_single_child_returns_false(self) -> None:
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A")])
        _annotate_tree_structure(tree, {"A"})

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A"], "A": []},
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_edge_gate_neither_child_diverges(self) -> None:
        gate = _make_gate(
            edge_divergent={
                "root": True,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_edge_gate_one_child_diverges(self) -> None:
        gate = _make_gate(
            edge_divergent={
                "root": True,
                "L": True,
                "R": False,
                "L1": True,
                "L2": True,
                "R1": True,
                "R2": True,
            },
        )
        assert gate.decision("root") is TraversalDecision.SPLIT

    def test_edge_gate_missing_annotations_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing edge_divergent values"):
            _make_gate(edge_divergent={})

    def test_sibling_gate_siblings_same(self) -> None:
        gate = _make_gate(
            sibling_different={
                "root": False,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_sibling_gate_siblings_different(self) -> None:
        assert _make_gate().decision("root") is TraversalDecision.SPLIT

    def test_sibling_gate_skipped_returns_false(self) -> None:
        gate = _make_gate(
            sibling_skipped={
                "root": True,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_sibling_gate_missing_annotations_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing sibling_different values"):
            _make_gate(sibling_different={})

    def test_passthrough_disabled_returns_false(self) -> None:
        gate = _make_gate(
            passthrough=False,
            sibling_different={
                "root": False,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_passthrough_when_sibling_gate_fails_with_descendant_signal(self) -> None:
        tree = _make_deep_tree()
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True
        gate = _make_gate(
            tree=tree,
            passthrough=True,
            sibling_different=sibling_different,
        )
        assert gate.decision("root") is TraversalDecision.PASS_THROUGH

    def test_passthrough_decision_uses_cached_gate_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = _make_deep_tree()
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True
        gate = _make_gate(
            tree=tree,
            passthrough=True,
            sibling_different=sibling_different,
        )

        def _fail_recompute(*_args, **_kwargs):
            raise AssertionError("passthrough decision recomputed gate predicates")

        monkeypatch.setattr(gate, "_passes_split_prerequisites", _fail_recompute)
        monkeypatch.setattr(gate, "_sibling_gate_is_open", _fail_recompute)

        assert gate.decision("root") is TraversalDecision.PASS_THROUGH

    def test_no_passthrough_when_sibling_gate_passes(self) -> None:
        gate = _make_gate(
            passthrough=True,
        )
        assert gate.decision("root") is TraversalDecision.SPLIT

    def test_no_passthrough_when_gates_1_2_fail(self) -> None:
        gate = _make_gate(
            passthrough=True,
            edge_divergent={
                "root": False,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY

    def test_no_passthrough_when_no_descendant_signal(self) -> None:
        gate = _make_gate(
            passthrough=True,
            sibling_different={
                "root": False,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY


class TestTreeDecompositionTraversal:
    def test_decompose_tree_splits_to_leaf_clusters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tree = _make_binary_tree()
        annotations_df = _make_annotations(
            tree,
            edge_divergent={node: True for node in tree.nodes},
            sibling_different={node: True for node in tree.nodes},
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            monkeypatch,
            passthrough=False,
        )

        cluster_leaf_sets = sorted(_cluster_leaf_sets(result), key=lambda leaves: min(leaves))
        assert cluster_leaf_sets == [{"L1"}, {"L2"}, {"R1"}, {"R2"}]

    def test_decompose_tree_reports_edge_and_sibling_alpha(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = _make_binary_tree()
        annotations_df = _make_annotations(
            tree,
            edge_divergent={node: True for node in tree.nodes},
            sibling_different={node: False for node in tree.nodes},
        )

        monkeypatch.setattr(TreeDecomposition, "_prepare_annotations", lambda self, df: df)
        decomposer = TreeDecomposition(
            tree=tree,
            annotations_df=annotations_df,
            edge_alpha=0.007,
            sibling_alpha=0.123,
            passthrough=False,
        )

        result = decomposer.decompose_tree()

        assert result["independence_analysis"]["edge_alpha"] == 0.007
        assert result["independence_analysis"]["sibling_alpha"] == 0.123

    def test_decompose_tree_passthrough_reaches_descendant_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = _make_deep_tree()
        edge_divergent = {node: True for node in tree.nodes}
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True

        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            monkeypatch,
            passthrough=True,
        )

        cluster_leaf_sets = sorted(_cluster_leaf_sets(result), key=lambda leaves: min(leaves))
        assert cluster_leaf_sets == [{"A1", "A2"}, {"C1", "C2"}, {"D1", "D2"}]

    def test_decompose_tree_without_passthrough_merges_at_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = _make_deep_tree()
        edge_divergent = {node: True for node in tree.nodes}
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True

        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            monkeypatch,
            passthrough=False,
        )

        cluster_leaf_sets = _cluster_leaf_sets(result)
        assert cluster_leaf_sets == [{"A1", "A2", "C1", "C2", "D1", "D2"}]

    def test_decompose_tree_passthrough_ignores_blocked_descendant_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = _make_deep_tree()
        edge_divergent = {node: True for node in tree.nodes}
        edge_divergent["C"] = False
        edge_divergent["D"] = False
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True

        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            monkeypatch,
            passthrough=True,
        )

        cluster_leaf_sets = _cluster_leaf_sets(result)
        assert cluster_leaf_sets == [{"A1", "A2", "C1", "C2", "D1", "D2"}]
