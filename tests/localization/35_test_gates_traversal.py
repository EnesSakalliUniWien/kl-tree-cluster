"""Tests for gate evaluation and the live TreeDecomposition traversal path."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    GateEvaluator,
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
    local_significant: dict[str, bool] | None = None,
    sibling_different: dict[str, bool] | None = None,
    sibling_skipped: dict[str, bool] | None = None,
    children_map: dict[str, list[str]] | None = None,
    descendant_leaf_sets: dict[str, set[str]] | None = None,
    has_descendant_split: dict[str, bool] | None = None,
    passthrough: bool = False,
) -> GateEvaluator:
    if tree is None:
        tree = _make_binary_tree()

    if children_map is None:
        children_map = {node: list(tree.successors(node)) for node in tree.nodes}

    if descendant_leaf_sets is None:
        descendant_leaf_sets = {}
        for node in tree.nodes:
            if tree.nodes[node].get("is_leaf", False):
                descendant_leaf_sets[node] = {node}
            else:
                descendant_leaf_sets[node] = {
                    child
                    for child in nx.descendants(tree, node)
                    if tree.nodes[child].get("is_leaf", False)
                }

    if local_significant is None:
        local_significant = {node: True for node in tree.nodes}

    if sibling_different is None:
        sibling_different = {node: True for node in tree.nodes}

    if sibling_skipped is None:
        sibling_skipped = {node: False for node in tree.nodes}

    return GateEvaluator(
        tree=tree,
        local_significant=local_significant,
        sibling_different=sibling_different,
        sibling_skipped=sibling_skipped,
        children_map=children_map,
        descendant_leaf_sets=descendant_leaf_sets,
        has_descendant_split=has_descendant_split or {},
        passthrough=passthrough,
    )


def _make_annotations(
    tree: nx.DiGraph,
    *,
    local_significant: dict[str, bool],
    sibling_different: dict[str, bool],
    sibling_skipped: dict[str, bool] | None = None,
) -> pd.DataFrame:
    if sibling_skipped is None:
        sibling_skipped = {node: False for node in tree.nodes}

    return pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": pd.Series(local_significant, dtype=bool),
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


class TestGateEvaluator:
    def test_gate1_nonbinary_node_returns_false(self) -> None:
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A"), ("root", "B"), ("root", "C")])
        _annotate_tree_structure(tree, {"A", "B", "C"})

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A", "B", "C"], "A": [], "B": [], "C": []},
            descendant_leaf_sets={"root": {"A", "B", "C"}, "A": {"A"}, "B": {"B"}, "C": {"C"}},
        )
        assert gate.should_split("root") is False

    def test_gate1_single_child_returns_false(self) -> None:
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A")])
        _annotate_tree_structure(tree, {"A"})

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A"], "A": []},
            descendant_leaf_sets={"root": {"A"}, "A": {"A"}},
        )
        assert gate.should_split("root") is False

    def test_gate2_neither_child_diverges(self) -> None:
        gate = _make_gate(
            local_significant={
                "root": True,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.should_split("root") is False

    def test_gate2_one_child_diverges(self) -> None:
        gate = _make_gate(
            local_significant={
                "root": True,
                "L": True,
                "R": False,
                "L1": True,
                "L2": True,
                "R1": True,
                "R2": True,
            },
        )
        assert gate.should_split("root") is True

    def test_gate2_missing_annotations_raises(self) -> None:
        gate = _make_gate(local_significant={})
        with pytest.raises(ValueError, match="Missing child-parent divergence"):
            gate.should_split("root")

    def test_gate3_siblings_same(self) -> None:
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
        assert gate.should_split("root") is False

    def test_gate3_siblings_different(self) -> None:
        assert _make_gate().should_split("root") is True

    def test_gate3_skipped_returns_false(self) -> None:
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
        assert gate.should_split("root") is False

    def test_gate3_missing_annotations_raises(self) -> None:
        gate = _make_gate(sibling_different={})
        with pytest.raises(ValueError, match="Sibling divergence annotations missing"):
            gate.should_split("root")

    def test_passthrough_disabled_returns_false(self) -> None:
        gate = _make_gate(
            passthrough=False,
            has_descendant_split={"root": True},
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
        assert gate.should_pass_through("root") is False

    def test_passthrough_when_gate3_fails_with_descendant_signal(self) -> None:
        gate = _make_gate(
            passthrough=True,
            has_descendant_split={"root": True, "L": False, "R": False},
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
        assert gate.should_pass_through("root") is True

    def test_no_passthrough_when_gate3_passes(self) -> None:
        gate = _make_gate(
            passthrough=True,
            has_descendant_split={"root": True},
        )
        assert gate.should_split("root") is True
        assert gate.should_pass_through("root") is False

    def test_no_passthrough_when_gates_1_2_fail(self) -> None:
        gate = _make_gate(
            passthrough=True,
            has_descendant_split={"root": True},
            local_significant={
                "root": False,
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        assert gate.should_pass_through("root") is False

    def test_no_passthrough_when_no_descendant_signal(self) -> None:
        gate = _make_gate(
            passthrough=True,
            has_descendant_split={"root": False},
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
        assert gate.should_pass_through("root") is False


class TestTreeDecompositionTraversal:
    def test_decompose_tree_splits_to_leaf_clusters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tree = _make_binary_tree()
        annotations_df = _make_annotations(
            tree,
            local_significant={node: True for node in tree.nodes},
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

    def test_decompose_tree_passthrough_reaches_descendant_split(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = _make_deep_tree()
        local_significant = {node: True for node in tree.nodes}
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True

        annotations_df = _make_annotations(
            tree,
            local_significant=local_significant,
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
        local_significant = {node: True for node in tree.nodes}
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True

        annotations_df = _make_annotations(
            tree,
            local_significant=local_significant,
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
