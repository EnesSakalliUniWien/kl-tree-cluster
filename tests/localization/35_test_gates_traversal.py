"""Unit tests for decomposition gate-traversal helpers and GateEvaluator.

Covers:
- ``iterate_worklist`` — LIFO pop, dedup via processed set
- ``process_node`` — split/merge dispatch
- ``GateEvaluator`` — gate evaluation logic with mocked annotations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.traversal import (
    GateEvaluator,
    iterate_worklist,
    process_node,
)

# =====================================================================
# Helpers
# =====================================================================


def _make_binary_tree() -> nx.DiGraph:
    """Small binary tree for gate tests.

    Structure::

            root
           /    \\
          L      R
         / \\   / \\
        L1  L2 R1  R2
    """
    tree = nx.DiGraph()
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
    for node in tree.nodes:
        tree.nodes[node]["is_leaf"] = node in ("L1", "L2", "R1", "R2")
        tree.nodes[node]["label"] = node
    return tree


def _make_gate(
    tree: nx.DiGraph | None = None,
    *,
    local_significant: dict | None = None,
    sibling_different: dict | None = None,
    sibling_skipped: dict | None = None,
    children_map: dict | None = None,
    descendant_leaf_sets: dict | None = None,
) -> GateEvaluator:
    """Build a GateEvaluator with explicit overrides."""
    if tree is None:
        tree = _make_binary_tree()

    if children_map is None:
        children_map = {
            n: list(tree.successors(n))
            for n in tree.nodes
            if not tree.nodes[n].get("is_leaf", False)
        }
        # leaves map to empty list
        for n in tree.nodes:
            children_map.setdefault(n, [])

    if descendant_leaf_sets is None:
        descendant_leaf_sets = {}
        for n in tree.nodes:
            if tree.nodes[n].get("is_leaf", False):
                descendant_leaf_sets[n] = {n}
            else:
                descendant_leaf_sets[n] = {
                    d for d in nx.descendants(tree, n) if tree.nodes[d].get("is_leaf")
                }

    if local_significant is None:
        # default: all children are significant
        local_significant = {n: True for n in tree.nodes}

    if sibling_different is None:
        # default: all internal nodes have "different" siblings
        sibling_different = {n: True for n in tree.nodes}

    if sibling_skipped is None:
        sibling_skipped = {n: False for n in tree.nodes}

    return GateEvaluator(
        tree=tree,
        local_significant=local_significant,
        sibling_different=sibling_different,
        sibling_skipped=sibling_skipped,
        children_map=children_map,
        descendant_leaf_sets=descendant_leaf_sets,
    )


def _dummy_test_divergence(node_a: str, node_b: str):
    """Return a fixed (stat, df, p_value) triple."""
    return (1.0, 1.0, 0.5)


# =====================================================================
# iterate_worklist
# =====================================================================


class TestIterateWorklist:
    def test_basic_lifo(self):
        """Yields items in LIFO (stack) order."""
        nodes = ["A", "B", "C"]
        processed: set[str] = set()
        result = list(iterate_worklist(nodes, processed))
        assert result == ["C", "B", "A"]

    def test_dedup_via_processed(self):
        """Already-processed nodes are skipped."""
        nodes = ["A", "B", "C"]
        processed = {"B"}
        result = list(iterate_worklist(nodes, processed))
        assert result == ["C", "A"]
        assert "B" in processed

    def test_marks_as_processed(self):
        """All yielded nodes are added to the processed set."""
        nodes = ["X", "Y"]
        processed: set[str] = set()
        list(iterate_worklist(nodes, processed))
        assert processed == {"X", "Y"}

    def test_empty_worklist(self):
        nodes: list[str] = []
        processed: set[str] = set()
        result = list(iterate_worklist(nodes, processed))
        assert result == []

    def test_mutations_between_yields(self):
        """Caller can push new items onto the worklist between yields."""
        nodes = ["root"]
        processed: set[str] = set()
        collected = []
        for node in iterate_worklist(nodes, processed):
            collected.append(node)
            if node == "root":
                nodes.append("child_R")
                nodes.append("child_L")
        assert collected == ["root", "child_L", "child_R"]

    def test_duplicate_push_is_skipped(self):
        """Pushing a node twice only yields it once."""
        nodes = ["A"]
        processed: set[str] = set()
        collected = []
        for node in iterate_worklist(nodes, processed):
            collected.append(node)
            if node == "A":
                nodes.append("B")
                nodes.append("B")  # duplicate
        assert collected == ["A", "B"]


# =====================================================================
# process_node (v1)
# =====================================================================


class TestProcessNode:
    def test_split_pushes_children(self):
        """When gate says split, children are pushed right-then-left."""
        gate = _make_gate()
        worklist: list[str] = []
        clusters: list[set[str]] = []

        process_node("root", gate, worklist, clusters)

        # right pushed first, then left (so left pops first)
        assert worklist == ["R", "L"]
        assert clusters == []

    def test_merge_collects_leaves(self):
        """When gate says merge, all descendant leaves become a cluster."""
        gate = _make_gate(
            # Gate 2 fails → merge
            local_significant={
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
                "root": True,
            },
        )
        worklist: list[str] = []
        clusters: list[set[str]] = []

        process_node("root", gate, worklist, clusters)

        assert worklist == []
        assert len(clusters) == 1
        assert clusters[0] == {"L1", "L2", "R1", "R2"}

    def test_split_at_inner_node(self):
        """Splitting at inner node L pushes L1 and L2."""
        gate = _make_gate()
        worklist: list[str] = []
        clusters: list[set[str]] = []

        process_node("L", gate, worklist, clusters)

        assert worklist == ["L2", "L1"]

    def test_merge_at_inner_node(self):
        """Merge at L collects {L1, L2}."""
        gate = _make_gate(
            sibling_different={
                "root": True,
                "L": False,
                "R": True,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
            },
        )
        worklist: list[str] = []
        clusters: list[set[str]] = []

        process_node("L", gate, worklist, clusters)

        assert worklist == []
        assert clusters[0] == {"L1", "L2"}


# =====================================================================
# GateEvaluator
# =====================================================================


class TestGateEvaluator:
    """Unit tests for GateEvaluator gate logic."""

    # --- Gate 1: Binary structure ---

    def test_gate1_nonbinary_node_returns_false(self):
        """Non-binary node (3 children) → should_split returns False."""
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A"), ("root", "B"), ("root", "C")])
        for n in tree.nodes:
            tree.nodes[n]["is_leaf"] = n != "root"

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A", "B", "C"], "A": [], "B": [], "C": []},
            descendant_leaf_sets={"root": {"A", "B", "C"}, "A": {"A"}, "B": {"B"}, "C": {"C"}},
        )
        assert gate.should_split("root") is False

    def test_gate1_single_child_returns_false(self):
        """Single child → should_split returns False."""
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A")])
        for n in tree.nodes:
            tree.nodes[n]["is_leaf"] = n != "root"

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A"], "A": []},
            descendant_leaf_sets={"root": {"A"}, "A": {"A"}},
        )
        assert gate.should_split("root") is False

    # --- Gate 2: Child-parent divergence ---

    def test_gate2_neither_child_diverges(self):
        """Neither child edge-significant → merge (no signal)."""
        gate = _make_gate(
            local_significant={
                "L": False,
                "R": False,
                "L1": False,
                "L2": False,
                "R1": False,
                "R2": False,
                "root": True,
            },
        )
        assert gate.should_split("root") is False

    def test_gate2_one_child_diverges(self):
        """One child significant, other not → passes Gate 2."""
        gate = _make_gate(
            local_significant={
                "L": True,
                "R": False,
                "L1": True,
                "L2": True,
                "R1": True,
                "R2": True,
                "root": True,
            },
        )
        # Gate 2 passes (left diverges), Gate 3 also passes (default: different)
        assert gate.should_split("root") is True

    def test_gate2_missing_annotations_raises(self):
        """Missing edge annotations raise ValueError."""
        gate = _make_gate(local_significant={})
        with pytest.raises(ValueError, match="Missing child-parent divergence"):
            gate.should_split("root")

    # --- Gate 3: Sibling divergence ---

    def test_gate3_siblings_same(self):
        """Siblings same → merge."""
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

    def test_gate3_siblings_different(self):
        """Siblings different → split."""
        gate = _make_gate()  # default: all different
        assert gate.should_split("root") is True

    def test_gate3_skipped_returns_false(self):
        """Skipped sibling test is conservative → merge."""
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

    def test_gate3_missing_annotations_raises(self):
        """Missing sibling divergence annotations raise ValueError."""
        gate = _make_gate(sibling_different={})
        with pytest.raises(ValueError, match="Sibling divergence annotations missing"):
            gate.should_split("root")

    # --- All gates combined ---

    def test_all_gates_pass(self):
        """Binary + both children diverge + siblings different → split."""
        gate = _make_gate()  # defaults: all significant, all different
        assert gate.should_split("root") is True


# =====================================================================
# Full-traversal integration smoke test
# =====================================================================


class TestTraversalIntegration:
    """Smoke test combining iterate_worklist + process_node."""

    def test_v1_full_split_traversal(self):
        """Fully splitting tree gives 4 leaf clusters."""
        gate = _make_gate()  # all gates pass at all levels
        worklist = ["root"]
        processed: set[str] = set()
        clusters: list[set[str]] = []

        for node in iterate_worklist(worklist, processed):
            process_node(node, gate, worklist, clusters)

        # Should have 4 singleton-ish leaf clusters
        # root splits → L, R
        # L splits → L1, L2  (leaves, gate has empty children, so depends on gate)
        # For leaves the children_map has [] so len != 2 → merge (leaf cluster)
        all_leaves = set()
        for c in clusters:
            all_leaves.update(c)
        assert all_leaves == {"L1", "L2", "R1", "R2"}

    def test_v1_merge_at_root(self):
        """Merge at root gives single cluster with all leaves."""
        gate = _make_gate(
            local_significant={n: False for n in ["L", "R", "L1", "L2", "R1", "R2", "root"]},
        )
        worklist = ["root"]
        processed: set[str] = set()
        clusters: list[set[str]] = []

        for node in iterate_worklist(worklist, processed):
            process_node(node, gate, worklist, clusters)

        assert len(clusters) == 1
        assert clusters[0] == {"L1", "L2", "R1", "R2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
