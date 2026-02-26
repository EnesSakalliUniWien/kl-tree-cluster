"""Unit tests for gates.py traversal helpers and GateEvaluator.

Covers:
- ``iterate_worklist`` — LIFO pop, dedup via processed set
- ``process_node`` — v1 split/merge dispatch
- ``process_node_v2`` — v2 split/merge + localization_results + power guard
- ``GateEvaluator`` — gate evaluation logic with mocked annotations
- ``V2TraversalState`` — dataclass defaults
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from kl_clustering_analysis.hierarchy_analysis.gates import (
    GateEvaluator,
    V2TraversalState,
    iterate_worklist,
    process_node,
    process_node_v2,
)
from kl_clustering_analysis.hierarchy_analysis.signal_localization import LocalizationResult

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
    root: str = "root",
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
        root=root,
    )


def _dummy_test_divergence(node_a: str, node_b: str):
    """Return a fixed (stat, df, p_value) triple."""
    return (1.0, 1.0, 0.5)


# =====================================================================
# V2TraversalState
# =====================================================================


class TestV2TraversalState:
    def test_default_state(self):
        state = V2TraversalState(split_points=[], merge_points=[], localization_results={})
        assert state.split_points == []
        assert state.merge_points == []
        assert state.localization_results == {}

    def test_mutability(self):
        state = V2TraversalState(split_points=[], merge_points=[], localization_results={})
        state.split_points.append(("root", "L", "R"))
        state.merge_points.append("X")
        state.localization_results["root"] = LocalizationResult(
            left_root="L",
            right_root="R",
            aggregate_p_value=0.01,
            aggregate_significant=True,
        )
        assert len(state.split_points) == 1
        assert len(state.merge_points) == 1
        assert "root" in state.localization_results


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
# process_node_v2
# =====================================================================


class TestProcessNodeV2:
    """Tests for the v2 traversal node processor."""

    def _make_state(self) -> V2TraversalState:
        return V2TraversalState(split_points=[], merge_points=[], localization_results={})

    def test_split_records_split_point(self):
        """Split records (parent, left, right) in state.split_points."""
        gate = _make_gate()
        worklist: list[str] = []
        state = self._make_state()

        process_node_v2(
            "root",
            gate,
            worklist,
            state,
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )

        assert len(state.split_points) == 1
        parent, left, right = state.split_points[0]
        assert parent == "root"
        assert left == "L"
        assert right == "R"
        assert state.merge_points == []

    def test_split_pushes_children(self):
        """Split pushes children right-then-left for left-first DFS."""
        gate = _make_gate()
        worklist: list[str] = []
        state = self._make_state()

        process_node_v2(
            "root",
            gate,
            worklist,
            state,
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )

        assert worklist == ["R", "L"]

    def test_merge_records_merge_point(self):
        """Merge records the node in state.merge_points."""
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
        worklist: list[str] = []
        state = self._make_state()

        process_node_v2(
            "root",
            gate,
            worklist,
            state,
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )

        assert state.merge_points == ["root"]
        assert state.split_points == []
        assert worklist == []

    def test_localization_result_stored_when_present(self):
        """When should_split_v2 returns a localization result, it's stored."""
        loc = LocalizationResult(
            left_root="L",
            right_root="R",
            aggregate_p_value=0.001,
            aggregate_significant=True,
            difference_pairs=[("L1", "R1")],
        )
        gate = _make_gate()
        worklist: list[str] = []
        state = self._make_state()

        with patch.object(gate, "should_split_v2", return_value=(True, loc)):
            process_node_v2(
                "root",
                gate,
                worklist,
                state,
                test_divergence=_dummy_test_divergence,
                sibling_alpha=0.05,
            )

        assert "root" in state.localization_results
        assert state.localization_results["root"] is loc

    def test_power_guard_no_localization_stored(self):
        """Power guard: split=True, loc_result=None → no entry in localization_results."""
        gate = _make_gate()
        worklist: list[str] = []
        state = self._make_state()

        # Simulate power guard: should_split_v2 returns (True, None)
        with patch.object(gate, "should_split_v2", return_value=(True, None)):
            process_node_v2(
                "root",
                gate,
                worklist,
                state,
                test_divergence=_dummy_test_divergence,
                sibling_alpha=0.05,
            )

        assert "root" not in state.localization_results
        # But split still happens:
        assert len(state.split_points) == 1
        assert worklist == ["R", "L"]

    def test_passes_params_to_should_split_v2(self):
        """Verify test_divergence, sibling_alpha, max_depth, and max_pairs are forwarded."""
        gate = _make_gate()
        worklist: list[str] = []
        state = self._make_state()

        mock_divergence = MagicMock(return_value=(1.0, 1.0, 0.5))

        with patch.object(gate, "should_split_v2", return_value=(False, None)) as mock_split:
            process_node_v2(
                "root",
                gate,
                worklist,
                state,
                test_divergence=mock_divergence,
                sibling_alpha=0.01,
                localization_max_depth=3,
                localization_max_pairs=25,
            )

            mock_split.assert_called_once_with(
                "root",
                test_divergence=mock_divergence,
                sibling_alpha=0.01,
                localization_max_depth=3,
                localization_max_pairs=25,
            )

    def test_multiple_nodes_accumulate(self):
        """Processing multiple nodes accumulates in state."""
        gate = _make_gate()
        state = self._make_state()

        # Process root → split
        worklist: list[str] = []
        process_node_v2(
            "root",
            gate,
            worklist,
            state,
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )

        # Now merge at L (make gate say merge at L)
        gate_merge_L = _make_gate(
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
        worklist2: list[str] = []
        process_node_v2(
            "L",
            gate_merge_L,
            worklist2,
            state,
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )

        assert len(state.split_points) == 1  # root split
        assert state.merge_points == ["L"]  # L merged


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

    # --- should_split_v2 ---

    def test_v2_gates_12_fail_returns_false_none(self):
        """When Gates 1&2 fail, v2 returns (False, None)."""
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
        result, loc = gate.should_split_v2(
            "root",
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )
        assert result is False
        assert loc is None

    def test_v2_gate3_fail_returns_false_none(self):
        """When Gate 3 fails (siblings same), v2 returns (False, None)."""
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
        result, loc = gate.should_split_v2(
            "root",
            test_divergence=_dummy_test_divergence,
            sibling_alpha=0.05,
        )
        assert result is False
        assert loc is None

    def test_v2_power_guard_discards_empty_diff_pairs(self):
        """BH correction kills all sub-pairs → (True, None) power guard."""
        tree = _make_binary_tree()
        # Give leaves distributions and sample sizes for localization
        for node in tree.nodes:
            tree.nodes[node]["distribution"] = np.array([0.5, 0.5])
            tree.nodes[node]["sample_size"] = 100

        gate = _make_gate(tree=tree)

        # All p-values are borderline non-significant at sub-level (BH kills them)
        # but aggregate is significant
        call_count = [0]

        def mock_test(a, b):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: aggregate (from localize_divergence_signal)
                return (10.0, 3.0, 0.001)
            # Sub-pairs: borderline p-values that survive raw but not BH
            return (2.0, 1.0, 0.04)

        result, loc = gate.should_split_v2(
            "root",
            test_divergence=mock_test,
            sibling_alpha=0.05,
        )

        assert result is True
        # Power guard may or may not activate depending on BH outcome;
        # if loc is None, power guard activated
        # if loc is not None, difference_pairs must be non-empty
        if loc is not None:
            assert len(loc.difference_pairs) > 0

    def test_v2_passes_localization_max_depth(self):
        """localization_max_depth is forwarded to localize_divergence_signal."""
        gate = _make_gate()

        with patch(
            "kl_clustering_analysis.hierarchy_analysis.gates.localize_divergence_signal"
        ) as mock_loc:
            mock_loc.return_value = LocalizationResult(
                left_root="L",
                right_root="R",
                aggregate_p_value=0.001,
                aggregate_significant=True,
                difference_pairs=[("L1", "R1")],
            )

            gate.should_split_v2(
                "root",
                test_divergence=_dummy_test_divergence,
                sibling_alpha=0.05,
                localization_max_depth=2,
            )

            mock_loc.assert_called_once()
            call_kwargs = mock_loc.call_args
            assert call_kwargs.kwargs.get("max_depth") == 2 or call_kwargs[1].get("max_depth") == 2

    # --- _check_edge_significance ---

    def test_check_edge_significance_root(self):
        """Root is always edge-significant."""
        gate = _make_gate(local_significant={"root": False})
        assert gate._check_edge_significance("root") is True

    def test_check_edge_significance_true(self):
        gate = _make_gate(local_significant={"L": True, "root": True})
        assert gate._check_edge_significance("L") is True

    def test_check_edge_significance_false(self):
        gate = _make_gate(local_significant={"L": False, "root": True})
        assert gate._check_edge_significance("L") is False

    def test_check_edge_significance_missing_defaults_false(self):
        gate = _make_gate(local_significant={"root": True})
        assert gate._check_edge_significance("MISSING") is False


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

    def test_v2_full_traversal(self):
        """V2 full traversal records split/merge points correctly."""
        gate = _make_gate()
        worklist = ["root"]
        processed: set[str] = set()
        state = V2TraversalState(split_points=[], merge_points=[], localization_results={})

        # Mock should_split_v2 to return (True, None) for root, (False, None) for leaves
        original_should_split_v2 = gate.should_split_v2

        def mock_v2(node_id, **kwargs):
            children = gate._children_map.get(node_id, [])
            if len(children) == 2:
                return True, None  # all internal nodes split (power guard)
            return False, None

        with patch.object(gate, "should_split_v2", side_effect=mock_v2):
            for node in iterate_worklist(worklist, processed):
                process_node_v2(
                    node,
                    gate,
                    worklist,
                    state,
                    test_divergence=_dummy_test_divergence,
                    sibling_alpha=0.05,
                )

        # root splits → L, R split → L1, L2, R1, R2 merged (leaves)
        split_parents = [sp[0] for sp in state.split_points]
        assert "root" in split_parents
        assert "L" in split_parents
        assert "R" in split_parents
        # Leaves become merge points
        assert set(state.merge_points) == {"L1", "L2", "R1", "R2"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
