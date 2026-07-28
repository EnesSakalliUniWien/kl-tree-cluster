"""Tests for gate evaluation and the live TreeDecomposition traversal path."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from tree_break_selection.hierarchy_analysis.cluster_assignments import (
    ClusterBoundary,
    build_cluster_assignments,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.gate_evaluator import (
    GateEvaluator,
    TraversalDecision,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.spectral_transport import (
    annotate_spectral_transport_passthrough_support,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.tree.poset_tree import PosetTree


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
    passthrough_supported: dict[str, bool] | None = None,
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
        passthrough_supported=passthrough_supported,
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

    edge_p_values = {node: 0.01 if edge_divergent[node] else 0.90 for node in tree.nodes}
    sibling_p_values = {
        node: 0.02 if sibling_different[node] and not sibling_skipped[node] else 0.80
        for node in tree.nodes
    }

    return pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": pd.Series(edge_divergent, dtype=bool),
            "Child_Parent_Divergence_P_Value": pd.Series(edge_p_values, dtype=float),
            "Child_Parent_Divergence_P_Value_BH": pd.Series(edge_p_values, dtype=float),
            "Child_Parent_Divergence_Tested": True,
            "Child_Parent_Divergence_Ancestor_Blocked": False,
            "Sibling_BH_Different": pd.Series(sibling_different, dtype=bool),
            "Sibling_Divergence_Skipped": pd.Series(sibling_skipped, dtype=bool),
            "Sibling_Divergence_P_Value": pd.Series(sibling_p_values, dtype=float),
            "Sibling_Divergence_P_Value_Corrected": pd.Series(sibling_p_values, dtype=float),
            "Sibling_Test_Statistic": pd.Series(
                {
                    node: 8.0 if sibling_different[node] and not sibling_skipped[node] else 0.5
                    for node in tree.nodes
                },
                dtype=float,
            ),
            "Sibling_Degrees_of_Freedom": 1.0,
            "Sibling_Test_Method": "synthetic_projected_wald",
        }
    ).reindex(list(tree.nodes))


def _decompose_with_annotations(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    passthrough: bool,
    spectral_transport_passthrough_guard: bool = False,
) -> dict[str, object]:
    decomposer = TreeDecomposition(
        tree=tree,
        annotations_df=annotations_df,
        passthrough=passthrough,
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        trace_level="full",
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
        0: {
            "root_node": "left_subtree",
            "leaves": ["L1", "L2"],
            "leaf_signature": ("L1", "L2"),
            "size": 2,
        },
        1: {
            "root_node": "right_leaf",
            "leaves": ["R"],
            "leaf_signature": ("R",),
            "size": 1,
        },
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
        assert (
            gate.passthrough_audit_status("root")["passthrough_split_prerequisites_open"] is False
        )

    def test_gate1_single_child_returns_false(self) -> None:
        tree = nx.DiGraph()
        tree.add_edges_from([("root", "A")])
        _annotate_tree_structure(tree, {"A"})

        gate = _make_gate(
            tree=tree,
            children_map={"root": ["A"], "A": []},
        )
        assert gate.decision("root") is TraversalDecision.BOUNDARY
        assert (
            gate.passthrough_audit_status("root")["passthrough_split_prerequisites_open"] is False
        )

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
        assert (
            gate.passthrough_audit_status("root")["passthrough_split_prerequisites_open"] is False
        )

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
        assert gate.passthrough_audit_status("root") == {
            "passthrough_enabled": False,
            "passthrough_split_prerequisites_open": True,
            "passthrough_sibling_gate_open": False,
            "passthrough_descendant_split_available": False,
            "passthrough_candidate": False,
            "passthrough_supported": True,
            "passthrough_bottleneck": "",
            "passthrough_decision_reason": "passthrough_disabled",
        }

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
        assert gate.passthrough_audit_status("root") == {
            "passthrough_enabled": True,
            "passthrough_split_prerequisites_open": True,
            "passthrough_sibling_gate_open": False,
            "passthrough_descendant_split_available": True,
            "passthrough_candidate": True,
            "passthrough_supported": True,
            "passthrough_bottleneck": "",
            "passthrough_decision_reason": "pass_through",
        }

    def test_spectral_support_guard_blocks_passthrough(self) -> None:
        tree = _make_deep_tree()
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True
        passthrough_supported = {node: True for node in tree.nodes}
        passthrough_supported["root"] = False

        gate = _make_gate(
            tree=tree,
            passthrough=True,
            sibling_different=sibling_different,
            passthrough_supported=passthrough_supported,
        )

        assert gate.decision("root") is TraversalDecision.BOUNDARY
        assert gate.passthrough_support_status("root") == {
            "passthrough_supported": False,
            "passthrough_bottleneck": "",
        }
        assert gate.passthrough_audit_status("root") == {
            "passthrough_enabled": True,
            "passthrough_split_prerequisites_open": True,
            "passthrough_sibling_gate_open": False,
            "passthrough_descendant_split_available": True,
            "passthrough_candidate": True,
            "passthrough_supported": False,
            "passthrough_bottleneck": "",
            "passthrough_decision_reason": "passthrough_support_blocked",
        }

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
        assert (
            gate.passthrough_audit_status("root")["passthrough_decision_reason"]
            == "sibling_gate_open_split"
        )

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
        assert (
            gate.passthrough_audit_status("root")["passthrough_decision_reason"]
            == "split_prerequisites_closed"
        )

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
        assert (
            gate.passthrough_audit_status("root")["passthrough_decision_reason"]
            == "no_descendant_split"
        )


class TestTreeDecompositionTraversal:
    def test_decompose_tree_splits_to_leaf_clusters(self) -> None:
        tree = _make_binary_tree()
        annotations_df = _make_annotations(
            tree,
            edge_divergent={node: True for node in tree.nodes},
            sibling_different={node: True for node in tree.nodes},
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            passthrough=False,
        )

        cluster_leaf_sets = sorted(_cluster_leaf_sets(result), key=lambda leaves: min(leaves))
        assert cluster_leaf_sets == [{"L1"}, {"L2"}, {"R1"}, {"R2"}]

    def test_direct_annotations_do_not_claim_unknown_alpha(self) -> None:
        tree = _make_binary_tree()
        annotations_df = _make_annotations(
            tree,
            edge_divergent={node: True for node in tree.nodes},
            sibling_different={node: False for node in tree.nodes},
        )

        decomposer = TreeDecomposition(
            tree=tree,
            annotations_df=annotations_df,
            passthrough=False,
        )

        result = decomposer.decompose_tree()

        assert result["independence_analysis"]["edge_alpha"] is None
        assert result["independence_analysis"]["sibling_alpha"] is None

    def test_decompose_tree_default_trace_level_is_compact(self) -> None:
        tree = _make_binary_tree()
        annotations_df = _make_annotations(
            tree,
            edge_divergent={node: True for node in tree.nodes},
            sibling_different={node: False for node in tree.nodes},
        )

        decomposer = TreeDecomposition(
            tree=tree,
            annotations_df=annotations_df,
            passthrough=False,
        )

        result = decomposer.decompose_tree()

        assert "traversal_trace" not in result
        assert "full_edge_traversal_trace" not in result
        assert result["traversal_counters"] == {
            "live_nodes_visited": 1,
            "live_internal_tuples": 1,
            "live_split_count": 0,
            "live_pass_through_count": 0,
            "live_boundary_count": 1,
            "live_passthrough_candidate_count": 0,
            "live_passthrough_support_blocked_count": 0,
        }

    def test_decompose_tree_passthrough_reaches_descendant_split(self) -> None:
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
            passthrough=True,
        )

        cluster_leaf_sets = sorted(_cluster_leaf_sets(result), key=lambda leaves: min(leaves))
        assert cluster_leaf_sets == [{"A1", "A2"}, {"C1", "C2"}, {"D1", "D2"}]
        root_trace = result["traversal_trace"][0]
        assert root_trace["passthrough_candidate"] is True
        assert root_trace["passthrough_decision_reason"] == "pass_through"
        assert result["traversal_counters"]["live_passthrough_candidate_count"] == 1

    def test_decompose_tree_without_passthrough_merges_at_root(self) -> None:
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
            passthrough=False,
        )

        cluster_leaf_sets = _cluster_leaf_sets(result)
        assert cluster_leaf_sets == [{"A1", "A2", "C1", "C2", "D1", "D2"}]

    def test_full_edge_traversal_walks_past_sibling_closed_root(self) -> None:
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
            passthrough=False,
        )

        live_trace = result["traversal_trace"]
        full_trace = result["full_edge_traversal_trace"]
        assert [row["node_id"] for row in live_trace] == ["root"]
        assert {row["node_id"] for row in full_trace} == set(tree.nodes)
        assert full_trace[0]["node_id"] == "root"
        assert full_trace[0]["actual_decision"] == "boundary"
        assert full_trace[0]["actual_visited"] is True
        assert full_trace[0]["edge_traversal_action"] == "continue"
        assert any(row["node_id"] == "B" and not row["actual_visited"] for row in full_trace)

    def test_full_edge_traversal_stops_when_child_edges_close(self) -> None:
        tree = _make_deep_tree()
        edge_divergent = {node: False for node in tree.nodes}
        edge_divergent["A"] = True
        edge_divergent["B"] = True
        sibling_different = {node: False for node in tree.nodes}

        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            passthrough=False,
        )

        full_trace = result["full_edge_traversal_trace"]
        assert [row["node_id"] for row in full_trace] == ["root", "A", "B"]
        stop_reasons = {row["node_id"]: row["edge_traversal_stop_reason"] for row in full_trace}
        assert stop_reasons == {
            "root": "edge_open_continue",
            "A": "edge_closed",
            "B": "edge_closed",
        }

    def test_full_edge_traversal_records_branch_lengths_and_counters(self) -> None:
        tree = _make_deep_tree()
        tree.edges["root", "A"]["branch_length"] = 1.25
        tree.edges["root", "B"]["branch_length"] = 2.5
        edge_divergent = {node: False for node in tree.nodes}
        edge_divergent["A"] = True
        edge_divergent["B"] = True
        sibling_different = {node: False for node in tree.nodes}

        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            passthrough=False,
        )

        root_row = result["full_edge_traversal_trace"][0]
        assert root_row["left_child"] == "A"
        assert root_row["right_child"] == "B"
        assert root_row["left_edge_test_tuple"] == ("root", "A")
        assert root_row["right_edge_test_tuple"] == ("root", "B")
        assert root_row["sibling_test_tuple"] == ("root", "A", "B")
        assert root_row["left_edge_p_value"] == 0.01
        assert root_row["left_edge_p_value_bh"] == 0.01
        assert root_row["left_edge_tested"] is True
        assert root_row["right_edge_p_value"] == 0.01
        assert root_row["right_edge_p_value_bh"] == 0.01
        assert root_row["right_edge_tested"] is True
        assert root_row["sibling_p_value"] == 0.80
        assert root_row["sibling_p_value_corrected"] == 0.80
        assert root_row["sibling_test_statistic"] == 0.5
        assert root_row["sibling_degrees_of_freedom"] == 1.0
        assert root_row["sibling_test_method"] == "synthetic_projected_wald"
        assert root_row["descendant_leaf_signature"] == (
            "A1",
            "A2",
            "C1",
            "C2",
            "D1",
            "D2",
        )
        assert root_row["passthrough_enabled"] is False
        assert root_row["passthrough_candidate"] is False
        assert root_row["passthrough_decision_reason"] == "passthrough_disabled"
        assert root_row["left_branch_length"] == 1.25
        assert root_row["right_branch_length"] == 2.5
        assert root_row["left_branch_length_missing"] is False
        assert root_row["right_branch_length_missing"] is False

        counters = result["traversal_counters"]
        assert counters["live_nodes_visited"] == len(result["traversal_trace"])
        assert counters["live_internal_tuples"] == 1
        assert counters["live_boundary_count"] == 1
        assert counters["full_edge_nodes_visited"] == len(result["full_edge_traversal_trace"])
        assert counters["full_edge_internal_tuples"] == 3
        assert counters["full_edge_continue_count"] == 1
        assert counters["full_edge_stop_count"] == 2
        assert counters["full_edge_closed_stop_count"] == 2
        assert counters["full_edge_passthrough_candidate_count"] == 0

    def test_decompose_tree_passthrough_ignores_blocked_descendant_signal(self) -> None:
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
            passthrough=True,
        )

        cluster_leaf_sets = _cluster_leaf_sets(result)
        assert cluster_leaf_sets == [{"A1", "A2", "C1", "C2", "D1", "D2"}]

    def test_decompose_tree_selected_family_guard_blocks_passthrough(self) -> None:
        tree = _make_deep_tree()
        edge_divergent = {node: True for node in tree.nodes}
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True
        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )
        annotations_df["Selective_Permutation_Guard_Would_Block"] = False
        annotations_df["Selective_Permutation_Guard_Blocked"] = False
        annotations_df.loc["root", "Selective_Permutation_Guard_Would_Block"] = True

        decomposer = TreeDecomposition(
            tree=tree,
            annotations_df=annotations_df,
            passthrough=True,
            selected_family_passthrough_guard=True,
            trace_level="full",
        )

        result = decomposer.decompose_tree()

        assert _cluster_leaf_sets(result) == [{"A1", "A2", "C1", "C2", "D1", "D2"}]
        root_trace = result["traversal_trace"][0]
        assert root_trace["decision"] == "boundary"
        assert root_trace["passthrough_candidate"] is True
        assert root_trace["passthrough_supported"] is False
        assert root_trace["passthrough_bottleneck"] == ("selected_family_passthrough_guard_blocked")
        assert root_trace["passthrough_decision_reason"] == "passthrough_support_blocked"
        assert result["traversal_counters"]["live_passthrough_support_blocked_count"] == 1

    def test_decompose_tree_spectral_support_guard_blocks_passthrough(self) -> None:
        tree = _make_deep_tree()
        edge_divergent = {node: True for node in tree.nodes}
        sibling_different = {node: False for node in tree.nodes}
        sibling_different["B"] = True
        annotations_df = _make_annotations(
            tree,
            edge_divergent=edge_divergent,
            sibling_different=sibling_different,
        )
        annotations_df["Spectral_Transport_Pass_Through_Supported"] = True
        annotations_df["Spectral_Transport_Bottleneck"] = "supported_mp_mode_path"
        annotations_df.loc["root", "Spectral_Transport_Pass_Through_Supported"] = False
        annotations_df.loc["root", "Spectral_Transport_Bottleneck"] = (
            "spectral_transport_bottleneck"
        )

        result = _decompose_with_annotations(
            tree,
            annotations_df,
            passthrough=True,
            spectral_transport_passthrough_guard=True,
        )

        assert _cluster_leaf_sets(result) == [{"A1", "A2", "C1", "C2", "D1", "D2"}]
        root_trace = result["traversal_trace"][0]
        assert root_trace["decision"] == "boundary"
        assert root_trace["left_edge_test_tuple"] == ("root", "A")
        assert root_trace["right_edge_test_tuple"] == ("root", "B")
        assert root_trace["sibling_test_tuple"] == ("root", "A", "B")
        assert root_trace["left_edge_p_value"] == 0.01
        assert root_trace["right_edge_p_value"] == 0.01
        assert root_trace["sibling_p_value"] == 0.80
        assert root_trace["sibling_p_value_corrected"] == 0.80
        assert root_trace["descendant_leaf_signature"] == (
            "A1",
            "A2",
            "C1",
            "C2",
            "D1",
            "D2",
        )
        assert root_trace["passthrough_supported"] is False
        assert root_trace["passthrough_bottleneck"] == "spectral_transport_bottleneck"
        assert root_trace["passthrough_candidate"] is True
        assert root_trace["passthrough_decision_reason"] == "passthrough_support_blocked"
        assert result["traversal_counters"]["live_passthrough_support_blocked_count"] == 1


def test_spectral_transport_annotation_supports_coherent_passthrough_path() -> None:
    tree = _make_deep_tree()
    edge_divergent = {node: True for node in tree.nodes}
    sibling_different = {node: False for node in tree.nodes}
    sibling_different["B"] = True
    annotations = _make_annotations(
        tree,
        edge_divergent=edge_divergent,
        sibling_different=sibling_different,
    )
    spectral_context = SimpleNamespace(
        principal_component_projections_by_node={
            node: np.asarray([[1.0, 0.0]]) for node in tree.nodes
        },
        principal_component_eigenvalues_by_node={node: np.asarray([4.0]) for node in tree.nodes},
        raw_mp_signal_counts_by_node={node: 1 for node in tree.nodes},
    )

    out = annotate_spectral_transport_passthrough_support(
        tree,
        annotations,
        spectral_context,
        max_cost=0.1,
    )

    assert bool(out.loc["root", "Spectral_Transport_Pass_Through_Supported"])
    assert out.loc["root", "Spectral_Transport_Bottleneck"] == "supported_mp_mode_path"


def test_spectral_transport_annotation_blocks_rotated_passthrough_path() -> None:
    tree = _make_deep_tree()
    edge_divergent = {node: True for node in tree.nodes}
    sibling_different = {node: False for node in tree.nodes}
    sibling_different["B"] = True
    annotations = _make_annotations(
        tree,
        edge_divergent=edge_divergent,
        sibling_different=sibling_different,
    )
    spectral_context = SimpleNamespace(
        principal_component_projections_by_node={
            **{node: np.asarray([[1.0, 0.0]]) for node in tree.nodes},
            "B": np.asarray([[0.0, 1.0]]),
        },
        principal_component_eigenvalues_by_node={node: np.asarray([4.0]) for node in tree.nodes},
        raw_mp_signal_counts_by_node={node: 1 for node in tree.nodes},
    )

    out = annotate_spectral_transport_passthrough_support(
        tree,
        annotations,
        spectral_context,
        max_cost=0.1,
    )

    assert not bool(out.loc["root", "Spectral_Transport_Pass_Through_Supported"])
    assert bool(out.loc["root", "Spectral_Transport_Pass_Through_Blocked"])


def test_spectral_transport_annotation_blocks_floor_only_when_mp_blocks_required() -> None:
    tree = _make_deep_tree()
    edge_divergent = {node: True for node in tree.nodes}
    sibling_different = {node: False for node in tree.nodes}
    sibling_different["B"] = True
    annotations = _make_annotations(
        tree,
        edge_divergent=edge_divergent,
        sibling_different=sibling_different,
    )
    spectral_context = SimpleNamespace(
        principal_component_projections_by_node={
            node: np.asarray([[1.0, 0.0]]) for node in tree.nodes
        },
        principal_component_eigenvalues_by_node={node: np.asarray([4.0]) for node in tree.nodes},
        raw_mp_signal_counts_by_node={node: 0 for node in tree.nodes},
    )

    out = annotate_spectral_transport_passthrough_support(
        tree,
        annotations,
        spectral_context,
        max_cost=0.1,
    )

    assert not bool(out.loc["root", "Spectral_Transport_Pass_Through_Supported"])
    assert bool(out.loc["root", "Spectral_Transport_Pass_Through_Blocked"])
    assert out.loc["root", "Spectral_Transport_Bottleneck"] == ("spectral_transport_bottleneck")


def test_spectral_transport_annotation_does_not_veto_floor_only_when_mp_blocks_not_required() -> (
    None
):
    tree = _make_deep_tree()
    edge_divergent = {node: True for node in tree.nodes}
    sibling_different = {node: False for node in tree.nodes}
    sibling_different["B"] = True
    annotations = _make_annotations(
        tree,
        edge_divergent=edge_divergent,
        sibling_different=sibling_different,
    )
    spectral_context = SimpleNamespace(
        principal_component_projections_by_node={
            node: np.asarray([[1.0, 0.0]]) for node in tree.nodes
        },
        principal_component_eigenvalues_by_node={node: np.asarray([4.0]) for node in tree.nodes},
        raw_mp_signal_counts_by_node={node: 0 for node in tree.nodes},
    )

    out = annotate_spectral_transport_passthrough_support(
        tree,
        annotations,
        spectral_context,
        max_cost=0.1,
        require_mp_blocks=False,
    )

    assert bool(out.loc["root", "Spectral_Transport_Pass_Through_Supported"])
    assert not bool(out.loc["root", "Spectral_Transport_Pass_Through_Blocked"])
    assert out.loc["root", "Spectral_Transport_Bottleneck"] == ("unmeasured_no_matched_mp_path")
