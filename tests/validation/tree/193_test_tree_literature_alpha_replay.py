from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from benchmarks.validation.tree.tree_literature_alpha_replay import (
    DEFAULT_POLICIES,
    PolicySpec,
    replay_cell,
    write_replay_artifacts,
)


def _row(
    node: str,
    *,
    trace_index: int,
    depth: int,
    left_child: str = "",
    right_child: str = "",
    leaves: list[str] | None = None,
    p_value: float | None = None,
    corrected_p_value: float | None = None,
    edge_open: bool = True,
    left_edge_p: float = 1e-8,
    right_edge_p: float = 1e-8,
    dense_p: float = 1.0,
    sparse_p: float = 1.0,
) -> dict[str, object]:
    return {
        "case_id": "synthetic_case",
        "test_case": 1,
        "tree_inference": "average",
        "run_id": "run_average",
        "trace_type": "full_edge_traversal_trace",
        "trace_index": trace_index,
        "node_id": node,
        "left_child": left_child,
        "right_child": right_child,
        "sibling_p_value": p_value,
        "sibling_p_value_corrected": corrected_p_value
        if corrected_p_value is not None
        else p_value,
        "sibling_sparse_p_value": sparse_p,
        "sibling_dense_p_value": dense_p,
        "sibling_fixed_coordinate_bh_p_value": sparse_p,
        "sibling_fixed_global_p_value": dense_p,
        "left_edge_p_value_bh": left_edge_p,
        "right_edge_p_value_bh": right_edge_p,
        "edge_gate_open": edge_open,
        "depth": depth,
        "n_children": int(bool(left_child)) + int(bool(right_child)),
        "descendant_leaf_signature": json.dumps(leaves or [node]),
    }


def _policy(policy_id: str) -> PolicySpec:
    return next(policy for policy in DEFAULT_POLICIES if policy.policy_id == policy_id)


def test_hierarchical_fdr_does_not_test_children_when_parent_not_rejected() -> None:
    cell = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="A",
                right_child="B",
                leaves=["a1", "a2", "b1", "b2"],
                p_value=0.50,
            ),
            _row("A", trace_index=1, depth=1, leaves=["a1", "a2"], p_value=0.001),
        ]
    )

    decisions, summary = replay_cell(cell, _policy("yekutieli_hierarchical_fdr_bh"))

    assert summary["opened_nodes"] == 0
    assert [row["node_id"] for row in decisions] == ["root"]


def test_dependence_robust_by_is_stricter_than_depth_bh() -> None:
    cell = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="A",
                right_child="B",
                leaves=["a1", "a2", "b1", "b2"],
                p_value=0.001,
            ),
            _row("A", trace_index=1, depth=1, leaves=["a1", "a2"], p_value=0.04),
            _row("B", trace_index=2, depth=1, leaves=["b1", "b2"], p_value=0.041),
        ]
    )
    bh = PolicySpec(
        policy_id="bh",
        literature_family="hierarchical FDR",
        implementation_status="test",
        adjustment="hierarchical_bh",
        p_value_column="sibling_p_value",
        base_alpha=0.05,
    )
    by = PolicySpec(
        policy_id="by",
        literature_family="hierarchical dependence",
        implementation_status="test",
        adjustment="hierarchical_by",
        p_value_column="sibling_p_value",
        base_alpha=0.05,
    )

    _bh_decisions, bh_summary = replay_cell(cell, bh)
    _by_decisions, by_summary = replay_cell(cell, by)

    assert bh_summary["opened_nodes"] == 3
    assert by_summary["opened_nodes"] == 1


def test_graphical_recycling_conserves_child_alpha() -> None:
    cell = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="A",
                right_child="B",
                leaves=["a1", "a2", "b1", "b2"],
                p_value=0.001,
            ),
            _row("A", trace_index=1, depth=1, leaves=["a1", "a2"], p_value=0.008),
            _row("B", trace_index=2, depth=1, leaves=["b1", "b2"], p_value=0.90),
        ]
    )

    decisions, summary = replay_cell(cell, _policy("bretz_graphical_gatekeeping_recycle"))
    child_alpha = {
        row["node_id"]: row["local_alpha"] for row in decisions if row["node_id"] in {"A", "B"}
    }

    assert summary["opened_nodes"] == 1
    assert child_alpha == {"A": 0.005, "B": 0.005}


def test_exact_selective_policy_fails_closed_without_selected_p_values() -> None:
    cell = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="A",
                right_child="B",
                leaves=["a1", "b1"],
                p_value=0.001,
            )
        ]
    )

    decisions, summary = replay_cell(cell, _policy("gao_selective_inference_required"))

    assert summary["opened_nodes"] == 0
    assert decisions[0]["blocked_reason"] == "missing_exact_selected_clustering_p_value"


def test_adaptive_proxy_blocks_tiny_child_split_even_when_alpha_opens() -> None:
    leaves = [f"s{i}" for i in range(100)]
    cell = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="tiny",
                right_child="rest",
                leaves=leaves,
                p_value=0.05,
                corrected_p_value=0.05,
                dense_p=1e-20,
                sparse_p=1e-8,
            ),
        ]
    )

    decisions, summary = replay_cell(cell, _policy("trace_adaptive_alpha_spending_proxy"))

    assert decisions[0]["local_alpha"] == 0.20
    assert decisions[0]["blocked_reason"] == "child_balance_guard_failed"
    assert summary["opened_nodes"] == 0


def test_adaptive_proxy_can_pass_through_to_supported_descendant_split() -> None:
    root_leaves = [f"s{i}" for i in range(200)]
    a_leaves = root_leaves[:100]
    b_leaves = root_leaves[100:]
    cell = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="A",
                right_child="B",
                leaves=root_leaves,
                p_value=0.90,
                corrected_p_value=0.90,
                dense_p=1e-20,
                sparse_p=1e-8,
            ),
            _row(
                "A",
                trace_index=1,
                depth=1,
                left_child="A1",
                right_child="A2",
                leaves=a_leaves,
                p_value=0.05,
                corrected_p_value=0.05,
                dense_p=1e-20,
                sparse_p=1e-8,
            ),
            _row("A1", trace_index=2, depth=2, leaves=a_leaves[:50], edge_open=False),
            _row("A2", trace_index=3, depth=2, leaves=a_leaves[50:], edge_open=False),
            _row("B", trace_index=4, depth=1, leaves=b_leaves, edge_open=False),
        ]
    )

    decisions, summary = replay_cell(cell, _policy("trace_adaptive_alpha_spending_proxy"))
    by_node = {row["node_id"]: row for row in decisions}

    assert by_node["root"]["pass_through"] is True
    assert by_node["A"]["rejected"] is True
    assert summary["opened_nodes"] == 1
    assert summary["pass_through_nodes"] == 1
    assert summary["cluster_count"] == 3


def test_writer_outputs_all_replay_artifacts(tmp_path: Path) -> None:
    trace = pd.DataFrame(
        [
            _row(
                "root",
                trace_index=0,
                depth=0,
                left_child="A",
                right_child="B",
                leaves=["a1", "b1"],
                p_value=0.001,
                corrected_p_value=0.001,
            )
        ]
    )
    taxonomy = pd.DataFrame(
        [
            {
                "case_id": "synthetic_case",
                "test_case": 1,
                "case_category": "synthetic",
                "true_clusters": 2,
                "ok_cells": 1,
                "loss_bucket": "sibling_gate_closed_after_edge_open",
            }
        ]
    )
    trace_path = tmp_path / "trace.csv"
    taxonomy_path = tmp_path / "taxonomy.csv"
    output_dir = tmp_path / "out"
    trace.to_csv(trace_path, index=False)
    taxonomy.to_csv(taxonomy_path, index=False)

    manifest = write_replay_artifacts(
        trace_path=trace_path,
        taxonomy_path=taxonomy_path,
        output_dir=output_dir,
    )

    assert manifest["n_method_summary_rows"] == len(DEFAULT_POLICIES)
    assert manifest["n_requirement_audit_rows"] == 6
    for name in (
        "literature_policy_replay_decisions.csv",
        "literature_policy_replay_cell_summary.csv",
        "literature_policy_replay_case_summary.csv",
        "literature_policy_replay_method_summary.csv",
        "literature_policy_replay_requirement_audit.csv",
        "literature_policy_replay_report.md",
        "literature_policy_replay_manifest.json",
    ):
        assert (output_dir / name).exists()

    audit = pd.read_csv(output_dir / "literature_policy_replay_requirement_audit.csv")
    assert set(audit["requested_family"]) == {
        "Hierarchical FDR",
        "TreeBH / multiresolution tree testing",
        "Hierarchical procedures under dependence",
        "Graphical gatekeeping / alpha recycling",
        "Selective inference for clustering",
        "Adaptive alpha spending for hierarchical clustering",
    }
    assert "blocked_pending_selected_p_values" in set(audit["production_interpretation"])
    assert "diagnostic_only_no_promotion" in set(audit["production_interpretation"])
