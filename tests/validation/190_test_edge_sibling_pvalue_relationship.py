from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.edge_sibling_pvalue_relationship import (
    build_edge_sibling_pair_table,
    build_long_distribution_table,
    run_edge_sibling_pvalue_relationship,
    summarize_edge_sibling_pairs,
)


def _node_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_a", "case_a"],
            "data_role": ["selected_null", "selected_null"],
            "method_id": ["fixed", "fixed"],
            "replicate": [0, 0],
            "data_seed": [11, 11],
            "node_id": ["N1", "L1"],
            "decision_class": ["accepted_internal_split", "leaf_fragment"],
            "traversal_decision": ["split", "not_visited"],
            "visited": [True, False],
            "n_children": [2, 0],
            "outgoing_left_child_id": ["L1", ""],
            "outgoing_right_child_id": ["L2", ""],
            "outgoing_left_edge_p_value": [0.01, float("nan")],
            "outgoing_left_edge_bh_p_value": [0.02, float("nan")],
            "outgoing_right_edge_p_value": [0.2, float("nan")],
            "outgoing_right_edge_bh_p_value": [0.4, float("nan")],
            "sibling_p_value": [0.03, float("nan")],
            "sibling_p_value_corrected": [0.06, float("nan")],
            "sibling_open": [True, False],
            "sibling_test_method": ["fixed_coordinate_bh", ""],
            "sibling_gate_p_value_calibration": ["fixed_subspace_bh", ""],
            "sibling_sparse_p_value": [0.08, float("nan")],
            "sibling_sparse_method": ["fixed_coordinate_bh", ""],
            "sibling_dense_p_value": [0.01, float("nan")],
            "sibling_dense_method": ["fixed_global_chi_square", ""],
            "sibling_projection_dimension": [3.0, float("nan")],
        }
    )


def test_build_edge_sibling_pair_table_links_parent_to_outgoing_edges() -> None:
    pairs = build_edge_sibling_pair_table(_node_decisions())

    assert pairs.shape[0] == 1
    row = pairs.iloc[0]
    assert row["parent_node_id"] == "N1"
    assert row["left_child_id"] == "L1"
    assert row["right_child_id"] == "L2"
    assert row["min_edge_bh_p_value"] == pytest.approx(0.02)
    assert row["sibling_p_value_corrected"] == pytest.approx(0.06)
    assert row["sibling_sparse_p_value"] == pytest.approx(0.08)
    assert row["sibling_dense_p_value"] == pytest.approx(0.01)
    assert row["sibling_gate_p_value_calibration"] == "fixed_subspace_bh"
    assert bool(row["sibling_tested"]) is True
    assert row["edge_pair_max_neglog10_bh"] == pytest.approx(-np.log10(0.02))


def test_summary_and_long_table_report_related_edge_and_sibling_distributions() -> None:
    pairs = build_edge_sibling_pair_table(_node_decisions())
    long = build_long_distribution_table(pairs)
    summary = summarize_edge_sibling_pairs(pairs)

    assert set(long["test_type"]) == {
        "left_edge_bh",
        "right_edge_bh",
        "sibling_active_corrected",
        "sibling_sparse",
        "sibling_dense",
    }
    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert row["n_parent_sibling_rows"] == 1
    assert row["n_sibling_tested"] == 1
    assert row["n_sibling_open"] == 1
    assert row["min_edge_bh_p_value"] == pytest.approx(0.02)
    assert row["min_sibling_corrected_p_value"] == pytest.approx(0.06)
    assert row["median_sparse_sibling_neglog10"] == pytest.approx(-np.log10(0.08))
    assert row["median_dense_sibling_neglog10"] == pytest.approx(-np.log10(0.01))


def test_run_edge_sibling_pvalue_relationship_writes_report_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "nodes.csv"
        output_dir = Path(tmpdir) / "out"
        _node_decisions().to_csv(input_path, index=False)

        outputs = run_edge_sibling_pvalue_relationship(
            node_decisions_path=input_path,
            output_dir=output_dir,
            title="Test Relationship",
        )

        assert set(outputs) == {
            "pair_table",
            "long_table",
            "summary",
            "plot",
            "html_report",
            "manifest",
        }
        for path in outputs.values():
            assert path.exists()
