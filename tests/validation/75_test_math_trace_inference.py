import json
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.math_trace.infer_benchmark_math import infer_math
from benchmarks.diagnostics.math_trace.selected_tail_law import (
    compute_selected_tail_variables,
)
from benchmarks.diagnostics.math_trace.trace_schema import REQUIRED_NODE_TRACE_COLUMNS


def _trace_row(**overrides):
    row = {column: 1 for column in REQUIRED_NODE_TRACE_COLUMNS}
    row.update(
        {
            "case_id": "trace_case",
            "replicate_id": 0,
            "node_id": "n1",
            "parent_id": "root",
            "left_child": "l",
            "right_child": "r",
            "child_balance": 0.5,
            "barycentric_leverage": 0.0,
            "edge_left_raw_p": 0.01,
            "edge_right_raw_p": 0.02,
            "edge_left_bh_p": 0.02,
            "edge_right_bh_p": 0.03,
            "edge_path_open": True,
            "edge_action": 4.6,
            "sibling_raw_stat": 12.0,
            "sibling_raw_p": 0.01,
            "sibling_projection_dimension": 2,
            "reference_scale": 1.0,
            "degrees_of_freedom": 4,
            "selected_ratio": 3.0,
            "selected_subspace_cos2": 0.75,
            "selected_subspace_tan2": 1.0 / 3.0,
            "lambda_k_over_mp": 1.2,
            "selected_eigenvalue_mass": 0.8,
            "internal_support_status": "supported",
            "n_supported_records": 100,
            "n_strict_null_records": 50,
            "n_stopped_records": 50,
            "n_eff_family": 40,
            "inflation_c_hat": 1.5,
            "sibling_adjusted_p": 0.015,
            "sibling_bh_p": 0.02,
            "traversal_decision": "split",
            "failure_label": "",
        }
    )
    row.update(overrides)
    return row


def test_infer_math_writes_trace_outputs(tmp_path: Path):
    trace = pd.DataFrame(
        [
            _trace_row(node_id="n1"),
            _trace_row(
                node_id="n2",
                internal_support_status="unsupported",
                n_supported_records=0,
            ),
        ]
    )
    trace_path = tmp_path / "node_decision_trace.csv"
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "math"
    trace.to_csv(trace_path, index=False)
    manifest_path.write_text(json.dumps({"case_id": "trace_case"}), encoding="utf-8")

    summary = infer_math(
        output_dir=output_dir,
        manifest_json=manifest_path,
        node_decision_trace_csv=trace_path,
    )

    assert summary["n_trace_rows"] == 2
    assert summary["failure_label_counts"]["calibration_support_undefined"] == 1
    assert (output_dir / "math_inference_report.md").exists()
    assert (output_dir / "failure_attribution.csv").exists()
    assert (output_dir / "support_threshold_audit.csv").exists()


def test_selected_tail_variables_omit_unused_derived_columns() -> None:
    table = pd.DataFrame(
        {
            "selected_ratio": [3.0],
            "edge_left_raw_p": [0.01],
            "edge_right_raw_p": [0.02],
            "selected_subspace_cos2": [0.75],
        }
    )

    variables = compute_selected_tail_variables(table)

    assert variables["selected_ratio"].tolist() == [3.0]
    assert "edge_action_from_p" not in variables.columns
    assert "selected_subspace_tan2_from_cos2" not in variables.columns
