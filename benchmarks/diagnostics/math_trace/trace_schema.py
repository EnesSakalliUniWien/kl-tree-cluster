"""Schema helpers for traceable benchmark math diagnostics."""

from __future__ import annotations

import pandas as pd

REQUIRED_NODE_TRACE_COLUMNS = [
    "case_id",
    "replicate_id",
    "node_id",
    "parent_id",
    "left_child",
    "right_child",
    "parent_depth",
    "parent_sample_size",
    "left_sample_size",
    "right_sample_size",
    "child_balance",
    "barycentric_leverage",
    "edge_left_raw_p",
    "edge_right_raw_p",
    "edge_left_bh_p",
    "edge_right_bh_p",
    "edge_path_open",
    "edge_action",
    "sibling_raw_stat",
    "sibling_raw_p",
    "sibling_projection_dimension",
    "reference_scale",
    "degrees_of_freedom",
    "selected_ratio",
    "selected_subspace_cos2",
    "selected_subspace_tan2",
    "lambda_k_over_mp",
    "selected_eigenvalue_mass",
    "internal_support_status",
    "n_supported_records",
    "n_strict_null_records",
    "n_stopped_records",
    "n_eff_family",
    "inflation_c_hat",
    "sibling_adjusted_p",
    "sibling_bh_p",
    "traversal_decision",
    "failure_label",
]

OPTIONAL_CLASSIFIER_COLUMNS = [
    "tree_oracle_unrecoverable",
    "covariance_null_invalid",
    "fixed_projection_pvalues_not_uniform",
    "selected_mp_calibration_failed",
    "selected_tail_context_not_admissible",
    "sibling_BH_false_split_rate_high",
    "traversal_overrides_local_evidence_badly",
]

OPTIONAL_GEOMETRY_TRACE_COLUMNS = [
    "geometry_parent_radius",
    "geometry_angle_to_leading_axis_deg",
    "geometry_independent_radius_fraction",
    "geometry_sibling_separation_parent_ratio",
    "geometry_sibling_separation_child_ratio",
    "geometry_abs_common_axis_gap",
    "action_budget_proxy",
    "action_budget_proxy_capped",
    "angular_shell_risk_score",
    "is_null_context",
    "is_signal_context",
    "geometry_label_provenance",
]


def missing_required_columns(table: pd.DataFrame) -> list[str]:
    """Return required node-trace columns absent from a table."""
    return [column for column in REQUIRED_NODE_TRACE_COLUMNS if column not in table.columns]


def validate_node_decision_trace(table: pd.DataFrame) -> None:
    """Raise a compact error if the node decision trace is structurally incomplete."""
    missing = missing_required_columns(table)
    if missing:
        raise ValueError(f"node_decision_trace.csv is missing required columns: {missing}")
