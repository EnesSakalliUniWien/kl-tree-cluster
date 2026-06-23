"""Selected-neighborhood distribution diagnostics for traversal states.

This panel compares the previous neighborhood-calibrated method idea with the
current strict support-gated path at the point where traversal actually acts:
accepted splits, stopped boundaries, and pass-through ancestors. It is
diagnostic-only and does not promote recovery splits.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    SIBLING_GATE_PROFILES,
)

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_selected_neighborhood_distribution_not_calibration"
SCHEMA_VERSION = "selected_neighborhood_distribution_panel/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "selected_neighborhood_distribution_panel"
)

JOIN_KEYS = ("case_id", "data_role", "replicate", "node_id")

NODE_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "parent_id",
    "depth",
    "traversal_decision",
    "decision_class",
    "n_children",
    "n_descendant_leaves",
    "child_parent_edge_open",
    "sibling_open",
    "sibling_p_value",
    "sibling_projection_dimension",
    "root_stability_guard_blocked",
    "root_selective_guard_blocked",
    "selected_family_guard_blocked",
    "topology_incidence_role",
    "topology_pass_through_candidate",
}

TOPOLOGY_VALUE_COLUMNS = (
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "balance_product",
    "outgoing_balance_edge_product",
    "neighborhood_scale",
    "distance_to_stopping_edge",
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
)

LAW_VALUE_COLUMNS = (
    "topology_neighborhood_log_component",
    "topology_neighborhood_tau_b",
    "topology_neighborhood_tau_t",
    "topology_neighborhood_tau_s",
    "topology_neighborhood_h_k",
    "topology_neighborhood_nearest_stable_distance",
    "topology_neighborhood_nearest_signal_distance",
    "topology_neighborhood_support_count",
    "topology_neighborhood_signal_count",
    "topology_neighborhood_selected_nonnull_excluded_count",
    "topology_neighborhood_support_status",
    "neighborhood_scale_log_component",
    "neighborhood_scale_support_truth_count",
    "neighborhood_scale_support_negative_count",
    "neighborhood_scale_support_status",
    "support_truth_count",
    "support_negative_count",
    "support_status",
    "guarded_recovery_status",
    "recover_internal_split",
)

NUMERIC_METRICS = (
    "depth",
    "branch_length_to_parent",
    "n_children",
    "n_descendant_leaves",
    "sibling_p_value",
    "sibling_projection_dimension",
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "balance_product",
    "outgoing_balance_edge_product",
    "neighborhood_scale",
    "distance_to_stopping_edge",
    "topology_neighborhood_log_component",
    "topology_neighborhood_tau_b",
    "topology_neighborhood_tau_t",
    "topology_neighborhood_tau_s",
    "topology_neighborhood_h_k",
    "topology_neighborhood_nearest_stable_distance",
    "topology_neighborhood_nearest_signal_distance",
    "topology_neighborhood_support_count",
    "topology_neighborhood_signal_count",
    "topology_neighborhood_selected_nonnull_excluded_count",
    "neighborhood_scale_log_component",
    "neighborhood_scale_support_truth_count",
    "neighborhood_scale_support_negative_count",
    "support_truth_count",
    "support_negative_count",
)

JOINT_METRIC_PAIRS = (
    ("incoming_branch_balance", "outgoing_balance"),
    ("balance_product", "outgoing_edge_norm_balance"),
    ("balance_product", "neighborhood_scale"),
    ("balance_product", "topology_neighborhood_tau_b"),
    ("outgoing_edge_norm_balance", "outgoing_fragment_risk_proxy_score"),
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "parent_id",
    "branch_length_to_parent",
    "depth",
    "traversal_decision",
    "decision_class",
    "traversal_state",
    "traversal_stop_reason",
    "n_children",
    "n_descendant_leaves",
    "child_parent_edge_open",
    "sibling_open",
    "sibling_p_value",
    "sibling_projection_dimension",
    "root_stability_guard_blocked",
    "root_selective_guard_blocked",
    "selected_family_guard_blocked",
    "explicit_guard_blocked",
    "topology_incidence_role",
    "topology_pass_through_candidate",
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "neighborhood_evidence_family",
    *TOPOLOGY_VALUE_COLUMNS[3:],
    *LAW_VALUE_COLUMNS,
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "data_role",
    "traversal_state",
    "traversal_stop_reason",
    "guard_truth_role",
    "metric",
    "row_count",
    "finite_count",
    "missing_count",
    "mean",
    "std",
    "min",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "max",
    "distribution_status",
)

JOINT_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "data_role",
    "traversal_state",
    "metric_pair",
    "left_metric",
    "right_metric",
    "row_count",
    "finite_pair_count",
    "pearson_correlation",
    "spearman_correlation",
    "joint_status",
)

RUN_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "split_count",
    "boundary_count",
    "pass_through_count",
    "old_neighborhood_row_count",
    "current_neighborhood_row_count",
    "old_and_current_neighborhood_row_count",
    "recover_internal_split_count",
    "production_status",
)

COVERAGE_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "data_role",
    "traversal_state",
    "traversal_stop_reason",
    "row_count",
    "traversal_only_count",
    "old_topology_neighborhood_count",
    "current_directed_topology_count",
    "old_and_current_count",
    "old_neighborhood_row_count",
    "current_neighborhood_row_count",
    "old_neighborhood_fraction",
    "current_neighborhood_fraction",
    "old_and_current_fraction",
    "recover_internal_split_count",
    "coverage_status",
)

CASE_COVERAGE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "method_id",
    "data_role",
    "traversal_state",
    "traversal_stop_reason",
    "row_count",
    "traversal_only_count",
    "old_topology_neighborhood_count",
    "current_directed_topology_count",
    "old_and_current_count",
    "old_neighborhood_row_count",
    "current_neighborhood_row_count",
    "old_neighborhood_fraction",
    "current_neighborhood_fraction",
    "old_and_current_fraction",
    "recover_internal_split_count",
    "coverage_status",
)

METHOD_CONTRAST_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "left_method_id",
    "right_method_id",
    "paired_node_count",
    "traversal_decision_agreement_count",
    "traversal_decision_agreement_fraction",
    "traversal_state_agreement_count",
    "traversal_state_agreement_fraction",
    "traversal_stop_reason_agreement_count",
    "traversal_stop_reason_agreement_fraction",
    "left_split_count",
    "right_split_count",
    "left_pass_through_count",
    "right_pass_through_count",
    "left_boundary_count",
    "right_boundary_count",
    "both_old_and_current_count",
    "left_only_old_and_current_count",
    "right_only_old_and_current_count",
    "neither_old_and_current_count",
    "left_explicit_guard_blocked_count",
    "right_explicit_guard_blocked_count",
    "contrast_status",
)

CANDIDATE_METHOD_CONTRAST_COLUMNS = METHOD_CONTRAST_COLUMNS

CANDIDATE_METHOD_CONTRAST_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "left_method_id",
    "right_method_id",
    "candidate_reason",
    "left_traversal_decision",
    "right_traversal_decision",
    "left_traversal_state",
    "right_traversal_state",
    "left_traversal_stop_reason",
    "right_traversal_stop_reason",
    "left_neighborhood_evidence_family",
    "right_neighborhood_evidence_family",
    "left_explicit_guard_blocked",
    "right_explicit_guard_blocked",
    "left_depth",
    "right_depth",
    "left_n_descendant_leaves",
    "right_n_descendant_leaves",
    "left_child_parent_edge_open",
    "right_child_parent_edge_open",
    "left_sibling_open",
    "right_sibling_open",
    "left_sibling_p_value",
    "right_sibling_p_value",
    "left_sibling_projection_dimension",
    "right_sibling_projection_dimension",
    "left_topology_pass_through_candidate",
    "right_topology_pass_through_candidate",
    "left_balance_product",
    "right_balance_product",
    "left_outgoing_edge_norm_balance",
    "right_outgoing_edge_norm_balance",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "left_descendant_pass_through_count",
    "right_descendant_pass_through_count",
    "left_descendant_guard_blocked_count",
    "right_descendant_guard_blocked_count",
    "left_descendant_stable_boundary_count",
    "right_descendant_stable_boundary_count",
    "traversal_decision_agrees",
    "traversal_state_agrees",
    "traversal_stop_reason_agrees",
    "candidate_contrast_status",
)

CANDIDATE_AMBIGUITY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "ambiguity_bucket",
    "row_count",
    "decision_divergence_count",
    "left_split_count",
    "left_pass_through_count",
    "right_split_count",
    "right_pass_through_count",
    "left_guard_blocked_count",
    "right_guard_blocked_count",
    "both_old_and_current_count",
    "traversal_only_pair_count",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "candidate_reason_examples",
    "ambiguity_status",
)

CANDIDATE_LOCAL_FEATURE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "ambiguity_bucket",
    "profile_side",
    "method_id",
    "row_count",
    "edge_open_count",
    "sibling_open_count",
    "pass_through_candidate_count",
    "guard_blocked_count",
    "finite_sibling_p_count",
    "median_sibling_p_value",
    "finite_projection_dimension_count",
    "median_projection_dimension",
    "finite_depth_count",
    "median_depth",
    "min_depth",
    "max_depth",
    "finite_descendant_leaves_count",
    "median_descendant_leaves",
    "finite_balance_product_count",
    "median_balance_product",
    "finite_outgoing_edge_norm_balance_count",
    "median_outgoing_edge_norm_balance",
    "local_feature_status",
)

CANDIDATE_LAW_TARGET_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "ambiguity_bucket",
    "row_count",
    "decision_divergence_count",
    "both_old_and_current_count",
    "traversal_only_pair_count",
    "left_edge_open_count",
    "left_sibling_open_count",
    "left_pass_through_candidate_count",
    "left_guard_blocked_count",
    "left_median_sibling_p_value",
    "left_median_depth",
    "left_median_descendant_leaves",
    "left_finite_balance_product_count",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "law_target",
    "evidence_gap",
    "production_action",
)

STOP_RULE_COMPARISON_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "ambiguity_bucket",
    "row_count",
    "decision_divergence_count",
    "left_method_id",
    "right_method_id",
    "left_edge_open_count",
    "right_edge_open_count",
    "left_sibling_open_count",
    "right_sibling_open_count",
    "left_pass_through_count",
    "right_pass_through_count",
    "left_boundary_count",
    "right_boundary_count",
    "left_guard_blocked_count",
    "right_guard_blocked_count",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "left_descendant_pass_through_count",
    "right_descendant_pass_through_count",
    "left_old_and_current_count",
    "right_old_and_current_count",
    "traversal_only_pair_count",
    "stop_rule_pattern",
    "stop_rule_interpretation",
)

RETENTION_EVIDENCE_COLUMNS = (
    "schema_version",
    "study_role",
    "stop_rule_pattern",
    "data_role",
    "ambiguity_bucket",
    "row_count",
    "decision_divergence_count",
    "left_edge_open_count",
    "left_sibling_open_count",
    "left_pass_through_count",
    "right_pass_through_count",
    "right_guard_blocked_count",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "traversal_only_pair_count",
    "old_and_current_pair_count",
    "finite_sibling_p_count",
    "median_sibling_p_value",
    "finite_depth_count",
    "median_depth",
    "finite_descendant_leaves_count",
    "median_descendant_leaves",
    "finite_balance_product_count",
    "median_balance_product",
    "finite_outgoing_edge_norm_balance_count",
    "median_outgoing_edge_norm_balance",
    "retention_evidence_status",
)

RETENTION_GAP_COLUMNS = (
    "schema_version",
    "study_role",
    "stop_rule_pattern",
    "data_role",
    "ambiguity_bucket",
    "row_count",
    "retention_evidence_status",
    "observed_evidence",
    "missing_evidence",
    "method_implication",
    "next_law_requirement",
    "production_action",
)

METHOD_CONTRACT_COLUMNS = (
    "schema_version",
    "study_role",
    "contract_component",
    "component_status",
    "shared_by_methods",
    "conditional_profile_behavior",
    "refined_profile_behavior",
    "evidence_source",
    "evidence_summary",
    "missing_evidence",
    "next_law_requirement",
    "production_action",
)

PROFILE_CONFIG_CONTRACT_COLUMNS = (
    "schema_version",
    "study_role",
    "left_method_id",
    "right_method_id",
    "profile_field",
    "left_value",
    "right_value",
    "values_match",
    "profile_contract_status",
    "method_interpretation",
)

METHOD_READINESS_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "readiness_scope",
    "readiness_status",
    "supported_behavior",
    "unsupported_behavior",
    "evidence_source",
    "blocking_evidence",
    "required_next_law",
    "production_action",
)


@dataclass(frozen=True)
class SelectedNeighborhoodDistributionPanelConfig:
    """Runtime contract for selected-neighborhood distribution diagnostics."""

    node_decisions_path: Path
    output_dir: Path
    topology_rows_path: Path | None = None
    conditional_law_rows_path: Path | None = None

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_distribution_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_distribution_summary.csv"

    @property
    def joint_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_joint_summary.csv"

    @property
    def run_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_run_summary.csv"

    @property
    def coverage_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_coverage_summary.csv"

    @property
    def case_coverage_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_case_coverage_summary.csv"

    @property
    def method_contrast_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_method_contrast_summary.csv"

    @property
    def candidate_method_contrast_summary_path(self) -> Path:
        return self.output_dir / (
            "selected_neighborhood_candidate_method_contrast_summary.csv"
        )

    @property
    def candidate_method_contrast_rows_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_candidate_method_contrast_rows.csv"

    @property
    def candidate_ambiguity_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_candidate_ambiguity_summary.csv"

    @property
    def candidate_local_feature_summary_path(self) -> Path:
        return self.output_dir / (
            "selected_neighborhood_candidate_local_feature_summary.csv"
        )

    @property
    def candidate_law_target_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_candidate_law_target_summary.csv"

    @property
    def stop_rule_comparison_summary_path(self) -> Path:
        return self.output_dir / (
            "selected_neighborhood_stop_rule_comparison_summary.csv"
        )

    @property
    def retention_evidence_summary_path(self) -> Path:
        return self.output_dir / (
            "selected_neighborhood_retention_evidence_summary.csv"
        )

    @property
    def retention_gap_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_retention_gap_summary.csv"

    @property
    def method_contract_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_method_contract_summary.csv"

    @property
    def profile_config_contract_summary_path(self) -> Path:
        return self.output_dir / (
            "selected_neighborhood_profile_config_contract_summary.csv"
        )

    @property
    def method_readiness_summary_path(self) -> Path:
        return self.output_dir / "selected_neighborhood_method_readiness_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_columns(rows: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(rows.columns))
    if missing:
        raise ValueError(f"{label} rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in rows:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _string_series(rows: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in rows:
        return pd.Series(default, index=rows.index, dtype=object)
    return rows[column].fillna(default).astype(str)


def _bool_series(rows: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in rows:
        return pd.Series(default, index=rows.index, dtype=bool)
    values = rows[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.astype("object")
    normalized = normalized.where(pd.notna(normalized), str(default).lower())
    normalized = normalized.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes"})


def _dedupe_for_join(rows: pd.DataFrame, value_columns: Sequence[str]) -> pd.DataFrame:
    available = [column for column in value_columns if column in rows.columns]
    if not available:
        return pd.DataFrame(columns=list(JOIN_KEYS))
    required = set(JOIN_KEYS)
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Join rows are missing columns: {missing!r}")
    return rows.loc[:, [*JOIN_KEYS, *available]].drop_duplicates(
        subset=list(JOIN_KEYS),
        keep="first",
    )


def _merge_optional(
    base: pd.DataFrame,
    optional_rows: pd.DataFrame | None,
    *,
    value_columns: Sequence[str],
    suffix: str,
) -> pd.DataFrame:
    if optional_rows is None or optional_rows.empty:
        return base
    joinable = _dedupe_for_join(optional_rows, value_columns)
    if joinable.empty:
        return base
    return base.merge(joinable, on=list(JOIN_KEYS), how="left", suffixes=("", suffix))


def _classify_state(decision: str) -> str:
    decision = str(decision)
    if decision == "split":
        return "split"
    if decision == "pass_through":
        return "pass_through"
    return "boundary"


def _classify_stop_reason(row: pd.Series) -> str:
    state = _classify_state(str(row.get("traversal_decision", "")))
    if state == "split":
        return "accepted_split"
    if state == "pass_through":
        return "sibling_closed_descendant_split_available"
    if bool(row.get("explicit_guard_blocked", False)):
        return "explicit_guard_blocked"
    n_children = _finite_float(row.get("n_children", math.nan))
    if math.isfinite(n_children) and int(n_children) != 2:
        return "non_binary_or_leaf_boundary"
    if not bool(row.get("child_parent_edge_open", False)):
        return "edge_closed"
    if bool(row.get("sibling_open", False)):
        return "boundary_after_open_sibling"
    return "sibling_closed_no_descendant_split"


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _neighborhood_evidence_family(row: pd.Series) -> str:
    has_current = any(
        math.isfinite(_finite_float(row.get(column, math.nan)))
        for column in (
            "incoming_branch_balance",
            "outgoing_balance",
            "balance_product",
            "outgoing_edge_norm_balance",
            "neighborhood_scale",
        )
    )
    has_old = any(
        math.isfinite(_finite_float(row.get(column, math.nan)))
        for column in (
            "topology_neighborhood_tau_b",
            "topology_neighborhood_tau_t",
            "topology_neighborhood_tau_s",
            "topology_neighborhood_h_k",
            "distance_to_stopping_edge",
        )
    )
    if has_old and has_current:
        return "old_and_current"
    if has_old:
        return "old_topology_neighborhood"
    if has_current:
        return "current_directed_topology"
    return "traversal_only"


def build_selected_neighborhood_distribution_rows(
    *,
    node_decisions: pd.DataFrame,
    topology_rows: pd.DataFrame | None = None,
    conditional_law_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join traversal rows with optional topology evidence and classify states."""
    _validate_columns(node_decisions, NODE_REQUIRED_COLUMNS, "Node-decision")
    rows = node_decisions.copy()
    rows = _merge_optional(
        rows,
        topology_rows,
        value_columns=TOPOLOGY_VALUE_COLUMNS,
        suffix="_topology",
    )
    rows = _merge_optional(
        rows,
        conditional_law_rows,
        value_columns=LAW_VALUE_COLUMNS,
        suffix="_law",
    )
    rows["schema_version"] = SCHEMA_VERSION
    rows["study_role"] = STUDY_ROLE
    rows["traversal_state"] = _string_series(rows, "traversal_decision").map(
        _classify_state
    )
    rows["explicit_guard_blocked"] = (
        _bool_series(rows, "root_stability_guard_blocked")
        | _bool_series(rows, "root_selective_guard_blocked")
        | _bool_series(rows, "selected_family_guard_blocked")
    )
    rows["child_parent_edge_open"] = _bool_series(rows, "child_parent_edge_open")
    rows["sibling_open"] = _bool_series(rows, "sibling_open")
    rows["root_stability_guard_blocked"] = _bool_series(
        rows,
        "root_stability_guard_blocked",
    )
    rows["root_selective_guard_blocked"] = _bool_series(
        rows,
        "root_selective_guard_blocked",
    )
    rows["selected_family_guard_blocked"] = _bool_series(
        rows,
        "selected_family_guard_blocked",
    )
    if "recover_internal_split" in rows:
        rows["recover_internal_split"] = _bool_series(rows, "recover_internal_split")
    else:
        rows["recover_internal_split"] = False
    rows["traversal_stop_reason"] = [
        _classify_stop_reason(row) for _, row in rows.iterrows()
    ]
    rows["neighborhood_evidence_family"] = [
        _neighborhood_evidence_family(row) for _, row in rows.iterrows()
    ]

    for column in ROW_COLUMNS:
        if column not in rows:
            rows[column] = math.nan
    for column in NUMERIC_METRICS:
        rows[column] = _numeric(rows, column)
    for column in (
        "case_id",
        "data_role",
        "method_id",
        "node_id",
        "parent_id",
        "traversal_decision",
        "decision_class",
        "topology_incidence_role",
        "guard_truth_role",
        "topology_support_role",
        "topology_signal_role",
        "topology_neighborhood_support_status",
        "neighborhood_scale_support_status",
        "support_status",
        "guarded_recovery_status",
    ):
        if column in rows:
            rows[column] = _string_series(rows, column)
    return rows.loc[:, ROW_COLUMNS].copy()


def _distribution_record(
    *,
    method_id: str,
    data_role: str,
    traversal_state: str,
    traversal_stop_reason: str,
    guard_truth_role: str,
    metric: str,
    values: pd.Series,
    row_count: int,
) -> dict[str, object]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = finite.dropna()
    if finite.empty:
        stats = {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
        status = "distribution_no_finite_values"
    else:
        stats = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "min": float(finite.min()),
            "p05": float(finite.quantile(0.05)),
            "p25": float(finite.quantile(0.25)),
            "p50": float(finite.quantile(0.50)),
            "p75": float(finite.quantile(0.75)),
            "p95": float(finite.quantile(0.95)),
            "max": float(finite.max()),
        }
        status = (
            "distribution_observed"
            if int(finite.shape[0]) >= 2
            else "distribution_singleton_diagnostic"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "method_id": method_id,
        "data_role": data_role,
        "traversal_state": traversal_state,
        "traversal_stop_reason": traversal_stop_reason,
        "guard_truth_role": guard_truth_role,
        "metric": metric,
        "row_count": int(row_count),
        "finite_count": int(finite.shape[0]),
        "missing_count": int(row_count - finite.shape[0]),
        **stats,
        "distribution_status": status,
    }


def summarize_selected_neighborhood_distributions(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric neighborhood distributions by traversal state."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    group_columns = [
        "method_id",
        "data_role",
        "traversal_state",
        "traversal_stop_reason",
        "guard_truth_role",
    ]
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        row_count = int(group.shape[0])
        for metric in NUMERIC_METRICS:
            records.append(
                _distribution_record(
                    method_id=str(keys[0]),
                    data_role=str(keys[1]),
                    traversal_state=str(keys[2]),
                    traversal_stop_reason=str(keys[3]),
                    guard_truth_role=str(keys[4]),
                    metric=metric,
                    values=_numeric(group, metric),
                    row_count=row_count,
                )
            )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def summarize_selected_neighborhood_joint_distributions(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected joint neighborhood relationships."""
    if rows.empty:
        return pd.DataFrame(columns=JOINT_COLUMNS)
    records: list[dict[str, object]] = []
    group_columns = ["method_id", "data_role", "traversal_state"]
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        for left, right in JOINT_METRIC_PAIRS:
            pair = pd.DataFrame(
                {
                    "left": _numeric(group, left),
                    "right": _numeric(group, right),
                }
            ).replace([np.inf, -np.inf], np.nan)
            finite = pair.dropna()
            if finite.shape[0] >= 2 and (
                finite["left"].nunique(dropna=True) < 2
                or finite["right"].nunique(dropna=True) < 2
            ):
                pearson = math.nan
                spearman = math.nan
                status = "joint_distribution_constant_input"
            elif finite.shape[0] >= 2:
                pearson = float(finite["left"].corr(finite["right"], method="pearson"))
                spearman = float(
                    finite["left"].corr(finite["right"], method="spearman")
                )
                status = "joint_distribution_observed"
            elif finite.shape[0] == 1:
                pearson = math.nan
                spearman = math.nan
                status = "joint_distribution_singleton_diagnostic"
            else:
                pearson = math.nan
                spearman = math.nan
                status = "joint_distribution_no_finite_values"
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "method_id": str(keys[0]),
                    "data_role": str(keys[1]),
                    "traversal_state": str(keys[2]),
                    "metric_pair": f"{left}__{right}",
                    "left_metric": left,
                    "right_metric": right,
                    "row_count": int(group.shape[0]),
                    "finite_pair_count": int(finite.shape[0]),
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "joint_status": status,
                }
            )
    return pd.DataFrame.from_records(records, columns=JOINT_COLUMNS)


def summarize_selected_neighborhood_run(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize run-level state counts."""
    if rows.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "split_count": 0,
                    "boundary_count": 0,
                    "pass_through_count": 0,
                    "old_neighborhood_row_count": 0,
                    "current_neighborhood_row_count": 0,
                    "old_and_current_neighborhood_row_count": 0,
                    "recover_internal_split_count": 0,
                    "production_status": "diagnostic_only_no_rows",
                }
            ],
            columns=RUN_SUMMARY_COLUMNS,
        )
    family = rows["neighborhood_evidence_family"].astype(str)
    state = rows["traversal_state"].astype(str)
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "split_count": int(state.eq("split").sum()),
                "boundary_count": int(state.eq("boundary").sum()),
                "pass_through_count": int(state.eq("pass_through").sum()),
                "old_neighborhood_row_count": int(
                    family.isin(
                        {
                            "old_topology_neighborhood",
                            "old_and_current",
                        }
                    ).sum()
                ),
                "current_neighborhood_row_count": int(
                    family.isin(
                        {
                            "current_directed_topology",
                            "old_and_current",
                        }
                    ).sum()
                ),
                "old_and_current_neighborhood_row_count": int(
                    family.eq("old_and_current").sum()
                ),
                "recover_internal_split_count": int(
                    rows["recover_internal_split"].astype(bool).sum()
                    if "recover_internal_split" in rows
                    else 0
                ),
                "production_status": "diagnostic_only_not_calibration",
            }
        ],
        columns=RUN_SUMMARY_COLUMNS,
    )


def _coverage_status(
    *,
    row_count: int,
    old_count: int,
    current_count: int,
    old_and_current_count: int,
) -> str:
    if row_count <= 0:
        return "coverage_no_rows"
    if old_and_current_count == row_count:
        return "coverage_complete_old_current"
    if old_count == 0 and current_count == 0:
        return "coverage_traversal_only"
    if old_count > 0 and current_count == 0:
        return "coverage_old_only"
    if old_count == 0 and current_count > 0:
        return "coverage_current_only"
    return "coverage_partial_old_current"


def _summarize_coverage(
    rows: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    output_columns: Sequence[str],
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=output_columns)
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        key_values = dict(zip(group_columns, keys, strict=True))
        family = group["neighborhood_evidence_family"].astype(str)
        row_count = int(group.shape[0])
        traversal_only_count = int(family.eq("traversal_only").sum())
        old_topology_count = int(family.eq("old_topology_neighborhood").sum())
        current_topology_count = int(family.eq("current_directed_topology").sum())
        old_and_current_count = int(family.eq("old_and_current").sum())
        old_count = old_topology_count + old_and_current_count
        current_count = current_topology_count + old_and_current_count
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                **{column: str(value) for column, value in key_values.items()},
                "row_count": row_count,
                "traversal_only_count": traversal_only_count,
                "old_topology_neighborhood_count": old_topology_count,
                "current_directed_topology_count": current_topology_count,
                "old_and_current_count": old_and_current_count,
                "old_neighborhood_row_count": old_count,
                "current_neighborhood_row_count": current_count,
                "old_neighborhood_fraction": old_count / row_count,
                "current_neighborhood_fraction": current_count / row_count,
                "old_and_current_fraction": old_and_current_count / row_count,
                "recover_internal_split_count": int(
                    group["recover_internal_split"].astype(bool).sum()
                    if "recover_internal_split" in group
                    else 0
                ),
                "coverage_status": _coverage_status(
                    row_count=row_count,
                    old_count=old_count,
                    current_count=current_count,
                    old_and_current_count=old_and_current_count,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)


def summarize_selected_neighborhood_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize old/current neighborhood evidence coverage by traversal state."""
    return _summarize_coverage(
        rows,
        group_columns=[
            "method_id",
            "data_role",
            "traversal_state",
            "traversal_stop_reason",
        ],
        output_columns=COVERAGE_COLUMNS,
    )


def summarize_selected_neighborhood_case_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize old/current evidence coverage by case and traversal state."""
    return _summarize_coverage(
        rows,
        group_columns=[
            "case_id",
            "method_id",
            "data_role",
            "traversal_state",
            "traversal_stop_reason",
        ],
        output_columns=CASE_COVERAGE_COLUMNS,
    )


def _count_state(rows: pd.DataFrame, column: str, state: str) -> int:
    return int(rows[column].astype(str).eq(state).sum())


def _candidate_paired_nodes(paired: pd.DataFrame) -> pd.Series:
    left_state = paired["traversal_state_left"].astype(str)
    right_state = paired["traversal_state_right"].astype(str)
    left_old_current = paired["neighborhood_evidence_family_left"].astype(str).eq(
        "old_and_current"
    )
    right_old_current = paired["neighborhood_evidence_family_right"].astype(str).eq(
        "old_and_current"
    )
    return (
        left_state.isin({"split", "pass_through"})
        | right_state.isin({"split", "pass_through"})
        | paired["explicit_guard_blocked_left"].astype(bool)
        | paired["explicit_guard_blocked_right"].astype(bool)
        | left_old_current
        | right_old_current
    )


def _candidate_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if str(row["traversal_state_left"]) == "split":
        reasons.append("left_split")
    if str(row["traversal_state_right"]) == "split":
        reasons.append("right_split")
    if str(row["traversal_state_left"]) == "pass_through":
        reasons.append("left_pass_through")
    if str(row["traversal_state_right"]) == "pass_through":
        reasons.append("right_pass_through")
    if bool(row["explicit_guard_blocked_left"]):
        reasons.append("left_guard_blocked")
    if bool(row["explicit_guard_blocked_right"]):
        reasons.append("right_guard_blocked")
    if str(row["neighborhood_evidence_family_left"]) == "old_and_current":
        reasons.append("left_old_and_current")
    if str(row["neighborhood_evidence_family_right"]) == "old_and_current":
        reasons.append("right_old_and_current")
    return "|".join(reasons) if reasons else "not_candidate"


def _candidate_contrast_status(row: pd.Series) -> str:
    decision_agrees = str(row["traversal_decision_left"]) == str(
        row["traversal_decision_right"]
    )
    stop_reason_agrees = str(row["traversal_stop_reason_left"]) == str(
        row["traversal_stop_reason_right"]
    )
    if not decision_agrees:
        return "candidate_decision_diverges"
    if not stop_reason_agrees:
        return "candidate_stop_reason_diverges"
    return "candidate_decision_agrees"


def _descendant_outcome_lookup(
    rows: pd.DataFrame,
) -> dict[tuple[str, str, str, int, str], dict[str, int]]:
    child_map: dict[tuple[str, str, str, int], dict[str, list[str]]] = {}
    row_map: dict[tuple[str, str, str, int, str], pd.Series] = {}
    for _, row in rows.iterrows():
        group_key = (
            str(row["case_id"]),
            str(row["data_role"]),
            str(row["method_id"]),
            int(row["replicate"]),
        )
        node_id = str(row["node_id"])
        parent_id = str(row["parent_id"])
        row_map[(*group_key, node_id)] = row
        child_map.setdefault(group_key, {}).setdefault(parent_id, []).append(node_id)

    outcome_cache: dict[tuple[str, str, str, int, str], dict[str, int]] = {}

    def outcomes_for(
        case_id: str,
        data_role: str,
        method_id: str,
        replicate: int,
        node_id: str,
    ) -> dict[str, int]:
        key = (case_id, data_role, method_id, int(replicate), node_id)
        if key in outcome_cache:
            return outcome_cache[key]
        group_key = (case_id, data_role, method_id, int(replicate))
        children = child_map.get(group_key, {})
        stack = list(children.get(node_id, []))
        accepted_split_count = 0
        pass_through_count = 0
        guard_blocked_count = 0
        stable_boundary_count = 0
        while stack:
            child = stack.pop()
            child_row = row_map.get((*group_key, child))
            if child_row is not None:
                decision_class = str(child_row.get("decision_class", ""))
                traversal_state = str(child_row.get("traversal_state", ""))
                if decision_class == "accepted_internal_split":
                    accepted_split_count += 1
                if traversal_state == "pass_through":
                    pass_through_count += 1
                if bool(child_row.get("explicit_guard_blocked", False)):
                    guard_blocked_count += 1
                if decision_class == "stable_boundary":
                    stable_boundary_count += 1
            stack.extend(children.get(child, []))
        outcome_cache[key] = {
            "accepted_split_count": int(accepted_split_count),
            "pass_through_count": int(pass_through_count),
            "guard_blocked_count": int(guard_blocked_count),
            "stable_boundary_count": int(stable_boundary_count),
        }
        return outcome_cache[key]

    # Materialize for all nodes once so row construction stays simple.
    for case_id, data_role, method_id, replicate, node_id in list(row_map):
        outcomes_for(case_id, data_role, method_id, replicate, node_id)
    return outcome_cache


def _candidate_ambiguity_bucket(row: pd.Series) -> str:
    if bool(row["traversal_decision_agrees"]):
        return "candidate_agreement"
    left_active = str(row["left_traversal_state"]) in {"split", "pass_through"}
    right_suppressed = str(row["right_traversal_decision"]) in {
        "boundary",
        "not_visited",
    }
    right_active = str(row["right_traversal_state"]) in {"split", "pass_through"}
    left_suppressed = str(row["left_traversal_decision"]) in {
        "boundary",
        "not_visited",
    }
    if str(row["data_role"]) == "selected_null" and left_active and right_suppressed:
        return "conservative_selected_null_suppression"
    if str(row["data_role"]) == "signal" and left_active and right_suppressed:
        return "possible_signal_over_suppression"
    if right_active and left_suppressed:
        return "refined_profile_more_open"
    return "other_candidate_divergence"


def _method_contrast_records(
    paired: pd.DataFrame,
    *,
    left_method: str,
    right_method: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for keys, group in paired.groupby(["case_id", "data_role"], sort=True):
        paired_count = int(group.shape[0])
        decision_agree = group["traversal_decision_left"].astype(str).eq(
            group["traversal_decision_right"].astype(str)
        )
        state_agree = group["traversal_state_left"].astype(str).eq(
            group["traversal_state_right"].astype(str)
        )
        stop_agree = group["traversal_stop_reason_left"].astype(str).eq(
            group["traversal_stop_reason_right"].astype(str)
        )
        left_old_current = group["neighborhood_evidence_family_left"].astype(str).eq(
            "old_and_current"
        )
        right_old_current = group["neighborhood_evidence_family_right"].astype(str).eq(
            "old_and_current"
        )
        both_old_current = left_old_current & right_old_current
        left_only_old_current = left_old_current & ~right_old_current
        right_only_old_current = ~left_old_current & right_old_current
        neither_old_current = ~left_old_current & ~right_old_current
        status = (
            "method_decisions_identical"
            if bool(decision_agree.all())
            else "method_decisions_diverge"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(keys[0]),
                "data_role": str(keys[1]),
                "left_method_id": left_method,
                "right_method_id": right_method,
                "paired_node_count": paired_count,
                "traversal_decision_agreement_count": int(decision_agree.sum()),
                "traversal_decision_agreement_fraction": float(
                    decision_agree.mean()
                ),
                "traversal_state_agreement_count": int(state_agree.sum()),
                "traversal_state_agreement_fraction": float(state_agree.mean()),
                "traversal_stop_reason_agreement_count": int(stop_agree.sum()),
                "traversal_stop_reason_agreement_fraction": float(stop_agree.mean()),
                "left_split_count": _count_state(
                    group,
                    "traversal_state_left",
                    "split",
                ),
                "right_split_count": _count_state(
                    group,
                    "traversal_state_right",
                    "split",
                ),
                "left_pass_through_count": _count_state(
                    group,
                    "traversal_state_left",
                    "pass_through",
                ),
                "right_pass_through_count": _count_state(
                    group,
                    "traversal_state_right",
                    "pass_through",
                ),
                "left_boundary_count": _count_state(
                    group,
                    "traversal_state_left",
                    "boundary",
                ),
                "right_boundary_count": _count_state(
                    group,
                    "traversal_state_right",
                    "boundary",
                ),
                "both_old_and_current_count": int(both_old_current.sum()),
                "left_only_old_and_current_count": int(left_only_old_current.sum()),
                "right_only_old_and_current_count": int(
                    right_only_old_current.sum()
                ),
                "neither_old_and_current_count": int(neither_old_current.sum()),
                "left_explicit_guard_blocked_count": int(
                    group["explicit_guard_blocked_left"].astype(bool).sum()
                ),
                "right_explicit_guard_blocked_count": int(
                    group["explicit_guard_blocked_right"].astype(bool).sum()
                ),
                "contrast_status": status,
            }
        )
    return records


def _paired_method_rows(rows: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    method_ids = sorted(rows["method_id"].astype(str).dropna().unique())
    if len(method_ids) < 2:
        return []

    paired_rows: list[tuple[str, str, pd.DataFrame]] = []
    pair_keys = ["case_id", "data_role", "replicate", "node_id"]
    value_columns = [
        *pair_keys,
        "method_id",
        "depth",
        "n_descendant_leaves",
        "child_parent_edge_open",
        "sibling_open",
        "sibling_p_value",
        "sibling_projection_dimension",
        "topology_pass_through_candidate",
        "balance_product",
        "outgoing_edge_norm_balance",
        "traversal_decision",
        "traversal_state",
        "traversal_stop_reason",
        "neighborhood_evidence_family",
        "explicit_guard_blocked",
    ]
    compact = rows.loc[:, value_columns].drop_duplicates(
        subset=[*pair_keys, "method_id"],
        keep="first",
    )
    for left_method, right_method in combinations(method_ids, 2):
        left = compact.loc[compact["method_id"].astype(str).eq(left_method)]
        right = compact.loc[compact["method_id"].astype(str).eq(right_method)]
        paired = left.merge(
            right,
            on=pair_keys,
            how="inner",
            suffixes=("_left", "_right"),
        )
        if not paired.empty:
            paired_rows.append((left_method, right_method, paired))
    return paired_rows


def summarize_selected_neighborhood_method_contrast(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare traversal decisions for method pairs on the same selected nodes."""
    if rows.empty:
        return pd.DataFrame(columns=METHOD_CONTRAST_COLUMNS)
    paired_rows = _paired_method_rows(rows)
    if not paired_rows:
        return pd.DataFrame(columns=METHOD_CONTRAST_COLUMNS)
    records: list[dict[str, object]] = []
    for left_method, right_method, paired in paired_rows:
        records.extend(
            _method_contrast_records(
                paired,
                left_method=left_method,
                right_method=right_method,
            )
        )
    return pd.DataFrame.from_records(records, columns=METHOD_CONTRAST_COLUMNS)


def summarize_selected_neighborhood_candidate_method_contrast(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Compare methods only on paired nodes that are traversal candidates."""
    if rows.empty:
        return pd.DataFrame(columns=CANDIDATE_METHOD_CONTRAST_COLUMNS)
    paired_rows = _paired_method_rows(rows)
    if not paired_rows:
        return pd.DataFrame(columns=CANDIDATE_METHOD_CONTRAST_COLUMNS)
    records: list[dict[str, object]] = []
    for left_method, right_method, paired in paired_rows:
        candidates = paired.loc[_candidate_paired_nodes(paired)].copy()
        if candidates.empty:
            continue
        records.extend(
            _method_contrast_records(
                candidates,
                left_method=left_method,
                right_method=right_method,
            )
        )
    return pd.DataFrame.from_records(records, columns=CANDIDATE_METHOD_CONTRAST_COLUMNS)


def build_selected_neighborhood_candidate_method_contrast_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return node-level paired candidate rows for method comparison."""
    if rows.empty:
        return pd.DataFrame(columns=CANDIDATE_METHOD_CONTRAST_ROW_COLUMNS)
    paired_rows = _paired_method_rows(rows)
    if not paired_rows:
        return pd.DataFrame(columns=CANDIDATE_METHOD_CONTRAST_ROW_COLUMNS)
    descendant_outcomes = _descendant_outcome_lookup(rows)
    records: list[dict[str, object]] = []
    for left_method, right_method, paired in paired_rows:
        candidates = paired.loc[_candidate_paired_nodes(paired)].copy()
        if candidates.empty:
            continue
        for _, row in candidates.sort_values(
            ["case_id", "data_role", "replicate", "node_id"]
        ).iterrows():
            decision_agrees = str(row["traversal_decision_left"]) == str(
                row["traversal_decision_right"]
            )
            state_agrees = str(row["traversal_state_left"]) == str(
                row["traversal_state_right"]
            )
            stop_reason_agrees = str(row["traversal_stop_reason_left"]) == str(
                row["traversal_stop_reason_right"]
            )
            left_descendants = descendant_outcomes.get(
                (
                    str(row["case_id"]),
                    str(row["data_role"]),
                    left_method,
                    int(row["replicate"]),
                    str(row["node_id"]),
                ),
                {},
            )
            right_descendants = descendant_outcomes.get(
                (
                    str(row["case_id"]),
                    str(row["data_role"]),
                    right_method,
                    int(row["replicate"]),
                    str(row["node_id"]),
                ),
                {},
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "case_id": str(row["case_id"]),
                    "data_role": str(row["data_role"]),
                    "replicate": int(row["replicate"]),
                    "node_id": str(row["node_id"]),
                    "left_method_id": left_method,
                    "right_method_id": right_method,
                    "candidate_reason": _candidate_reason(row),
                    "left_traversal_decision": str(row["traversal_decision_left"]),
                    "right_traversal_decision": str(row["traversal_decision_right"]),
                    "left_traversal_state": str(row["traversal_state_left"]),
                    "right_traversal_state": str(row["traversal_state_right"]),
                    "left_traversal_stop_reason": str(
                        row["traversal_stop_reason_left"]
                    ),
                    "right_traversal_stop_reason": str(
                        row["traversal_stop_reason_right"]
                    ),
                    "left_neighborhood_evidence_family": str(
                        row["neighborhood_evidence_family_left"]
                    ),
                    "right_neighborhood_evidence_family": str(
                        row["neighborhood_evidence_family_right"]
                    ),
                    "left_explicit_guard_blocked": bool(
                        row["explicit_guard_blocked_left"]
                    ),
                    "right_explicit_guard_blocked": bool(
                        row["explicit_guard_blocked_right"]
                    ),
                    "left_depth": _finite_float(row["depth_left"]),
                    "right_depth": _finite_float(row["depth_right"]),
                    "left_n_descendant_leaves": _finite_float(
                        row["n_descendant_leaves_left"]
                    ),
                    "right_n_descendant_leaves": _finite_float(
                        row["n_descendant_leaves_right"]
                    ),
                    "left_child_parent_edge_open": bool(
                        row["child_parent_edge_open_left"]
                    ),
                    "right_child_parent_edge_open": bool(
                        row["child_parent_edge_open_right"]
                    ),
                    "left_sibling_open": bool(row["sibling_open_left"]),
                    "right_sibling_open": bool(row["sibling_open_right"]),
                    "left_sibling_p_value": _finite_float(
                        row["sibling_p_value_left"]
                    ),
                    "right_sibling_p_value": _finite_float(
                        row["sibling_p_value_right"]
                    ),
                    "left_sibling_projection_dimension": _finite_float(
                        row["sibling_projection_dimension_left"]
                    ),
                    "right_sibling_projection_dimension": _finite_float(
                        row["sibling_projection_dimension_right"]
                    ),
                    "left_topology_pass_through_candidate": bool(
                        row["topology_pass_through_candidate_left"]
                    ),
                    "right_topology_pass_through_candidate": bool(
                        row["topology_pass_through_candidate_right"]
                    ),
                    "left_balance_product": _finite_float(row["balance_product_left"]),
                    "right_balance_product": _finite_float(
                        row["balance_product_right"]
                    ),
                    "left_outgoing_edge_norm_balance": _finite_float(
                        row["outgoing_edge_norm_balance_left"]
                    ),
                    "right_outgoing_edge_norm_balance": _finite_float(
                        row["outgoing_edge_norm_balance_right"]
                    ),
                    "left_descendant_accepted_split_count": int(
                        left_descendants.get("accepted_split_count", 0)
                    ),
                    "right_descendant_accepted_split_count": int(
                        right_descendants.get("accepted_split_count", 0)
                    ),
                    "left_descendant_pass_through_count": int(
                        left_descendants.get("pass_through_count", 0)
                    ),
                    "right_descendant_pass_through_count": int(
                        right_descendants.get("pass_through_count", 0)
                    ),
                    "left_descendant_guard_blocked_count": int(
                        left_descendants.get("guard_blocked_count", 0)
                    ),
                    "right_descendant_guard_blocked_count": int(
                        right_descendants.get("guard_blocked_count", 0)
                    ),
                    "left_descendant_stable_boundary_count": int(
                        left_descendants.get("stable_boundary_count", 0)
                    ),
                    "right_descendant_stable_boundary_count": int(
                        right_descendants.get("stable_boundary_count", 0)
                    ),
                    "traversal_decision_agrees": bool(decision_agrees),
                    "traversal_state_agrees": bool(state_agrees),
                    "traversal_stop_reason_agrees": bool(stop_reason_agrees),
                    "candidate_contrast_status": _candidate_contrast_status(row),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=CANDIDATE_METHOD_CONTRAST_ROW_COLUMNS,
    )


def summarize_selected_neighborhood_candidate_ambiguity(
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize candidate-level method disagreements into ambiguity buckets."""
    if candidate_rows.empty:
        return pd.DataFrame(columns=CANDIDATE_AMBIGUITY_COLUMNS)
    rows = candidate_rows.copy()
    rows["ambiguity_bucket"] = [
        _candidate_ambiguity_bucket(row) for _, row in rows.iterrows()
    ]
    records: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "ambiguity_bucket"]
    for keys, group in rows.groupby(group_columns, sort=True):
        decision_divergent = ~group["traversal_decision_agrees"].astype(bool)
        both_old_current = group["left_neighborhood_evidence_family"].astype(str).eq(
            "old_and_current"
        ) & group["right_neighborhood_evidence_family"].astype(str).eq(
            "old_and_current"
        )
        traversal_only_pair = group["left_neighborhood_evidence_family"].astype(str).eq(
            "traversal_only"
        ) & group["right_neighborhood_evidence_family"].astype(str).eq(
            "traversal_only"
        )
        reason_examples = sorted(set(group["candidate_reason"].astype(str)))[:5]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(keys[0]),
                "data_role": str(keys[1]),
                "ambiguity_bucket": str(keys[2]),
                "row_count": int(group.shape[0]),
                "decision_divergence_count": int(decision_divergent.sum()),
                "left_split_count": _count_state(
                    group,
                    "left_traversal_state",
                    "split",
                ),
                "left_pass_through_count": _count_state(
                    group,
                    "left_traversal_state",
                    "pass_through",
                ),
                "right_split_count": _count_state(
                    group,
                    "right_traversal_state",
                    "split",
                ),
                "right_pass_through_count": _count_state(
                    group,
                    "right_traversal_state",
                    "pass_through",
                ),
                "left_guard_blocked_count": int(
                    group["left_explicit_guard_blocked"].astype(bool).sum()
                ),
                "right_guard_blocked_count": int(
                    group["right_explicit_guard_blocked"].astype(bool).sum()
                ),
                "both_old_and_current_count": int(both_old_current.sum()),
                "traversal_only_pair_count": int(traversal_only_pair.sum()),
                "left_descendant_accepted_split_count": int(
                    group["left_descendant_accepted_split_count"].sum()
                ),
                "right_descendant_accepted_split_count": int(
                    group["right_descendant_accepted_split_count"].sum()
                ),
                "candidate_reason_examples": ";".join(reason_examples),
                "ambiguity_status": (
                    "candidate_ambiguity_requires_law"
                    if bool(decision_divergent.any())
                    else "candidate_agreement_diagnostic"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=CANDIDATE_AMBIGUITY_COLUMNS)


def _finite_median(values: pd.Series) -> tuple[int, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = finite.dropna()
    if finite.empty:
        return 0, math.nan
    return int(finite.shape[0]), float(finite.median())


def _finite_min_max(values: pd.Series) -> tuple[int, float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = finite.dropna()
    if finite.empty:
        return 0, math.nan, math.nan
    return int(finite.shape[0]), float(finite.min()), float(finite.max())


def summarize_selected_neighborhood_candidate_local_features(
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize local traversal geometry by ambiguity bucket and profile side."""
    if candidate_rows.empty:
        return pd.DataFrame(columns=CANDIDATE_LOCAL_FEATURE_COLUMNS)
    rows = candidate_rows.copy()
    if "ambiguity_bucket" not in rows:
        rows["ambiguity_bucket"] = [
            _candidate_ambiguity_bucket(row) for _, row in rows.iterrows()
        ]

    records: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "ambiguity_bucket"]
    for keys, group in rows.groupby(group_columns, sort=True):
        for side in ("left", "right"):
            sibling_p_count, sibling_p_median = _finite_median(
                group[f"{side}_sibling_p_value"]
            )
            projection_count, projection_median = _finite_median(
                group[f"{side}_sibling_projection_dimension"]
            )
            depth_count, depth_median = _finite_median(group[f"{side}_depth"])
            _, depth_min, depth_max = _finite_min_max(group[f"{side}_depth"])
            leaves_count, leaves_median = _finite_median(
                group[f"{side}_n_descendant_leaves"]
            )
            balance_count, balance_median = _finite_median(
                group[f"{side}_balance_product"]
            )
            edge_norm_count, edge_norm_median = _finite_median(
                group[f"{side}_outgoing_edge_norm_balance"]
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "case_id": str(keys[0]),
                    "data_role": str(keys[1]),
                    "ambiguity_bucket": str(keys[2]),
                    "profile_side": side,
                    "method_id": str(group[f"{side}_method_id"].iloc[0])
                    if f"{side}_method_id" in group
                    else "",
                    "row_count": int(group.shape[0]),
                    "edge_open_count": int(
                        group[f"{side}_child_parent_edge_open"].astype(bool).sum()
                    ),
                    "sibling_open_count": int(
                        group[f"{side}_sibling_open"].astype(bool).sum()
                    ),
                    "pass_through_candidate_count": int(
                        group[f"{side}_topology_pass_through_candidate"]
                        .astype(bool)
                        .sum()
                    ),
                    "guard_blocked_count": int(
                        group[f"{side}_explicit_guard_blocked"].astype(bool).sum()
                    ),
                    "finite_sibling_p_count": sibling_p_count,
                    "median_sibling_p_value": sibling_p_median,
                    "finite_projection_dimension_count": projection_count,
                    "median_projection_dimension": projection_median,
                    "finite_depth_count": depth_count,
                    "median_depth": depth_median,
                    "min_depth": depth_min,
                    "max_depth": depth_max,
                    "finite_descendant_leaves_count": leaves_count,
                    "median_descendant_leaves": leaves_median,
                    "finite_balance_product_count": balance_count,
                    "median_balance_product": balance_median,
                    "finite_outgoing_edge_norm_balance_count": edge_norm_count,
                    "median_outgoing_edge_norm_balance": edge_norm_median,
                    "local_feature_status": (
                        "local_features_observed"
                        if sibling_p_count or balance_count
                        else "local_features_missing"
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=CANDIDATE_LOCAL_FEATURE_COLUMNS)


def _law_target_for_bucket(
    *,
    ambiguity_bucket: str,
    row_count: int,
    both_old_and_current_count: int,
    traversal_only_pair_count: int,
    left_pass_through_candidate_count: int,
    left_finite_balance_product_count: int,
    left_descendant_accepted_split_count: int = 0,
) -> tuple[str, str, str]:
    if ambiguity_bucket == "candidate_agreement":
        return (
            "agreement_context_not_calibration",
            "no_disagreement_to_resolve",
            "diagnostic_only_no_promotion",
        )
    if ambiguity_bucket == "conservative_selected_null_suppression":
        evidence_gap = (
            "partial_topology_evidence"
            if both_old_and_current_count > 0
            else "traversal_only_false_positive_context"
        )
        return (
            "validate_false_positive_suppression_law",
            evidence_gap,
            "retain_fail_closed_refined_guard",
        )
    if ambiguity_bucket == "possible_signal_over_suppression":
        missing_topology = left_finite_balance_product_count == 0
        mostly_pass_through = left_pass_through_candidate_count >= max(1, row_count // 2)
        evidence_gap = (
            "missing_topology_evidence_for_downstream_splits"
            if missing_topology and left_descendant_accepted_split_count > 0
            else
            "missing_topology_evidence_for_pass_through"
            if missing_topology and mostly_pass_through
            else "incomplete_signal_recovery_evidence"
        )
        return (
            "derive_traversal_only_pass_through_retention_law",
            evidence_gap,
            "fail_closed_until_law_validated",
        )
    if ambiguity_bucket == "refined_profile_more_open":
        return (
            "inspect_refined_opening_cases",
            "opposite_direction_disagreement",
            "diagnostic_only_no_promotion",
        )
    if traversal_only_pair_count == row_count:
        return (
            "inspect_other_traversal_only_divergence",
            "traversal_only_unclassified_disagreement",
            "fail_closed_until_law_validated",
        )
    return (
        "inspect_other_candidate_divergence",
        "mixed_evidence_unclassified_disagreement",
        "diagnostic_only_no_promotion",
    )


def summarize_selected_neighborhood_candidate_law_targets(
    ambiguity: pd.DataFrame,
    local_features: pd.DataFrame,
) -> pd.DataFrame:
    """Convert ambiguity buckets into explicit diagnostic law targets."""
    if ambiguity.empty:
        return pd.DataFrame(columns=CANDIDATE_LAW_TARGET_COLUMNS)
    left_local = (
        local_features.loc[local_features["profile_side"].astype(str).eq("left")]
        if not local_features.empty
        else pd.DataFrame(columns=CANDIDATE_LOCAL_FEATURE_COLUMNS)
    )
    local_index = {
        (str(row["case_id"]), str(row["data_role"]), str(row["ambiguity_bucket"])): row
        for _, row in left_local.iterrows()
    }
    records: list[dict[str, object]] = []
    for _, row in ambiguity.iterrows():
        key = (
            str(row["case_id"]),
            str(row["data_role"]),
            str(row["ambiguity_bucket"]),
        )
        local = local_index.get(key, {})
        row_count = int(row["row_count"])
        both_old_and_current_count = int(row["both_old_and_current_count"])
        traversal_only_pair_count = int(row["traversal_only_pair_count"])
        pass_through_candidate_count = int(
            local.get("pass_through_candidate_count", 0)
        )
        finite_balance_product_count = int(
            local.get("finite_balance_product_count", 0)
        )
        left_descendant_split_count = int(
            row.get("left_descendant_accepted_split_count", 0)
        )
        right_descendant_split_count = int(
            row.get("right_descendant_accepted_split_count", 0)
        )
        law_target, evidence_gap, production_action = _law_target_for_bucket(
            ambiguity_bucket=str(row["ambiguity_bucket"]),
            row_count=row_count,
            both_old_and_current_count=both_old_and_current_count,
            traversal_only_pair_count=traversal_only_pair_count,
            left_pass_through_candidate_count=pass_through_candidate_count,
            left_finite_balance_product_count=finite_balance_product_count,
            left_descendant_accepted_split_count=left_descendant_split_count,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "ambiguity_bucket": str(row["ambiguity_bucket"]),
                "row_count": row_count,
                "decision_divergence_count": int(row["decision_divergence_count"]),
                "both_old_and_current_count": both_old_and_current_count,
                "traversal_only_pair_count": traversal_only_pair_count,
                "left_edge_open_count": int(local.get("edge_open_count", 0)),
                "left_sibling_open_count": int(local.get("sibling_open_count", 0)),
                "left_pass_through_candidate_count": pass_through_candidate_count,
                "left_guard_blocked_count": int(local.get("guard_blocked_count", 0)),
                "left_median_sibling_p_value": _finite_float(
                    local.get("median_sibling_p_value", math.nan)
                ),
                "left_median_depth": _finite_float(
                    local.get("median_depth", math.nan)
                ),
                "left_median_descendant_leaves": _finite_float(
                    local.get("median_descendant_leaves", math.nan)
                ),
                "left_finite_balance_product_count": finite_balance_product_count,
                "left_descendant_accepted_split_count": left_descendant_split_count,
                "right_descendant_accepted_split_count": right_descendant_split_count,
                "law_target": law_target,
                "evidence_gap": evidence_gap,
                "production_action": production_action,
            }
        )
    return pd.DataFrame.from_records(records, columns=CANDIDATE_LAW_TARGET_COLUMNS)


def _stop_rule_pattern(
    *,
    ambiguity_bucket: str,
    decision_divergence_count: int,
    left_pass_through_count: int,
    right_pass_through_count: int,
    left_descendant_accepted_split_count: int,
    right_descendant_accepted_split_count: int,
    right_guard_blocked_count: int,
    traversal_only_pair_count: int,
    row_count: int,
) -> tuple[str, str]:
    if decision_divergence_count == 0:
        return (
            "paired_candidate_agreement",
            "Candidate decisions agree; use only as context, not calibration.",
        )
    if (
        left_pass_through_count > right_pass_through_count
        and left_descendant_accepted_split_count
        > right_descendant_accepted_split_count
    ):
        interpretation = (
            "Conditional traversal walks through sibling-closed candidates to "
            "downstream accepted splits while the refined profile stops them."
        )
        if traversal_only_pair_count == row_count:
            interpretation += " Joined topology-neighborhood evidence is absent."
        if ambiguity_bucket == "conservative_selected_null_suppression":
            interpretation += (
                " Because this occurs in selected-null rows, descendant splits "
                "cannot be a retention rule."
            )
        return (
            "left_pass_through_downstream_split_right_stops",
            interpretation,
        )
    if right_guard_blocked_count > 0:
        return (
            "right_refined_guard_blocks_left_candidate",
            (
                "The refined profile closes candidate movement with an explicit "
                "guard; this can be useful null suppression or possible signal "
                "over-suppression depending on the selected-neighborhood law."
            ),
        )
    return (
        "candidate_stop_rule_divergence_unclassified",
        (
            "Candidate traversal states diverge, but the current diagnostic "
            "features do not isolate a stop/pass-through mechanism."
        ),
    )


def summarize_selected_neighborhood_stop_rule_comparison(
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how candidate rows differ under the traversal stop rule."""
    if candidate_rows.empty:
        return pd.DataFrame(columns=STOP_RULE_COMPARISON_COLUMNS)
    rows = candidate_rows.copy()
    if "ambiguity_bucket" not in rows:
        rows["ambiguity_bucket"] = [
            _candidate_ambiguity_bucket(row) for _, row in rows.iterrows()
        ]
    records: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "ambiguity_bucket"]
    for keys, group in rows.groupby(group_columns, sort=True):
        row_count = int(group.shape[0])
        decision_divergent = ~group["traversal_decision_agrees"].astype(bool)
        left_old_current = group["left_neighborhood_evidence_family"].astype(str).eq(
            "old_and_current"
        )
        right_old_current = group["right_neighborhood_evidence_family"].astype(str).eq(
            "old_and_current"
        )
        traversal_only_pair = group["left_neighborhood_evidence_family"].astype(str).eq(
            "traversal_only"
        ) & group["right_neighborhood_evidence_family"].astype(str).eq(
            "traversal_only"
        )
        left_pass_through_count = _count_state(
            group,
            "left_traversal_state",
            "pass_through",
        )
        right_pass_through_count = _count_state(
            group,
            "right_traversal_state",
            "pass_through",
        )
        right_guard_blocked_count = int(
            group["right_explicit_guard_blocked"].astype(bool).sum()
        )
        left_descendant_split_count = int(
            group["left_descendant_accepted_split_count"].sum()
        )
        right_descendant_split_count = int(
            group["right_descendant_accepted_split_count"].sum()
        )
        pattern, interpretation = _stop_rule_pattern(
            ambiguity_bucket=str(keys[2]),
            decision_divergence_count=int(decision_divergent.sum()),
            left_pass_through_count=left_pass_through_count,
            right_pass_through_count=right_pass_through_count,
            left_descendant_accepted_split_count=left_descendant_split_count,
            right_descendant_accepted_split_count=right_descendant_split_count,
            right_guard_blocked_count=right_guard_blocked_count,
            traversal_only_pair_count=int(traversal_only_pair.sum()),
            row_count=row_count,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(keys[0]),
                "data_role": str(keys[1]),
                "ambiguity_bucket": str(keys[2]),
                "row_count": row_count,
                "decision_divergence_count": int(decision_divergent.sum()),
                "left_method_id": str(group["left_method_id"].iloc[0]),
                "right_method_id": str(group["right_method_id"].iloc[0]),
                "left_edge_open_count": int(
                    group["left_child_parent_edge_open"].astype(bool).sum()
                ),
                "right_edge_open_count": int(
                    group["right_child_parent_edge_open"].astype(bool).sum()
                ),
                "left_sibling_open_count": int(
                    group["left_sibling_open"].astype(bool).sum()
                ),
                "right_sibling_open_count": int(
                    group["right_sibling_open"].astype(bool).sum()
                ),
                "left_pass_through_count": left_pass_through_count,
                "right_pass_through_count": right_pass_through_count,
                "left_boundary_count": _count_state(
                    group,
                    "left_traversal_state",
                    "boundary",
                ),
                "right_boundary_count": _count_state(
                    group,
                    "right_traversal_state",
                    "boundary",
                ),
                "left_guard_blocked_count": int(
                    group["left_explicit_guard_blocked"].astype(bool).sum()
                ),
                "right_guard_blocked_count": right_guard_blocked_count,
                "left_descendant_accepted_split_count": left_descendant_split_count,
                "right_descendant_accepted_split_count": right_descendant_split_count,
                "left_descendant_pass_through_count": int(
                    group["left_descendant_pass_through_count"].sum()
                ),
                "right_descendant_pass_through_count": int(
                    group["right_descendant_pass_through_count"].sum()
                ),
                "left_old_and_current_count": int(left_old_current.sum()),
                "right_old_and_current_count": int(right_old_current.sum()),
                "traversal_only_pair_count": int(traversal_only_pair.sum()),
                "stop_rule_pattern": pattern,
                "stop_rule_interpretation": interpretation,
            }
        )
    return pd.DataFrame.from_records(records, columns=STOP_RULE_COMPARISON_COLUMNS)


def _retention_evidence_status(
    *,
    stop_rule_pattern: str,
    data_role: str,
    row_count: int,
    traversal_only_pair_count: int,
    finite_balance_product_count: int,
) -> str:
    if stop_rule_pattern != "left_pass_through_downstream_split_right_stops":
        return "retention_evidence_context_only"
    if data_role == "selected_null":
        return "retention_evidence_false_positive_risk"
    if traversal_only_pair_count == row_count and finite_balance_product_count == 0:
        return "retention_evidence_signal_unresolved_no_topology"
    if finite_balance_product_count > 0:
        return "retention_evidence_signal_with_topology_context"
    return "retention_evidence_signal_incomplete"


def summarize_selected_neighborhood_retention_evidence(
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize evidence available for pass-through retention candidates."""
    if candidate_rows.empty:
        return pd.DataFrame(columns=RETENTION_EVIDENCE_COLUMNS)
    rows = candidate_rows.copy()
    if "ambiguity_bucket" not in rows:
        rows["ambiguity_bucket"] = [
            _candidate_ambiguity_bucket(row) for _, row in rows.iterrows()
        ]
    if "stop_rule_pattern" not in rows:
        pattern_values: list[str] = []
        for _, row in rows.iterrows():
            pattern, _ = _stop_rule_pattern(
                ambiguity_bucket=str(row["ambiguity_bucket"]),
                decision_divergence_count=(
                    0 if bool(row["traversal_decision_agrees"]) else 1
                ),
                left_pass_through_count=(
                    1 if str(row["left_traversal_state"]) == "pass_through" else 0
                ),
                right_pass_through_count=(
                    1 if str(row["right_traversal_state"]) == "pass_through" else 0
                ),
                left_descendant_accepted_split_count=int(
                    row["left_descendant_accepted_split_count"]
                ),
                right_descendant_accepted_split_count=int(
                    row["right_descendant_accepted_split_count"]
                ),
                right_guard_blocked_count=(
                    1 if bool(row["right_explicit_guard_blocked"]) else 0
                ),
                traversal_only_pair_count=(
                    1
                    if str(row["left_neighborhood_evidence_family"])
                    == "traversal_only"
                    and str(row["right_neighborhood_evidence_family"])
                    == "traversal_only"
                    else 0
                ),
                row_count=1,
            )
            pattern_values.append(pattern)
        rows["stop_rule_pattern"] = pattern_values

    records: list[dict[str, object]] = []
    group_columns = ["stop_rule_pattern", "data_role", "ambiguity_bucket"]
    for keys, group in rows.groupby(group_columns, sort=True):
        row_count = int(group.shape[0])
        decision_divergent = ~group["traversal_decision_agrees"].astype(bool)
        traversal_only_pair = group["left_neighborhood_evidence_family"].astype(str).eq(
            "traversal_only"
        ) & group["right_neighborhood_evidence_family"].astype(str).eq(
            "traversal_only"
        )
        old_and_current_pair = group["left_neighborhood_evidence_family"].astype(
            str
        ).eq("old_and_current") & group["right_neighborhood_evidence_family"].astype(
            str
        ).eq("old_and_current")
        sibling_p_count, sibling_p_median = _finite_median(
            group["left_sibling_p_value"]
        )
        depth_count, depth_median = _finite_median(group["left_depth"])
        leaves_count, leaves_median = _finite_median(
            group["left_n_descendant_leaves"]
        )
        balance_count, balance_median = _finite_median(group["left_balance_product"])
        edge_norm_count, edge_norm_median = _finite_median(
            group["left_outgoing_edge_norm_balance"]
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "stop_rule_pattern": str(keys[0]),
                "data_role": str(keys[1]),
                "ambiguity_bucket": str(keys[2]),
                "row_count": row_count,
                "decision_divergence_count": int(decision_divergent.sum()),
                "left_edge_open_count": int(
                    group["left_child_parent_edge_open"].astype(bool).sum()
                ),
                "left_sibling_open_count": int(
                    group["left_sibling_open"].astype(bool).sum()
                ),
                "left_pass_through_count": _count_state(
                    group,
                    "left_traversal_state",
                    "pass_through",
                ),
                "right_pass_through_count": _count_state(
                    group,
                    "right_traversal_state",
                    "pass_through",
                ),
                "right_guard_blocked_count": int(
                    group["right_explicit_guard_blocked"].astype(bool).sum()
                ),
                "left_descendant_accepted_split_count": int(
                    group["left_descendant_accepted_split_count"].sum()
                ),
                "right_descendant_accepted_split_count": int(
                    group["right_descendant_accepted_split_count"].sum()
                ),
                "traversal_only_pair_count": int(traversal_only_pair.sum()),
                "old_and_current_pair_count": int(old_and_current_pair.sum()),
                "finite_sibling_p_count": sibling_p_count,
                "median_sibling_p_value": sibling_p_median,
                "finite_depth_count": depth_count,
                "median_depth": depth_median,
                "finite_descendant_leaves_count": leaves_count,
                "median_descendant_leaves": leaves_median,
                "finite_balance_product_count": balance_count,
                "median_balance_product": balance_median,
                "finite_outgoing_edge_norm_balance_count": edge_norm_count,
                "median_outgoing_edge_norm_balance": edge_norm_median,
                "retention_evidence_status": _retention_evidence_status(
                    stop_rule_pattern=str(keys[0]),
                    data_role=str(keys[1]),
                    row_count=row_count,
                    traversal_only_pair_count=int(traversal_only_pair.sum()),
                    finite_balance_product_count=balance_count,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=RETENTION_EVIDENCE_COLUMNS)


def _retention_gap_components(row: pd.Series) -> tuple[str, str, str, str, str]:
    status = str(row["retention_evidence_status"])
    observed = (
        f"edge_open={int(row['left_edge_open_count'])}/{int(row['row_count'])};"
        f"sibling_open={int(row['left_sibling_open_count'])}/{int(row['row_count'])};"
        "left_descendant_splits="
        f"{int(row['left_descendant_accepted_split_count'])};"
        "right_descendant_splits="
        f"{int(row['right_descendant_accepted_split_count'])};"
        f"traversal_only_pairs={int(row['traversal_only_pair_count'])};"
        f"finite_balance_product={int(row['finite_balance_product_count'])}"
    )
    if status == "retention_evidence_false_positive_risk":
        return (
            observed,
            "selected_null_rows_share_pass_through_downstream_split_pattern",
            "refined_guard_suppression_remains_required",
            "derive_null-side selected-pass-through false-positive law",
            "retain_fail_closed_refined_guard",
        )
    if status == "retention_evidence_signal_unresolved_no_topology":
        return (
            observed,
            "missing_structural_topology_coverage_at_signal_pass_through_rows",
            "conditional_profile_may_recover_signal_but_is_unvalidated",
            "derive signal-side topology likelihood for retained pass-through walks",
            "fail_closed_until_topology_law_validated",
        )
    if status == "retention_evidence_signal_with_topology_context":
        return (
            observed,
            "topology_context_present_but_not_calibrated",
            "diagnostic_candidate_only",
            "validate topology evidence against selected-null pass-through risk",
            "diagnostic_only_no_promotion",
        )
    return (
        observed,
        "not_a_retention_decision_row",
        "context_only_for_method_comparison",
        "no retention-law update from this row",
        "diagnostic_only_no_promotion",
    )


def summarize_selected_neighborhood_retention_gaps(
    retention_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Translate retention evidence into explicit method-law gaps."""
    if retention_evidence.empty:
        return pd.DataFrame(columns=RETENTION_GAP_COLUMNS)
    records: list[dict[str, object]] = []
    for _, row in retention_evidence.iterrows():
        (
            observed,
            missing,
            implication,
            next_requirement,
            production_action,
        ) = _retention_gap_components(row)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "stop_rule_pattern": str(row["stop_rule_pattern"]),
                "data_role": str(row["data_role"]),
                "ambiguity_bucket": str(row["ambiguity_bucket"]),
                "row_count": int(row["row_count"]),
                "retention_evidence_status": str(row["retention_evidence_status"]),
                "observed_evidence": observed,
                "missing_evidence": missing,
                "method_implication": implication,
                "next_law_requirement": next_requirement,
                "production_action": production_action,
            }
        )
    return pd.DataFrame.from_records(records, columns=RETENTION_GAP_COLUMNS)


def _gap_row_by_status(
    gaps: pd.DataFrame,
    status: str,
) -> pd.Series | None:
    if gaps.empty:
        return None
    matches = gaps.loc[gaps["retention_evidence_status"].astype(str).eq(status)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _gap_summary(row: pd.Series | None) -> tuple[str, str, str, str]:
    if row is None:
        return (
            "evidence_row_missing",
            "evidence_row_missing",
            "derive missing diagnostic evidence before method comparison",
            "fail_closed_until_evidence_present",
        )
    return (
        str(row["observed_evidence"]),
        str(row["missing_evidence"]),
        str(row["next_law_requirement"]),
        str(row["production_action"]),
    )


def summarize_selected_neighborhood_method_contract(
    retention_gaps: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize what is shared and what differs between the two profiles."""
    null_row = _gap_row_by_status(
        retention_gaps,
        "retention_evidence_false_positive_risk",
    )
    signal_row = _gap_row_by_status(
        retention_gaps,
        "retention_evidence_signal_unresolved_no_topology",
    )
    (
        null_evidence,
        null_missing,
        null_next,
        null_action,
    ) = _gap_summary(null_row)
    (
        signal_evidence,
        signal_missing,
        signal_next,
        signal_action,
    ) = _gap_summary(signal_row)
    records = [
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "contract_component": "base_traversal_skeleton",
            "component_status": "shared_binary_edge_sibling_passthrough",
            "shared_by_methods": True,
            "conditional_profile_behavior": (
                "binary plus edge plus sibling gates with pass-through"
            ),
            "refined_profile_behavior": (
                "binary plus edge plus sibling gates with pass-through"
            ),
            "evidence_source": "gate_evaluator_contract",
            "evidence_summary": (
                "Both profiles use the same traversal action skeleton; "
                "candidate differences arise after gate annotation."
            ),
            "missing_evidence": "none_for_base_skeleton",
            "next_law_requirement": (
                "condition selected pass-through retention, not base traversal"
            ),
            "production_action": "diagnostic_only_no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "contract_component": "selected_null_pass_through_control",
            "component_status": "refined_guard_required",
            "shared_by_methods": False,
            "conditional_profile_behavior": (
                "retains pass-through walks that can reach downstream splits"
            ),
            "refined_profile_behavior": (
                "suppresses selected-null pass-through/downstream split pattern"
            ),
            "evidence_source": "retention_gap_summary:false_positive_risk",
            "evidence_summary": null_evidence,
            "missing_evidence": null_missing,
            "next_law_requirement": null_next,
            "production_action": null_action,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "contract_component": "signal_pass_through_retention",
            "component_status": "conditional_recovery_unvalidated",
            "shared_by_methods": False,
            "conditional_profile_behavior": (
                "retains deep signal pass-through walks to downstream splits"
            ),
            "refined_profile_behavior": (
                "blocks the same walk until topology evidence is available"
            ),
            "evidence_source": "retention_gap_summary:signal_unresolved_no_topology",
            "evidence_summary": signal_evidence,
            "missing_evidence": signal_missing,
            "next_law_requirement": signal_next,
            "production_action": signal_action,
        },
    ]
    return pd.DataFrame.from_records(records, columns=METHOD_CONTRACT_COLUMNS)


def _profile_field_value(profile_id: str, field_name: str) -> object:
    profile = SIBLING_GATE_PROFILES.get(str(profile_id))
    if profile is None:
        return ""
    return getattr(profile, field_name)


def _profile_contract_status(field_name: str, values_match: bool) -> tuple[str, str]:
    if values_match:
        if field_name in {
            "sibling_gate_method",
            "sibling_gate_alpha_penalty",
        }:
            return (
                "shared_fixed_coordinate_sibling_gate",
                "Both profiles use the same fixed-coordinate sibling gate layer.",
            )
        if field_name.startswith("root_stability"):
            return (
                "shared_selected_root_stability_guard",
                "Both profiles use the same selected-root stability guard settings.",
            )
        return (
            "shared_profile_value",
            "The profile field does not distinguish the two methods.",
        )
    if field_name.startswith("root_selective_permutation_guard"):
        return (
            "refined_selected_family_guard_difference",
            (
                "The refined profile adds selected-family pass-through guard "
                "behavior that the conditional diagnostic profile does not apply."
            ),
        )
    if field_name == "status":
        return (
            "diagnostic_status_difference",
            "The profiles carry different diagnostic maturity labels.",
        )
    return (
        "profile_difference_requires_review",
        "This profile field differs and should be reviewed before comparison.",
    )


def _method_ids_from_candidate_rows(candidate_rows: pd.DataFrame) -> tuple[str, str]:
    if candidate_rows.empty:
        return (
            "fixed_coordinate_conditional_topology_diagnostic_v1",
            "fixed_coordinate_global_passthrough_refined_v1",
        )
    return (
        str(candidate_rows["left_method_id"].iloc[0]),
        str(candidate_rows["right_method_id"].iloc[0]),
    )


def summarize_selected_neighborhood_profile_config_contract(
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Compare concrete sibling-gate profile settings for the paired methods."""
    left_method, right_method = _method_ids_from_candidate_rows(candidate_rows)
    profile_fields = (
        "sibling_gate_method",
        "sibling_gate_alpha_penalty",
        "root_stability_guard_threshold",
        "root_stability_subsample_replicates",
        "root_stability_feature_fraction",
        "root_stability_seed",
        "status",
        "root_selective_permutation_guard_replicates",
        "root_selective_permutation_guard_alpha",
        "root_selective_permutation_guard_scope",
    )
    records: list[dict[str, object]] = []
    for field_name in profile_fields:
        left_value = _profile_field_value(left_method, field_name)
        right_value = _profile_field_value(right_method, field_name)
        values_match = left_value == right_value
        status, interpretation = _profile_contract_status(
            field_name,
            bool(values_match),
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "left_method_id": left_method,
                "right_method_id": right_method,
                "profile_field": field_name,
                "left_value": "" if left_value is None else str(left_value),
                "right_value": "" if right_value is None else str(right_value),
                "values_match": bool(values_match),
                "profile_contract_status": status,
                "method_interpretation": interpretation,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=PROFILE_CONFIG_CONTRACT_COLUMNS,
    )


def _method_ids_from_profile_contract(
    profile_config_contract: pd.DataFrame,
) -> tuple[str, str]:
    if profile_config_contract.empty:
        return (
            "fixed_coordinate_conditional_topology_diagnostic_v1",
            "fixed_coordinate_global_passthrough_refined_v1",
        )
    return (
        str(profile_config_contract["left_method_id"].iloc[0]),
        str(profile_config_contract["right_method_id"].iloc[0]),
    )


def _contract_row(
    method_contract: pd.DataFrame,
    component: str,
) -> pd.Series | None:
    if method_contract.empty:
        return None
    matches = method_contract.loc[
        method_contract["contract_component"].astype(str).eq(component)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _contract_value(
    row: pd.Series | None,
    column: str,
    default: str,
) -> str:
    if row is None:
        return default
    return str(row[column])


def summarize_selected_neighborhood_method_readiness(
    *,
    method_contract: pd.DataFrame,
    profile_config_contract: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize what each compared method is currently ready to support."""
    conditional_method, refined_method = _method_ids_from_profile_contract(
        profile_config_contract
    )
    base = _contract_row(method_contract, "base_traversal_skeleton")
    selected_null = _contract_row(
        method_contract,
        "selected_null_pass_through_control",
    )
    signal = _contract_row(method_contract, "signal_pass_through_retention")
    records = [
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "method_id": conditional_method,
            "readiness_scope": "base_traversal_skeleton",
            "readiness_status": "diagnostic_ready_shared_skeleton",
            "supported_behavior": _contract_value(
                base,
                "conditional_profile_behavior",
                "binary plus edge plus sibling gates with pass-through",
            ),
            "unsupported_behavior": "production promotion from traversal skeleton alone",
            "evidence_source": "method_contract_summary:base_traversal_skeleton",
            "blocking_evidence": _contract_value(
                base,
                "missing_evidence",
                "none_for_base_skeleton",
            ),
            "required_next_law": _contract_value(
                base,
                "next_law_requirement",
                "condition selected pass-through retention",
            ),
            "production_action": "diagnostic_only_no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "method_id": refined_method,
            "readiness_scope": "base_traversal_skeleton",
            "readiness_status": "diagnostic_ready_shared_skeleton",
            "supported_behavior": _contract_value(
                base,
                "refined_profile_behavior",
                "binary plus edge plus sibling gates with pass-through",
            ),
            "unsupported_behavior": "production promotion from traversal skeleton alone",
            "evidence_source": "method_contract_summary:base_traversal_skeleton",
            "blocking_evidence": _contract_value(
                base,
                "missing_evidence",
                "none_for_base_skeleton",
            ),
            "required_next_law": _contract_value(
                base,
                "next_law_requirement",
                "condition selected pass-through retention",
            ),
            "production_action": "diagnostic_only_no_promotion",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "method_id": refined_method,
            "readiness_scope": "selected_null_pass_through_control",
            "readiness_status": "ready_as_fail_closed_guard_candidate",
            "supported_behavior": _contract_value(
                selected_null,
                "refined_profile_behavior",
                "suppresses selected-null pass-through/downstream split pattern",
            ),
            "unsupported_behavior": (
                "relaxing selected-family pass-through guard without null law"
            ),
            "evidence_source": "method_contract_summary:selected_null_control",
            "blocking_evidence": _contract_value(
                selected_null,
                "missing_evidence",
                "selected-null false-positive law missing",
            ),
            "required_next_law": _contract_value(
                selected_null,
                "next_law_requirement",
                "derive null-side selected-pass-through false-positive law",
            ),
            "production_action": _contract_value(
                selected_null,
                "production_action",
                "retain_fail_closed_refined_guard",
            ),
        },
        {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "method_id": conditional_method,
            "readiness_scope": "signal_pass_through_retention",
            "readiness_status": "not_ready_missing_topology_likelihood",
            "supported_behavior": _contract_value(
                signal,
                "conditional_profile_behavior",
                "retains deep signal pass-through walks to downstream splits",
            ),
            "unsupported_behavior": "production recovery of deep pass-through signal",
            "evidence_source": "method_contract_summary:signal_retention",
            "blocking_evidence": _contract_value(
                signal,
                "missing_evidence",
                "missing structural topology coverage",
            ),
            "required_next_law": _contract_value(
                signal,
                "next_law_requirement",
                "derive signal-side topology likelihood",
            ),
            "production_action": _contract_value(
                signal,
                "production_action",
                "fail_closed_until_topology_law_validated",
            ),
        },
    ]
    return pd.DataFrame.from_records(records, columns=METHOD_READINESS_COLUMNS)


def run_selected_neighborhood_distribution_panel(
    config: SelectedNeighborhoodDistributionPanelConfig,
) -> dict[str, Path]:
    """Run selected-neighborhood distribution diagnostics and write outputs."""
    node_decisions = pd.read_csv(config.node_decisions_path, keep_default_na=False)
    topology_rows = (
        pd.read_csv(config.topology_rows_path, keep_default_na=False)
        if config.topology_rows_path is not None
        else None
    )
    conditional_law_rows = (
        pd.read_csv(config.conditional_law_rows_path, keep_default_na=False)
        if config.conditional_law_rows_path is not None
        else None
    )
    rows = build_selected_neighborhood_distribution_rows(
        node_decisions=node_decisions,
        topology_rows=topology_rows,
        conditional_law_rows=conditional_law_rows,
    )
    summary = summarize_selected_neighborhood_distributions(rows)
    joint = summarize_selected_neighborhood_joint_distributions(rows)
    run_summary = summarize_selected_neighborhood_run(rows)
    coverage = summarize_selected_neighborhood_coverage(rows)
    case_coverage = summarize_selected_neighborhood_case_coverage(rows)
    method_contrast = summarize_selected_neighborhood_method_contrast(rows)
    candidate_method_contrast = (
        summarize_selected_neighborhood_candidate_method_contrast(rows)
    )
    candidate_method_contrast_rows = (
        build_selected_neighborhood_candidate_method_contrast_rows(rows)
    )
    candidate_ambiguity = summarize_selected_neighborhood_candidate_ambiguity(
        candidate_method_contrast_rows
    )
    candidate_local_features = (
        summarize_selected_neighborhood_candidate_local_features(
            candidate_method_contrast_rows
        )
    )
    candidate_law_targets = summarize_selected_neighborhood_candidate_law_targets(
        candidate_ambiguity,
        candidate_local_features,
    )
    stop_rule_comparison = summarize_selected_neighborhood_stop_rule_comparison(
        candidate_method_contrast_rows
    )
    retention_evidence = summarize_selected_neighborhood_retention_evidence(
        candidate_method_contrast_rows
    )
    retention_gaps = summarize_selected_neighborhood_retention_gaps(
        retention_evidence
    )
    method_contract = summarize_selected_neighborhood_method_contract(
        retention_gaps
    )
    profile_config_contract = (
        summarize_selected_neighborhood_profile_config_contract(
            candidate_method_contrast_rows
        )
    )
    method_readiness = summarize_selected_neighborhood_method_readiness(
        method_contract=method_contract,
        profile_config_contract=profile_config_contract,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    joint.to_csv(config.joint_summary_path, index=False)
    run_summary.to_csv(config.run_summary_path, index=False)
    coverage.to_csv(config.coverage_summary_path, index=False)
    case_coverage.to_csv(config.case_coverage_summary_path, index=False)
    method_contrast.to_csv(config.method_contrast_summary_path, index=False)
    candidate_method_contrast.to_csv(
        config.candidate_method_contrast_summary_path,
        index=False,
    )
    candidate_method_contrast_rows.to_csv(
        config.candidate_method_contrast_rows_path,
        index=False,
    )
    candidate_ambiguity.to_csv(config.candidate_ambiguity_summary_path, index=False)
    candidate_local_features.to_csv(
        config.candidate_local_feature_summary_path,
        index=False,
    )
    candidate_law_targets.to_csv(
        config.candidate_law_target_summary_path,
        index=False,
    )
    stop_rule_comparison.to_csv(
        config.stop_rule_comparison_summary_path,
        index=False,
    )
    retention_evidence.to_csv(
        config.retention_evidence_summary_path,
        index=False,
    )
    retention_gaps.to_csv(
        config.retention_gap_summary_path,
        index=False,
    )
    method_contract.to_csv(
        config.method_contract_summary_path,
        index=False,
    )
    profile_config_contract.to_csv(
        config.profile_config_contract_summary_path,
        index=False,
    )
    method_readiness.to_csv(
        config.method_readiness_summary_path,
        index=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "inputs": {
            "node_decisions": str(config.node_decisions_path),
            "topology_rows": (
                None
                if config.topology_rows_path is None
                else str(config.topology_rows_path)
            ),
            "conditional_law_rows": (
                None
                if config.conditional_law_rows_path is None
                else str(config.conditional_law_rows_path)
            ),
        },
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "joint_summary": str(config.joint_summary_path),
            "run_summary": str(config.run_summary_path),
            "coverage_summary": str(config.coverage_summary_path),
            "case_coverage_summary": str(config.case_coverage_summary_path),
            "method_contrast_summary": str(config.method_contrast_summary_path),
            "candidate_method_contrast_summary": str(
                config.candidate_method_contrast_summary_path
            ),
            "candidate_method_contrast_rows": str(
                config.candidate_method_contrast_rows_path
            ),
            "candidate_ambiguity_summary": str(
                config.candidate_ambiguity_summary_path
            ),
            "candidate_local_feature_summary": str(
                config.candidate_local_feature_summary_path
            ),
            "candidate_law_target_summary": str(
                config.candidate_law_target_summary_path
            ),
            "stop_rule_comparison_summary": str(
                config.stop_rule_comparison_summary_path
            ),
            "retention_evidence_summary": str(
                config.retention_evidence_summary_path
            ),
            "retention_gap_summary": str(config.retention_gap_summary_path),
            "method_contract_summary": str(config.method_contract_summary_path),
            "profile_config_contract_summary": str(
                config.profile_config_contract_summary_path
            ),
            "method_readiness_summary": str(config.method_readiness_summary_path),
        },
        "production_status": str(run_summary["production_status"].iloc[0]),
        "interpretation": (
            "Diagnostic-only distribution panel. Neighborhood variables are "
            "summarized at split, boundary, and pass-through traversal states; "
            "no recovery split is promoted."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "joint_summary": config.joint_summary_path,
        "run_summary": config.run_summary_path,
        "coverage_summary": config.coverage_summary_path,
        "case_coverage_summary": config.case_coverage_summary_path,
        "method_contrast_summary": config.method_contrast_summary_path,
        "candidate_method_contrast_summary": (
            config.candidate_method_contrast_summary_path
        ),
        "candidate_method_contrast_rows": config.candidate_method_contrast_rows_path,
        "candidate_ambiguity_summary": config.candidate_ambiguity_summary_path,
        "candidate_local_feature_summary": (
            config.candidate_local_feature_summary_path
        ),
        "candidate_law_target_summary": config.candidate_law_target_summary_path,
        "stop_rule_comparison_summary": config.stop_rule_comparison_summary_path,
        "retention_evidence_summary": config.retention_evidence_summary_path,
        "retention_gap_summary": config.retention_gap_summary_path,
        "method_contract_summary": config.method_contract_summary_path,
        "profile_config_contract_summary": (
            config.profile_config_contract_summary_path
        ),
        "method_readiness_summary": config.method_readiness_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-decisions-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topology-rows-path", type=Path)
    parser.add_argument("--conditional-law-rows-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_selected_neighborhood_distribution_panel(
        SelectedNeighborhoodDistributionPanelConfig(
            node_decisions_path=args.node_decisions_path,
            output_dir=args.output_dir,
            topology_rows_path=args.topology_rows_path,
            conditional_law_rows_path=args.conditional_law_rows_path,
        )
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
