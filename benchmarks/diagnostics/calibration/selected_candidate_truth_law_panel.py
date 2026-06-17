"""Selected-candidate truth-law diagnostics.

This panel is an oracle benchmark diagnostic, not a calibration rule. It asks
whether selected traversal candidates are true branch transitions in their own
immediate child split, or whether they are retained pass-through fragments.
The goal is to validate the next selected-neighborhood law before any
production promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.diagnostics.calibration.selected_pass_through_branch_recovery_conditioning import (
    _feature_geometry_for_partition,
    build_generated_branch_positive_support_fixture,
)
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
)

STUDY_ROLE = "diagnostic_selected_candidate_truth_law_not_calibration"
SCHEMA_VERSION = "selected_candidate_truth_law_panel/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.selected_candidate_truth_law_panel"
)

DEFAULT_BRANCH_ARI_FLOOR = 0.50
DEFAULT_PARTIAL_ARI_FLOOR = 0.25
DEFAULT_CHILD_PURITY_FLOOR = 0.65

TruthLabelMaps = Mapping[tuple[str, str, str, int], tuple[int, Mapping[str, int]]]
FeatureDataMaps = Mapping[tuple[str, str, str, int], tuple[int, pd.DataFrame]]

CANDIDATE_REQUIRED_COLUMNS = {
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
    "left_sibling_open",
    "right_sibling_open",
    "right_explicit_guard_blocked",
}

ASSIGNMENT_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "sample_id",
    "path_node_ids",
}

TRAVERSAL_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "data_seed",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "own_split_method_id",
    "candidate_reason",
    "left_traversal_state",
    "right_traversal_state",
    "left_traversal_decision",
    "right_traversal_decision",
    "left_sibling_open",
    "right_sibling_open",
    "right_explicit_guard_blocked",
    "candidate_context_status",
    "left_depth",
    "right_depth",
    "left_n_descendant_leaves",
    "right_n_descendant_leaves",
    "left_sibling_p_value",
    "right_sibling_p_value",
    "min_sibling_p_value",
    "max_sibling_p_value",
    "negative_log10_min_sibling_p_value",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "descendant_accepted_split_delta",
    "left_descendant_pass_through_count",
    "right_descendant_pass_through_count",
    "descendant_pass_through_delta",
    "left_descendant_stable_boundary_count",
    "right_descendant_stable_boundary_count",
    "own_split_status",
    "own_split_child_count",
    "own_split_parent_sample_count",
    "own_split_left_child_id",
    "own_split_right_child_id",
    "own_split_left_sample_count",
    "own_split_right_sample_count",
    "own_split_node_cluster_count",
    "own_split_ari",
    "own_split_child_mean_purity",
    "own_split_child_majority_distinct",
    "own_split_left_majority_label",
    "own_split_right_majority_label",
    "own_split_truth_class",
    "candidate_truth_diagnostic_status",
    "feature_geometry_status",
    "feature_geometry_node_sample_count",
    "feature_geometry_left_child_count",
    "feature_geometry_right_child_count",
    "feature_node_pairwise_jaccard",
    "feature_left_pairwise_jaccard",
    "feature_right_pairwise_jaccard",
    "feature_homogeneity_gain_min",
    "feature_heterogeneity_gain_max",
    "feature_subspace_consensus_jaccard_topk",
    "feature_heterogeneity_subspace_consensus_jaccard_topk",
    "feature_child_contrast_norm",
    "feature_branch_geometry_score",
    "feature_barycentric_geometry_score",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "signal_row_count",
    "selected_null_control_count",
    "branch_recovery_count",
    "partial_branch_recovery_count",
    "false_fragment_count",
    "unresolved_count",
    "unavailable_count",
    "pass_through_branch_recovery_count",
    "split_branch_recovery_count",
    "max_pass_through_ari",
    "max_split_ari",
    "diagnostic_status",
    "next_required_step",
    "production_action",
)

FEATURE_METRIC_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "branch_count",
    "negative_count",
    "finite_branch_count",
    "finite_negative_count",
    "best_direction",
    "best_auc",
    "branch_min",
    "branch_median",
    "branch_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_status",
)

FEATURE_METRIC_STATE_COLUMNS = (
    "schema_version",
    "study_role",
    "state_scope",
    "scope_row_count",
    "metric",
    "branch_count",
    "negative_count",
    "finite_branch_count",
    "finite_negative_count",
    "best_direction",
    "best_auc",
    "branch_min",
    "branch_median",
    "branch_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_status",
)

CONTEXT_METRIC_STATE_COLUMNS = (
    "schema_version",
    "study_role",
    "state_scope",
    "scope_row_count",
    "metric",
    "branch_count",
    "negative_count",
    "finite_branch_count",
    "finite_negative_count",
    "best_direction",
    "best_auc",
    "branch_min",
    "branch_median",
    "branch_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_status",
)

FAMILY_LIKELIHOOD_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "ambiguity_bucket",
    "stop_rule_pattern",
    "family_row_count",
    "branch_recovery_count",
    "partial_branch_recovery_count",
    "false_fragment_count",
    "selected_null_control_count",
    "unresolved_count",
    "unavailable_count",
    "pass_through_candidate_count",
    "split_candidate_count",
    "descendant_accepted_split_total",
    "median_min_sibling_p_value",
    "min_min_sibling_p_value",
    "max_min_sibling_p_value",
    "median_feature_branch_geometry_score",
    "max_feature_branch_geometry_score",
    "matched_selected_null_family_count",
    "matched_selected_null_candidate_count",
    "matched_selected_null_control_count",
    "matched_selected_null_collision",
    "family_truth_signal",
    "family_likelihood_status",
    "next_required_step",
    "production_action",
)

FAMILY_LIKELIHOOD_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "ambiguity_bucket",
    "stop_rule_pattern",
    "family_likelihood_status",
    "family_count",
    "candidate_count",
    "branch_recovery_count",
    "partial_branch_recovery_count",
    "false_fragment_count",
    "selected_null_control_count",
    "unresolved_count",
    "pass_through_candidate_count",
    "split_candidate_count",
    "matched_selected_null_collision_family_count",
    "clean_branch_family_count",
    "colliding_branch_family_count",
    "diagnostic_status",
    "next_required_step",
    "production_action",
)

COLLISION_LAW_COMPONENT_COLUMNS = (
    "schema_version",
    "study_role",
    "component_id",
    "support_scope",
    "family_count",
    "candidate_count",
    "branch_recovery_count",
    "partial_branch_recovery_count",
    "false_fragment_count",
    "selected_null_control_count",
    "unresolved_count",
    "clean_branch_family_count",
    "colliding_branch_family_count",
    "matched_selected_null_collision_family_count",
    "fragment_mixed_branch_family_count",
    "pass_through_branch_family_count",
    "selected_null_suppression_family_count",
    "component_status",
    "next_required_step",
    "production_action",
)

ACCEPTED_SPLIT_FILTER_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "ambiguity_bucket",
    "stop_rule_pattern",
    "family_likelihood_status",
    "accepted_split_filter_role",
    "family_row_count",
    "branch_recovery_count",
    "partial_branch_recovery_count",
    "false_fragment_count",
    "unresolved_count",
    "max_feature_homogeneity_gain_min",
    "median_feature_homogeneity_gain_min",
    "min_feature_homogeneity_gain_min",
    "max_feature_branch_geometry_score",
    "median_feature_branch_geometry_score",
    "min_feature_branch_geometry_score",
    "max_feature_child_contrast_norm",
    "median_feature_child_contrast_norm",
    "min_feature_child_contrast_norm",
    "max_feature_barycentric_geometry_score",
    "median_feature_barycentric_geometry_score",
    "min_min_sibling_p_value",
    "median_min_sibling_p_value",
    "max_min_sibling_p_value",
    "median_negative_log10_min_sibling_p_value",
)

ACCEPTED_SPLIT_FILTER_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "positive_family_count",
    "negative_family_count",
    "finite_positive_family_count",
    "finite_negative_family_count",
    "best_direction",
    "best_auc",
    "positive_min",
    "positive_median",
    "positive_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_status",
    "diagnostic_status",
    "production_action",
)

ACCEPTED_SPLIT_PAIR_FILTER_COLUMNS = (
    "schema_version",
    "study_role",
    "metric_left",
    "metric_right",
    "direction_left",
    "direction_right",
    "positive_family_count",
    "negative_family_count",
    "finite_positive_pair_count",
    "finite_negative_pair_count",
    "zero_negative_positive_pass_count",
    "zero_negative_positive_recall",
    "threshold_left",
    "threshold_right",
    "zero_negative_status",
    "diagnostic_status",
    "production_action",
)

ACCEPTED_SPLIT_FRONTIER_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "metric_spec",
    "metric_count",
    "positive_family_count",
    "negative_family_count",
    "finite_positive_family_count",
    "finite_negative_family_count",
    "negative_dominated_positive_count",
    "negative_dominated_positive_fraction",
    "frontier_positive_count",
    "frontier_positive_fraction",
    "diagnostic_status",
    "next_required_step",
    "production_action",
)

SELECTED_FAMILY_FRONTIER_LAW_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "case_id",
    "data_role",
    "replicate",
    "accepted_split_filter_role",
    "family_likelihood_status",
    "frontier_margin_to_negative",
    "frontier_margin_scale",
    "frontier_non_dominated",
    "support_status",
    "support_prior_log_odds",
    "posterior_style_log_odds",
    "posterior_style_probability",
    "selected_family_frontier_law_status",
    "production_action",
)

SELECTED_FAMILY_FRONTIER_LAW_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "metric_spec",
    "metric_count",
    "positive_family_count",
    "negative_family_count",
    "finite_positive_family_count",
    "finite_negative_family_count",
    "clean_non_dominated_count",
    "clean_dominated_count",
    "clean_non_dominated_fraction",
    "clean_margin_min",
    "clean_margin_median",
    "clean_margin_max",
    "beta_prior_alpha",
    "beta_prior_beta",
    "beta_posterior_alpha",
    "beta_posterior_beta",
    "posterior_clean_frontier_mean",
    "posterior_clean_frontier_lower90",
    "diagnostic_status",
    "next_required_step",
    "production_action",
)

SELECTED_FAMILY_FRONTIER_ABLATION_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "ablation_id",
    "removed_metric",
    "metric_spec",
    "metric_count",
    "finite_positive_family_count",
    "finite_negative_family_count",
    "clean_non_dominated_count",
    "clean_dominated_count",
    "clean_non_dominated_fraction",
    "clean_margin_min",
    "clean_margin_p10",
    "clean_margin_median",
    "clean_margin_max",
    "fragility_status",
    "next_required_step",
    "production_action",
)

FRONTIER_WITNESS_GAP_COLUMNS = (
    "gap_max_feature_homogeneity_gain_min",
    "gap_median_feature_homogeneity_gain_min",
    "gap_max_feature_branch_geometry_score",
    "gap_median_feature_branch_geometry_score",
    "gap_max_feature_child_contrast_norm",
    "gap_max_feature_barycentric_geometry_score",
    "gap_median_min_sibling_p_value",
    "gap_median_negative_log10_min_sibling_p_value",
)

SELECTED_FAMILY_FRONTIER_WITNESS_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "metric_spec",
    "metric_count",
    "case_id",
    "data_role",
    "replicate",
    "accepted_split_filter_role",
    "witness_case_id",
    "witness_data_role",
    "witness_replicate",
    "witness_accepted_split_filter_role",
    "frontier_margin_to_witness",
    "frontier_margin_status",
    "active_metric",
    "active_metric_direction",
    "active_metric_gap",
    "active_metric_family_value",
    "active_metric_witness_value",
    *FRONTIER_WITNESS_GAP_COLUMNS,
)

SIBLING_RESCUE_AUDIT_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "case_id",
    "replicate",
    "witness_case_id",
    "witness_replicate",
    "frontier_margin_status",
    "active_metric",
    "frontier_margin_to_witness",
    "sibling_gap",
    "log_sibling_gap",
    "structural_gap_max",
    "structural_positive_gap_count",
    "structural_negative_gap_count",
    "structural_tie_gap_count",
    "sibling_rescue_status",
    "production_action",
)

SIBLING_RESCUE_AUDIT_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "row_count",
    "sibling_active_count",
    "sibling_margin_thin_count",
    "sibling_only_no_structural_support_count",
    "sibling_only_thin_no_structural_support_count",
    "structural_active_count",
    "structural_support_available_count",
    "summary_status",
    "next_required_step",
    "production_action",
)

SIBLING_RESCUE_GUARD_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "frontier_id",
    "row_count",
    "retained_after_guard_count",
    "blocked_by_guard_count",
    "retained_after_guard_fraction",
    "blocked_sibling_only_count",
    "blocked_sibling_only_thin_count",
    "blocked_sibling_dominated_or_tied_count",
    "guard_status",
    "next_required_step",
    "production_action",
)

ACCEPTED_SPLIT_FRONTIER_SPECS: tuple[
    tuple[str, tuple[tuple[str, str], ...]],
    ...,
] = (
    (
        "feature_strength_high",
        (
            ("max_feature_homogeneity_gain_min", "high"),
            ("max_feature_branch_geometry_score", "high"),
            ("max_feature_child_contrast_norm", "high"),
        ),
    ),
    (
        "feature_strength_with_fragment_penalty",
        (
            ("median_feature_homogeneity_gain_min", "high"),
            ("median_feature_branch_geometry_score", "high"),
            ("max_feature_barycentric_geometry_score", "low"),
        ),
    ),
    (
        "binary_transfer_best_pair_context",
        (
            ("max_feature_child_contrast_norm", "low"),
            ("max_feature_barycentric_geometry_score", "low"),
            ("median_min_sibling_p_value", "low"),
        ),
    ),
    (
        "full_family_context_frontier",
        (
            ("median_feature_homogeneity_gain_min", "high"),
            ("median_feature_branch_geometry_score", "high"),
            ("max_feature_child_contrast_norm", "low"),
            ("max_feature_barycentric_geometry_score", "low"),
            ("median_min_sibling_p_value", "low"),
        ),
    ),
    (
        "full_family_log_sibling_context_frontier",
        (
            ("median_feature_homogeneity_gain_min", "high"),
            ("median_feature_branch_geometry_score", "high"),
            ("max_feature_child_contrast_norm", "low"),
            ("max_feature_barycentric_geometry_score", "low"),
            ("median_negative_log10_min_sibling_p_value", "high"),
        ),
    ),
)

STATE_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "left_traversal_state",
    "right_traversal_state",
    "own_split_truth_class",
    "row_count",
    "finite_ari_count",
    "median_ari",
    "max_ari",
)


@dataclass(frozen=True)
class SelectedCandidateTruthLawPanelConfig:
    """Runtime contract for selected-candidate truth diagnostics."""

    output_dir: Path
    candidate_rows_path: Path | None = None
    gene_assignments_path: Path | None = None
    traversal_rows_path: Path | None = None
    use_generated_support_fixture: bool = False
    truth_suite: str = "binary"
    branch_ari_floor: float = DEFAULT_BRANCH_ARI_FLOOR
    partial_ari_floor: float = DEFAULT_PARTIAL_ARI_FLOOR
    child_purity_floor: float = DEFAULT_CHILD_PURITY_FLOOR
    top_k: int = 12
    max_pairwise_samples: int = 200
    min_frontier_law_support: int = 3
    min_frontier_law_margin: float = 1e-8

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "selected_candidate_truth_law_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_truth_law_summary.csv"

    @property
    def state_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_truth_law_state_summary.csv"

    @property
    def feature_metric_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_feature_metric_summary.csv"

    @property
    def feature_metric_state_summary_path(self) -> Path:
        return (
            self.output_dir / "selected_candidate_feature_metric_state_summary.csv"
        )

    @property
    def context_metric_state_summary_path(self) -> Path:
        return (
            self.output_dir / "selected_candidate_context_metric_state_summary.csv"
        )

    @property
    def family_likelihood_rows_path(self) -> Path:
        return self.output_dir / "selected_candidate_family_likelihood_rows.csv"

    @property
    def family_likelihood_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_family_likelihood_summary.csv"

    @property
    def collision_law_components_path(self) -> Path:
        return self.output_dir / "selected_candidate_collision_law_components.csv"

    @property
    def accepted_split_filter_rows_path(self) -> Path:
        return self.output_dir / "selected_candidate_accepted_split_filter_rows.csv"

    @property
    def accepted_split_filter_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_accepted_split_filter_summary.csv"

    @property
    def accepted_split_pair_filter_summary_path(self) -> Path:
        return (
            self.output_dir
            / "selected_candidate_accepted_split_pair_filter_summary.csv"
        )

    @property
    def accepted_split_frontier_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_accepted_split_frontier_summary.csv"

    @property
    def selected_family_frontier_law_rows_path(self) -> Path:
        return self.output_dir / "selected_candidate_frontier_law_rows.csv"

    @property
    def selected_family_frontier_law_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_frontier_law_summary.csv"

    @property
    def selected_family_frontier_ablation_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_frontier_ablation_summary.csv"

    @property
    def selected_family_frontier_witness_rows_path(self) -> Path:
        return self.output_dir / "selected_candidate_frontier_witness_rows.csv"

    @property
    def sibling_rescue_audit_rows_path(self) -> Path:
        return self.output_dir / "selected_candidate_sibling_rescue_audit_rows.csv"

    @property
    def sibling_rescue_audit_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_sibling_rescue_audit_summary.csv"

    @property
    def sibling_rescue_guard_summary_path(self) -> Path:
        return self.output_dir / "selected_candidate_sibling_rescue_guard_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def generated_support_candidate_rows_path(self) -> Path:
        return self.output_dir / "generated_support_candidate_rows.csv"

    @property
    def generated_support_gene_assignments_path(self) -> Path:
        return self.output_dir / "generated_support_gene_assignments.csv"


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _negative_log10(value: object) -> float:
    number = _finite_float(value)
    if not math.isfinite(number):
        return math.nan
    return float(-math.log10(max(number, np.finfo(float).tiny)))


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


def _optional_finite(row: pd.Series, column: str) -> float:
    return _finite_float(row[column]) if column in row.index else math.nan


def _optional_int(row: pd.Series, column: str) -> int:
    value = _optional_finite(row, column)
    return int(value) if math.isfinite(value) else 0


def _candidate_context_metrics(row: pd.Series) -> dict[str, object]:
    optional_columns = {
        "left_depth",
        "right_depth",
        "left_n_descendant_leaves",
        "right_n_descendant_leaves",
        "left_sibling_p_value",
        "right_sibling_p_value",
        "left_descendant_accepted_split_count",
        "right_descendant_accepted_split_count",
        "left_descendant_pass_through_count",
        "right_descendant_pass_through_count",
        "left_descendant_stable_boundary_count",
        "right_descendant_stable_boundary_count",
    }
    available = optional_columns & set(row.index)
    left_p = _optional_finite(row, "left_sibling_p_value")
    right_p = _optional_finite(row, "right_sibling_p_value")
    finite_p = [value for value in [left_p, right_p] if math.isfinite(value)]
    min_p = min(finite_p) if finite_p else math.nan
    max_p = max(finite_p) if finite_p else math.nan
    clipped_min_p = max(min_p, np.finfo(float).tiny) if math.isfinite(min_p) else math.nan
    left_split = _optional_int(row, "left_descendant_accepted_split_count")
    right_split = _optional_int(row, "right_descendant_accepted_split_count")
    left_pass = _optional_int(row, "left_descendant_pass_through_count")
    right_pass = _optional_int(row, "right_descendant_pass_through_count")
    return {
        "candidate_context_status": (
            "candidate_context_observed"
            if available
            else "candidate_context_missing_optional_fields"
        ),
        "left_depth": _optional_finite(row, "left_depth"),
        "right_depth": _optional_finite(row, "right_depth"),
        "left_n_descendant_leaves": _optional_finite(
            row,
            "left_n_descendant_leaves",
        ),
        "right_n_descendant_leaves": _optional_finite(
            row,
            "right_n_descendant_leaves",
        ),
        "left_sibling_p_value": left_p,
        "right_sibling_p_value": right_p,
        "min_sibling_p_value": min_p,
        "max_sibling_p_value": max_p,
        "negative_log10_min_sibling_p_value": (
            -math.log10(clipped_min_p) if math.isfinite(clipped_min_p) else math.nan
        ),
        "left_descendant_accepted_split_count": left_split,
        "right_descendant_accepted_split_count": right_split,
        "descendant_accepted_split_delta": int(left_split - right_split),
        "left_descendant_pass_through_count": left_pass,
        "right_descendant_pass_through_count": right_pass,
        "descendant_pass_through_delta": int(left_pass - right_pass),
        "left_descendant_stable_boundary_count": _optional_int(
            row,
            "left_descendant_stable_boundary_count",
        ),
        "right_descendant_stable_boundary_count": _optional_int(
            row,
            "right_descendant_stable_boundary_count",
        ),
    }


def _input_data_role(data_role: object) -> str:
    return "null" if str(data_role) == "selected_null" else str(data_role)


def _output_data_role(data_role: object) -> str:
    return "selected_null" if str(data_role) == "null" else str(data_role)


def _split_path(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _next_path_node(path: Sequence[str], node_id: str) -> str:
    try:
        index = list(path).index(node_id)
    except ValueError:
        return ""
    next_index = index + 1
    return str(path[next_index]) if next_index < len(path) else ""


def _majority_label(labels: Sequence[int]) -> int | None:
    if not labels:
        return None
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    if values.size == 0:
        return None
    return int(values[int(np.argmax(counts))])


def _purity_from_labels(labels: Sequence[int]) -> float:
    if not labels:
        return math.nan
    _values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    if counts.size == 0:
        return math.nan
    return float(np.max(counts) / np.sum(counts))


def _validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing!r}")


def _run_data_and_truth_maps_by_run(
    traversal_rows: pd.DataFrame,
    *,
    suite: str,
) -> tuple[
    dict[tuple[str, str, str, int], tuple[int, dict[str, int]]],
    dict[tuple[str, str, str, int], tuple[int, pd.DataFrame]],
]:
    _validate_columns(
        traversal_rows,
        TRAVERSAL_REQUIRED_COLUMNS,
        label="traversal rows",
    )
    case_names = sorted(set(traversal_rows["case_id"].dropna().astype(str)))
    cases = {
        str(case["name"]): dict(case)
        for case in _select_cases(suite=suite, case_names=case_names)
    }
    truth_maps: dict[tuple[str, str, str, int], tuple[int, dict[str, int]]] = {}
    data_maps: dict[tuple[str, str, str, int], tuple[int, pd.DataFrame]] = {}
    run_columns = ["case_id", "data_role", "method_id", "replicate", "data_seed"]
    for _, row in (
        traversal_rows[run_columns]
        .drop_duplicates()
        .sort_values(run_columns)
        .iterrows()
    ):
        case_id = str(row["case_id"])
        data_role = _output_data_role(row["data_role"])
        method_id = str(row["method_id"])
        replicate = int(float(row["replicate"]))
        data_seed = int(float(row["data_seed"]))
        case = cases.get(case_id)
        if case is None:
            continue
        (
            _case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        data, _feature_space, truth_labels, _true_clusters = _generate_data_with_truth(
            case=case,
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            data_role=_input_data_role(data_role),
            seed=data_seed,
        )
        key = (case_id, data_role, method_id, replicate)
        truth_maps[key] = (
            data_seed,
            {
                str(sample_id): int(label)
                for sample_id, label in zip(
                    data.index.astype(str),
                    np.asarray(truth_labels, dtype=int),
                )
            },
        )
        data_maps[key] = (data_seed, data)
    return truth_maps, data_maps


def _truth_label_maps_by_run(
    traversal_rows: pd.DataFrame,
    *,
    suite: str,
) -> dict[tuple[str, str, str, int], tuple[int, dict[str, int]]]:
    truth_maps, _data_maps = _run_data_and_truth_maps_by_run(
        traversal_rows,
        suite=suite,
    )
    return truth_maps


def build_generated_support_candidate_rows(node_rows: pd.DataFrame) -> pd.DataFrame:
    """Convert generated pass-through support rows into split-candidate rows."""
    required = {
        "case_id",
        "data_role",
        "replicate",
        "truth_downstream_split_node_id",
        "truth_geometry_role",
        "left_method_id",
    }
    _validate_columns(node_rows, required, label="generated support node rows")
    records: list[dict[str, object]] = []
    for _, row in node_rows.sort_values(
        ["case_id", "data_role", "replicate", "node_id"]
    ).iterrows():
        truth_role = str(row["truth_geometry_role"])
        if str(row["data_role"]) == "selected_null":
            reason = "generated_selected_null_control_split"
        elif truth_role == "truth_recovery_pass_through_positive":
            reason = "generated_full_branch_recovery_split"
        elif truth_role == "partial_truth_recovery_pass_through_candidate":
            reason = "generated_partial_branch_recovery_split"
        elif truth_role == "barycentric_mixture_pass_through_candidate":
            reason = "generated_barycentric_mixture_split"
        elif truth_role == "fragment_false_pass_through":
            reason = "generated_false_fragment_split"
        else:
            reason = "generated_unresolved_split"
        records.append(
            {
                "case_id": str(row["case_id"]),
                "data_role": _output_data_role(row["data_role"]),
                "replicate": int(float(row["replicate"])),
                "node_id": str(row["truth_downstream_split_node_id"]),
                "left_method_id": str(row["left_method_id"]),
                "right_method_id": str(row["left_method_id"]),
                "candidate_reason": reason,
                "left_traversal_decision": "split",
                "right_traversal_decision": "split",
                "left_traversal_state": "split",
                "right_traversal_state": "split",
                "left_sibling_open": True,
                "right_sibling_open": True,
                "right_explicit_guard_blocked": False,
            }
        )
    return pd.DataFrame.from_records(records)


def _generated_support_traversal_rows(node_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "data_role", "replicate", "left_method_id", "data_seed"}
    _validate_columns(node_rows, required, label="generated support node rows")
    rows = node_rows.copy()
    return (
        pd.DataFrame(
            {
                "case_id": rows["case_id"].astype(str),
                "data_role": rows["data_role"].map(_output_data_role),
                "method_id": rows["left_method_id"].astype(str),
                "replicate": pd.to_numeric(
                    rows["replicate"],
                    errors="coerce",
                )
                .fillna(-1)
                .astype(int),
                "data_seed": pd.to_numeric(
                    rows["data_seed"],
                    errors="coerce",
                )
                .fillna(-1)
                .astype(int),
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _assignment_groups(
    gene_assignments: pd.DataFrame,
) -> dict[tuple[str, str, str, int], pd.DataFrame]:
    _validate_columns(
        gene_assignments,
        ASSIGNMENT_REQUIRED_COLUMNS,
        label="gene assignments",
    )
    rows = gene_assignments.copy()
    rows["data_role"] = rows["data_role"].map(_output_data_role)
    rows["replicate"] = (
        pd.to_numeric(rows["replicate"], errors="coerce").fillna(-1).astype(int)
    )
    groups: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    for key, group in rows.groupby(
        ["case_id", "data_role", "method_id", "replicate"],
        sort=False,
    ):
        case_id, data_role, method_id, replicate = key
        groups[(str(case_id), str(data_role), str(method_id), int(replicate))] = group
    return groups


def _empty_own_split_metrics(*, status: str) -> dict[str, object]:
    return {
        "own_split_status": status,
        "own_split_child_count": 0,
        "own_split_parent_sample_count": 0,
        "own_split_left_child_id": "",
        "own_split_right_child_id": "",
        "own_split_left_sample_count": 0,
        "own_split_right_sample_count": 0,
        "own_split_node_cluster_count": 0,
        "own_split_ari": math.nan,
        "own_split_child_mean_purity": math.nan,
        "own_split_child_majority_distinct": False,
        "own_split_left_majority_label": math.nan,
        "own_split_right_majority_label": math.nan,
    }


def _empty_feature_geometry_metrics(*, status: str) -> dict[str, object]:
    return {
        "feature_geometry_status": status,
        "feature_geometry_node_sample_count": 0,
        "feature_geometry_left_child_count": 0,
        "feature_geometry_right_child_count": 0,
        "feature_node_pairwise_jaccard": math.nan,
        "feature_left_pairwise_jaccard": math.nan,
        "feature_right_pairwise_jaccard": math.nan,
        "feature_homogeneity_gain_min": math.nan,
        "feature_heterogeneity_gain_max": math.nan,
        "feature_subspace_consensus_jaccard_topk": math.nan,
        "feature_heterogeneity_subspace_consensus_jaccard_topk": math.nan,
        "feature_child_contrast_norm": math.nan,
        "feature_branch_geometry_score": math.nan,
        "feature_barycentric_geometry_score": math.nan,
    }


def _own_split_child_samples(
    *,
    assignments: pd.DataFrame,
    node_id: str,
) -> tuple[str, str, list[str], list[str], int, int]:
    parsed = assignments[["sample_id", "path_node_ids"]].copy()
    parsed["path"] = parsed["path_node_ids"].map(_split_path)
    parsed = parsed.loc[parsed["path"].map(lambda path: node_id in path)].copy()
    if parsed.empty:
        return "", "", [], [], 0, 0
    parsed["next_child"] = parsed["path"].map(lambda path: _next_path_node(path, node_id))
    parsed = parsed.loc[parsed["next_child"].astype(str).ne("")]
    if parsed.empty:
        return "", "", [], [], 0, 0
    child_counts = parsed["next_child"].astype(str).value_counts()
    if child_counts.shape[0] < 2:
        return "", "", [], [], int(child_counts.shape[0]), int(parsed.shape[0])
    left_child, right_child = child_counts.index[:2].astype(str).tolist()
    left_samples = [
        str(sample_id)
        for sample_id in parsed.loc[
            parsed["next_child"].astype(str).eq(left_child), "sample_id"
        ]
    ]
    right_samples = [
        str(sample_id)
        for sample_id in parsed.loc[
            parsed["next_child"].astype(str).eq(right_child), "sample_id"
        ]
    ]
    return (
        left_child,
        right_child,
        left_samples,
        right_samples,
        int(child_counts.shape[0]),
        int(parsed.shape[0]),
    )


def _own_split_truth_metrics(
    *,
    assignments: pd.DataFrame,
    truth_by_sample: Mapping[str, int],
    node_id: str,
) -> dict[str, object]:
    (
        left_child,
        right_child,
        left_samples,
        right_samples,
        child_count,
        parent_sample_count,
    ) = _own_split_child_samples(assignments=assignments, node_id=node_id)
    if parent_sample_count == 0:
        return _empty_own_split_metrics(status="own_split_node_not_in_paths")
    if child_count == 0:
        return _empty_own_split_metrics(status="own_split_leaf_or_missing_children")
    if child_count < 2:
        metrics = _empty_own_split_metrics(status="own_split_children_missing")
        metrics["own_split_child_count"] = child_count
        metrics["own_split_parent_sample_count"] = parent_sample_count
        return metrics
    left_labels = [
        int(truth_by_sample[str(sample_id)])
        for sample_id in left_samples
        if str(sample_id) in truth_by_sample
    ]
    right_labels = [
        int(truth_by_sample[str(sample_id)])
        for sample_id in right_samples
        if str(sample_id) in truth_by_sample
    ]
    if not left_labels or not right_labels:
        metrics = _empty_own_split_metrics(status="own_split_truth_labels_missing")
        metrics.update(
            {
                "own_split_child_count": child_count,
                "own_split_parent_sample_count": parent_sample_count,
                "own_split_left_child_id": left_child,
                "own_split_right_child_id": right_child,
                "own_split_left_sample_count": int(len(left_labels)),
                "own_split_right_sample_count": int(len(right_labels)),
            }
        )
        return metrics

    truth = np.asarray(left_labels + right_labels, dtype=int)
    membership = np.concatenate(
        [np.zeros(len(left_labels), dtype=int), np.ones(len(right_labels), dtype=int)]
    )
    left_majority = _majority_label(left_labels)
    right_majority = _majority_label(right_labels)
    child_purities = [_purity_from_labels(left_labels), _purity_from_labels(right_labels)]
    ari = (
        float(adjusted_rand_score(truth, membership))
        if np.unique(truth).size > 1
        else 0.0
    )
    return {
        "own_split_status": "own_split_truth_observed",
        "own_split_child_count": child_count,
        "own_split_parent_sample_count": int(len(left_labels) + len(right_labels)),
        "own_split_left_child_id": left_child,
        "own_split_right_child_id": right_child,
        "own_split_left_sample_count": int(len(left_labels)),
        "own_split_right_sample_count": int(len(right_labels)),
        "own_split_node_cluster_count": int(np.unique(truth).size),
        "own_split_ari": ari,
        "own_split_child_mean_purity": float(np.nanmean(child_purities)),
        "own_split_child_majority_distinct": bool(
            left_majority is not None
            and right_majority is not None
            and left_majority != right_majority
        ),
        "own_split_left_majority_label": (
            math.nan if left_majority is None else int(left_majority)
        ),
        "own_split_right_majority_label": (
            math.nan if right_majority is None else int(right_majority)
        ),
    }


def _own_split_feature_geometry_metrics(
    *,
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    node_id: str,
    seed: int,
    top_k: int,
    max_pairwise_samples: int,
) -> dict[str, object]:
    (
        _left_child,
        _right_child,
        left_samples,
        right_samples,
        child_count,
        parent_sample_count,
    ) = _own_split_child_samples(assignments=assignments, node_id=node_id)
    if parent_sample_count == 0:
        return _empty_feature_geometry_metrics(
            status="feature_geometry_node_not_in_paths"
        )
    if child_count == 0:
        return _empty_feature_geometry_metrics(
            status="feature_geometry_leaf_or_missing_children"
        )
    if child_count < 2:
        metrics = _empty_feature_geometry_metrics(
            status="feature_geometry_children_missing"
        )
        metrics["feature_geometry_node_sample_count"] = parent_sample_count
        return metrics
    data_index = set(data.index.astype(str))
    left_ids = [sample_id for sample_id in left_samples if sample_id in data_index]
    right_ids = [sample_id for sample_id in right_samples if sample_id in data_index]
    parent_ids = left_ids + right_ids
    if len(parent_ids) < 2 or not left_ids or not right_ids:
        metrics = _empty_feature_geometry_metrics(
            status="feature_geometry_insufficient_samples"
        )
        metrics["feature_geometry_node_sample_count"] = len(parent_ids)
        metrics["feature_geometry_left_child_count"] = len(left_ids)
        metrics["feature_geometry_right_child_count"] = len(right_ids)
        return metrics
    geometry = _feature_geometry_for_partition(
        data=data,
        parent_sample_ids=parent_ids,
        left_sample_ids=left_ids,
        right_sample_ids=right_ids,
        top_k=int(top_k),
        max_pairwise_samples=int(max_pairwise_samples),
        seed=int(seed),
    )
    return {
        **_empty_feature_geometry_metrics(status="feature_geometry_observed"),
        **geometry,
        "feature_geometry_status": "feature_geometry_observed",
    }


def _classify_own_split(
    *,
    data_role: str,
    metrics: Mapping[str, object],
    branch_ari_floor: float,
    partial_ari_floor: float,
    child_purity_floor: float,
) -> str:
    if data_role == "selected_null":
        return "selected_null_control"
    if metrics.get("own_split_status") != "own_split_truth_observed":
        return "own_split_unavailable"
    ari = _finite_float(metrics.get("own_split_ari"))
    purity = _finite_float(metrics.get("own_split_child_mean_purity"))
    distinct = _bool_value(metrics.get("own_split_child_majority_distinct"))
    if not math.isfinite(ari) or not math.isfinite(purity):
        return "own_split_unavailable"
    if distinct and ari >= branch_ari_floor and purity >= child_purity_floor:
        return "own_split_branch_recovery"
    if distinct and ari >= partial_ari_floor and purity >= child_purity_floor:
        return "own_split_partial_branch_recovery"
    if not distinct and purity >= child_purity_floor:
        return "own_split_false_fragment"
    return "own_split_unresolved"


def _diagnostic_status_for_class(truth_class: str) -> str:
    if truth_class == "own_split_branch_recovery":
        return "candidate_branch_recovery_observed"
    if truth_class == "own_split_partial_branch_recovery":
        return "candidate_partial_branch_recovery_observed"
    if truth_class == "own_split_false_fragment":
        return "candidate_false_fragment_observed"
    if truth_class == "selected_null_control":
        return "candidate_selected_null_control_observed"
    if truth_class == "own_split_unavailable":
        return "candidate_truth_unavailable"
    return "candidate_truth_unresolved"


def build_selected_candidate_truth_law_rows(
    candidate_rows: pd.DataFrame,
    gene_assignments: pd.DataFrame,
    traversal_rows: pd.DataFrame | None = None,
    *,
    truth_label_maps: TruthLabelMaps | None = None,
    feature_data_maps: FeatureDataMaps | None = None,
    suite: str = "binary",
    branch_ari_floor: float = DEFAULT_BRANCH_ARI_FLOOR,
    partial_ari_floor: float = DEFAULT_PARTIAL_ARI_FLOOR,
    child_purity_floor: float = DEFAULT_CHILD_PURITY_FLOOR,
    top_k: int = 12,
    max_pairwise_samples: int = 200,
) -> pd.DataFrame:
    """Attach benchmark-truth own-split labels to selected candidates."""
    if candidate_rows.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    _validate_columns(
        candidate_rows,
        CANDIDATE_REQUIRED_COLUMNS,
        label="candidate rows",
    )
    if truth_label_maps is None:
        if traversal_rows is None:
            raise ValueError("traversal rows are required when truth maps are absent")
        truth_label_maps, generated_feature_data_maps = _run_data_and_truth_maps_by_run(
            traversal_rows,
            suite=suite,
        )
        if feature_data_maps is None:
            feature_data_maps = generated_feature_data_maps
    elif feature_data_maps is None and traversal_rows is not None:
        _generated_truth_maps, feature_data_maps = _run_data_and_truth_maps_by_run(
            traversal_rows,
            suite=suite,
        )
    assignment_groups = _assignment_groups(gene_assignments)

    rows = candidate_rows.copy()
    rows["data_role"] = rows["data_role"].map(_output_data_role)
    rows["replicate"] = (
        pd.to_numeric(rows["replicate"], errors="coerce").fillna(-1).astype(int)
    )

    records: list[dict[str, object]] = []
    for _, row in rows.sort_values(
        ["case_id", "data_role", "replicate", "node_id"]
    ).iterrows():
        case_id = str(row["case_id"])
        data_role = str(row["data_role"])
        replicate = int(row["replicate"])
        node_id = str(row["node_id"])
        method_id = str(row["left_method_id"])
        key = (case_id, data_role, method_id, replicate)
        truth_entry = truth_label_maps.get(key)
        assignments = assignment_groups.get(key)
        if truth_entry is None:
            metrics = _empty_own_split_metrics(
                status="own_split_missing_truth_run"
            )
        elif assignments is None or assignments.empty:
            metrics = _empty_own_split_metrics(
                status="own_split_missing_assignments"
            )
        else:
            _data_seed, truth_by_sample = truth_entry
            metrics = _own_split_truth_metrics(
                assignments=assignments,
                truth_by_sample=truth_by_sample,
                node_id=node_id,
            )
        feature_entry = None if feature_data_maps is None else feature_data_maps.get(key)
        if feature_entry is None:
            feature_metrics = _empty_feature_geometry_metrics(
                status="feature_geometry_missing_feature_run"
            )
        elif assignments is None or assignments.empty:
            feature_metrics = _empty_feature_geometry_metrics(
                status="feature_geometry_missing_assignments"
            )
        else:
            feature_seed, data = feature_entry
            feature_metrics = _own_split_feature_geometry_metrics(
                data=data,
                assignments=assignments,
                node_id=node_id,
                seed=feature_seed,
                top_k=top_k,
                max_pairwise_samples=max_pairwise_samples,
            )
        context_metrics = _candidate_context_metrics(row)
        truth_class = _classify_own_split(
            data_role=data_role,
            metrics=metrics,
            branch_ari_floor=float(branch_ari_floor),
            partial_ari_floor=float(partial_ari_floor),
            child_purity_floor=float(child_purity_floor),
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": case_id,
                "data_role": data_role,
                "replicate": replicate,
                "node_id": node_id,
                "own_split_method_id": method_id,
                "candidate_reason": str(row["candidate_reason"]),
                "left_traversal_state": str(row["left_traversal_state"]),
                "right_traversal_state": str(row["right_traversal_state"]),
                "left_traversal_decision": str(row["left_traversal_decision"]),
                "right_traversal_decision": str(row["right_traversal_decision"]),
                "left_sibling_open": _bool_value(row["left_sibling_open"]),
                "right_sibling_open": _bool_value(row["right_sibling_open"]),
                "right_explicit_guard_blocked": _bool_value(
                    row["right_explicit_guard_blocked"]
                ),
                **context_metrics,
                **metrics,
                "own_split_truth_class": truth_class,
                "candidate_truth_diagnostic_status": _diagnostic_status_for_class(
                    truth_class
                ),
                **feature_metrics,
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _max_metric(rows: pd.DataFrame, mask: pd.Series, metric: str) -> float:
    values = pd.to_numeric(rows.loc[mask, metric], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.max()) if not values.empty else math.nan


def _pass_through_mask(rows: pd.DataFrame) -> pd.Series:
    return rows["left_traversal_state"].eq("pass_through") | rows[
        "right_traversal_state"
    ].eq("pass_through")


def _split_mask(rows: pd.DataFrame) -> pd.Series:
    return rows["left_traversal_state"].eq("split") | rows[
        "right_traversal_state"
    ].eq("split")


def summarize_selected_candidate_truth_law(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether branch recovery occurs in split or pass-through rows."""
    if rows.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "signal_row_count": 0,
                    "selected_null_control_count": 0,
                    "branch_recovery_count": 0,
                    "partial_branch_recovery_count": 0,
                    "false_fragment_count": 0,
                    "unresolved_count": 0,
                    "unavailable_count": 0,
                    "pass_through_branch_recovery_count": 0,
                    "split_branch_recovery_count": 0,
                    "max_pass_through_ari": math.nan,
                    "max_split_ari": math.nan,
                    "diagnostic_status": "candidate_truth_rows_missing",
                    "next_required_step": "run_candidate_truth_law_panel_on_selected_candidates",
                    "production_action": "diagnostic_only_no_promotion",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )

    truth = rows["own_split_truth_class"].astype(str)
    branch_mask = truth.eq("own_split_branch_recovery")
    partial_mask = truth.eq("own_split_partial_branch_recovery")
    false_mask = truth.eq("own_split_false_fragment")
    unresolved_mask = truth.eq("own_split_unresolved")
    unavailable_mask = truth.eq("own_split_unavailable")
    selected_null_mask = truth.eq("selected_null_control")
    pass_mask = _pass_through_mask(rows)
    split_mask = _split_mask(rows)

    branch_count = int(branch_mask.sum())
    pass_branch_count = int((branch_mask & pass_mask).sum())
    split_branch_count = int((branch_mask & split_mask).sum())
    false_count = int(false_mask.sum())
    if pass_branch_count:
        diagnostic_status = "pass_through_branch_recovery_observed_diagnostic_only"
        next_step = "derive_selected_neighborhood_law_for_pass_through_recovery"
    elif branch_count:
        diagnostic_status = "branch_recovery_observed_only_on_accepted_splits"
        next_step = "preserve_accepted_branch_splits_and_block_pass_through_fragments"
    elif false_count:
        diagnostic_status = "pass_through_fragment_risk_observed"
        next_step = "derive_fragment_suppression_conditioning_before_promotion"
    else:
        diagnostic_status = "branch_recovery_support_missing"
        next_step = "construct_signal_side_branch_recovery_support_fixture"

    record = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "row_count": int(rows.shape[0]),
        "signal_row_count": int(rows["data_role"].astype(str).eq("signal").sum()),
        "selected_null_control_count": int(selected_null_mask.sum()),
        "branch_recovery_count": branch_count,
        "partial_branch_recovery_count": int(partial_mask.sum()),
        "false_fragment_count": false_count,
        "unresolved_count": int(unresolved_mask.sum()),
        "unavailable_count": int(unavailable_mask.sum()),
        "pass_through_branch_recovery_count": pass_branch_count,
        "split_branch_recovery_count": split_branch_count,
        "max_pass_through_ari": _max_metric(rows, pass_mask, "own_split_ari"),
        "max_split_ari": _max_metric(rows, split_mask, "own_split_ari"),
        "diagnostic_status": diagnostic_status,
        "next_required_step": next_step,
        "production_action": "diagnostic_only_no_promotion",
    }
    return pd.DataFrame.from_records([record], columns=SUMMARY_COLUMNS)


def summarize_selected_candidate_truth_law_states(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate truth classes by directed traversal state."""
    if rows.empty:
        return pd.DataFrame(columns=STATE_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    group_columns = [
        "data_role",
        "left_traversal_state",
        "right_traversal_state",
        "own_split_truth_class",
    ]
    for key, group in rows.groupby(group_columns, sort=True):
        data_role, left_state, right_state, truth_class = [str(value) for value in key]
        values = pd.to_numeric(group["own_split_ari"], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": data_role,
                "left_traversal_state": left_state,
                "right_traversal_state": right_state,
                "own_split_truth_class": truth_class,
                "row_count": int(group.shape[0]),
                "finite_ari_count": int(values.shape[0]),
                "median_ari": float(values.median()) if not values.empty else math.nan,
                "max_ari": float(values.max()) if not values.empty else math.nan,
            }
        )
    return pd.DataFrame.from_records(records, columns=STATE_SUMMARY_COLUMNS)


def _finite_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.dropna().to_numpy(dtype=float)


def _auc(positive_values: np.ndarray, negative_values: np.ndarray) -> float:
    if positive_values.size == 0 or negative_values.size == 0:
        return math.nan
    wins = 0.0
    total = 0
    for value in positive_values:
        wins += float(np.sum(value > negative_values))
        wins += 0.5 * float(np.sum(value == negative_values))
        total += int(negative_values.size)
    return float(wins / total) if total else math.nan


def _metric_record(
    rows: pd.DataFrame,
    *,
    metric: str,
) -> dict[str, object]:
    truth = rows["own_split_truth_class"].astype(str)
    branch = _finite_array(rows.loc[truth.eq("own_split_branch_recovery"), metric])
    negative = _finite_array(
        rows.loc[
            truth.isin(
                [
                    "own_split_false_fragment",
                    "selected_null_control",
                ]
            ),
            metric,
        ]
    )
    auc_high = _auc(branch, negative)
    auc_low = _auc(-branch, -negative)
    if math.isfinite(auc_low) and (
        not math.isfinite(auc_high) or auc_low > auc_high
    ):
        best_direction = "low"
        best_auc = auc_low
        separates = branch.size > 0 and negative.size > 0 and branch.max() < negative.min()
    else:
        best_direction = "high"
        best_auc = auc_high
        separates = branch.size > 0 and negative.size > 0 and branch.min() > negative.max()
    if branch.size == 0 or negative.size == 0:
        zero_status = "insufficient_branch_or_negative_support"
    elif separates:
        zero_status = f"zero_negative_separates_branch_{best_direction}"
    else:
        zero_status = "zero_negative_does_not_separate_branch"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "metric": metric,
        "branch_count": int(truth.eq("own_split_branch_recovery").sum()),
        "negative_count": int(
            truth.isin(["own_split_false_fragment", "selected_null_control"]).sum()
        ),
        "finite_branch_count": int(branch.size),
        "finite_negative_count": int(negative.size),
        "best_direction": best_direction,
        "best_auc": best_auc,
        "branch_min": float(np.min(branch)) if branch.size else math.nan,
        "branch_median": float(np.median(branch)) if branch.size else math.nan,
        "branch_max": float(np.max(branch)) if branch.size else math.nan,
        "negative_min": float(np.min(negative)) if negative.size else math.nan,
        "negative_median": float(np.median(negative)) if negative.size else math.nan,
        "negative_max": float(np.max(negative)) if negative.size else math.nan,
        "zero_negative_status": zero_status,
    }


def summarize_selected_candidate_feature_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-oracle feature metrics against oracle candidate classes."""
    if rows.empty:
        return pd.DataFrame(columns=FEATURE_METRIC_COLUMNS)
    metrics = [
        "feature_homogeneity_gain_min",
        "feature_subspace_consensus_jaccard_topk",
        "feature_child_contrast_norm",
        "feature_branch_geometry_score",
        "feature_barycentric_geometry_score",
    ]
    records = [_metric_record(rows, metric=metric) for metric in metrics]
    return pd.DataFrame.from_records(records, columns=FEATURE_METRIC_COLUMNS)


def _state_scope_masks(rows: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    pass_mask = _pass_through_mask(rows)
    split_mask = _split_mask(rows)
    split_split = rows["left_traversal_state"].eq("split") & rows[
        "right_traversal_state"
    ].eq("split")
    pass_boundary = (
        rows["left_traversal_state"].eq("pass_through")
        & rows["right_traversal_state"].eq("boundary")
    ) | (
        rows["right_traversal_state"].eq("pass_through")
        & rows["left_traversal_state"].eq("boundary")
    )
    pass_pass = rows["left_traversal_state"].eq("pass_through") & rows[
        "right_traversal_state"
    ].eq("pass_through")
    masks = [
        ("all_candidates", pd.Series(True, index=rows.index)),
        ("split_no_pass_through", split_mask & ~pass_mask),
        ("pass_through_any", pass_mask),
        ("split_split", split_split),
        ("pass_through_boundary", pass_boundary),
        ("pass_through_pass_through", pass_pass),
    ]
    return [(label, mask) for label, mask in masks if int(mask.sum()) > 0]


def summarize_selected_candidate_feature_metrics_by_state(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize feature metrics separately by directed traversal state scope."""
    if rows.empty:
        return pd.DataFrame(columns=FEATURE_METRIC_STATE_COLUMNS)
    records: list[dict[str, object]] = []
    for scope, mask in _state_scope_masks(rows):
        scoped = rows.loc[mask].copy()
        metrics = summarize_selected_candidate_feature_metrics(scoped)
        for _, metric_row in metrics.iterrows():
            record = dict(metric_row)
            record["state_scope"] = scope
            record["scope_row_count"] = int(scoped.shape[0])
            records.append(record)
    return pd.DataFrame.from_records(records, columns=FEATURE_METRIC_STATE_COLUMNS)


def summarize_selected_candidate_context_metrics_by_state(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize traversal/context metrics by directed traversal state scope."""
    if rows.empty:
        return pd.DataFrame(columns=CONTEXT_METRIC_STATE_COLUMNS)
    metrics = [
        "left_depth",
        "left_n_descendant_leaves",
        "min_sibling_p_value",
        "negative_log10_min_sibling_p_value",
        "left_descendant_accepted_split_count",
        "right_descendant_accepted_split_count",
        "descendant_accepted_split_delta",
        "left_descendant_pass_through_count",
        "descendant_pass_through_delta",
        "left_descendant_stable_boundary_count",
    ]
    records: list[dict[str, object]] = []
    for scope, mask in _state_scope_masks(rows):
        scoped = rows.loc[mask].copy()
        for metric in metrics:
            record = _metric_record(scoped, metric=metric)
            record["state_scope"] = scope
            record["scope_row_count"] = int(scoped.shape[0])
            records.append(record)
    return pd.DataFrame.from_records(records, columns=CONTEXT_METRIC_STATE_COLUMNS)


def _candidate_ambiguity_bucket(row: pd.Series) -> str:
    if str(row["left_traversal_decision"]) == str(row["right_traversal_decision"]):
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


def _candidate_stop_rule_pattern(row: pd.Series) -> str:
    left_state = str(row["left_traversal_state"])
    right_decision = str(row["right_traversal_decision"])
    right_blocked = _bool_value(row["right_explicit_guard_blocked"])
    left_descendant_splits = _optional_int(
        row,
        "left_descendant_accepted_split_count",
    )
    if (
        left_state == "pass_through"
        and right_decision in {"boundary", "not_visited"}
        and left_descendant_splits > 0
    ):
        return "left_pass_through_downstream_split_right_stops"
    if (
        left_state == "split"
        and right_decision in {"boundary", "not_visited"}
        and right_blocked
    ):
        return "right_refined_guard_blocks_left_candidate"
    if str(row["left_traversal_decision"]) == str(row["right_traversal_decision"]):
        return "paired_candidate_agreement"
    return "candidate_stop_rule_divergence_unclassified"


def _count_truth(rows: pd.DataFrame, truth_class: str) -> int:
    if rows.empty:
        return 0
    return int(rows["own_split_truth_class"].astype(str).eq(truth_class).sum())


def _count_left_state(rows: pd.DataFrame, state: str) -> int:
    if rows.empty:
        return 0
    return int(rows["left_traversal_state"].astype(str).eq(state).sum())


def _finite_series(values: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _finite_summary(values: pd.Series) -> tuple[float, float, float]:
    finite = _finite_series(values)
    if finite.empty:
        return math.nan, math.nan, math.nan
    return float(finite.median()), float(finite.min()), float(finite.max())


def _family_status(
    *,
    data_role: str,
    branch_count: int,
    partial_count: int,
    false_count: int,
    selected_null_count: int,
    unresolved_count: int,
    matched_selected_null_collision: bool,
) -> tuple[str, str, str]:
    if data_role == "selected_null" or selected_null_count:
        return (
            "selected_null_suppression_required",
            "validate_selected_null_family_suppression_before_relaxation",
            "retain_fail_closed_refined_guard",
        )
    if branch_count and not false_count and not matched_selected_null_collision:
        return (
            "clean_branch_family_diagnostic_candidate",
            "validate_clean_branch_family_transfer_before_promotion",
            "diagnostic_only_no_promotion",
        )
    if branch_count and (false_count or matched_selected_null_collision):
        return (
            "branch_family_collides_with_fragment_or_selected_null",
            "derive_collision_aware_selected_neighborhood_law",
            "fail_closed_until_collision_law_validated",
        )
    if partial_count:
        return (
            "partial_branch_family_unresolved",
            "derive_partial_branch_recovery_law_or_keep_multiscale",
            "fail_closed_until_partial_law_validated",
        )
    if false_count:
        return (
            "fragment_family_suppression_required",
            "preserve_fragment_suppression_and_selected_null_controls",
            "retain_fail_closed_refined_guard",
        )
    if unresolved_count:
        return (
            "unresolved_family_fail_closed",
            "collect_more_family_truth_evidence",
            "fail_closed_until_evidence_present",
        )
    return (
        "unavailable_family_fail_closed",
        "collect_candidate_family_truth_evidence",
        "fail_closed_until_evidence_present",
    )


def summarize_selected_candidate_family_likelihood_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected-candidate truth at the run/family pattern level.

    This is an oracle diagnostic for the selected-neighborhood law. It does not
    fit or apply a production likelihood; it identifies whether branch-positive
    families are clean, fragment-mixed, or collide with matched selected-null
    families under the same selected stop-rule pattern.
    """
    if rows.empty:
        return pd.DataFrame(columns=FAMILY_LIKELIHOOD_ROW_COLUMNS)
    prepared = rows.copy()
    prepared["ambiguity_bucket"] = [
        _candidate_ambiguity_bucket(row) for _, row in prepared.iterrows()
    ]
    prepared["stop_rule_pattern"] = [
        _candidate_stop_rule_pattern(row) for _, row in prepared.iterrows()
    ]
    selected_null_groups = {
        (
            str(row["case_id"]),
            int(row["replicate"]),
            str(row["ambiguity_bucket"]),
            str(row["stop_rule_pattern"]),
        ): row
        for _, row in (
            prepared.loc[prepared["data_role"].astype(str).eq("selected_null")]
            .groupby(
                [
                    "case_id",
                    "replicate",
                    "ambiguity_bucket",
                    "stop_rule_pattern",
                ],
                sort=True,
            )
            .agg(
                matched_selected_null_family_count=("node_id", "size"),
                matched_selected_null_control_count=(
                    "own_split_truth_class",
                    lambda values: int(
                        values.astype(str).eq("selected_null_control").sum()
                    ),
                ),
            )
            .reset_index()
            .iterrows()
        )
    }
    records: list[dict[str, object]] = []
    group_columns = [
        "case_id",
        "data_role",
        "replicate",
        "ambiguity_bucket",
        "stop_rule_pattern",
    ]
    for keys, group in prepared.groupby(group_columns, sort=True):
        case_id, data_role, replicate, ambiguity_bucket, stop_rule_pattern = keys
        branch_count = _count_truth(group, "own_split_branch_recovery")
        partial_count = _count_truth(group, "own_split_partial_branch_recovery")
        false_count = _count_truth(group, "own_split_false_fragment")
        selected_null_count = _count_truth(group, "selected_null_control")
        unresolved_count = _count_truth(group, "own_split_unresolved")
        unavailable_count = _count_truth(group, "own_split_unavailable")
        median_p, min_p, max_p = _finite_summary(group["min_sibling_p_value"])
        feature_median, _feature_min, feature_max = _finite_summary(
            group["feature_branch_geometry_score"]
        )
        selected_null_match = selected_null_groups.get(
            (
                str(case_id),
                int(replicate),
                str(ambiguity_bucket).replace(
                    "possible_signal_over_suppression",
                    "conservative_selected_null_suppression",
                ),
                str(stop_rule_pattern),
            )
        )
        matched_family_count = 0
        matched_candidate_count = 0
        matched_control_count = 0
        if selected_null_match is not None and str(data_role) != "selected_null":
            matched_family_count = 1
            matched_candidate_count = int(
                selected_null_match["matched_selected_null_family_count"]
            )
            matched_control_count = int(
                selected_null_match["matched_selected_null_control_count"]
            )
        matched_collision = bool(
            matched_control_count > 0
            and str(stop_rule_pattern) != "paired_candidate_agreement"
        )
        status, next_step, action = _family_status(
            data_role=str(data_role),
            branch_count=branch_count,
            partial_count=partial_count,
            false_count=false_count,
            selected_null_count=selected_null_count,
            unresolved_count=unresolved_count,
            matched_selected_null_collision=matched_collision,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "ambiguity_bucket": str(ambiguity_bucket),
                "stop_rule_pattern": str(stop_rule_pattern),
                "family_row_count": int(group.shape[0]),
                "branch_recovery_count": branch_count,
                "partial_branch_recovery_count": partial_count,
                "false_fragment_count": false_count,
                "selected_null_control_count": selected_null_count,
                "unresolved_count": unresolved_count,
                "unavailable_count": unavailable_count,
                "pass_through_candidate_count": _count_left_state(
                    group,
                    "pass_through",
                ),
                "split_candidate_count": _count_left_state(group, "split"),
                "descendant_accepted_split_total": int(
                    pd.to_numeric(
                        group["left_descendant_accepted_split_count"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "median_min_sibling_p_value": median_p,
                "min_min_sibling_p_value": min_p,
                "max_min_sibling_p_value": max_p,
                "median_feature_branch_geometry_score": feature_median,
                "max_feature_branch_geometry_score": feature_max,
                "matched_selected_null_family_count": matched_family_count,
                "matched_selected_null_candidate_count": matched_candidate_count,
                "matched_selected_null_control_count": matched_control_count,
                "matched_selected_null_collision": matched_collision,
                "family_truth_signal": (
                    "branch"
                    if branch_count
                    else "partial"
                    if partial_count
                    else "selected_null"
                    if selected_null_count
                    else "fragment"
                    if false_count
                    else "unresolved"
                    if unresolved_count
                    else "unavailable"
                ),
                "family_likelihood_status": status,
                "next_required_step": next_step,
                "production_action": action,
            }
        )
    return pd.DataFrame.from_records(records, columns=FAMILY_LIKELIHOOD_ROW_COLUMNS)


def _family_summary_status(group: pd.DataFrame) -> tuple[str, str, str]:
    branch_families = group["branch_recovery_count"].astype(int).gt(0)
    colliding_branch = branch_families & (
        group["matched_selected_null_collision"].astype(bool)
        | group["false_fragment_count"].astype(int).gt(0)
    )
    clean_branch = branch_families & ~colliding_branch
    selected_null = group["selected_null_control_count"].astype(int).gt(0)
    if bool(colliding_branch.any()):
        return (
            "branch_family_collision_observed",
            "derive_collision_aware_selected_neighborhood_law",
            "fail_closed_until_collision_law_validated",
        )
    if bool(clean_branch.any()):
        return (
            "clean_branch_families_observed_diagnostic",
            "validate_clean_family_likelihood_transfer",
            "diagnostic_only_no_promotion",
        )
    if bool(selected_null.any()):
        return (
            "selected_null_family_risk_observed",
            "preserve_selected_null_suppression",
            "retain_fail_closed_refined_guard",
        )
    return (
        "no_branch_family_support",
        "collect_selected_family_branch_support",
        "fail_closed_until_evidence_present",
    )


def summarize_selected_candidate_family_likelihood(
    family_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected-family diagnostic statuses by traversal context."""
    if family_rows.empty:
        return pd.DataFrame(columns=FAMILY_LIKELIHOOD_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    group_columns = [
        "data_role",
        "ambiguity_bucket",
        "stop_rule_pattern",
        "family_likelihood_status",
    ]
    for keys, group in family_rows.groupby(group_columns, sort=True):
        status, next_step, action = _family_summary_status(group)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": str(keys[0]),
                "ambiguity_bucket": str(keys[1]),
                "stop_rule_pattern": str(keys[2]),
                "family_likelihood_status": str(keys[3]),
                "family_count": int(group.shape[0]),
                "candidate_count": int(group["family_row_count"].astype(int).sum()),
                "branch_recovery_count": int(
                    group["branch_recovery_count"].astype(int).sum()
                ),
                "partial_branch_recovery_count": int(
                    group["partial_branch_recovery_count"].astype(int).sum()
                ),
                "false_fragment_count": int(
                    group["false_fragment_count"].astype(int).sum()
                ),
                "selected_null_control_count": int(
                    group["selected_null_control_count"].astype(int).sum()
                ),
                "unresolved_count": int(group["unresolved_count"].astype(int).sum()),
                "pass_through_candidate_count": int(
                    group["pass_through_candidate_count"].astype(int).sum()
                ),
                "split_candidate_count": int(
                    group["split_candidate_count"].astype(int).sum()
                ),
                "matched_selected_null_collision_family_count": int(
                    group["matched_selected_null_collision"].astype(bool).sum()
                ),
                "clean_branch_family_count": int(
                    (
                        group["branch_recovery_count"].astype(int).gt(0)
                        & ~group["matched_selected_null_collision"].astype(bool)
                        & group["false_fragment_count"].astype(int).eq(0)
                    ).sum()
                ),
                "colliding_branch_family_count": int(
                    (
                        group["branch_recovery_count"].astype(int).gt(0)
                        & (
                            group["matched_selected_null_collision"].astype(bool)
                            | group["false_fragment_count"].astype(int).gt(0)
                        )
                    ).sum()
                ),
                "diagnostic_status": status,
                "next_required_step": next_step,
                "production_action": action,
            }
        )
    return pd.DataFrame.from_records(records, columns=FAMILY_LIKELIHOOD_SUMMARY_COLUMNS)


def _collision_component_status(
    *,
    component_id: str,
    family_count: int,
    clean_branch_count: int,
    colliding_branch_count: int,
    selected_null_count: int,
    false_fragment_count: int,
    branch_count: int,
) -> tuple[str, str, str]:
    if family_count == 0:
        return (
            "component_support_missing",
            "collect_selected_family_component_support",
            "fail_closed_until_evidence_present",
        )
    if component_id == "accepted_split_preservation":
        if clean_branch_count and not colliding_branch_count:
            return (
                "accepted_split_preservation_clean_diagnostic",
                "validate accepted-split feature/topology filter before promotion",
                "diagnostic_only_no_promotion",
            )
        if clean_branch_count and colliding_branch_count:
            return (
                "accepted_split_preservation_needs_fragment_filter",
                "derive accepted-split fragment filter before promotion",
                "diagnostic_only_no_promotion",
            )
        if colliding_branch_count:
            return (
                "accepted_split_branch_collision_observed",
                "derive accepted-split collision filter",
                "fail_closed_until_collision_law_validated",
            )
        return (
            "accepted_split_branch_support_missing",
            "collect accepted-split branch support",
            "fail_closed_until_evidence_present",
        )
    if component_id == "pass_through_retention":
        if branch_count and colliding_branch_count == branch_count:
            return (
                "pass_through_branch_families_all_collide",
                "derive collision-aware pass-through retention law",
                "fail_closed_until_collision_law_validated",
            )
        if branch_count and colliding_branch_count:
            return (
                "pass_through_branch_families_partly_collide",
                "separate clean and colliding pass-through regimes",
                "fail_closed_until_collision_law_validated",
            )
        if branch_count:
            return (
                "pass_through_branch_clean_diagnostic_candidate",
                "validate pass-through clean-family transfer",
                "diagnostic_only_no_promotion",
            )
        if false_fragment_count:
            return (
                "pass_through_fragment_suppression_required",
                "preserve refined pass-through guard",
                "retain_fail_closed_refined_guard",
            )
        return (
            "pass_through_retention_support_missing",
            "collect real pass-through branch support",
            "fail_closed_until_evidence_present",
        )
    if component_id == "selected_null_suppression":
        if selected_null_count:
            return (
                "selected_null_suppression_required",
                "preserve selected-family guard unless null law is derived",
                "retain_fail_closed_refined_guard",
            )
        return (
            "selected_null_support_missing",
            "collect selected-null family controls",
            "fail_closed_until_evidence_present",
        )
    if component_id == "guard_stop_fragment_suppression":
        if false_fragment_count or selected_null_count:
            return (
                "guard_stop_fragment_or_null_suppression_required",
                "preserve guard-stop suppression",
                "retain_fail_closed_refined_guard",
            )
        return (
            "guard_stop_support_missing",
            "collect guard-stop fragment/null controls",
            "fail_closed_until_evidence_present",
        )
    return (
        "component_unclassified",
        "inspect selected-family component",
        "diagnostic_only_no_promotion",
    )


def _component_record(
    family_rows: pd.DataFrame,
    *,
    component_id: str,
    support_scope: str,
) -> dict[str, object]:
    if family_rows.empty:
        family_count = 0
        candidate_count = 0
        branch_count = 0
        partial_count = 0
        false_count = 0
        selected_null_count = 0
        unresolved_count = 0
        clean_branch_count = 0
        colliding_branch_count = 0
        null_collision_count = 0
        fragment_mixed_branch_count = 0
        pass_branch_count = 0
        selected_null_family_count = 0
    else:
        family_count = int(family_rows.shape[0])
        candidate_count = int(family_rows["family_row_count"].astype(int).sum())
        branch_count = int(family_rows["branch_recovery_count"].astype(int).sum())
        partial_count = int(
            family_rows["partial_branch_recovery_count"].astype(int).sum()
        )
        false_count = int(family_rows["false_fragment_count"].astype(int).sum())
        selected_null_count = int(
            family_rows["selected_null_control_count"].astype(int).sum()
        )
        unresolved_count = int(family_rows["unresolved_count"].astype(int).sum())
        branch_family = family_rows["branch_recovery_count"].astype(int).gt(0)
        clean_branch = branch_family & family_rows[
            "family_likelihood_status"
        ].astype(str).eq("clean_branch_family_diagnostic_candidate")
        colliding_branch = branch_family & family_rows[
            "family_likelihood_status"
        ].astype(str).eq("branch_family_collides_with_fragment_or_selected_null")
        null_collision = family_rows["matched_selected_null_collision"].astype(bool)
        fragment_mixed_branch = branch_family & family_rows[
            "false_fragment_count"
        ].astype(int).gt(0)
        pass_branch = branch_family & family_rows["stop_rule_pattern"].astype(str).eq(
            "left_pass_through_downstream_split_right_stops"
        )
        selected_null_family = family_rows["family_likelihood_status"].astype(str).eq(
            "selected_null_suppression_required"
        )
        clean_branch_count = int(clean_branch.sum())
        colliding_branch_count = int(colliding_branch.sum())
        null_collision_count = int(null_collision.sum())
        fragment_mixed_branch_count = int(fragment_mixed_branch.sum())
        pass_branch_count = int(pass_branch.sum())
        selected_null_family_count = int(selected_null_family.sum())
    status, next_step, action = _collision_component_status(
        component_id=component_id,
        family_count=family_count,
        clean_branch_count=clean_branch_count,
        colliding_branch_count=colliding_branch_count,
        selected_null_count=selected_null_count,
        false_fragment_count=false_count,
        branch_count=pass_branch_count
        if component_id == "pass_through_retention"
        else clean_branch_count + colliding_branch_count,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "component_id": component_id,
        "support_scope": support_scope,
        "family_count": family_count,
        "candidate_count": candidate_count,
        "branch_recovery_count": branch_count,
        "partial_branch_recovery_count": partial_count,
        "false_fragment_count": false_count,
        "selected_null_control_count": selected_null_count,
        "unresolved_count": unresolved_count,
        "clean_branch_family_count": clean_branch_count,
        "colliding_branch_family_count": colliding_branch_count,
        "matched_selected_null_collision_family_count": null_collision_count,
        "fragment_mixed_branch_family_count": fragment_mixed_branch_count,
        "pass_through_branch_family_count": pass_branch_count,
        "selected_null_suppression_family_count": selected_null_family_count,
        "component_status": status,
        "next_required_step": next_step,
        "production_action": action,
    }


def summarize_selected_candidate_collision_law_components(
    family_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the selected-family law into explicit diagnostic components."""
    if family_rows.empty:
        components = [
            ("accepted_split_preservation", "signal:paired_candidate_agreement"),
            (
                "pass_through_retention",
                "signal:left_pass_through_downstream_split_right_stops",
            ),
            (
                "selected_null_suppression",
                "selected_null:selected_stop_rule_patterns",
            ),
            (
                "guard_stop_fragment_suppression",
                "signal:right_refined_guard_blocks_left_candidate",
            ),
        ]
        return pd.DataFrame.from_records(
            [
                _component_record(
                    pd.DataFrame(columns=FAMILY_LIKELIHOOD_ROW_COLUMNS),
                    component_id=component_id,
                    support_scope=support_scope,
                )
                for component_id, support_scope in components
            ],
            columns=COLLISION_LAW_COMPONENT_COLUMNS,
        )
    signal = family_rows["data_role"].astype(str).eq("signal")
    selected_null = family_rows["data_role"].astype(str).eq("selected_null")
    paired = family_rows["stop_rule_pattern"].astype(str).eq(
        "paired_candidate_agreement"
    )
    pass_through = family_rows["stop_rule_pattern"].astype(str).eq(
        "left_pass_through_downstream_split_right_stops"
    )
    guard_stop = family_rows["stop_rule_pattern"].astype(str).eq(
        "right_refined_guard_blocks_left_candidate"
    )
    selected_stop = pass_through | guard_stop
    component_inputs = [
        (
            "accepted_split_preservation",
            "signal:paired_candidate_agreement",
            family_rows.loc[signal & paired],
        ),
        (
            "pass_through_retention",
            "signal:left_pass_through_downstream_split_right_stops",
            family_rows.loc[signal & pass_through],
        ),
        (
            "selected_null_suppression",
            "selected_null:selected_stop_rule_patterns",
            family_rows.loc[selected_null & selected_stop],
        ),
        (
            "guard_stop_fragment_suppression",
            "signal:right_refined_guard_blocks_left_candidate",
            family_rows.loc[signal & guard_stop],
        ),
    ]
    return pd.DataFrame.from_records(
        [
            _component_record(
                component_rows,
                component_id=component_id,
                support_scope=support_scope,
            )
            for component_id, support_scope, component_rows in component_inputs
        ],
        columns=COLLISION_LAW_COMPONENT_COLUMNS,
    )


def _accepted_split_filter_role(family_status: str) -> str:
    if family_status == "clean_branch_family_diagnostic_candidate":
        return "clean_branch_family"
    if family_status == "branch_family_collides_with_fragment_or_selected_null":
        return "fragment_mixed_branch_family"
    if family_status == "fragment_family_suppression_required":
        return "fragment_only_family"
    if family_status == "partial_branch_family_unresolved":
        return "partial_branch_family"
    if family_status == "unresolved_family_fail_closed":
        return "unresolved_family"
    return "other_family"


def _metric_min_median_max(group: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    finite = _finite_series(group[metric]) if metric in group else pd.Series(dtype=float)
    if finite.empty:
        return math.nan, math.nan, math.nan
    return float(finite.min()), float(finite.median()), float(finite.max())


def build_selected_candidate_accepted_split_filter_rows(
    rows: pd.DataFrame,
    family_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build family-level non-oracle metrics for accepted split preservation."""
    if rows.empty or family_rows.empty:
        return pd.DataFrame(columns=ACCEPTED_SPLIT_FILTER_ROW_COLUMNS)
    prepared = rows.copy()
    prepared["ambiguity_bucket"] = [
        _candidate_ambiguity_bucket(row) for _, row in prepared.iterrows()
    ]
    prepared["stop_rule_pattern"] = [
        _candidate_stop_rule_pattern(row) for _, row in prepared.iterrows()
    ]
    accepted = prepared.loc[
        prepared["data_role"].astype(str).eq("signal")
        & prepared["stop_rule_pattern"].astype(str).eq("paired_candidate_agreement")
    ].copy()
    if accepted.empty:
        return pd.DataFrame(columns=ACCEPTED_SPLIT_FILTER_ROW_COLUMNS)
    family_index = {
        (
            str(row["case_id"]),
            str(row["data_role"]),
            int(row["replicate"]),
            str(row["ambiguity_bucket"]),
            str(row["stop_rule_pattern"]),
        ): row
        for _, row in family_rows.iterrows()
    }
    records: list[dict[str, object]] = []
    group_columns = [
        "case_id",
        "data_role",
        "replicate",
        "ambiguity_bucket",
        "stop_rule_pattern",
    ]
    for keys, group in accepted.groupby(group_columns, sort=True):
        case_id, data_role, replicate, ambiguity_bucket, stop_rule_pattern = keys
        family = family_index.get(
            (
                str(case_id),
                str(data_role),
                int(replicate),
                str(ambiguity_bucket),
                str(stop_rule_pattern),
            )
        )
        if family is None:
            continue
        family_status = str(family["family_likelihood_status"])
        hom_min, hom_median, hom_max = _metric_min_median_max(
            group,
            "feature_homogeneity_gain_min",
        )
        branch_min, branch_median, branch_max = _metric_min_median_max(
            group,
            "feature_branch_geometry_score",
        )
        contrast_min, contrast_median, contrast_max = _metric_min_median_max(
            group,
            "feature_child_contrast_norm",
        )
        bary_min, bary_median, bary_max = _metric_min_median_max(
            group,
            "feature_barycentric_geometry_score",
        )
        sibling_min, sibling_median, sibling_max = _metric_min_median_max(
            group,
            "min_sibling_p_value",
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "ambiguity_bucket": str(ambiguity_bucket),
                "stop_rule_pattern": str(stop_rule_pattern),
                "family_likelihood_status": family_status,
                "accepted_split_filter_role": _accepted_split_filter_role(
                    family_status
                ),
                "family_row_count": int(family["family_row_count"]),
                "branch_recovery_count": int(family["branch_recovery_count"]),
                "partial_branch_recovery_count": int(
                    family["partial_branch_recovery_count"]
                ),
                "false_fragment_count": int(family["false_fragment_count"]),
                "unresolved_count": int(family["unresolved_count"]),
                "max_feature_homogeneity_gain_min": hom_max,
                "median_feature_homogeneity_gain_min": hom_median,
                "min_feature_homogeneity_gain_min": hom_min,
                "max_feature_branch_geometry_score": branch_max,
                "median_feature_branch_geometry_score": branch_median,
                "min_feature_branch_geometry_score": branch_min,
                "max_feature_child_contrast_norm": contrast_max,
                "median_feature_child_contrast_norm": contrast_median,
                "min_feature_child_contrast_norm": contrast_min,
                "max_feature_barycentric_geometry_score": bary_max,
                "median_feature_barycentric_geometry_score": bary_median,
                "min_min_sibling_p_value": sibling_min,
                "median_min_sibling_p_value": sibling_median,
                "max_min_sibling_p_value": sibling_max,
                "median_negative_log10_min_sibling_p_value": _negative_log10(
                    sibling_median
                ),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=ACCEPTED_SPLIT_FILTER_ROW_COLUMNS,
    )


def _accepted_filter_metric_record(
    rows: pd.DataFrame,
    *,
    metric: str,
) -> dict[str, object]:
    role = rows["accepted_split_filter_role"].astype(str)
    positive_mask = role.eq("clean_branch_family")
    negative_mask = role.isin(
        ["fragment_mixed_branch_family", "fragment_only_family"]
    )
    positive = _finite_array(rows.loc[positive_mask, metric])
    negative = _finite_array(rows.loc[negative_mask, metric])
    auc_high = _auc(positive, negative)
    auc_low = _auc(-positive, -negative)
    if math.isfinite(auc_low) and (
        not math.isfinite(auc_high) or auc_low > auc_high
    ):
        best_direction = "low"
        best_auc = auc_low
        separates = (
            positive.size > 0 and negative.size > 0 and positive.max() < negative.min()
        )
    else:
        best_direction = "high"
        best_auc = auc_high
        separates = (
            positive.size > 0 and negative.size > 0 and positive.min() > negative.max()
        )
    if positive.size == 0 or negative.size == 0:
        zero_status = "insufficient_positive_or_negative_family_support"
    elif separates:
        zero_status = f"zero_negative_separates_clean_family_{best_direction}"
    else:
        zero_status = "zero_negative_does_not_separate_clean_family"
    if zero_status.startswith("zero_negative_separates"):
        diagnostic_status = "accepted_split_filter_candidate_metric"
        production_action = "diagnostic_only_no_promotion"
    elif zero_status == "zero_negative_does_not_separate_clean_family":
        diagnostic_status = "accepted_split_filter_metric_overlaps_fragments"
        production_action = "fail_closed_until_fragment_filter_validated"
    else:
        diagnostic_status = "accepted_split_filter_support_insufficient"
        production_action = "fail_closed_until_evidence_present"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "metric": metric,
        "positive_family_count": int(positive_mask.sum()),
        "negative_family_count": int(negative_mask.sum()),
        "finite_positive_family_count": int(positive.size),
        "finite_negative_family_count": int(negative.size),
        "best_direction": best_direction,
        "best_auc": best_auc,
        "positive_min": float(np.min(positive)) if positive.size else math.nan,
        "positive_median": float(np.median(positive)) if positive.size else math.nan,
        "positive_max": float(np.max(positive)) if positive.size else math.nan,
        "negative_min": float(np.min(negative)) if negative.size else math.nan,
        "negative_median": float(np.median(negative)) if negative.size else math.nan,
        "negative_max": float(np.max(negative)) if negative.size else math.nan,
        "zero_negative_status": zero_status,
        "diagnostic_status": diagnostic_status,
        "production_action": production_action,
    }


def summarize_selected_candidate_accepted_split_filter(
    filter_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate accepted split family metrics against fragment-mixed families."""
    if filter_rows.empty:
        return pd.DataFrame(columns=ACCEPTED_SPLIT_FILTER_SUMMARY_COLUMNS)
    metrics = [
        "max_feature_homogeneity_gain_min",
        "median_feature_homogeneity_gain_min",
        "max_feature_branch_geometry_score",
        "median_feature_branch_geometry_score",
        "max_feature_child_contrast_norm",
        "median_feature_child_contrast_norm",
        "max_feature_barycentric_geometry_score",
        "median_feature_barycentric_geometry_score",
        "min_min_sibling_p_value",
        "median_min_sibling_p_value",
        "median_negative_log10_min_sibling_p_value",
    ]
    return pd.DataFrame.from_records(
        [
            _accepted_filter_metric_record(filter_rows, metric=metric)
            for metric in metrics
        ],
        columns=ACCEPTED_SPLIT_FILTER_SUMMARY_COLUMNS,
    )


def _direction_pass(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "low":
        return values <= threshold
    return values >= threshold


def _pair_filter_record(
    filter_rows: pd.DataFrame,
    *,
    metric_left: str,
    metric_right: str,
    direction_left: str,
    direction_right: str,
) -> dict[str, object]:
    role = filter_rows["accepted_split_filter_role"].astype(str)
    positive_mask = role.eq("clean_branch_family")
    negative_mask = role.isin(
        ["fragment_mixed_branch_family", "fragment_only_family"]
    )
    positive = filter_rows.loc[positive_mask, [metric_left, metric_right]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    negative = filter_rows.loc[negative_mask, [metric_left, metric_right]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    positive = positive.replace([np.inf, -np.inf], np.nan).dropna()
    negative = negative.replace([np.inf, -np.inf], np.nan).dropna()
    positive_values = positive.to_numpy(dtype=float)
    negative_values = negative.to_numpy(dtype=float)
    best_count = 0
    best_left = math.nan
    best_right = math.nan
    if positive_values.size and negative_values.size:
        left_thresholds = np.unique(positive_values[:, 0])
        right_thresholds = np.unique(positive_values[:, 1])
        for threshold_left in left_thresholds:
            pos_left = _direction_pass(
                positive_values[:, 0],
                float(threshold_left),
                direction_left,
            )
            neg_left = _direction_pass(
                negative_values[:, 0],
                float(threshold_left),
                direction_left,
            )
            for threshold_right in right_thresholds:
                neg_pass = neg_left & _direction_pass(
                    negative_values[:, 1],
                    float(threshold_right),
                    direction_right,
                )
                if bool(np.any(neg_pass)):
                    continue
                pos_pass = pos_left & _direction_pass(
                    positive_values[:, 1],
                    float(threshold_right),
                    direction_right,
                )
                pass_count = int(np.sum(pos_pass))
                if pass_count > best_count:
                    best_count = pass_count
                    best_left = float(threshold_left)
                    best_right = float(threshold_right)
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    finite_positive_count = int(positive_values.shape[0])
    finite_negative_count = int(negative_values.shape[0])
    recall = (
        float(best_count / finite_positive_count)
        if finite_positive_count
        else math.nan
    )
    if finite_positive_count == 0 or finite_negative_count == 0:
        zero_status = "insufficient_positive_or_negative_family_support"
        diagnostic_status = "accepted_split_pair_filter_support_insufficient"
        production_action = "fail_closed_until_evidence_present"
    elif best_count == finite_positive_count:
        zero_status = "pairwise_full_zero_negative_filter"
        diagnostic_status = "accepted_split_pair_filter_full_diagnostic"
        production_action = "diagnostic_only_no_promotion"
    elif best_count > 0:
        zero_status = "pairwise_partial_zero_negative_filter"
        diagnostic_status = "accepted_split_pair_filter_partial_only"
        production_action = "fail_closed_until_fragment_filter_validated"
    else:
        zero_status = "pairwise_no_zero_negative_filter"
        diagnostic_status = "accepted_split_pair_filter_not_found"
        production_action = "fail_closed_until_fragment_filter_validated"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "metric_left": metric_left,
        "metric_right": metric_right,
        "direction_left": direction_left,
        "direction_right": direction_right,
        "positive_family_count": positive_count,
        "negative_family_count": negative_count,
        "finite_positive_pair_count": finite_positive_count,
        "finite_negative_pair_count": finite_negative_count,
        "zero_negative_positive_pass_count": best_count,
        "zero_negative_positive_recall": recall,
        "threshold_left": best_left,
        "threshold_right": best_right,
        "zero_negative_status": zero_status,
        "diagnostic_status": diagnostic_status,
        "production_action": production_action,
    }


def summarize_selected_candidate_accepted_split_pair_filter(
    filter_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Search diagnostic pairwise filters for accepted split family fragments."""
    if filter_rows.empty:
        return pd.DataFrame(columns=ACCEPTED_SPLIT_PAIR_FILTER_COLUMNS)
    metrics = [
        "max_feature_homogeneity_gain_min",
        "median_feature_homogeneity_gain_min",
        "max_feature_branch_geometry_score",
        "median_feature_branch_geometry_score",
        "max_feature_child_contrast_norm",
        "median_feature_child_contrast_norm",
        "max_feature_barycentric_geometry_score",
        "median_feature_barycentric_geometry_score",
        "min_min_sibling_p_value",
        "median_min_sibling_p_value",
        "median_negative_log10_min_sibling_p_value",
    ]
    records: list[dict[str, object]] = []
    for metric_left, metric_right in combinations(metrics, 2):
        for direction_left, direction_right in product(["high", "low"], repeat=2):
            records.append(
                _pair_filter_record(
                    filter_rows,
                    metric_left=metric_left,
                    metric_right=metric_right,
                    direction_left=direction_left,
                    direction_right=direction_right,
                )
            )
    return pd.DataFrame.from_records(
        records,
        columns=ACCEPTED_SPLIT_PAIR_FILTER_COLUMNS,
    )


def _frontier_matrix(
    rows: pd.DataFrame,
    metric_specs: Sequence[tuple[str, str]],
) -> np.ndarray:
    if rows.empty:
        return np.empty((0, len(metric_specs)), dtype=float)
    columns: list[np.ndarray] = []
    for metric, direction in metric_specs:
        values = pd.to_numeric(rows[metric], errors="coerce").to_numpy(dtype=float)
        columns.append(values if direction == "high" else -values)
    return np.column_stack(columns)


def _frontier_record(
    filter_rows: pd.DataFrame,
    *,
    frontier_id: str,
    metric_specs: Sequence[tuple[str, str]],
) -> dict[str, object]:
    role = filter_rows["accepted_split_filter_role"].astype(str)
    positive = filter_rows.loc[role.eq("clean_branch_family")].copy()
    negative = filter_rows.loc[
        role.isin(["fragment_mixed_branch_family", "fragment_only_family"])
    ].copy()
    required_metrics = [metric for metric, _direction in metric_specs]
    positive = positive.dropna(subset=required_metrics)
    negative = negative.dropna(subset=required_metrics)
    positive_matrix = _frontier_matrix(positive, metric_specs)
    negative_matrix = _frontier_matrix(negative, metric_specs)
    dominated_count = 0
    if positive_matrix.size and negative_matrix.size:
        for positive_vector in positive_matrix:
            dominates = np.all(negative_matrix >= positive_vector, axis=1) & np.any(
                negative_matrix > positive_vector,
                axis=1,
            )
            dominated_count += int(bool(np.any(dominates)))
    finite_positive_count = int(positive_matrix.shape[0])
    finite_negative_count = int(negative_matrix.shape[0])
    frontier_count = int(finite_positive_count - dominated_count)
    dominated_fraction = (
        float(dominated_count / finite_positive_count)
        if finite_positive_count
        else math.nan
    )
    frontier_fraction = (
        float(frontier_count / finite_positive_count)
        if finite_positive_count
        else math.nan
    )
    if finite_positive_count == 0 or finite_negative_count == 0:
        status = "frontier_support_insufficient"
        next_step = "collect_accepted_split_frontier_support"
        action = "fail_closed_until_evidence_present"
    elif dominated_count == 0:
        status = "clean_families_not_negative_dominated_diagnostic"
        next_step = "test monotone frontier transfer before promotion"
        action = "diagnostic_only_no_promotion"
    elif dominated_count == finite_positive_count:
        status = "all_clean_families_negative_dominated"
        next_step = "derive non-monotone family law or keep fail-closed"
        action = "fail_closed_until_fragment_filter_validated"
    else:
        status = "some_clean_families_negative_dominated"
        next_step = "derive higher-order fragment law for dominated clean families"
        action = "fail_closed_until_fragment_filter_validated"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "frontier_id": frontier_id,
        "metric_spec": ";".join(
            f"{metric}:{direction}" for metric, direction in metric_specs
        ),
        "metric_count": int(len(metric_specs)),
        "positive_family_count": int(role.eq("clean_branch_family").sum()),
        "negative_family_count": int(
            role.isin(["fragment_mixed_branch_family", "fragment_only_family"]).sum()
        ),
        "finite_positive_family_count": finite_positive_count,
        "finite_negative_family_count": finite_negative_count,
        "negative_dominated_positive_count": dominated_count,
        "negative_dominated_positive_fraction": dominated_fraction,
        "frontier_positive_count": frontier_count,
        "frontier_positive_fraction": frontier_fraction,
        "diagnostic_status": status,
        "next_required_step": next_step,
        "production_action": action,
    }


def summarize_selected_candidate_accepted_split_frontiers(
    filter_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize monotone multi-metric frontiers for accepted split families."""
    if filter_rows.empty:
        return pd.DataFrame(columns=ACCEPTED_SPLIT_FRONTIER_COLUMNS)
    return pd.DataFrame.from_records(
        [
            _frontier_record(
                filter_rows,
                frontier_id=frontier_id,
                metric_specs=metric_specs,
            )
            for frontier_id, metric_specs in ACCEPTED_SPLIT_FRONTIER_SPECS
        ],
        columns=ACCEPTED_SPLIT_FRONTIER_COLUMNS,
    )


def _logistic(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if value >= 0:
        exp_neg = math.exp(-value)
        return float(1.0 / (1.0 + exp_neg))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _frontier_margin_to_negative(
    vector: np.ndarray,
    negative_matrix: np.ndarray,
) -> float:
    """Return the continuous non-domination margin against negative controls.

    With all metrics oriented so larger is better, a positive margin means that
    for every negative control there is at least one coordinate where this
    family is strictly better. A non-positive margin means at least one
    negative control dominates or ties the family.
    """
    if vector.size == 0 or negative_matrix.size == 0:
        return math.nan
    finite_negatives = negative_matrix[
        np.isfinite(negative_matrix).all(axis=1),
    ]
    if finite_negatives.size == 0 or not np.isfinite(vector).all():
        return math.nan
    pair_margins = np.max(vector - finite_negatives, axis=1)
    return float(np.min(pair_margins)) if pair_margins.size else math.nan


def _frontier_margin_scale(margins: Sequence[float]) -> float:
    finite = np.asarray(
        [abs(float(value)) for value in margins if math.isfinite(float(value))],
        dtype=float,
    )
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 1.0
    scale = float(np.median(finite))
    return scale if math.isfinite(scale) and scale > 0.0 else 1.0


def _support_status(
    *,
    finite_positive_count: int,
    finite_negative_count: int,
    min_support: int,
) -> str:
    positive_ok = finite_positive_count >= min_support
    negative_ok = finite_negative_count >= min_support
    if positive_ok and negative_ok:
        return "frontier_law_support_sufficient"
    if not positive_ok and not negative_ok:
        return "frontier_law_positive_and_negative_support_thin"
    if not positive_ok:
        return "frontier_law_positive_support_thin"
    return "frontier_law_negative_support_thin"


def _beta_lower_credible_approx(
    *,
    alpha: float,
    beta: float,
    z_value: float = 1.6448536269514722,
) -> float:
    total = alpha + beta
    if total <= 0.0:
        return math.nan
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    lower = mean - z_value * math.sqrt(max(variance, 0.0))
    return float(min(1.0, max(0.0, lower)))


def _frontier_law_row_status(
    *,
    role: str,
    margin: float,
    support_status: str,
    min_margin: float,
) -> tuple[str, str]:
    support_ok = support_status == "frontier_law_support_sufficient"
    if role == "clean_branch_family":
        if not math.isfinite(margin):
            return (
                "clean_family_frontier_margin_unavailable",
                "fail_closed_until_evidence_present",
            )
        if margin <= 0.0:
            return (
                "clean_family_negative_dominated_fail_closed",
                "fail_closed_until_fragment_filter_validated",
            )
        if support_ok:
            if margin <= min_margin:
                return (
                    "clean_family_frontier_margin_thin_diagnostic",
                    "diagnostic_only_no_promotion",
                )
            return (
                "clean_family_frontier_supported_diagnostic",
                "diagnostic_only_no_promotion",
            )
        return (
            "clean_family_frontier_support_insufficient_fail_closed",
            "fail_closed_until_evidence_present",
        )
    if role in {"fragment_mixed_branch_family", "fragment_only_family"}:
        return (
            "fragment_family_frontier_control",
            "retain_fail_closed_fragment_suppression",
        )
    return (
        "frontier_law_truth_role_not_scored",
        "diagnostic_only_no_promotion",
    )


def build_selected_candidate_frontier_law_rows(
    filter_rows: pd.DataFrame,
    *,
    min_support: int = 3,
    min_margin: float = 1e-8,
) -> pd.DataFrame:
    """Build posterior-style selected-family frontier diagnostic rows.

    This is intentionally not a production calibration rule. It uses labeled
    diagnostic families to ask whether each clean accepted split has a positive
    continuous margin against fragment controls under predeclared frontiers.
    """
    if filter_rows.empty:
        return pd.DataFrame(columns=SELECTED_FAMILY_FRONTIER_LAW_ROW_COLUMNS)
    records: list[dict[str, object]] = []
    for frontier_id, metric_specs in ACCEPTED_SPLIT_FRONTIER_SPECS:
        required_metrics = [metric for metric, _direction in metric_specs]
        scored = filter_rows.dropna(subset=required_metrics).copy()
        scored_role = scored["accepted_split_filter_role"].astype(str)
        positive = scored.loc[scored_role.eq("clean_branch_family")].copy()
        negative = scored.loc[
            scored_role.isin(["fragment_mixed_branch_family", "fragment_only_family"])
        ].copy()
        negative_matrix = _frontier_matrix(negative, metric_specs)
        positive_matrix = _frontier_matrix(positive, metric_specs)
        clean_margins = [
            _frontier_margin_to_negative(vector, negative_matrix)
            for vector in positive_matrix
        ]
        margin_scale = _frontier_margin_scale(clean_margins)
        support = _support_status(
            finite_positive_count=int(positive_matrix.shape[0]),
            finite_negative_count=int(negative_matrix.shape[0]),
            min_support=int(min_support),
        )
        prior_log_odds = math.log(
            (float(positive_matrix.shape[0]) + 1.0)
            / (float(negative_matrix.shape[0]) + 1.0)
        )
        for _, row in scored.iterrows():
            row_role = str(row["accepted_split_filter_role"])
            vector = _frontier_matrix(pd.DataFrame([row]), metric_specs)[0]
            margin = _frontier_margin_to_negative(vector, negative_matrix)
            scaled_margin = 0.0
            if math.isfinite(margin):
                scaled_margin = margin / margin_scale
            log_odds = prior_log_odds + scaled_margin
            law_status, action = _frontier_law_row_status(
                role=row_role,
                margin=margin,
                support_status=support,
                min_margin=float(min_margin),
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "frontier_id": frontier_id,
                    "case_id": str(row["case_id"]),
                    "data_role": str(row["data_role"]),
                    "replicate": int(row["replicate"]),
                    "accepted_split_filter_role": row_role,
                    "family_likelihood_status": str(row["family_likelihood_status"]),
                    "frontier_margin_to_negative": margin,
                    "frontier_margin_scale": margin_scale,
                    "frontier_non_dominated": bool(
                        math.isfinite(margin) and margin > 0.0
                    ),
                    "support_status": support,
                    "support_prior_log_odds": prior_log_odds,
                    "posterior_style_log_odds": log_odds,
                    "posterior_style_probability": _logistic(log_odds),
                    "selected_family_frontier_law_status": law_status,
                    "production_action": action,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=SELECTED_FAMILY_FRONTIER_LAW_ROW_COLUMNS,
    )


def _frontier_law_summary_status(
    *,
    finite_positive_count: int,
    finite_negative_count: int,
    clean_dominated_count: int,
    clean_margin_min: float,
    min_support: int,
    min_margin: float,
) -> tuple[str, str, str]:
    support = _support_status(
        finite_positive_count=finite_positive_count,
        finite_negative_count=finite_negative_count,
        min_support=min_support,
    )
    if support != "frontier_law_support_sufficient":
        return (
            "selected_family_frontier_law_support_insufficient",
            "collect selected-family frontier controls before calibration",
            "fail_closed_until_evidence_present",
        )
    if clean_dominated_count:
        return (
            "selected_family_frontier_law_leaky",
            "derive higher-order selected-family law or keep fail-closed",
            "fail_closed_until_fragment_filter_validated",
        )
    if math.isfinite(clean_margin_min) and clean_margin_min <= min_margin:
        return (
            "selected_family_frontier_law_margin_thin_diagnostic",
            "validate numerical and benchmark stability before promotion",
            "diagnostic_only_no_promotion",
        )
    return (
        "selected_family_frontier_law_candidate_diagnostic",
        "validate frontier law transfer on held-out benchmark families",
        "diagnostic_only_no_promotion",
    )


def summarize_selected_candidate_frontier_law(
    law_rows: pd.DataFrame,
    *,
    min_support: int = 3,
    min_margin: float = 1e-8,
) -> pd.DataFrame:
    """Summarize the selected-family frontier law diagnostics."""
    if law_rows.empty:
        return pd.DataFrame(columns=SELECTED_FAMILY_FRONTIER_LAW_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    spec_by_id = {
        frontier_id: metric_specs
        for frontier_id, metric_specs in ACCEPTED_SPLIT_FRONTIER_SPECS
    }
    for frontier_id, group in law_rows.groupby("frontier_id", sort=True):
        role = group["accepted_split_filter_role"].astype(str)
        positive = group.loc[role.eq("clean_branch_family")]
        negative = group.loc[
            role.isin(["fragment_mixed_branch_family", "fragment_only_family"])
        ]
        positive_margins = _finite_array(positive["frontier_margin_to_negative"])
        non_dominated = positive["frontier_non_dominated"].astype(bool)
        clean_non_dominated_count = int(non_dominated.sum())
        finite_positive_count = int(positive_margins.size)
        finite_negative_count = int(
            np.isfinite(
                pd.to_numeric(
                    negative["frontier_margin_to_negative"],
                    errors="coerce",
                ).to_numpy(dtype=float)
            ).sum()
        )
        clean_dominated_count = int(finite_positive_count - clean_non_dominated_count)
        clean_margin_min = (
            float(np.min(positive_margins)) if positive_margins.size else math.nan
        )
        posterior_alpha = 1.0 + float(clean_non_dominated_count)
        posterior_beta = 1.0 + float(clean_dominated_count)
        status, next_step, action = _frontier_law_summary_status(
            finite_positive_count=finite_positive_count,
            finite_negative_count=finite_negative_count,
            clean_dominated_count=clean_dominated_count,
            clean_margin_min=clean_margin_min,
            min_support=int(min_support),
            min_margin=float(min_margin),
        )
        metric_specs = spec_by_id[str(frontier_id)]
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "frontier_id": str(frontier_id),
                "metric_spec": ";".join(
                    f"{metric}:{direction}" for metric, direction in metric_specs
                ),
                "metric_count": int(len(metric_specs)),
                "positive_family_count": int(role.eq("clean_branch_family").sum()),
                "negative_family_count": int(
                    role.isin(
                        ["fragment_mixed_branch_family", "fragment_only_family"]
                    ).sum()
                ),
                "finite_positive_family_count": finite_positive_count,
                "finite_negative_family_count": finite_negative_count,
                "clean_non_dominated_count": clean_non_dominated_count,
                "clean_dominated_count": clean_dominated_count,
                "clean_non_dominated_fraction": (
                    float(clean_non_dominated_count / finite_positive_count)
                    if finite_positive_count
                    else math.nan
                ),
                "clean_margin_min": clean_margin_min,
                "clean_margin_median": (
                    float(np.median(positive_margins))
                    if positive_margins.size
                    else math.nan
                ),
                "clean_margin_max": (
                    float(np.max(positive_margins))
                    if positive_margins.size
                    else math.nan
                ),
                "beta_prior_alpha": 1.0,
                "beta_prior_beta": 1.0,
                "beta_posterior_alpha": posterior_alpha,
                "beta_posterior_beta": posterior_beta,
                "posterior_clean_frontier_mean": posterior_alpha
                / (posterior_alpha + posterior_beta),
                "posterior_clean_frontier_lower90": _beta_lower_credible_approx(
                    alpha=posterior_alpha,
                    beta=posterior_beta,
                ),
                "diagnostic_status": status,
                "next_required_step": next_step,
                "production_action": action,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SELECTED_FAMILY_FRONTIER_LAW_SUMMARY_COLUMNS,
    )


def _frontier_ablation_status(
    *,
    finite_positive_count: int,
    finite_negative_count: int,
    clean_dominated_count: int,
    clean_margin_min: float,
    min_support: int,
) -> tuple[str, str, str]:
    support = _support_status(
        finite_positive_count=finite_positive_count,
        finite_negative_count=finite_negative_count,
        min_support=min_support,
    )
    if support != "frontier_law_support_sufficient":
        return (
            "frontier_ablation_support_insufficient",
            "collect fragment-control support for frontier ablation",
            "fail_closed_until_evidence_present",
        )
    if clean_dominated_count:
        return (
            "frontier_ablation_leaky",
            "identify missing conditioning coordinate or keep fail-closed",
            "fail_closed_until_fragment_filter_validated",
        )
    if math.isfinite(clean_margin_min) and clean_margin_min <= 1e-8:
        return (
            "frontier_ablation_zero_leakage_but_margin_thin",
            "validate numerical and benchmark stability before promotion",
            "diagnostic_only_no_promotion",
        )
    return (
        "frontier_ablation_zero_leakage_with_margin",
        "validate frontier transfer before promotion",
        "diagnostic_only_no_promotion",
    )


def _frontier_ablation_record(
    filter_rows: pd.DataFrame,
    *,
    frontier_id: str,
    ablation_id: str,
    removed_metric: str,
    metric_specs: Sequence[tuple[str, str]],
    min_support: int,
) -> dict[str, object]:
    role = filter_rows["accepted_split_filter_role"].astype(str)
    positive = filter_rows.loc[role.eq("clean_branch_family")].copy()
    negative = filter_rows.loc[
        role.isin(["fragment_mixed_branch_family", "fragment_only_family"])
    ].copy()
    required_metrics = [metric for metric, _direction in metric_specs]
    positive = positive.dropna(subset=required_metrics)
    negative = negative.dropna(subset=required_metrics)
    positive_matrix = _frontier_matrix(positive, metric_specs)
    negative_matrix = _frontier_matrix(negative, metric_specs)
    margins = np.asarray(
        [
            _frontier_margin_to_negative(vector, negative_matrix)
            for vector in positive_matrix
        ],
        dtype=float,
    )
    finite_margins = margins[np.isfinite(margins)]
    non_dominated = finite_margins > 0.0
    clean_non_dominated_count = int(np.sum(non_dominated))
    finite_positive_count = int(finite_margins.size)
    finite_negative_count = int(negative_matrix.shape[0])
    clean_dominated_count = int(finite_positive_count - clean_non_dominated_count)
    clean_margin_min = (
        float(np.min(finite_margins)) if finite_margins.size else math.nan
    )
    status, next_step, action = _frontier_ablation_status(
        finite_positive_count=finite_positive_count,
        finite_negative_count=finite_negative_count,
        clean_dominated_count=clean_dominated_count,
        clean_margin_min=clean_margin_min,
        min_support=int(min_support),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "frontier_id": frontier_id,
        "ablation_id": ablation_id,
        "removed_metric": removed_metric,
        "metric_spec": ";".join(
            f"{metric}:{direction}" for metric, direction in metric_specs
        ),
        "metric_count": int(len(metric_specs)),
        "finite_positive_family_count": finite_positive_count,
        "finite_negative_family_count": finite_negative_count,
        "clean_non_dominated_count": clean_non_dominated_count,
        "clean_dominated_count": clean_dominated_count,
        "clean_non_dominated_fraction": (
            float(clean_non_dominated_count / finite_positive_count)
            if finite_positive_count
            else math.nan
        ),
        "clean_margin_min": clean_margin_min,
        "clean_margin_p10": (
            float(np.quantile(finite_margins, 0.10))
            if finite_margins.size
            else math.nan
        ),
        "clean_margin_median": (
            float(np.median(finite_margins)) if finite_margins.size else math.nan
        ),
        "clean_margin_max": (
            float(np.max(finite_margins)) if finite_margins.size else math.nan
        ),
        "fragility_status": status,
        "next_required_step": next_step,
        "production_action": action,
    }


def summarize_selected_candidate_frontier_ablation(
    filter_rows: pd.DataFrame,
    *,
    min_support: int = 3,
) -> pd.DataFrame:
    """Report how frontier laws change when each coordinate is removed."""
    if filter_rows.empty:
        return pd.DataFrame(columns=SELECTED_FAMILY_FRONTIER_ABLATION_COLUMNS)
    records: list[dict[str, object]] = []
    for frontier_id, metric_specs in ACCEPTED_SPLIT_FRONTIER_SPECS:
        records.append(
            _frontier_ablation_record(
                filter_rows,
                frontier_id=frontier_id,
                ablation_id="all_metrics",
                removed_metric="none",
                metric_specs=metric_specs,
                min_support=int(min_support),
            )
        )
        if len(metric_specs) <= 1:
            continue
        for remove_index, (removed_metric, _direction) in enumerate(metric_specs):
            ablated_specs = tuple(
                spec for index, spec in enumerate(metric_specs) if index != remove_index
            )
            records.append(
                _frontier_ablation_record(
                    filter_rows,
                    frontier_id=frontier_id,
                    ablation_id=f"remove_{removed_metric}",
                    removed_metric=removed_metric,
                    metric_specs=ablated_specs,
                    min_support=int(min_support),
                )
            )
    return pd.DataFrame.from_records(
        records,
        columns=SELECTED_FAMILY_FRONTIER_ABLATION_COLUMNS,
    )


def _frontier_margin_status(margin: float, *, min_margin: float) -> str:
    if not math.isfinite(margin):
        return "frontier_margin_unavailable"
    if margin <= 0.0:
        return "frontier_margin_dominated_or_tied"
    if margin <= min_margin:
        return "frontier_margin_thin"
    return "frontier_margin_positive"


def build_selected_candidate_frontier_witness_rows(
    filter_rows: pd.DataFrame,
    *,
    min_margin: float = 1e-8,
) -> pd.DataFrame:
    """Record nearest fragment witnesses for clean selected-family frontiers."""
    if filter_rows.empty:
        return pd.DataFrame(columns=SELECTED_FAMILY_FRONTIER_WITNESS_COLUMNS)
    records: list[dict[str, object]] = []
    all_gap_metrics = {
        "max_feature_homogeneity_gain_min": "gap_max_feature_homogeneity_gain_min",
        "median_feature_homogeneity_gain_min": "gap_median_feature_homogeneity_gain_min",
        "max_feature_branch_geometry_score": "gap_max_feature_branch_geometry_score",
        "median_feature_branch_geometry_score": (
            "gap_median_feature_branch_geometry_score"
        ),
        "max_feature_child_contrast_norm": "gap_max_feature_child_contrast_norm",
        "max_feature_barycentric_geometry_score": (
            "gap_max_feature_barycentric_geometry_score"
        ),
        "median_min_sibling_p_value": "gap_median_min_sibling_p_value",
        "median_negative_log10_min_sibling_p_value": (
            "gap_median_negative_log10_min_sibling_p_value"
        ),
    }
    role = filter_rows["accepted_split_filter_role"].astype(str)
    for frontier_id, metric_specs in ACCEPTED_SPLIT_FRONTIER_SPECS:
        required_metrics = [metric for metric, _direction in metric_specs]
        positive = (
            filter_rows.loc[role.eq("clean_branch_family")]
            .dropna(subset=required_metrics)
            .copy()
        )
        negative = (
            filter_rows.loc[
                role.isin(["fragment_mixed_branch_family", "fragment_only_family"])
            ]
            .dropna(subset=required_metrics)
            .copy()
        )
        if positive.empty or negative.empty:
            continue
        positive_matrix = _frontier_matrix(positive, metric_specs)
        negative_matrix = _frontier_matrix(negative, metric_specs)
        for positive_index, positive_vector in enumerate(positive_matrix):
            gaps = positive_vector[None, :] - negative_matrix
            pair_margins = np.max(gaps, axis=1)
            witness_position = int(np.argmin(pair_margins))
            witness_gaps = gaps[witness_position]
            active_position = int(np.argmax(witness_gaps))
            active_metric, active_direction = metric_specs[active_position]
            positive_row = positive.iloc[positive_index]
            witness_row = negative.iloc[witness_position]
            record = {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "frontier_id": frontier_id,
                "metric_spec": ";".join(
                    f"{metric}:{direction}" for metric, direction in metric_specs
                ),
                "metric_count": int(len(metric_specs)),
                "case_id": str(positive_row["case_id"]),
                "data_role": str(positive_row["data_role"]),
                "replicate": int(positive_row["replicate"]),
                "accepted_split_filter_role": str(
                    positive_row["accepted_split_filter_role"]
                ),
                "witness_case_id": str(witness_row["case_id"]),
                "witness_data_role": str(witness_row["data_role"]),
                "witness_replicate": int(witness_row["replicate"]),
                "witness_accepted_split_filter_role": str(
                    witness_row["accepted_split_filter_role"]
                ),
                "frontier_margin_to_witness": float(pair_margins[witness_position]),
                "frontier_margin_status": _frontier_margin_status(
                    float(pair_margins[witness_position]),
                    min_margin=float(min_margin),
                ),
                "active_metric": active_metric,
                "active_metric_direction": active_direction,
                "active_metric_gap": float(witness_gaps[active_position]),
                "active_metric_family_value": _finite_float(
                    positive_row[active_metric]
                ),
                "active_metric_witness_value": _finite_float(
                    witness_row[active_metric]
                ),
                **{column: math.nan for column in FRONTIER_WITNESS_GAP_COLUMNS},
            }
            for metric_index, (metric, _direction) in enumerate(metric_specs):
                gap_column = all_gap_metrics[metric]
                record[gap_column] = float(witness_gaps[metric_index])
            records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=SELECTED_FAMILY_FRONTIER_WITNESS_COLUMNS,
    )


def _sibling_rescue_status(
    *,
    active_metric: str,
    margin_status: str,
    structural_gap_max: float,
) -> tuple[str, str]:
    sibling_active = active_metric in {
        "median_min_sibling_p_value",
        "median_negative_log10_min_sibling_p_value",
    }
    structural_support = math.isfinite(structural_gap_max) and structural_gap_max > 0.0
    if not sibling_active:
        return (
            "structural_coordinate_rescue",
            "diagnostic_only_no_promotion",
        )
    if margin_status == "frontier_margin_dominated_or_tied":
        return (
            "sibling_rescue_dominated_or_tied",
            "fail_closed_until_fragment_filter_validated",
        )
    if structural_support:
        return (
            "sibling_rescue_with_structural_support_diagnostic",
            "diagnostic_only_no_promotion",
        )
    if margin_status == "frontier_margin_thin":
        return (
            "sibling_only_thin_no_structural_support",
            "fail_closed_until_sibling_rescue_law_validated",
        )
    return (
        "sibling_only_no_structural_support",
        "fail_closed_until_sibling_rescue_law_validated",
    )


def build_selected_candidate_sibling_rescue_audit_rows(
    witness_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Audit whether sibling-p frontier rescues have structural support."""
    if witness_rows.empty:
        return pd.DataFrame(columns=SIBLING_RESCUE_AUDIT_ROW_COLUMNS)
    structural_gap_columns = [
        "gap_max_feature_homogeneity_gain_min",
        "gap_median_feature_homogeneity_gain_min",
        "gap_max_feature_branch_geometry_score",
        "gap_median_feature_branch_geometry_score",
        "gap_max_feature_child_contrast_norm",
        "gap_max_feature_barycentric_geometry_score",
    ]
    records: list[dict[str, object]] = []
    for _, row in witness_rows.iterrows():
        structural_values = np.asarray(
            [
                _finite_float(row[column])
                for column in structural_gap_columns
                if column in row.index and math.isfinite(_finite_float(row[column]))
            ],
            dtype=float,
        )
        structural_gap_max = (
            float(np.max(structural_values)) if structural_values.size else math.nan
        )
        structural_positive = int(np.sum(structural_values > 0.0))
        structural_negative = int(np.sum(structural_values < 0.0))
        structural_tie = int(np.sum(structural_values == 0.0))
        status, action = _sibling_rescue_status(
            active_metric=str(row["active_metric"]),
            margin_status=str(row["frontier_margin_status"]),
            structural_gap_max=structural_gap_max,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "frontier_id": str(row["frontier_id"]),
                "case_id": str(row["case_id"]),
                "replicate": int(row["replicate"]),
                "witness_case_id": str(row["witness_case_id"]),
                "witness_replicate": int(row["witness_replicate"]),
                "frontier_margin_status": str(row["frontier_margin_status"]),
                "active_metric": str(row["active_metric"]),
                "frontier_margin_to_witness": _finite_float(
                    row["frontier_margin_to_witness"]
                ),
                "sibling_gap": _finite_float(
                    row.get("gap_median_min_sibling_p_value", math.nan)
                ),
                "log_sibling_gap": _finite_float(
                    row.get(
                        "gap_median_negative_log10_min_sibling_p_value",
                        math.nan,
                    )
                ),
                "structural_gap_max": structural_gap_max,
                "structural_positive_gap_count": structural_positive,
                "structural_negative_gap_count": structural_negative,
                "structural_tie_gap_count": structural_tie,
                "sibling_rescue_status": status,
                "production_action": action,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SIBLING_RESCUE_AUDIT_ROW_COLUMNS,
    )


def _sibling_rescue_summary_status(group: pd.DataFrame) -> tuple[str, str, str]:
    statuses = group["sibling_rescue_status"].astype(str)
    if statuses.eq("sibling_only_thin_no_structural_support").any():
        return (
            "sibling_only_thin_rescue_requires_law",
            "derive structural sibling-rescue law or keep fail-closed",
            "fail_closed_until_sibling_rescue_law_validated",
        )
    if statuses.eq("sibling_only_no_structural_support").any():
        return (
            "sibling_only_rescue_requires_law",
            "derive structural sibling-rescue law or keep fail-closed",
            "fail_closed_until_sibling_rescue_law_validated",
        )
    if statuses.eq("sibling_rescue_with_structural_support_diagnostic").any():
        return (
            "sibling_rescue_has_structural_support_diagnostic",
            "validate structural sibling-rescue transfer",
            "diagnostic_only_no_promotion",
        )
    return (
        "frontier_rescue_structural_coordinates_only",
        "validate structural frontier transfer",
        "diagnostic_only_no_promotion",
    )


def summarize_selected_candidate_sibling_rescue_audit(
    audit_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize sibling-p rescue usage by frontier."""
    if audit_rows.empty:
        return pd.DataFrame(columns=SIBLING_RESCUE_AUDIT_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for frontier_id, group in audit_rows.groupby("frontier_id", sort=True):
        statuses = group["sibling_rescue_status"].astype(str)
        active = group["active_metric"].astype(str)
        sibling_active = active.isin(
            [
                "median_min_sibling_p_value",
                "median_negative_log10_min_sibling_p_value",
            ]
        )
        structural_support = pd.to_numeric(
            group["structural_gap_max"],
            errors="coerce",
        ).gt(0.0)
        summary_status, next_step, action = _sibling_rescue_summary_status(group)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "frontier_id": str(frontier_id),
                "row_count": int(group.shape[0]),
                "sibling_active_count": int(sibling_active.sum()),
                "sibling_margin_thin_count": int(
                    (
                        sibling_active
                        & group["frontier_margin_status"].astype(str).eq(
                            "frontier_margin_thin"
                        )
                    ).sum()
                ),
                "sibling_only_no_structural_support_count": int(
                    statuses.eq("sibling_only_no_structural_support").sum()
                ),
                "sibling_only_thin_no_structural_support_count": int(
                    statuses.eq("sibling_only_thin_no_structural_support").sum()
                ),
                "structural_active_count": int((~sibling_active).sum()),
                "structural_support_available_count": int(structural_support.sum()),
                "summary_status": summary_status,
                "next_required_step": next_step,
                "production_action": action,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SIBLING_RESCUE_AUDIT_SUMMARY_COLUMNS,
    )


def _sibling_rescue_guard_status(
    *,
    row_count: int,
    blocked_count: int,
) -> tuple[str, str, str]:
    if row_count == 0:
        return (
            "sibling_rescue_guard_support_missing",
            "collect sibling-rescue witness rows",
            "fail_closed_until_evidence_present",
        )
    if blocked_count == 0:
        return (
            "sibling_rescue_guard_no_unsupported_rescues",
            "validate structural frontier transfer",
            "diagnostic_only_no_promotion",
        )
    if blocked_count == row_count:
        return (
            "sibling_rescue_guard_blocks_all_candidates",
            "derive sibling-rescue law before using this frontier",
            "fail_closed_until_sibling_rescue_law_validated",
        )
    return (
        "sibling_rescue_guard_blocks_unsupported_rescues",
        "quantify clean-family loss or derive sibling-rescue law",
        "fail_closed_until_sibling_rescue_law_validated",
    )


def summarize_selected_candidate_sibling_rescue_guard(
    audit_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize a conservative guard that blocks unsupported sibling rescues."""
    if audit_rows.empty:
        return pd.DataFrame(columns=SIBLING_RESCUE_GUARD_SUMMARY_COLUMNS)
    blocked_statuses = {
        "sibling_only_thin_no_structural_support",
        "sibling_only_no_structural_support",
        "sibling_rescue_dominated_or_tied",
    }
    records: list[dict[str, object]] = []
    for frontier_id, group in audit_rows.groupby("frontier_id", sort=True):
        statuses = group["sibling_rescue_status"].astype(str)
        blocked = statuses.isin(blocked_statuses)
        row_count = int(group.shape[0])
        blocked_count = int(blocked.sum())
        retained_count = int(row_count - blocked_count)
        guard_status, next_step, action = _sibling_rescue_guard_status(
            row_count=row_count,
            blocked_count=blocked_count,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "frontier_id": str(frontier_id),
                "row_count": row_count,
                "retained_after_guard_count": retained_count,
                "blocked_by_guard_count": blocked_count,
                "retained_after_guard_fraction": (
                    float(retained_count / row_count) if row_count else math.nan
                ),
                "blocked_sibling_only_count": int(
                    statuses.eq("sibling_only_no_structural_support").sum()
                ),
                "blocked_sibling_only_thin_count": int(
                    statuses.eq("sibling_only_thin_no_structural_support").sum()
                ),
                "blocked_sibling_dominated_or_tied_count": int(
                    statuses.eq("sibling_rescue_dominated_or_tied").sum()
                ),
                "guard_status": guard_status,
                "next_required_step": next_step,
                "production_action": action,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SIBLING_RESCUE_GUARD_SUMMARY_COLUMNS,
    )


def run_selected_candidate_truth_law_panel(
    config: SelectedCandidateTruthLawPanelConfig,
    *,
    truth_label_maps: TruthLabelMaps | None = None,
    feature_data_maps: FeatureDataMaps | None = None,
) -> dict[str, Path]:
    """Run the selected-candidate truth diagnostic and write output files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    generated_support_outputs: dict[str, Path] = {}
    if config.use_generated_support_fixture:
        support_node_rows, gene_assignments = build_generated_branch_positive_support_fixture(
            suite=config.truth_suite,
        )
        candidate_rows = build_generated_support_candidate_rows(support_node_rows)
        traversal_rows = _generated_support_traversal_rows(support_node_rows)
        candidate_rows.to_csv(
            config.generated_support_candidate_rows_path,
            index=False,
        )
        gene_assignments.to_csv(
            config.generated_support_gene_assignments_path,
            index=False,
        )
        generated_support_outputs = {
            "generated_support_candidate_rows": (
                config.generated_support_candidate_rows_path
            ),
            "generated_support_gene_assignments": (
                config.generated_support_gene_assignments_path
            ),
        }
    else:
        if config.candidate_rows_path is None or config.gene_assignments_path is None:
            raise ValueError(
                "candidate_rows_path and gene_assignments_path are required "
                "unless use_generated_support_fixture is enabled"
            )
        candidate_rows = pd.read_csv(config.candidate_rows_path, low_memory=False)
        gene_assignments = pd.read_csv(config.gene_assignments_path, low_memory=False)
        traversal_rows = (
            None
            if config.traversal_rows_path is None
            else pd.read_csv(config.traversal_rows_path, low_memory=False)
        )
    rows = build_selected_candidate_truth_law_rows(
        candidate_rows,
        gene_assignments,
        traversal_rows,
        truth_label_maps=truth_label_maps,
        feature_data_maps=feature_data_maps,
        suite=config.truth_suite,
        branch_ari_floor=config.branch_ari_floor,
        partial_ari_floor=config.partial_ari_floor,
        child_purity_floor=config.child_purity_floor,
        top_k=config.top_k,
        max_pairwise_samples=config.max_pairwise_samples,
    )
    summary = summarize_selected_candidate_truth_law(rows)
    state_summary = summarize_selected_candidate_truth_law_states(rows)
    feature_metrics = summarize_selected_candidate_feature_metrics(rows)
    feature_state_metrics = summarize_selected_candidate_feature_metrics_by_state(rows)
    context_state_metrics = summarize_selected_candidate_context_metrics_by_state(rows)
    family_likelihood_rows = summarize_selected_candidate_family_likelihood_rows(rows)
    family_likelihood_summary = summarize_selected_candidate_family_likelihood(
        family_likelihood_rows
    )
    collision_law_components = summarize_selected_candidate_collision_law_components(
        family_likelihood_rows
    )
    accepted_split_filter_rows = build_selected_candidate_accepted_split_filter_rows(
        rows,
        family_likelihood_rows,
    )
    accepted_split_filter_summary = summarize_selected_candidate_accepted_split_filter(
        accepted_split_filter_rows
    )
    accepted_split_pair_filter_summary = (
        summarize_selected_candidate_accepted_split_pair_filter(
            accepted_split_filter_rows
        )
    )
    accepted_split_frontier_summary = (
        summarize_selected_candidate_accepted_split_frontiers(
            accepted_split_filter_rows
        )
    )
    selected_family_frontier_law_rows = build_selected_candidate_frontier_law_rows(
        accepted_split_filter_rows,
        min_support=config.min_frontier_law_support,
        min_margin=config.min_frontier_law_margin,
    )
    selected_family_frontier_law_summary = (
        summarize_selected_candidate_frontier_law(
            selected_family_frontier_law_rows,
            min_support=config.min_frontier_law_support,
            min_margin=config.min_frontier_law_margin,
        )
    )
    selected_family_frontier_ablation_summary = (
        summarize_selected_candidate_frontier_ablation(
            accepted_split_filter_rows,
            min_support=config.min_frontier_law_support,
        )
    )
    selected_family_frontier_witness_rows = (
        build_selected_candidate_frontier_witness_rows(
            accepted_split_filter_rows,
            min_margin=config.min_frontier_law_margin,
        )
    )
    sibling_rescue_audit_rows = build_selected_candidate_sibling_rescue_audit_rows(
        selected_family_frontier_witness_rows
    )
    sibling_rescue_audit_summary = summarize_selected_candidate_sibling_rescue_audit(
        sibling_rescue_audit_rows
    )
    sibling_rescue_guard_summary = summarize_selected_candidate_sibling_rescue_guard(
        sibling_rescue_audit_rows
    )

    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    state_summary.to_csv(config.state_summary_path, index=False)
    feature_metrics.to_csv(config.feature_metric_summary_path, index=False)
    feature_state_metrics.to_csv(
        config.feature_metric_state_summary_path,
        index=False,
    )
    context_state_metrics.to_csv(
        config.context_metric_state_summary_path,
        index=False,
    )
    family_likelihood_rows.to_csv(
        config.family_likelihood_rows_path,
        index=False,
    )
    family_likelihood_summary.to_csv(
        config.family_likelihood_summary_path,
        index=False,
    )
    collision_law_components.to_csv(
        config.collision_law_components_path,
        index=False,
    )
    accepted_split_filter_rows.to_csv(
        config.accepted_split_filter_rows_path,
        index=False,
    )
    accepted_split_filter_summary.to_csv(
        config.accepted_split_filter_summary_path,
        index=False,
    )
    accepted_split_pair_filter_summary.to_csv(
        config.accepted_split_pair_filter_summary_path,
        index=False,
    )
    accepted_split_frontier_summary.to_csv(
        config.accepted_split_frontier_summary_path,
        index=False,
    )
    selected_family_frontier_law_rows.to_csv(
        config.selected_family_frontier_law_rows_path,
        index=False,
    )
    selected_family_frontier_law_summary.to_csv(
        config.selected_family_frontier_law_summary_path,
        index=False,
    )
    selected_family_frontier_ablation_summary.to_csv(
        config.selected_family_frontier_ablation_summary_path,
        index=False,
    )
    selected_family_frontier_witness_rows.to_csv(
        config.selected_family_frontier_witness_rows_path,
        index=False,
    )
    sibling_rescue_audit_rows.to_csv(
        config.sibling_rescue_audit_rows_path,
        index=False,
    )
    sibling_rescue_audit_summary.to_csv(
        config.sibling_rescue_audit_summary_path,
        index=False,
    )
    sibling_rescue_guard_summary.to_csv(
        config.sibling_rescue_guard_summary_path,
        index=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "inputs": {
            "candidate_rows": (
                str(config.generated_support_candidate_rows_path)
                if config.use_generated_support_fixture
                else str(config.candidate_rows_path)
            ),
            "gene_assignments": (
                str(config.generated_support_gene_assignments_path)
                if config.use_generated_support_fixture
                else str(config.gene_assignments_path)
            ),
            "traversal_rows": (
                "generated_support_fixture"
                if config.use_generated_support_fixture
                else (
                    None
                    if config.traversal_rows_path is None
                    else str(config.traversal_rows_path)
                )
            ),
            "use_generated_support_fixture": bool(
                config.use_generated_support_fixture
            ),
            "truth_suite": config.truth_suite,
        },
        "thresholds": {
            "branch_ari_floor": float(config.branch_ari_floor),
            "partial_ari_floor": float(config.partial_ari_floor),
            "child_purity_floor": float(config.child_purity_floor),
            "top_k": int(config.top_k),
            "max_pairwise_samples": int(config.max_pairwise_samples),
            "min_frontier_law_support": int(config.min_frontier_law_support),
            "min_frontier_law_margin": float(config.min_frontier_law_margin),
        },
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
            "state_summary": str(config.state_summary_path),
            "feature_metric_summary": str(config.feature_metric_summary_path),
            "feature_metric_state_summary": str(
                config.feature_metric_state_summary_path
            ),
            "context_metric_state_summary": str(
                config.context_metric_state_summary_path
            ),
            "family_likelihood_rows": str(config.family_likelihood_rows_path),
            "family_likelihood_summary": str(
                config.family_likelihood_summary_path
            ),
            "collision_law_components": str(config.collision_law_components_path),
            "accepted_split_filter_rows": str(
                config.accepted_split_filter_rows_path
            ),
            "accepted_split_filter_summary": str(
                config.accepted_split_filter_summary_path
            ),
            "accepted_split_pair_filter_summary": str(
                config.accepted_split_pair_filter_summary_path
            ),
            "accepted_split_frontier_summary": str(
                config.accepted_split_frontier_summary_path
            ),
            "selected_family_frontier_law_rows": str(
                config.selected_family_frontier_law_rows_path
            ),
            "selected_family_frontier_law_summary": str(
                config.selected_family_frontier_law_summary_path
            ),
            "selected_family_frontier_ablation_summary": str(
                config.selected_family_frontier_ablation_summary_path
            ),
            "selected_family_frontier_witness_rows": str(
                config.selected_family_frontier_witness_rows_path
            ),
            "sibling_rescue_audit_rows": str(
                config.sibling_rescue_audit_rows_path
            ),
            "sibling_rescue_audit_summary": str(
                config.sibling_rescue_audit_summary_path
            ),
            "sibling_rescue_guard_summary": str(
                config.sibling_rescue_guard_summary_path
            ),
        },
        "row_count": int(rows.shape[0]),
        "family_count": int(family_likelihood_rows.shape[0]),
        "production_action": "diagnostic_only_no_promotion",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "state_summary": config.state_summary_path,
        "feature_metric_summary": config.feature_metric_summary_path,
        "feature_metric_state_summary": config.feature_metric_state_summary_path,
        "context_metric_state_summary": config.context_metric_state_summary_path,
        "family_likelihood_rows": config.family_likelihood_rows_path,
        "family_likelihood_summary": config.family_likelihood_summary_path,
        "collision_law_components": config.collision_law_components_path,
        "accepted_split_filter_rows": config.accepted_split_filter_rows_path,
        "accepted_split_filter_summary": config.accepted_split_filter_summary_path,
        "accepted_split_pair_filter_summary": (
            config.accepted_split_pair_filter_summary_path
        ),
        "accepted_split_frontier_summary": (
            config.accepted_split_frontier_summary_path
        ),
        "selected_family_frontier_law_rows": (
            config.selected_family_frontier_law_rows_path
        ),
        "selected_family_frontier_law_summary": (
            config.selected_family_frontier_law_summary_path
        ),
        "selected_family_frontier_ablation_summary": (
            config.selected_family_frontier_ablation_summary_path
        ),
        "selected_family_frontier_witness_rows": (
            config.selected_family_frontier_witness_rows_path
        ),
        "sibling_rescue_audit_rows": config.sibling_rescue_audit_rows_path,
        "sibling_rescue_audit_summary": config.sibling_rescue_audit_summary_path,
        "sibling_rescue_guard_summary": config.sibling_rescue_guard_summary_path,
        "manifest": config.manifest_path,
        **generated_support_outputs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether selected traversal candidates are own-split truth "
            "recoveries or pass-through fragments."
        )
    )
    parser.add_argument("--candidate-rows-path", type=Path)
    parser.add_argument("--traversal-rows-path", type=Path)
    parser.add_argument("--gene-assignments-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--use-generated-support-fixture",
        action="store_true",
        help=(
            "Build generated branch-positive/fragment/selected-null support "
            "inputs and audit their immediate split candidates."
        ),
    )
    parser.add_argument("--truth-suite", default="binary")
    parser.add_argument(
        "--branch-ari-floor",
        type=float,
        default=DEFAULT_BRANCH_ARI_FLOOR,
    )
    parser.add_argument(
        "--partial-ari-floor",
        type=float,
        default=DEFAULT_PARTIAL_ARI_FLOOR,
    )
    parser.add_argument(
        "--child-purity-floor",
        type=float,
        default=DEFAULT_CHILD_PURITY_FLOOR,
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-pairwise-samples", type=int, default=200)
    parser.add_argument("--min-frontier-law-support", type=int, default=3)
    parser.add_argument("--min-frontier-law-margin", type=float, default=1e-8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_selected_candidate_truth_law_panel(
        SelectedCandidateTruthLawPanelConfig(
            candidate_rows_path=args.candidate_rows_path,
            traversal_rows_path=args.traversal_rows_path,
            gene_assignments_path=args.gene_assignments_path,
            output_dir=args.output_dir,
            use_generated_support_fixture=args.use_generated_support_fixture,
            truth_suite=args.truth_suite,
            branch_ari_floor=args.branch_ari_floor,
            partial_ari_floor=args.partial_ari_floor,
            child_purity_floor=args.child_purity_floor,
            top_k=args.top_k,
            max_pairwise_samples=args.max_pairwise_samples,
            min_frontier_law_support=args.min_frontier_law_support,
            min_frontier_law_margin=args.min_frontier_law_margin,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
