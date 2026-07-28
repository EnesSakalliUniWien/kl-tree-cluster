"""Topology conditioning scan for context-negative emergent overlap rows.

Raw child-parent edge significance does not separate the context-negative
emergent truth row from negative ambiguous rows. This post-run panel scans the
next candidate family: selected neighborhood and topology coordinates around
the same internal junction.

The panel is diagnostic-only. A metric is a candidate only if an observed
threshold retains all truth-recovery rows with zero negative leakage; with one
truth row this remains a single-positive hypothesis that needs transfer
validation before it can become a traversal law.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.overlap.overlap_weak_zone_separability import (
    rank_auc,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_context_negative_topology_conditioning"
SCHEMA_VERSION = "overlap_context_negative_topology_conditioning/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap.overlap_context_negative_topology_conditioning"
)

CONTEXT_NEGATIVE_STATUS = "context_negative_emergent_mode_ambiguous"

MODE_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "bayesian_incidence_mode_status",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "branch_alignment_score",
}

BRANCH_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "depth",
    "decision_class",
    "traversal_decision",
    "sibling_open",
    "sibling_p_value",
    "selected_family_guard_blocked",
    "selected_family_p_value",
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
    "incoming_branch_balance",
    "outgoing_balance",
    "metric_family_alignment_score",
}

TRANSFER_GAP_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
    "soft_structure_pass",
    "default_internal_node_candidate",
    "blocking_components",
}

INCOME_OUTCOME_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "incoming_parent_depth",
    "incoming_parent_context_margin",
    "incoming_parent_homogeneity_gain_min",
    "incoming_parent_subspace_consensus_jaccard_topk",
    "outgoing_depth",
    "outgoing_homogeneity_gain_min",
    "outgoing_subspace_consensus_jaccard_topk",
    "outgoing_size_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "context_transition_delta",
    "subspace_transition_delta",
    "homogeneity_transition_delta",
}

CATEGORICAL_COLUMNS = (
    "decision_class",
    "traversal_decision",
    "sibling_open",
    "selected_family_guard_blocked",
    "soft_structure_pass",
    "default_internal_node_candidate",
    "blocking_components",
)

BASE_NUMERIC_METRICS = (
    "depth",
    "incoming_parent_depth",
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
    "incoming_branch_balance",
    "outgoing_balance",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
    "branch_alignment_score",
    "metric_family_alignment_score",
    "subspace_consensus_jaccard_topk",
    "size_balance",
    "edge_norm_balance",
    "fragment_risk_proxy_score",
    "incoming_parent_context_margin",
    "incoming_parent_homogeneity_gain_min",
    "incoming_parent_subspace_consensus_jaccard_topk",
    "outgoing_homogeneity_gain_min",
    "outgoing_subspace_consensus_jaccard_topk",
    "outgoing_size_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "context_transition_delta",
    "subspace_transition_delta",
    "homogeneity_transition_delta",
)

DERIVED_NUMERIC_METRICS = (
    "node_parent_share",
    "incoming_sibling_parent_share",
    "left_node_share",
    "right_node_share",
    "min_child_parent_share",
    "max_child_parent_share",
    "outgoing_minus_incoming_balance",
    "outgoing_to_incoming_balance_ratio",
    "balance_product",
    "sibling_neglog10_p_value",
    "selected_family_neglog10_p_value",
    "context_transition_abs",
    "subspace_transition_abs",
    "homogeneity_transition_abs",
    "outgoing_balance_edge_product",
    "outgoing_balance_edge_subspace_product",
    "neighborhood_topology_coherence_score",
    "context_negative_junction_score",
)

DEFAULT_TOPOLOGY_METRICS = BASE_NUMERIC_METRICS + DERIVED_NUMERIC_METRICS

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "parent_id",
    "branch_length_to_parent",
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "bayesian_incidence_mode_status",
    *CATEGORICAL_COLUMNS,
    "neighborhood_scale",
    "neighborhood_scale_source",
    *DEFAULT_TOPOLOGY_METRICS,
)

METRIC_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "truth_count",
    "negative_count",
    "finite_truth_count",
    "finite_negative_count",
    "best_direction",
    "best_auc",
    "high_direction_auc",
    "low_direction_auc",
    "truth_min",
    "truth_median",
    "truth_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "zero_negative_threshold",
    "zero_negative_direction",
    "zero_negative_truth_count",
    "zero_negative_truth_retention",
    "zero_negative_negative_count",
    "zero_negative_value_margin",
    "zero_negative_status",
)

CATEGORY_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "column",
    "category",
    "truth_selected_count",
    "negative_selected_count",
    "truth_total",
    "negative_total",
    "truth_retention",
    "negative_selection_rate",
    "category_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "context_negative_emergent_count",
    "truth_recovery_count",
    "negative_count",
    "metric_count",
    "separator_metric_count",
    "category_separator_count",
    "best_separator_metric",
    "best_separator_direction",
    "best_separator_threshold",
    "best_separator_truth_retention",
    "best_separator_value_margin",
    "diagnostic_status",
)

THRESHOLD_SCAN_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "direction",
    "threshold",
    "truth_selected_count",
    "negative_selected_count",
    "truth_total",
    "negative_total",
    "truth_retention",
    "negative_selection_rate",
)


@dataclass(frozen=True)
class OverlapContextNegativeTopologyConditioningConfig:
    """Runtime contract for context-negative topology conditioning diagnostics."""

    branch_rows_path: Path
    mode_rows_path: Path
    transfer_gap_rows_path: Path
    income_outcome_rows_path: Path
    output_dir: Path
    metrics: tuple[str, ...] = DEFAULT_TOPOLOGY_METRICS

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_topology_conditioning_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_topology_conditioning_metric_summary.csv"

    @property
    def category_summary_path(self) -> Path:
        return (
            self.output_dir / "overlap_context_negative_topology_conditioning_category_summary.csv"
        )

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_topology_conditioning_summary.csv"

    @property
    def threshold_scan_path(self) -> Path:
        return self.output_dir / "overlap_context_negative_topology_conditioning_threshold_scan.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _parse_metric_list(value: str) -> tuple[str, ...]:
    metrics = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not metrics:
        raise ValueError("At least one topology-conditioning metric is required.")
    unknown = sorted(set(metrics) - set(DEFAULT_TOPOLOGY_METRICS))
    if unknown:
        raise ValueError(f"Unknown topology-conditioning metrics: {unknown!r}")
    return metrics


def _validate_columns(rows: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{label} rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce")
    numer = pd.to_numeric(numerator, errors="coerce")
    return numer.divide(denom).where(denom.ne(0.0), np.nan)


def _neglog10_p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").clip(
        lower=np.nextafter(0.0, 1.0),
        upper=1.0,
    )
    return -np.log10(numeric)


def _topology_support_role(*, data_role: object, guard_truth_role: object) -> str:
    """Classify rows for topology-neighborhood empirical support."""
    role = str(guard_truth_role)
    data = str(data_role)
    if role == "truth_recovery":
        return ""
    if data in {"null", "selected_null"} or role == "null_like":
        return "strict_null"
    if data == "signal" or role in {"diffuse_or_wrong", "fragment_like"}:
        return "selected_nonnull"
    return ""


def _topology_signal_role(*, guard_truth_role: object) -> str:
    """Classify explicit topology signal rows for neighborhood diagnostics."""
    return "signal" if str(guard_truth_role) == "truth_recovery" else ""


def add_topology_conditioning_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with derived selected-neighborhood topology metrics."""
    enriched = rows.copy()
    n_parent = _numeric(enriched, "n_parent_context")
    n_node = _numeric(enriched, "n_node")
    n_incoming_sibling = _numeric(enriched, "n_incoming_sibling")
    n_left = _numeric(enriched, "n_left")
    n_right = _numeric(enriched, "n_right")
    incoming_balance = _numeric(enriched, "incoming_branch_balance")
    outgoing_balance = _numeric(enriched, "outgoing_balance")
    subspace = _numeric(enriched, "subspace_consensus_jaccard_topk")
    edge_norm_balance = _numeric(enriched, "edge_norm_balance")
    fragment_risk = _numeric(enriched, "fragment_risk_proxy_score")
    bayes = _numeric(enriched, "selected_family_log_bayes_factor_lower")
    context = _numeric(enriched, "continuous_context_min_margin")
    context_delta = _numeric(enriched, "context_transition_delta")
    subspace_delta = _numeric(enriched, "subspace_transition_delta")
    homogeneity_delta = _numeric(enriched, "homogeneity_transition_delta")

    enriched["node_parent_share"] = _safe_divide(n_node, n_parent)
    enriched["incoming_sibling_parent_share"] = _safe_divide(
        n_incoming_sibling,
        n_parent,
    )
    enriched["left_node_share"] = _safe_divide(n_left, n_node)
    enriched["right_node_share"] = _safe_divide(n_right, n_node)
    enriched["min_child_parent_share"] = _safe_divide(
        pd.concat([n_left, n_right], axis=1).min(axis=1),
        n_parent,
    )
    enriched["max_child_parent_share"] = _safe_divide(
        pd.concat([n_left, n_right], axis=1).max(axis=1),
        n_parent,
    )
    enriched["outgoing_minus_incoming_balance"] = outgoing_balance - incoming_balance
    enriched["outgoing_to_incoming_balance_ratio"] = _safe_divide(
        outgoing_balance,
        incoming_balance,
    )
    enriched["balance_product"] = outgoing_balance * incoming_balance
    enriched["sibling_neglog10_p_value"] = _neglog10_p(enriched["sibling_p_value"])
    enriched["selected_family_neglog10_p_value"] = _neglog10_p(enriched["selected_family_p_value"])
    enriched["context_transition_abs"] = context_delta.abs()
    enriched["subspace_transition_abs"] = subspace_delta.abs()
    enriched["homogeneity_transition_abs"] = homogeneity_delta.abs()
    enriched["outgoing_balance_edge_product"] = outgoing_balance * edge_norm_balance
    enriched["outgoing_balance_edge_subspace_product"] = (
        outgoing_balance * edge_norm_balance * subspace
    )
    enriched["neighborhood_topology_coherence_score"] = (
        outgoing_balance + edge_norm_balance + subspace - fragment_risk
    )
    enriched["context_negative_junction_score"] = (
        bayes
        + 4.0 * outgoing_balance
        + 2.0 * edge_norm_balance
        + 2.0 * subspace
        - fragment_risk
        + 80.0 * context
    )
    return enriched


def build_context_negative_topology_conditioning_rows(
    *,
    branch_rows: pd.DataFrame,
    mode_rows: pd.DataFrame,
    transfer_gap_rows: pd.DataFrame,
    income_outcome_rows: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_TOPOLOGY_METRICS,
) -> pd.DataFrame:
    """Join context-negative mode rows to topology and neighborhood evidence."""
    requested_metrics = tuple(metrics)
    _validate_columns(mode_rows, MODE_REQUIRED_COLUMNS, "Incidence-mode")
    _validate_columns(branch_rows, BRANCH_REQUIRED_COLUMNS, "Branch-incidence")
    _validate_columns(
        transfer_gap_rows,
        TRANSFER_GAP_REQUIRED_COLUMNS,
        "Transfer-gap",
    )
    _validate_columns(
        income_outcome_rows,
        INCOME_OUTCOME_REQUIRED_COLUMNS,
        "Income-outcome",
    )
    unknown = sorted(set(requested_metrics) - set(DEFAULT_TOPOLOGY_METRICS))
    if unknown:
        raise ValueError(f"Unknown topology-conditioning metrics: {unknown!r}")

    keys = ["case_id", "data_role", "replicate", "node_id"]
    mode_columns = [
        *keys,
        "guard_truth_role",
        "bayesian_incidence_mode_status",
        "selected_family_log_bayes_factor_lower",
        "continuous_context_min_margin",
        "branch_alignment_score",
    ]
    ambiguous = mode_rows.loc[
        mode_rows["bayesian_incidence_mode_status"].astype(str).eq(CONTEXT_NEGATIVE_STATUS),
        mode_columns,
    ].copy()
    branch_columns = [
        *keys,
        *(["incoming_parent_id"] if "incoming_parent_id" in branch_rows.columns else []),
        *(["branch_length_to_parent"] if "branch_length_to_parent" in branch_rows.columns else []),
        "depth",
        "decision_class",
        "traversal_decision",
        "sibling_open",
        "sibling_p_value",
        *(
            ["sibling_projection_dimension"]
            if "sibling_projection_dimension" in branch_rows.columns
            else []
        ),
        "selected_family_guard_blocked",
        "selected_family_p_value",
        "n_parent_context",
        "n_node",
        "n_incoming_sibling",
        "n_left",
        "n_right",
        "incoming_branch_balance",
        "outgoing_balance",
        "metric_family_alignment_score",
    ]
    gap_columns = [
        *keys,
        "subspace_consensus_jaccard_topk",
        "size_balance",
        "edge_norm_balance",
        "fragment_risk_proxy_score",
        "soft_structure_pass",
        "default_internal_node_candidate",
        "blocking_components",
    ]
    income_columns = [
        *keys,
        "incoming_parent_depth",
        "incoming_parent_context_margin",
        "incoming_parent_homogeneity_gain_min",
        "incoming_parent_subspace_consensus_jaccard_topk",
        "outgoing_depth",
        "outgoing_homogeneity_gain_min",
        "outgoing_subspace_consensus_jaccard_topk",
        "outgoing_size_balance",
        "outgoing_edge_norm_balance",
        "outgoing_fragment_risk_proxy_score",
        "context_transition_delta",
        "subspace_transition_delta",
        "homogeneity_transition_delta",
    ]
    rows = ambiguous.merge(
        branch_rows[branch_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    rows = rows.merge(
        transfer_gap_rows[gap_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    rows = rows.merge(
        income_outcome_rows[income_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if "sibling_projection_dimension" not in rows.columns:
        rows["sibling_projection_dimension"] = np.nan
    sibling_projection_dimension = _numeric(rows, "sibling_projection_dimension")
    rows["neighborhood_scale"] = sibling_projection_dimension.where(
        sibling_projection_dimension.gt(0.0),
        np.nan,
    )
    rows["neighborhood_scale_source"] = np.where(
        rows["neighborhood_scale"].notna(),
        "sibling_projection_dimension",
        "missing_sibling_projection_dimension",
    )
    rows = add_topology_conditioning_metrics(rows)

    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        record = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": str(row["case_id"]),
            "data_role": str(row["data_role"]),
            "replicate": int(row["replicate"]),
            "node_id": str(row["node_id"]),
            "parent_id": (
                str(row["incoming_parent_id"])
                if "incoming_parent_id" in row and pd.notna(row["incoming_parent_id"])
                else ""
            ),
            "branch_length_to_parent": _finite_float(row.get("branch_length_to_parent", math.nan)),
            "guard_truth_role": str(row["guard_truth_role"]),
            "topology_support_role": _topology_support_role(
                data_role=row["data_role"],
                guard_truth_role=row["guard_truth_role"],
            ),
            "topology_signal_role": _topology_signal_role(guard_truth_role=row["guard_truth_role"]),
            "bayesian_incidence_mode_status": str(row["bayesian_incidence_mode_status"]),
        }
        for column in CATEGORICAL_COLUMNS:
            record[column] = str(row[column]) if pd.notna(row[column]) else "missing"
        try:
            record["neighborhood_scale"] = float(row["neighborhood_scale"])
        except (TypeError, ValueError):
            record["neighborhood_scale"] = math.nan
        record["neighborhood_scale_source"] = (
            str(row["neighborhood_scale_source"])
            if pd.notna(row["neighborhood_scale_source"])
            else "missing_sibling_projection_dimension"
        )
        for metric in DEFAULT_TOPOLOGY_METRICS:
            try:
                record[metric] = float(row[metric])
            except (TypeError, ValueError):
                record[metric] = math.nan
        records.append(record)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _zero_negative_separator(
    *,
    truth_values: Sequence[float],
    negative_values: Sequence[float],
    direction: str,
) -> tuple[float, int, float, int, float, str]:
    truth = _finite(truth_values)
    negative = _finite(negative_values)
    if truth.size == 0 or negative.size == 0:
        return math.nan, 0, math.nan, 0, math.nan, "zero_negative_undefined"
    if direction == "greater_equal":
        negative_boundary = float(np.max(negative))
        threshold = float(np.nextafter(negative_boundary, math.inf))
        selected_truth = truth >= threshold
        selected_negative = negative >= threshold
        margin = float(np.min(truth) - negative_boundary)
    elif direction == "less_equal":
        negative_boundary = float(np.min(negative))
        threshold = float(np.nextafter(negative_boundary, -math.inf))
        selected_truth = truth <= threshold
        selected_negative = negative <= threshold
        margin = float(negative_boundary - np.max(truth))
    else:
        raise ValueError(f"Unknown separator direction: {direction!r}")
    truth_count = int(selected_truth.sum())
    negative_count = int(selected_negative.sum())
    retention = float(truth_count / truth.size)
    if truth_count == truth.size and negative_count == 0:
        status = "zero_negative_separates_all_truth"
    elif truth_count > 0 and negative_count == 0:
        status = "zero_negative_partial_truth_retention"
    elif truth_count == 0 and negative_count == 0:
        status = "zero_negative_no_truth_retention"
    else:
        status = "zero_negative_leakage"
    return threshold, truth_count, retention, negative_count, margin, status


def threshold_scan_for_metric(rows: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Scan observed thresholds for one topology-conditioning metric."""
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    values = _numeric(rows, metric)
    finite = values[np.isfinite(values)]
    thresholds = np.sort(finite.unique())
    records: list[dict[str, object]] = []
    truth_total = int(truth.sum())
    negative_total = int(negative.sum())
    for direction in ("greater_equal", "less_equal"):
        for threshold in thresholds:
            selected = values.ge(float(threshold))
            if direction == "less_equal":
                selected = values.le(float(threshold))
            truth_selected = int((selected & truth).sum())
            negative_selected = int((selected & negative).sum())
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "metric": metric,
                    "direction": direction,
                    "threshold": float(threshold),
                    "truth_selected_count": truth_selected,
                    "negative_selected_count": negative_selected,
                    "truth_total": truth_total,
                    "negative_total": negative_total,
                    "truth_retention": (
                        float(truth_selected / truth_total) if truth_total else math.nan
                    ),
                    "negative_selection_rate": (
                        float(negative_selected / negative_total) if negative_total else math.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=THRESHOLD_SCAN_COLUMNS)


def summarize_metric(rows: pd.DataFrame, *, metric: str) -> dict[str, object]:
    """Summarize one topology metric inside context-negative emergent rows."""
    roles = rows["guard_truth_role"].astype(str)
    truth_mask = roles.eq("truth_recovery")
    negative_mask = ~truth_mask
    values = _numeric(rows, metric)
    truth_values = _finite(values[truth_mask].to_numpy(dtype=float))
    negative_values = _finite(values[negative_mask].to_numpy(dtype=float))
    high_auc = rank_auc(truth_values, negative_values)
    low_auc = 1.0 - high_auc if math.isfinite(high_auc) else math.nan
    if not math.isfinite(high_auc):
        best_direction = "undefined"
        best_auc = math.nan
    elif high_auc >= low_auc:
        best_direction = "greater_equal"
        best_auc = high_auc
    else:
        best_direction = "less_equal"
        best_auc = low_auc

    high_sep = _zero_negative_separator(
        truth_values=truth_values,
        negative_values=negative_values,
        direction="greater_equal",
    )
    low_sep = _zero_negative_separator(
        truth_values=truth_values,
        negative_values=negative_values,
        direction="less_equal",
    )
    candidates = [
        ("greater_equal", *high_sep),
        ("less_equal", *low_sep),
    ]
    candidates.sort(
        key=lambda item: (
            item[3] if math.isfinite(item[3]) else -1.0,
            item[2],
            item[5] if math.isfinite(item[5]) else -math.inf,
            -item[4],
        ),
        reverse=True,
    )
    (
        separator_direction,
        separator_threshold,
        separator_truth_count,
        separator_retention,
        separator_negative_count,
        separator_margin,
        separator_status,
    ) = candidates[0]

    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "metric": metric,
        "truth_count": int(truth_mask.sum()),
        "negative_count": int(negative_mask.sum()),
        "finite_truth_count": int(truth_values.size),
        "finite_negative_count": int(negative_values.size),
        "best_direction": best_direction,
        "best_auc": float(best_auc) if math.isfinite(best_auc) else math.nan,
        "high_direction_auc": (float(high_auc) if math.isfinite(high_auc) else math.nan),
        "low_direction_auc": float(low_auc) if math.isfinite(low_auc) else math.nan,
        "truth_min": float(np.min(truth_values)) if truth_values.size else math.nan,
        "truth_median": (float(np.median(truth_values)) if truth_values.size else math.nan),
        "truth_max": float(np.max(truth_values)) if truth_values.size else math.nan,
        "negative_min": (float(np.min(negative_values)) if negative_values.size else math.nan),
        "negative_median": (
            float(np.median(negative_values)) if negative_values.size else math.nan
        ),
        "negative_max": (float(np.max(negative_values)) if negative_values.size else math.nan),
        "zero_negative_threshold": separator_threshold,
        "zero_negative_direction": separator_direction,
        "zero_negative_truth_count": separator_truth_count,
        "zero_negative_truth_retention": separator_retention,
        "zero_negative_negative_count": separator_negative_count,
        "zero_negative_value_margin": separator_margin,
        "zero_negative_status": separator_status,
    }


def summarize_categories(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize category-only topology selectors for the ambiguous rows."""
    if rows.empty:
        return pd.DataFrame(columns=CATEGORY_SUMMARY_COLUMNS)
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    negative = ~truth
    truth_total = int(truth.sum())
    negative_total = int(negative.sum())
    records: list[dict[str, object]] = []
    for column in CATEGORICAL_COLUMNS:
        categories = sorted(rows[column].astype(str).fillna("missing").unique())
        for category in categories:
            selected = rows[column].astype(str).fillna("missing").eq(category)
            truth_selected = int((selected & truth).sum())
            negative_selected = int((selected & negative).sum())
            if truth_selected == truth_total and negative_selected == 0:
                status = "category_separates_all_truth"
            elif truth_selected > 0 and negative_selected == 0:
                status = "category_partial_truth_retention"
            elif truth_selected == 0 and negative_selected == 0:
                status = "category_empty"
            else:
                status = "category_leaky_or_nonseparating"
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "column": column,
                    "category": category,
                    "truth_selected_count": truth_selected,
                    "negative_selected_count": negative_selected,
                    "truth_total": truth_total,
                    "negative_total": negative_total,
                    "truth_retention": (
                        float(truth_selected / truth_total) if truth_total else math.nan
                    ),
                    "negative_selection_rate": (
                        float(negative_selected / negative_total) if negative_total else math.nan
                    ),
                    "category_status": status,
                }
            )
    return pd.DataFrame.from_records(records, columns=CATEGORY_SUMMARY_COLUMNS)


def summarize_context_negative_topology_conditioning(
    rows: pd.DataFrame,
    *,
    metrics: Iterable[str] = DEFAULT_TOPOLOGY_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return overall, per-metric, and per-category topology summaries."""
    requested_metrics = tuple(metrics)
    if rows.empty:
        empty_metrics = pd.DataFrame(columns=METRIC_SUMMARY_COLUMNS)
        empty_categories = pd.DataFrame(columns=CATEGORY_SUMMARY_COLUMNS)
        summary = pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "context_negative_emergent_count": 0,
                    "truth_recovery_count": 0,
                    "negative_count": 0,
                    "metric_count": len(requested_metrics),
                    "separator_metric_count": 0,
                    "category_separator_count": 0,
                    "best_separator_metric": "",
                    "best_separator_direction": "",
                    "best_separator_threshold": math.nan,
                    "best_separator_truth_retention": math.nan,
                    "best_separator_value_margin": math.nan,
                    "diagnostic_status": "topology_conditioning_unavailable",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
        return summary, empty_metrics, empty_categories

    missing = sorted(set(requested_metrics) - set(rows.columns))
    if missing:
        raise ValueError(f"Conditioning rows are missing metrics: {missing!r}")

    metric_rows = pd.DataFrame.from_records(
        [summarize_metric(rows, metric=metric) for metric in requested_metrics],
        columns=METRIC_SUMMARY_COLUMNS,
    )
    category_rows = summarize_categories(rows)
    full = metric_rows["zero_negative_status"].eq("zero_negative_separates_all_truth")
    partial = metric_rows["zero_negative_status"].eq("zero_negative_partial_truth_retention")
    category_full = category_rows["category_status"].eq("category_separates_all_truth")
    if bool(full.any()):
        candidates = metric_rows.loc[full].sort_values(
            ["zero_negative_truth_retention", "best_auc", "zero_negative_value_margin"],
            ascending=[False, False, False],
        )
    elif bool(partial.any()):
        candidates = metric_rows.loc[partial].sort_values(
            ["zero_negative_truth_retention", "best_auc", "zero_negative_value_margin"],
            ascending=[False, False, False],
        )
    else:
        candidates = metric_rows.sort_values("best_auc", ascending=False)
    best = candidates.iloc[0] if not candidates.empty else pd.Series(dtype=object)

    roles = rows["guard_truth_role"].astype(str)
    truth_count = int(roles.eq("truth_recovery").sum())
    if bool(full.any()) and truth_count < 2:
        diagnostic_status = "topology_conditioning_single_truth_separator_candidate"
    elif bool(full.any()):
        diagnostic_status = "topology_conditioning_separator_candidate_found"
    elif bool(category_full.any()) and truth_count < 2:
        diagnostic_status = "topology_category_single_truth_separator_candidate"
    elif bool(category_full.any()):
        diagnostic_status = "topology_category_separator_candidate_found"
    elif bool(partial.any()):
        diagnostic_status = "topology_conditioning_partial_zero_negative_retention"
    else:
        diagnostic_status = "topology_conditioning_no_zero_negative_separator"

    summary = pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "context_negative_emergent_count": int(rows.shape[0]),
                "truth_recovery_count": truth_count,
                "negative_count": int(rows.shape[0] - truth_count),
                "metric_count": len(requested_metrics),
                "separator_metric_count": int(full.sum()),
                "category_separator_count": int(category_full.sum()),
                "best_separator_metric": str(best.get("metric", "")),
                "best_separator_direction": str(best.get("zero_negative_direction", "")),
                "best_separator_threshold": float(best.get("zero_negative_threshold", math.nan)),
                "best_separator_truth_retention": float(
                    best.get("zero_negative_truth_retention", math.nan)
                ),
                "best_separator_value_margin": float(
                    best.get("zero_negative_value_margin", math.nan)
                ),
                "diagnostic_status": diagnostic_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return summary, metric_rows, category_rows


def run_overlap_context_negative_topology_conditioning(
    config: OverlapContextNegativeTopologyConditioningConfig,
) -> dict[str, Path]:
    """Run topology conditioning diagnostics and write outputs."""
    branch_rows = pd.read_csv(config.branch_rows_path)
    mode_rows = pd.read_csv(config.mode_rows_path)
    transfer_gap_rows = pd.read_csv(config.transfer_gap_rows_path)
    income_outcome_rows = pd.read_csv(config.income_outcome_rows_path)
    rows = build_context_negative_topology_conditioning_rows(
        branch_rows=branch_rows,
        mode_rows=mode_rows,
        transfer_gap_rows=transfer_gap_rows,
        income_outcome_rows=income_outcome_rows,
        metrics=config.metrics,
    )
    summary, metric_summary, category_summary = summarize_context_negative_topology_conditioning(
        rows,
        metrics=config.metrics,
    )
    scans = []
    for metric in config.metrics:
        if metric in rows.columns:
            scan = threshold_scan_for_metric(rows, metric=metric)
            if not scan.empty:
                scans.append(scan)
    threshold_scan = (
        pd.concat(scans, ignore_index=True)
        if scans
        else pd.DataFrame(columns=THRESHOLD_SCAN_COLUMNS)
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    metric_summary.to_csv(config.metric_summary_path, index=False)
    category_summary.to_csv(config.category_summary_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    threshold_scan.to_csv(config.threshold_scan_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "branch_rows_path": str(config.branch_rows_path),
        "mode_rows_path": str(config.mode_rows_path),
        "transfer_gap_rows_path": str(config.transfer_gap_rows_path),
        "income_outcome_rows_path": str(config.income_outcome_rows_path),
        "context_negative_status": CONTEXT_NEGATIVE_STATUS,
        "metrics": list(config.metrics),
        "outputs": {
            "rows": str(config.rows_path),
            "metric_summary": str(config.metric_summary_path),
            "category_summary": str(config.category_summary_path),
            "summary": str(config.summary_path),
            "threshold_scan": str(config.threshold_scan_path),
        },
        "production_status": ("diagnostic_only_single_truth_topology_conditioning_scan"),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "metric_summary": config.metric_summary_path,
        "category_summary": config.category_summary_path,
        "summary": config.summary_path,
        "threshold_scan": config.threshold_scan_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-rows-path", type=Path, required=True)
    parser.add_argument("--mode-rows-path", type=Path, required=True)
    parser.add_argument("--transfer-gap-rows-path", type=Path, required=True)
    parser.add_argument("--income-outcome-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=_parse_metric_list,
        default=DEFAULT_TOPOLOGY_METRICS,
        help="Comma-separated topology-conditioning metrics to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_context_negative_topology_conditioning(
        OverlapContextNegativeTopologyConditioningConfig(
            branch_rows_path=args.branch_rows_path,
            mode_rows_path=args.mode_rows_path,
            transfer_gap_rows_path=args.transfer_gap_rows_path,
            income_outcome_rows_path=args.income_outcome_rows_path,
            output_dir=args.output_dir,
            metrics=tuple(args.metrics),
        )
    )


if __name__ == "__main__":
    main()
