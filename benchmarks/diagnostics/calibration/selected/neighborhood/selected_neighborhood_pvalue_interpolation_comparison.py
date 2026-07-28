"""Hold-out comparison of direct sibling p-values and interpolated p-like values.

The selected-neighborhood rows often contain a measured sibling p-value but no
precomputed child-level interpolated prior. This diagnostic reconstructs a
local tree-neighborhood interpolation from the measured rows, holds out the
target row, and compares the interpolated p-like value with the target's direct
sibling p-value.

The output is diagnostic-only. The interpolated quantity is not a calibrated
production p-value.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.overlap.overlap_conditional_topology_law_panel import (
    CachedTreeDistances,
    build_cached_tree_distances,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_measurability_law import (
    candidate_evidence_mask,
    compute_child_interpolated_null_prior,
)

SCHEMA_VERSION = "selected_neighborhood_pvalue_interpolation_comparison/v1"
STUDY_ROLE = "diagnostic_selected_neighborhood_pvalue_interpolation_comparison_not_calibration"
GENERATED_BY = "benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_pvalue_interpolation_comparison"

DEFAULT_ALPHA = 0.01
DEFAULT_SUPPORT_P_FLOOR = 0.10
DEFAULT_MIN_SUPPORT = 2
DEFAULT_FALLBACK_TAU_T = 1.0
DEFAULT_FALLBACK_TAU_S = 1.0
DEFAULT_FALLBACK_H_K = 1.0
PROBABILITY_TOLERANCE = 1e-12
DEFAULT_TAU_S_SENSITIVITY_GRID = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 100.0)

GROUP_COLUMNS = ("case_id", "data_role", "method_id", "replicate")
ELIGIBLE_SUPPORT_ROLES = {
    "strict_null",
    "edge_blocked",
    "stopped_or_null",
    "null_like",
    "stable",
}
SELECTED_NONNULL_ROLES = {"selected_nonnull", "selected_non_null", "nonnull"}
STABLE_DECISION_CLASSES = {
    "stable_boundary",
    "selected_root_blocked",
    "selected_family_blocked",
}

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
    "decision_class",
    "traversal_decision",
    "direct_sibling_p_value",
    "direct_sibling_open",
    "direct_significant",
    "interpolated_sibling_null_p_like",
    "interpolated_significant",
    "interpolated_minus_direct",
    "interpolated_to_direct_ratio",
    "negative_log10_direct_p",
    "negative_log10_interpolated_p_like",
    "negative_log10_delta",
    "support_anchor_count",
    "signal_anchor_count",
    "selected_nonnull_excluded_count",
    "support_weight",
    "effective_support",
    "stable_weighted_p_mean",
    "signal_attenuation",
    "nearest_support_distance",
    "nearest_signal_distance",
    "distance_to_stopping_edge",
    "tau_b",
    "tau_t",
    "tau_s",
    "h_k",
    "best_case_required_tau_s_for_alpha",
    "tree_distance_status",
    "support_anchor_rule",
    "signal_anchor_rule",
    "interpolation_status",
    "comparison_class",
    "behavior_label",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "comparison_class",
    "behavior_label",
    "row_count",
    "direct_significant_count",
    "interpolated_computed_count",
    "interpolated_significant_count",
    "interpolation_lower_count",
    "interpolation_higher_count",
    "median_direct_p",
    "median_interpolated_p_like",
    "median_negative_log10_delta",
    "median_best_case_required_tau_s_for_alpha",
    "p90_best_case_required_tau_s_for_alpha",
    "false_open_risk_count",
    "signal_catch_count",
    "signal_miss_count",
)

CASE_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "row_count",
    "direct_significant_count",
    "interpolated_computed_count",
    "interpolated_significant_count",
    "interpolation_lower_count",
    "interpolation_higher_count",
    "median_direct_p",
    "median_interpolated_p_like",
    "median_negative_log10_delta",
    "median_best_case_required_tau_s_for_alpha",
    "p90_best_case_required_tau_s_for_alpha",
    "dominant_behavior_label",
    "diagnostic_status",
)

TAU_S_SENSITIVITY_COLUMNS = (
    "schema_version",
    "study_role",
    "data_role",
    "method_id",
    "tau_s_threshold",
    "direct_significant_count",
    "best_case_significant_count",
    "best_case_significant_fraction",
    "sensitivity_label",
)

TAU_S_RANGE_COLUMNS = (
    "schema_version",
    "study_role",
    "method_id",
    "target_signal_fraction",
    "max_selected_null_fraction",
    "signal_direct_significant_count",
    "selected_null_direct_significant_count",
    "finite_signal_tau_s_count",
    "finite_selected_null_tau_s_count",
    "signal_tau_s_lower_bound",
    "selected_null_tau_s_upper_bound",
    "admissible_tau_s_width",
    "admissible_tau_s_midpoint",
    "tau_s_range_status",
)

DEFAULT_TAU_S_RANGE_SIGNAL_FRACTIONS = (0.25, 0.50, 0.75)
DEFAULT_TAU_S_RANGE_SELECTED_NULL_FRACTIONS = (0.01, 0.05, 0.10, 0.20)

TOPOLOGY_REGION_COLUMNS = ("case_id", "method_id", "replicate")
ROLE_TOPOLOGY_REGION_COLUMNS = ("case_id", "data_role", "method_id", "replicate")

REGION_BANDWIDTH_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "row_count",
    "direct_significant_count",
    "interpolated_computed_count",
    "interpolated_significant_count",
    "tau_b_finite_count",
    "tau_b_q10",
    "tau_b_median",
    "tau_b_q90",
    "tau_t_finite_count",
    "tau_t_q10",
    "tau_t_median",
    "tau_t_q90",
    "tau_s_finite_count",
    "tau_s_q10",
    "tau_s_median",
    "tau_s_q90",
    "h_k_finite_count",
    "h_k_q10",
    "h_k_median",
    "h_k_q90",
    "distance_to_stopping_edge_finite_count",
    "distance_to_stopping_edge_q10",
    "distance_to_stopping_edge_median",
    "distance_to_stopping_edge_q90",
    "best_case_required_tau_s_finite_count",
    "best_case_required_tau_s_q10",
    "best_case_required_tau_s_median",
    "best_case_required_tau_s_q90",
    "effective_support_median",
    "nearest_support_distance_median",
    "nearest_signal_distance_median",
    "region_bandwidth_status",
)

REGION_TAU_S_RANGE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "method_id",
    "replicate",
    "target_signal_fraction",
    "max_selected_null_fraction",
    "signal_direct_significant_count",
    "selected_null_direct_significant_count",
    "finite_signal_tau_s_count",
    "finite_selected_null_tau_s_count",
    "signal_tau_s_lower_bound",
    "selected_null_tau_s_upper_bound",
    "admissible_tau_s_width",
    "admissible_tau_s_midpoint",
    "signal_observed_tau_b_median",
    "selected_null_observed_tau_b_median",
    "signal_observed_tau_t_median",
    "selected_null_observed_tau_t_median",
    "signal_observed_tau_s_median",
    "selected_null_observed_tau_s_median",
    "signal_observed_h_k_median",
    "selected_null_observed_h_k_median",
    "tau_s_range_status",
)


@dataclass(frozen=True)
class HoldoutInterpolatedPValue:
    """Diagnostic hold-out interpolation result for one target row."""

    p_like: float
    support_weight: float
    effective_support: float
    stable_weighted_p_mean: float
    signal_attenuation: float
    support_anchor_count: int
    signal_anchor_count: int
    selected_nonnull_excluded_count: int
    nearest_support_distance: float
    nearest_signal_distance: float
    tau_t: float
    tau_s: float
    h_k: float
    tree_distance_status: str
    support_anchor_rule: str
    signal_anchor_rule: str
    status: str


@dataclass(frozen=True)
class GroupInterpolationContext:
    """Vectorized per-selected-tree context for hold-out interpolation."""

    group: pd.DataFrame
    cache_status: str
    row_positions: dict[object, int]
    distances: np.ndarray
    p_values: np.ndarray
    log_scales: np.ndarray
    tau_t_values: np.ndarray
    tau_s_values: np.ndarray
    h_k_values: np.ndarray
    support_mask: np.ndarray
    signal_mask: np.ndarray
    selected_nonnull_excluded_count: int
    support_anchor_rule: str
    signal_anchor_rule: str
    group_h_k: float
    fallback_tau_t: float
    fallback_tau_s: float
    fallback_h_k: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct selected-neighborhood sibling p-values with "
            "hold-out interpolated p-like values."
        )
    )
    parser.add_argument(
        "--rows",
        type=Path,
        required=True,
        help="selected_neighborhood_distribution_rows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for comparison rows, summaries, and manifest.json.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance threshold used for direct/interpolated comparison.",
    )
    parser.add_argument(
        "--support-p-floor",
        type=float,
        default=DEFAULT_SUPPORT_P_FLOOR,
        help="Inferred null-support anchor floor when explicit support roles are absent.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help="Minimum held-out null-support anchors required to interpolate.",
    )
    parser.add_argument(
        "--fallback-tau-t",
        type=float,
        default=DEFAULT_FALLBACK_TAU_T,
        help="Fallback tree-distance bandwidth for stable support weighting.",
    )
    parser.add_argument(
        "--fallback-tau-s",
        type=float,
        default=DEFAULT_FALLBACK_TAU_S,
        help="Fallback tree-distance bandwidth for signal attenuation.",
    )
    parser.add_argument(
        "--fallback-h-k",
        type=float,
        default=DEFAULT_FALLBACK_H_K,
        help="Fallback log-scale bandwidth when row/group scale bandwidth is absent.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Restrict target rows to direct/interpolated/guard traversal candidates.",
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _finite_positive(value: object, default: float) -> float:
    numeric = _finite_float(value)
    if math.isfinite(numeric) and numeric > 0.0:
        return float(numeric)
    return float(default)


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return str(value)


def _clean_role_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        return pd.Series("", index=rows.index, dtype=str)
    return (
        rows[column].fillna("").astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})
    )


def _probability_or_nan(value: object) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return math.nan
    if numeric < -PROBABILITY_TOLERANCE or numeric > 1.0 + PROBABILITY_TOLERANCE:
        return math.nan
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return float(numeric)


def _is_computed_status(status: str) -> bool:
    return status in {
        "interpolated_p_like_observed_diagnostic_only",
        "interpolated_p_like_no_signal_attenuation_diagnostic_only",
    }


def _tree_distances_available(status: str) -> bool:
    return str(status).startswith("cached_all_pairs_")


def _safe_neg_log10(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if value <= 0.0:
        return math.inf
    return -math.log10(value)


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return math.nan
    return float(finite.median())


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return math.nan
    return float(finite.quantile(float(quantile)))


def _best_case_required_tau_s_for_alpha(
    *,
    stable_weighted_p_mean: float,
    nearest_signal_distance: float,
    alpha: float,
) -> float:
    """Return optimistic signal bandwidth needed for p-like <= alpha.

    This assumes the best possible neighboring signal p-value, `p_signal = 0`.
    If even that requires a large bandwidth, the current interpolation failure
    is a locality/bandwidth bottleneck rather than a lack of stable support.
    """
    stable = float(stable_weighted_p_mean)
    distance = float(nearest_signal_distance)
    threshold = float(alpha)
    if not math.isfinite(stable) or not math.isfinite(distance):
        return math.nan
    if threshold <= 0.0:
        return math.nan
    if stable <= threshold:
        return 0.0
    required_attenuation = 1.0 - threshold / stable
    if required_attenuation <= 0.0:
        return 0.0
    if required_attenuation >= 1.0:
        return math.inf
    if distance <= 0.0:
        return 0.0
    return float(distance / (-math.log(required_attenuation)))


def _scale_value(row: pd.Series) -> float:
    for column in (
        "sibling_projection_dimension",
        "neighborhood_scale",
        "n_descendant_leaves",
    ):
        value = _finite_float(row.get(column, math.nan))
        if math.isfinite(value) and value > 0.0:
            return float(value)
    return math.nan


def _support_and_signal_masks(
    group: pd.DataFrame,
    *,
    alpha: float,
    support_p_floor: float,
) -> tuple[pd.Series, pd.Series, int, str, str]:
    p_values = pd.to_numeric(group.get("sibling_p_value"), errors="coerce")
    valid_probability = p_values.between(0.0, 1.0, inclusive="both")
    support_role = _clean_role_series(group, "topology_support_role")
    signal_role = _clean_role_series(group, "topology_signal_role")
    guard_role = _clean_role_series(group, "guard_truth_role")
    decision = _clean_role_series(group, "decision_class")
    sibling_open = (
        group["sibling_open"].fillna(False).map(_bool_value)
        if "sibling_open" in group
        else pd.Series(False, index=group.index)
    )

    has_explicit_roles = bool(
        support_role.str.len().gt(0).any() or signal_role.str.len().gt(0).any()
    )
    selected_nonnull = support_role.isin(SELECTED_NONNULL_ROLES)
    if has_explicit_roles:
        support_mask = (
            valid_probability
            & ~selected_nonnull
            & (support_role.isin(ELIGIBLE_SUPPORT_ROLES) | guard_role.eq("null_like"))
        )
        signal_mask = valid_probability & signal_role.eq("signal")
        support_rule = "explicit_topology_support_role"
        signal_rule = "explicit_topology_signal_role"
    else:
        stable_decision = decision.isin(STABLE_DECISION_CLASSES)
        support_mask = (
            valid_probability
            & p_values.ge(float(support_p_floor))
            & ~sibling_open
            & (stable_decision | decision.eq("") | decision.eq("boundary"))
        )
        signal_mask = valid_probability & (
            p_values.le(float(alpha)) | sibling_open | decision.eq("accepted_internal_split")
        )
        support_rule = "inferred_high_p_stable_boundary_anchor"
        signal_rule = "inferred_low_p_or_open_sibling_anchor"

    return (
        support_mask.fillna(False),
        signal_mask.fillna(False),
        int(selected_nonnull.sum()),
        support_rule,
        signal_rule,
    )


def _row_tau(row: pd.Series, column: str, fallback: float) -> float:
    return _finite_positive(row.get(column, math.nan), default=fallback)


def _group_h_k(
    group: pd.DataFrame,
    support_index: pd.Index,
    fallback: float,
) -> float:
    logs: list[float] = []
    for _, row in group.loc[support_index].iterrows():
        scale = _scale_value(row)
        if math.isfinite(scale) and scale > 0.0:
            logs.append(math.log(scale))
    if len(logs) > 1:
        std = float(np.std(logs, ddof=0))
        if math.isfinite(std) and std > 1e-12:
            return std
    return float(fallback)


def _group_log_scales(group: pd.DataFrame) -> np.ndarray:
    log_scales: list[float] = []
    for _, row in group.iterrows():
        scale = _scale_value(row)
        log_scales.append(math.log(scale) if math.isfinite(scale) and scale > 0.0 else math.nan)
    return np.asarray(log_scales, dtype=float)


def _group_positive_values(
    group: pd.DataFrame,
    column: str,
    fallback: float,
) -> np.ndarray:
    if column not in group:
        return np.full(len(group), float(fallback), dtype=float)
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0.0)
    values[invalid] = float(fallback)
    return values


def _distance_matrix_from_cache(
    group: pd.DataFrame,
    cache: CachedTreeDistances,
) -> np.ndarray:
    node_ids = group["node_id"].astype(str).to_numpy()
    node_to_positions: dict[str, list[int]] = {}
    for position, node_id in enumerate(node_ids):
        node_to_positions.setdefault(node_id, []).append(position)

    distances = np.full((len(group), len(group)), math.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    for (left, right), distance in cache.distances.items():
        left_positions = node_to_positions.get(str(left))
        right_positions = node_to_positions.get(str(right))
        if not left_positions or not right_positions:
            continue
        for left_position in left_positions:
            for right_position in right_positions:
                distances[left_position, right_position] = float(distance)
                distances[right_position, left_position] = float(distance)
    return distances


def _build_group_interpolation_context(
    group: pd.DataFrame,
    *,
    cache: CachedTreeDistances,
    support_mask: pd.Series,
    signal_mask: pd.Series,
    selected_nonnull_excluded_count: int,
    support_anchor_rule: str,
    signal_anchor_rule: str,
    fallback_tau_t: float,
    fallback_tau_s: float,
    fallback_h_k: float,
) -> GroupInterpolationContext:
    support_index = support_mask[support_mask].index
    group_h_k = _group_h_k(group, support_index, fallback=float(fallback_h_k))
    p_values = pd.to_numeric(group.get("sibling_p_value"), errors="coerce").to_numpy(dtype=float)
    p_values[(p_values < 0.0) | (p_values > 1.0)] = math.nan

    h_k_values = _group_positive_values(
        group,
        "topology_neighborhood_h_k",
        fallback=float(group_h_k),
    )
    return GroupInterpolationContext(
        group=group,
        cache_status=cache.status,
        row_positions={index: position for position, index in enumerate(group.index)},
        distances=_distance_matrix_from_cache(group, cache),
        p_values=p_values,
        log_scales=_group_log_scales(group),
        tau_t_values=_group_positive_values(
            group,
            "topology_neighborhood_tau_t",
            fallback=float(fallback_tau_t),
        ),
        tau_s_values=_group_positive_values(
            group,
            "topology_neighborhood_tau_s",
            fallback=float(fallback_tau_s),
        ),
        h_k_values=h_k_values,
        support_mask=support_mask.reindex(group.index).fillna(False).to_numpy(dtype=bool),
        signal_mask=signal_mask.reindex(group.index).fillna(False).to_numpy(dtype=bool),
        selected_nonnull_excluded_count=int(selected_nonnull_excluded_count),
        support_anchor_rule=support_anchor_rule,
        signal_anchor_rule=signal_anchor_rule,
        group_h_k=float(group_h_k),
        fallback_tau_t=float(fallback_tau_t),
        fallback_tau_s=float(fallback_tau_s),
        fallback_h_k=float(fallback_h_k),
    )


def compute_holdout_interpolated_p_like(
    *,
    group: pd.DataFrame,
    target_index: object,
    cache: CachedTreeDistances,
    support_mask: pd.Series,
    signal_mask: pd.Series,
    selected_nonnull_excluded_count: int = 0,
    support_anchor_rule: str = "",
    signal_anchor_rule: str = "",
    min_support: int = DEFAULT_MIN_SUPPORT,
    fallback_tau_t: float = DEFAULT_FALLBACK_TAU_T,
    fallback_tau_s: float = DEFAULT_FALLBACK_TAU_S,
    fallback_h_k: float = DEFAULT_FALLBACK_H_K,
    group_h_k: float | None = None,
) -> HoldoutInterpolatedPValue:
    """Compute a hold-out interpolated p-like value for one target row."""
    target_row = group.loc[target_index]
    target_node = target_row["node_id"]
    heldout_support = support_mask.copy()
    heldout_signal = signal_mask.copy()
    heldout_support.loc[target_index] = False
    heldout_signal.loc[target_index] = False
    support_index = heldout_support[heldout_support].index
    signal_index = heldout_signal[heldout_signal].index

    tau_t = _row_tau(
        target_row,
        "topology_neighborhood_tau_t",
        fallback=float(fallback_tau_t),
    )
    tau_s = _row_tau(
        target_row,
        "topology_neighborhood_tau_s",
        fallback=float(fallback_tau_s),
    )
    row_h_k = _finite_float(target_row.get("topology_neighborhood_h_k", math.nan))
    if math.isfinite(row_h_k) and row_h_k > 0.0:
        h_k = float(row_h_k)
    elif group_h_k is not None and math.isfinite(group_h_k) and group_h_k > 0.0:
        h_k = float(group_h_k)
    else:
        h_k = _group_h_k(group, support_index, fallback=float(fallback_h_k))

    if not cache.distances_available:
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=0.0,
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_index)),
            signal_anchor_count=int(len(signal_index)),
            selected_nonnull_excluded_count=int(selected_nonnull_excluded_count),
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=float(tau_t),
            tau_s=float(tau_s),
            h_k=float(h_k),
            tree_distance_status=cache.status,
            support_anchor_rule=support_anchor_rule,
            signal_anchor_rule=signal_anchor_rule,
            status="tree_distance_unavailable",
        )
    if len(support_index) < int(min_support):
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=0.0,
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_index)),
            signal_anchor_count=int(len(signal_index)),
            selected_nonnull_excluded_count=int(selected_nonnull_excluded_count),
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=float(tau_t),
            tau_s=float(tau_s),
            h_k=float(h_k),
            tree_distance_status=cache.status,
            support_anchor_rule=support_anchor_rule,
            signal_anchor_rule=signal_anchor_rule,
            status="support_bottleneck",
        )

    target_scale = _scale_value(target_row)
    target_log_scale = math.log(target_scale) if math.isfinite(target_scale) else math.nan
    stable_p_values: list[float] = []
    stable_weights: list[float] = []
    support_distances: list[float] = []
    for support_idx, support_row in group.loc[support_index].iterrows():
        p_value = _probability_or_nan(support_row.get("sibling_p_value", math.nan))
        if not math.isfinite(p_value):
            continue
        distance = cache.distance(target_node, support_row["node_id"])
        if not math.isfinite(distance):
            continue
        distance_weight = math.exp(-float(distance) / float(tau_t))
        support_scale = _scale_value(support_row)
        scale_weight = 1.0
        if math.isfinite(target_log_scale) and math.isfinite(support_scale) and h_k > 0.0:
            support_log_scale = math.log(support_scale)
            scale_weight = math.exp(
                -0.5 * ((support_log_scale - target_log_scale) / float(h_k)) ** 2
            )
        stable_p_values.append(float(p_value))
        stable_weights.append(float(distance_weight * scale_weight))
        support_distances.append(float(distance))

    if not stable_weights or sum(stable_weights) <= 0.0:
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=float(sum(stable_weights)),
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_index)),
            signal_anchor_count=int(len(signal_index)),
            selected_nonnull_excluded_count=int(selected_nonnull_excluded_count),
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=float(tau_t),
            tau_s=float(tau_s),
            h_k=float(h_k),
            tree_distance_status=cache.status,
            support_anchor_rule=support_anchor_rule,
            signal_anchor_rule=signal_anchor_rule,
            status="support_bottleneck_zero_weight",
        )

    signal_p_values: list[float] = []
    signal_distances: list[float] = []
    for signal_idx, signal_row in group.loc[signal_index].iterrows():
        p_value = _probability_or_nan(signal_row.get("sibling_p_value", math.nan))
        if not math.isfinite(p_value):
            continue
        distance = cache.distance(target_node, signal_row["node_id"])
        if not math.isfinite(distance):
            continue
        signal_p_values.append(float(p_value))
        signal_distances.append(float(distance))

    try:
        result = compute_child_interpolated_null_prior(
            stable_p_values=stable_p_values,
            stable_weights=stable_weights,
            signal_p_values=signal_p_values,
            signal_distances=signal_distances,
            tau_s=float(tau_s),
        )
    except ValueError:
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=float(sum(stable_weights)),
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_index)),
            signal_anchor_count=int(len(signal_index)),
            selected_nonnull_excluded_count=int(selected_nonnull_excluded_count),
            nearest_support_distance=float(min(support_distances)),
            nearest_signal_distance=math.inf,
            tau_t=float(tau_t),
            tau_s=float(tau_s),
            h_k=float(h_k),
            tree_distance_status=cache.status,
            support_anchor_rule=support_anchor_rule,
            signal_anchor_rule=signal_anchor_rule,
            status="invalid_probability_domain",
        )

    support_weight = float(sum(stable_weights))
    support_weight_square_sum = float(sum(weight * weight for weight in stable_weights))
    effective_support = (
        float((support_weight * support_weight) / support_weight_square_sum)
        if support_weight_square_sum > 0.0
        else 0.0
    )
    weighted_mean = float(
        sum(weight * value for weight, value in zip(stable_weights, stable_p_values))
        / support_weight
    )
    status = (
        "interpolated_p_like_observed_diagnostic_only"
        if signal_p_values
        else "interpolated_p_like_no_signal_attenuation_diagnostic_only"
    )
    if result.status == "invalid_probability_domain":
        status = "invalid_probability_domain"
    elif result.status == "support_bottleneck":
        status = "support_bottleneck_zero_weight"

    return HoldoutInterpolatedPValue(
        p_like=float(result.prior),
        support_weight=support_weight,
        effective_support=effective_support,
        stable_weighted_p_mean=weighted_mean,
        signal_attenuation=float(result.signal_attenuation),
        support_anchor_count=int(len(support_index)),
        signal_anchor_count=int(len(signal_index)),
        selected_nonnull_excluded_count=int(selected_nonnull_excluded_count),
        nearest_support_distance=float(min(support_distances)),
        nearest_signal_distance=(float(min(signal_distances)) if signal_distances else math.inf),
        tau_t=float(tau_t),
        tau_s=float(tau_s),
        h_k=float(h_k),
        tree_distance_status=cache.status,
        support_anchor_rule=support_anchor_rule,
        signal_anchor_rule=signal_anchor_rule,
        status=status,
    )


def compute_holdout_interpolated_p_like_fast(
    *,
    context: GroupInterpolationContext,
    target_index: object,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> HoldoutInterpolatedPValue:
    """Compute hold-out interpolation using a vectorized group context."""
    target_position = context.row_positions[target_index]
    tau_t = float(context.tau_t_values[target_position])
    tau_s = float(context.tau_s_values[target_position])
    h_k = float(context.h_k_values[target_position])

    support_mask = context.support_mask.copy()
    signal_mask = context.signal_mask.copy()
    support_mask[target_position] = False
    signal_mask[target_position] = False
    support_positions = np.flatnonzero(support_mask)
    signal_positions = np.flatnonzero(signal_mask)

    if not _tree_distances_available(context.cache_status):
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=0.0,
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_positions)),
            signal_anchor_count=int(len(signal_positions)),
            selected_nonnull_excluded_count=context.selected_nonnull_excluded_count,
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=tau_t,
            tau_s=tau_s,
            h_k=h_k,
            tree_distance_status=context.cache_status,
            support_anchor_rule=context.support_anchor_rule,
            signal_anchor_rule=context.signal_anchor_rule,
            status="tree_distance_unavailable",
        )

    if len(support_positions) < int(min_support):
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=0.0,
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_positions)),
            signal_anchor_count=int(len(signal_positions)),
            selected_nonnull_excluded_count=context.selected_nonnull_excluded_count,
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=tau_t,
            tau_s=tau_s,
            h_k=h_k,
            tree_distance_status=context.cache_status,
            support_anchor_rule=context.support_anchor_rule,
            signal_anchor_rule=context.signal_anchor_rule,
            status="support_bottleneck",
        )

    support_distances_all = context.distances[target_position, support_positions]
    stable_p_all = context.p_values[support_positions]
    valid_support = np.isfinite(support_distances_all) & np.isfinite(stable_p_all)
    support_distances = support_distances_all[valid_support]
    stable_p_values = stable_p_all[valid_support]

    if support_distances.size == 0:
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=0.0,
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_positions)),
            signal_anchor_count=int(len(signal_positions)),
            selected_nonnull_excluded_count=context.selected_nonnull_excluded_count,
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=tau_t,
            tau_s=tau_s,
            h_k=h_k,
            tree_distance_status=context.cache_status,
            support_anchor_rule=context.support_anchor_rule,
            signal_anchor_rule=context.signal_anchor_rule,
            status="support_bottleneck_zero_weight",
        )

    stable_weights = np.exp(-support_distances / tau_t)
    target_log_scale = context.log_scales[target_position]
    if math.isfinite(target_log_scale) and h_k > 0.0:
        support_log_scales = context.log_scales[support_positions][valid_support]
        finite_support_scale = np.isfinite(support_log_scales)
        scale_weights = np.ones_like(stable_weights)
        scale_weights[finite_support_scale] = np.exp(
            -0.5 * ((support_log_scales[finite_support_scale] - target_log_scale) / h_k) ** 2
        )
        stable_weights = stable_weights * scale_weights

    support_weight = float(stable_weights.sum())
    if support_weight <= 0.0:
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=support_weight,
            effective_support=0.0,
            stable_weighted_p_mean=math.nan,
            signal_attenuation=math.nan,
            support_anchor_count=int(len(support_positions)),
            signal_anchor_count=int(len(signal_positions)),
            selected_nonnull_excluded_count=context.selected_nonnull_excluded_count,
            nearest_support_distance=math.inf,
            nearest_signal_distance=math.inf,
            tau_t=tau_t,
            tau_s=tau_s,
            h_k=h_k,
            tree_distance_status=context.cache_status,
            support_anchor_rule=context.support_anchor_rule,
            signal_anchor_rule=context.signal_anchor_rule,
            status="support_bottleneck_zero_weight",
        )

    stable_weighted_p_mean = float(np.dot(stable_weights, stable_p_values) / support_weight)
    support_weight_square_sum = float(np.dot(stable_weights, stable_weights))
    effective_support = (
        float((support_weight * support_weight) / support_weight_square_sum)
        if support_weight_square_sum > 0.0
        else 0.0
    )

    signal_distances = np.array([], dtype=float)
    signal_attenuation = 0.0
    valid_signal_count = 0
    if len(signal_positions):
        signal_distances_all = context.distances[target_position, signal_positions]
        signal_p_all = context.p_values[signal_positions]
        valid_signal = np.isfinite(signal_distances_all) & np.isfinite(signal_p_all)
        signal_distances = signal_distances_all[valid_signal]
        signal_p_values = signal_p_all[valid_signal]
        valid_signal_count = int(signal_p_values.size)
        if valid_signal_count:
            signal_attenuation = float(
                np.max((1.0 - signal_p_values) * np.exp(-signal_distances / tau_s))
            )

    p_like = stable_weighted_p_mean * (1.0 - signal_attenuation)
    if p_like < -PROBABILITY_TOLERANCE or p_like > 1.0 + PROBABILITY_TOLERANCE:
        return HoldoutInterpolatedPValue(
            p_like=math.nan,
            support_weight=support_weight,
            effective_support=effective_support,
            stable_weighted_p_mean=stable_weighted_p_mean,
            signal_attenuation=signal_attenuation,
            support_anchor_count=int(len(support_positions)),
            signal_anchor_count=int(len(signal_positions)),
            selected_nonnull_excluded_count=context.selected_nonnull_excluded_count,
            nearest_support_distance=float(np.min(support_distances)),
            nearest_signal_distance=(
                float(np.min(signal_distances)) if signal_distances.size else math.inf
            ),
            tau_t=tau_t,
            tau_s=tau_s,
            h_k=h_k,
            tree_distance_status=context.cache_status,
            support_anchor_rule=context.support_anchor_rule,
            signal_anchor_rule=context.signal_anchor_rule,
            status="invalid_probability_domain",
        )

    status = (
        "interpolated_p_like_observed_diagnostic_only"
        if valid_signal_count
        else "interpolated_p_like_no_signal_attenuation_diagnostic_only"
    )
    return HoldoutInterpolatedPValue(
        p_like=float(min(max(p_like, 0.0), 1.0)),
        support_weight=support_weight,
        effective_support=effective_support,
        stable_weighted_p_mean=stable_weighted_p_mean,
        signal_attenuation=signal_attenuation,
        support_anchor_count=int(len(support_positions)),
        signal_anchor_count=int(len(signal_positions)),
        selected_nonnull_excluded_count=context.selected_nonnull_excluded_count,
        nearest_support_distance=float(np.min(support_distances)),
        nearest_signal_distance=(
            float(np.min(signal_distances)) if signal_distances.size else math.inf
        ),
        tau_t=tau_t,
        tau_s=tau_s,
        h_k=h_k,
        tree_distance_status=context.cache_status,
        support_anchor_rule=context.support_anchor_rule,
        signal_anchor_rule=context.signal_anchor_rule,
        status=status,
    )


def _comparison_class(
    *,
    direct_p: float,
    interpolated_p: float,
    interpolation_status: str,
) -> str:
    if not _is_computed_status(interpolation_status):
        return "interpolation_unavailable"
    if not math.isfinite(direct_p):
        return "direct_unavailable"
    delta = float(interpolated_p) - float(direct_p)
    if abs(delta) <= 1e-12:
        return "interpolation_equal_direct"
    if delta < 0.0:
        return "interpolation_lower_than_direct"
    return "interpolation_higher_than_direct"


def _behavior_label(
    *,
    data_role: str,
    direct_significant: bool,
    interpolated_significant: bool,
    interpolation_status: str,
) -> str:
    if not _is_computed_status(interpolation_status):
        return interpolation_status
    role = data_role.strip().lower()
    if role == "selected_null":
        if direct_significant and interpolated_significant:
            return "selected_null_false_signal_reinforced"
        if direct_significant and not interpolated_significant:
            return "selected_null_false_signal_suppressed"
        if not direct_significant and interpolated_significant:
            return "selected_null_false_open_risk"
        return "selected_null_conservative"
    if role == "signal":
        if direct_significant and interpolated_significant:
            return "signal_confirmed_by_interpolation"
        if direct_significant and not interpolated_significant:
            return "signal_not_caught_by_interpolation"
        if not direct_significant and interpolated_significant:
            return "signal_extra_open_candidate"
        return "signal_conservative_closed"
    if interpolated_significant and not direct_significant:
        return "unlabeled_extra_open_candidate"
    return "unlabeled_diagnostic"


def _build_group_rows(
    group: pd.DataFrame,
    *,
    target_index: pd.Index,
    alpha: float,
    support_p_floor: float,
    min_support: int,
    fallback_tau_t: float,
    fallback_tau_s: float,
    fallback_h_k: float,
) -> list[dict[str, object]]:
    cache = build_cached_tree_distances(group)
    support_mask, signal_mask, excluded_count, support_rule, signal_rule = (
        _support_and_signal_masks(
            group,
            alpha=float(alpha),
            support_p_floor=float(support_p_floor),
        )
    )
    context = _build_group_interpolation_context(
        group,
        cache=cache,
        support_mask=support_mask,
        signal_mask=signal_mask,
        selected_nonnull_excluded_count=excluded_count,
        support_anchor_rule=support_rule,
        signal_anchor_rule=signal_rule,
        fallback_tau_t=float(fallback_tau_t),
        fallback_tau_s=float(fallback_tau_s),
        fallback_h_k=float(fallback_h_k),
    )
    records: list[dict[str, object]] = []
    for idx in target_index:
        row = group.loc[idx]
        interpolation = compute_holdout_interpolated_p_like_fast(
            context=context,
            target_index=idx,
            min_support=int(min_support),
        )
        direct_p = _probability_or_nan(row.get("sibling_p_value", math.nan))
        direct_open = _bool_value(row.get("sibling_open", False))
        direct_significant = bool(
            direct_open or (math.isfinite(direct_p) and direct_p <= float(alpha))
        )
        interpolated_significant = bool(
            math.isfinite(interpolation.p_like) and interpolation.p_like <= float(alpha)
        )
        direct_log = _safe_neg_log10(direct_p)
        interpolated_log = _safe_neg_log10(interpolation.p_like)
        required_tau_s = _best_case_required_tau_s_for_alpha(
            stable_weighted_p_mean=interpolation.stable_weighted_p_mean,
            nearest_signal_distance=interpolation.nearest_signal_distance,
            alpha=float(alpha),
        )
        comparison = _comparison_class(
            direct_p=direct_p,
            interpolated_p=interpolation.p_like,
            interpolation_status=interpolation.status,
        )
        behavior = _behavior_label(
            data_role=_string_value(row, "data_role"),
            direct_significant=direct_significant,
            interpolated_significant=interpolated_significant,
            interpolation_status=interpolation.status,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": _string_value(row, "case_id"),
                "data_role": _string_value(row, "data_role"),
                "method_id": _string_value(row, "method_id"),
                "replicate": row.get("replicate", math.nan),
                "node_id": _string_value(row, "node_id"),
                "parent_id": _string_value(row, "parent_id"),
                "branch_length_to_parent": _finite_float(
                    row.get("branch_length_to_parent", math.nan)
                ),
                "decision_class": _string_value(row, "decision_class"),
                "traversal_decision": _string_value(row, "traversal_decision"),
                "direct_sibling_p_value": direct_p,
                "direct_sibling_open": direct_open,
                "direct_significant": direct_significant,
                "interpolated_sibling_null_p_like": interpolation.p_like,
                "interpolated_significant": interpolated_significant,
                "interpolated_minus_direct": (
                    float(interpolation.p_like - direct_p)
                    if math.isfinite(interpolation.p_like) and math.isfinite(direct_p)
                    else math.nan
                ),
                "interpolated_to_direct_ratio": (
                    float(interpolation.p_like / direct_p)
                    if (
                        math.isfinite(interpolation.p_like)
                        and math.isfinite(direct_p)
                        and direct_p > 0.0
                    )
                    else math.nan
                ),
                "negative_log10_direct_p": direct_log,
                "negative_log10_interpolated_p_like": interpolated_log,
                "negative_log10_delta": (
                    float(interpolated_log - direct_log)
                    if math.isfinite(interpolated_log) and math.isfinite(direct_log)
                    else math.nan
                ),
                "support_anchor_count": interpolation.support_anchor_count,
                "signal_anchor_count": interpolation.signal_anchor_count,
                "selected_nonnull_excluded_count": (interpolation.selected_nonnull_excluded_count),
                "support_weight": interpolation.support_weight,
                "effective_support": interpolation.effective_support,
                "stable_weighted_p_mean": interpolation.stable_weighted_p_mean,
                "signal_attenuation": interpolation.signal_attenuation,
                "nearest_support_distance": interpolation.nearest_support_distance,
                "nearest_signal_distance": interpolation.nearest_signal_distance,
                "distance_to_stopping_edge": _finite_float(
                    row.get("distance_to_stopping_edge", math.nan)
                ),
                "tau_b": _finite_float(row.get("topology_neighborhood_tau_b", math.nan)),
                "tau_t": interpolation.tau_t,
                "tau_s": interpolation.tau_s,
                "h_k": interpolation.h_k,
                "best_case_required_tau_s_for_alpha": required_tau_s,
                "tree_distance_status": interpolation.tree_distance_status,
                "support_anchor_rule": interpolation.support_anchor_rule,
                "signal_anchor_rule": interpolation.signal_anchor_rule,
                "interpolation_status": interpolation.status,
                "comparison_class": comparison,
                "behavior_label": behavior,
            }
        )
    return records


def build_pvalue_interpolation_comparison_rows(
    rows: pd.DataFrame,
    *,
    alpha: float = DEFAULT_ALPHA,
    support_p_floor: float = DEFAULT_SUPPORT_P_FLOOR,
    min_support: int = DEFAULT_MIN_SUPPORT,
    fallback_tau_t: float = DEFAULT_FALLBACK_TAU_T,
    fallback_tau_s: float = DEFAULT_FALLBACK_TAU_S,
    fallback_h_k: float = DEFAULT_FALLBACK_H_K,
    candidate_only: bool = False,
) -> pd.DataFrame:
    """Return hold-out direct-vs-interpolated p-value comparison rows."""
    target_mask = (
        candidate_evidence_mask(rows) if candidate_only else pd.Series(True, index=rows.index)
    )
    records: list[dict[str, object]] = []
    for _, group in rows.groupby(list(GROUP_COLUMNS), dropna=False, sort=True):
        group_targets = group.index.intersection(target_mask[target_mask].index)
        if group_targets.empty:
            continue
        records.extend(
            _build_group_rows(
                group,
                target_index=group_targets,
                alpha=float(alpha),
                support_p_floor=float(support_p_floor),
                min_support=int(min_support),
                fallback_tau_t=float(fallback_tau_t),
                fallback_tau_s=float(fallback_tau_s),
                fallback_h_k=float(fallback_h_k),
            )
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_pvalue_interpolation_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize hold-out interpolation comparison behavior."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    group_columns = ["data_role", "method_id", "comparison_class", "behavior_label"]
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        computed = group["interpolation_status"].map(_is_computed_status)
        false_open = group["behavior_label"].isin(
            {
                "selected_null_false_open_risk",
                "selected_null_false_signal_reinforced",
            }
        )
        signal_catch = group["behavior_label"].isin(
            {
                "signal_confirmed_by_interpolation",
                "signal_extra_open_candidate",
            }
        )
        signal_miss = group["behavior_label"].eq("signal_not_caught_by_interpolation")
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "data_role": keys[0],
                "method_id": keys[1],
                "comparison_class": keys[2],
                "behavior_label": keys[3],
                "row_count": int(len(group)),
                "direct_significant_count": int(group["direct_significant"].sum()),
                "interpolated_computed_count": int(computed.sum()),
                "interpolated_significant_count": int(group["interpolated_significant"].sum()),
                "interpolation_lower_count": int(
                    group["comparison_class"].eq("interpolation_lower_than_direct").sum()
                ),
                "interpolation_higher_count": int(
                    group["comparison_class"].eq("interpolation_higher_than_direct").sum()
                ),
                "median_direct_p": _safe_median(group["direct_sibling_p_value"]),
                "median_interpolated_p_like": _safe_median(
                    group["interpolated_sibling_null_p_like"]
                ),
                "median_negative_log10_delta": _safe_median(group["negative_log10_delta"]),
                "median_best_case_required_tau_s_for_alpha": _safe_median(
                    group["best_case_required_tau_s_for_alpha"]
                ),
                "p90_best_case_required_tau_s_for_alpha": _safe_quantile(
                    group["best_case_required_tau_s_for_alpha"],
                    0.90,
                ),
                "false_open_risk_count": int(false_open.sum()),
                "signal_catch_count": int(signal_catch.sum()),
                "signal_miss_count": int(signal_miss.sum()),
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def summarize_pvalue_interpolation_cases(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize interpolation comparison by benchmark case and method."""
    if rows.empty:
        return pd.DataFrame(columns=CASE_SUMMARY_COLUMNS)
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(["case_id", "data_role", "method_id"], dropna=False):
        computed = group["interpolation_status"].map(_is_computed_status)
        behavior_counts = group["behavior_label"].value_counts(dropna=False)
        dominant_behavior = str(behavior_counts.index[0]) if not behavior_counts.empty else ""
        if not bool(computed.any()):
            status = "interpolation_unavailable"
        elif bool(group["behavior_label"].str.contains("false_open").any()):
            status = "interpolation_false_open_risk"
        elif bool(group["behavior_label"].str.contains("not_caught").any()):
            status = "interpolation_misses_some_signal"
        else:
            status = "interpolation_behavior_observed_diagnostic_only"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": keys[0],
                "data_role": keys[1],
                "method_id": keys[2],
                "row_count": int(len(group)),
                "direct_significant_count": int(group["direct_significant"].sum()),
                "interpolated_computed_count": int(computed.sum()),
                "interpolated_significant_count": int(group["interpolated_significant"].sum()),
                "interpolation_lower_count": int(
                    group["comparison_class"].eq("interpolation_lower_than_direct").sum()
                ),
                "interpolation_higher_count": int(
                    group["comparison_class"].eq("interpolation_higher_than_direct").sum()
                ),
                "median_direct_p": _safe_median(group["direct_sibling_p_value"]),
                "median_interpolated_p_like": _safe_median(
                    group["interpolated_sibling_null_p_like"]
                ),
                "median_negative_log10_delta": _safe_median(group["negative_log10_delta"]),
                "median_best_case_required_tau_s_for_alpha": _safe_median(
                    group["best_case_required_tau_s_for_alpha"]
                ),
                "p90_best_case_required_tau_s_for_alpha": _safe_quantile(
                    group["best_case_required_tau_s_for_alpha"],
                    0.90,
                ),
                "dominant_behavior_label": dominant_behavior,
                "diagnostic_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=CASE_SUMMARY_COLUMNS)


def summarize_tau_s_sensitivity(
    rows: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_TAU_S_SENSITIVITY_GRID,
) -> pd.DataFrame:
    """Summarize optimistic signal-bandwidth tradeoffs by role and method."""
    if rows.empty:
        return pd.DataFrame(columns=TAU_S_SENSITIVITY_COLUMNS)
    direct = rows.loc[rows["direct_significant"].astype(bool)].copy()
    if direct.empty:
        return pd.DataFrame(columns=TAU_S_SENSITIVITY_COLUMNS)
    records: list[dict[str, object]] = []
    for keys, group in direct.groupby(["data_role", "method_id"], dropna=False):
        required = pd.to_numeric(
            group["best_case_required_tau_s_for_alpha"],
            errors="coerce",
        )
        finite_required = required[np.isfinite(required)]
        for threshold in thresholds:
            count = int(finite_required.le(float(threshold)).sum())
            total = int(len(group))
            role = str(keys[0])
            if role == "signal":
                label = "signal_recovered_best_case"
            elif role == "selected_null":
                label = "selected_null_reopened_best_case"
            else:
                label = "unlabeled_best_case"
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "data_role": keys[0],
                    "method_id": keys[1],
                    "tau_s_threshold": float(threshold),
                    "direct_significant_count": total,
                    "best_case_significant_count": count,
                    "best_case_significant_fraction": (float(count / total) if total else math.nan),
                    "sensitivity_label": label,
                }
            )
    return pd.DataFrame.from_records(records, columns=TAU_S_SENSITIVITY_COLUMNS)


def _finite_required_tau_s(group: pd.DataFrame) -> pd.Series:
    required = pd.to_numeric(
        group["best_case_required_tau_s_for_alpha"],
        errors="coerce",
    )
    return required[np.isfinite(required)]


def _tau_s_range_status(
    *,
    signal_count: int,
    finite_signal_count: int,
    selected_null_count: int,
    finite_selected_null_count: int,
    lower_bound: float,
    upper_bound: float,
) -> str:
    if signal_count <= 0:
        return "no_direct_signal_rows"
    if finite_signal_count <= 0 or not math.isfinite(lower_bound):
        return "no_finite_signal_tau_s_lower_bound"
    if selected_null_count <= 0:
        return "tau_s_range_open_ended_no_selected_null_direct_rows"
    if finite_selected_null_count <= 0 or not math.isfinite(upper_bound):
        return "no_finite_selected_null_tau_s_upper_bound"
    if lower_bound <= upper_bound:
        return "tau_s_range_admissible_diagnostic"
    return "tau_s_range_empty_selected_null_reopens_first"


def summarize_tau_s_range(
    rows: pd.DataFrame,
    *,
    target_signal_fractions: tuple[float, ...] = (DEFAULT_TAU_S_RANGE_SIGNAL_FRACTIONS),
    max_selected_null_fractions: tuple[float, ...] = (DEFAULT_TAU_S_RANGE_SELECTED_NULL_FRACTIONS),
) -> pd.DataFrame:
    """Estimate admissible tau_s intervals from optimistic row thresholds.

    For a target signal recovery fraction r, the lower endpoint is the
    r-quantile of required signal tau_s values. For a selected-null leak budget
    ell, the upper endpoint is the ell-quantile of required selected-null
    tau_s values. An interval exists only when lower <= upper.
    """
    if rows.empty:
        return pd.DataFrame(columns=TAU_S_RANGE_COLUMNS)
    direct = rows.loc[rows["direct_significant"].astype(bool)].copy()
    if direct.empty:
        return pd.DataFrame(columns=TAU_S_RANGE_COLUMNS)

    records: list[dict[str, object]] = []
    for method_id, method_group in direct.groupby("method_id", dropna=False):
        signal = method_group.loc[method_group["data_role"].astype(str).eq("signal")]
        selected_null = method_group.loc[method_group["data_role"].astype(str).eq("selected_null")]
        signal_required = _finite_required_tau_s(signal)
        selected_null_required = _finite_required_tau_s(selected_null)
        signal_count = int(len(signal))
        selected_null_count = int(len(selected_null))
        finite_signal_count = int(len(signal_required))
        finite_selected_null_count = int(len(selected_null_required))

        for signal_fraction in target_signal_fractions:
            target_fraction = float(signal_fraction)
            lower_bound = (
                float(signal_required.quantile(target_fraction))
                if finite_signal_count > 0
                else math.nan
            )
            for selected_null_fraction in max_selected_null_fractions:
                leak_fraction = float(selected_null_fraction)
                upper_bound = (
                    float(selected_null_required.quantile(leak_fraction))
                    if finite_selected_null_count > 0
                    else math.inf
                )
                status = _tau_s_range_status(
                    signal_count=signal_count,
                    finite_signal_count=finite_signal_count,
                    selected_null_count=selected_null_count,
                    finite_selected_null_count=finite_selected_null_count,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                width = (
                    float(upper_bound - lower_bound)
                    if math.isfinite(lower_bound) and math.isfinite(upper_bound)
                    else math.inf
                    if math.isfinite(lower_bound) and math.isinf(upper_bound)
                    else math.nan
                )
                midpoint = (
                    float((lower_bound + upper_bound) / 2.0)
                    if math.isfinite(lower_bound) and math.isfinite(upper_bound)
                    else math.nan
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "method_id": method_id,
                        "target_signal_fraction": target_fraction,
                        "max_selected_null_fraction": leak_fraction,
                        "signal_direct_significant_count": signal_count,
                        "selected_null_direct_significant_count": selected_null_count,
                        "finite_signal_tau_s_count": finite_signal_count,
                        "finite_selected_null_tau_s_count": (finite_selected_null_count),
                        "signal_tau_s_lower_bound": lower_bound,
                        "selected_null_tau_s_upper_bound": upper_bound,
                        "admissible_tau_s_width": width,
                        "admissible_tau_s_midpoint": midpoint,
                        "tau_s_range_status": status,
                    }
                )
    return pd.DataFrame.from_records(records, columns=TAU_S_RANGE_COLUMNS)


def _finite_numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _finite_count(values: pd.Series) -> int:
    return int(len(_finite_numeric_series(values)))


def _safe_quantile_or_nan(values: pd.Series, quantile: float) -> float:
    finite = _finite_numeric_series(values)
    if finite.empty:
        return math.nan
    return float(finite.quantile(float(quantile)))


def _observed_role_median(region: pd.DataFrame, role: str, column: str) -> float:
    role_rows = region.loc[region["data_role"].astype(str).eq(role)]
    if role_rows.empty or column not in role_rows:
        return math.nan
    return _safe_median(role_rows[column])


def _bandwidth_quantile_fields(group: pd.DataFrame, column: str, prefix: str) -> dict[str, object]:
    if column not in group:
        empty = pd.Series(dtype=float)
        return {
            f"{prefix}_finite_count": 0,
            f"{prefix}_q10": _safe_quantile_or_nan(empty, 0.10),
            f"{prefix}_median": _safe_quantile_or_nan(empty, 0.50),
            f"{prefix}_q90": _safe_quantile_or_nan(empty, 0.90),
        }
    values = group[column]
    return {
        f"{prefix}_finite_count": _finite_count(values),
        f"{prefix}_q10": _safe_quantile_or_nan(values, 0.10),
        f"{prefix}_median": _safe_quantile_or_nan(values, 0.50),
        f"{prefix}_q90": _safe_quantile_or_nan(values, 0.90),
    }


def _region_bandwidth_status(group: pd.DataFrame) -> str:
    row_count = int(len(group))
    tau_b_count = _finite_count(group["tau_b"]) if "tau_b" in group else 0
    tau_t_count = _finite_count(group["tau_t"]) if "tau_t" in group else 0
    tau_s_count = _finite_count(group["tau_s"]) if "tau_s" in group else 0
    h_k_count = _finite_count(group["h_k"]) if "h_k" in group else 0
    if (
        row_count > 0
        and tau_b_count == row_count
        and tau_t_count == row_count
        and tau_s_count == row_count
        and h_k_count == row_count
    ):
        return "observed_full_bandwidth_vector_diagnostic"
    if tau_b_count and tau_t_count and tau_s_count and h_k_count:
        return "observed_sparse_tau_b_bandwidth_vector_diagnostic"
    if tau_t_count and tau_s_count and h_k_count:
        return "observed_partial_bandwidth_vector_without_tau_b"
    return "bandwidth_vector_incomplete"


def summarize_region_bandwidths(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed bandwidth coordinates by selected topology region.

    A region here is a single selected tree for one benchmark case, data role,
    method, and replicate. This is role-specific because the signal and
    selected-null trees are generated separately.
    """
    if rows.empty:
        return pd.DataFrame(columns=REGION_BANDWIDTH_COLUMNS)

    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(
        list(ROLE_TOPOLOGY_REGION_COLUMNS),
        dropna=False,
        sort=True,
    ):
        computed = group["interpolation_status"].map(_is_computed_status)
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "study_role": STUDY_ROLE,
            "case_id": keys[0],
            "data_role": keys[1],
            "method_id": keys[2],
            "replicate": keys[3],
            "row_count": int(len(group)),
            "direct_significant_count": int(group["direct_significant"].sum()),
            "interpolated_computed_count": int(computed.sum()),
            "interpolated_significant_count": int(group["interpolated_significant"].sum()),
        }
        for column, prefix in (
            ("tau_b", "tau_b"),
            ("tau_t", "tau_t"),
            ("tau_s", "tau_s"),
            ("h_k", "h_k"),
            ("distance_to_stopping_edge", "distance_to_stopping_edge"),
            ("best_case_required_tau_s_for_alpha", "best_case_required_tau_s"),
        ):
            record.update(_bandwidth_quantile_fields(group, column, prefix))
        record.update(
            {
                "effective_support_median": _safe_median(group["effective_support"]),
                "nearest_support_distance_median": _safe_median(group["nearest_support_distance"]),
                "nearest_signal_distance_median": _safe_median(group["nearest_signal_distance"]),
                "region_bandwidth_status": _region_bandwidth_status(group),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=REGION_BANDWIDTH_COLUMNS)


def summarize_region_tau_s_ranges(
    rows: pd.DataFrame,
    *,
    target_signal_fractions: tuple[float, ...] = (DEFAULT_TAU_S_RANGE_SIGNAL_FRACTIONS),
    max_selected_null_fractions: tuple[float, ...] = (DEFAULT_TAU_S_RANGE_SELECTED_NULL_FRACTIONS),
) -> pd.DataFrame:
    """Estimate tau_s intervals separately for each benchmark topology region."""
    if rows.empty:
        return pd.DataFrame(columns=REGION_TAU_S_RANGE_COLUMNS)
    direct = rows.loc[rows["direct_significant"].astype(bool)].copy()
    if direct.empty:
        return pd.DataFrame(columns=REGION_TAU_S_RANGE_COLUMNS)

    all_region_groups = {
        tuple(keys): group
        for keys, group in rows.groupby(
            list(TOPOLOGY_REGION_COLUMNS),
            dropna=False,
            sort=True,
        )
    }

    records: list[dict[str, object]] = []
    for keys, region_direct in direct.groupby(
        list(TOPOLOGY_REGION_COLUMNS),
        dropna=False,
        sort=True,
    ):
        region_key = tuple(keys)
        region = all_region_groups.get(region_key, region_direct)
        signal = region_direct.loc[region_direct["data_role"].astype(str).eq("signal")]
        selected_null = region_direct.loc[
            region_direct["data_role"].astype(str).eq("selected_null")
        ]
        signal_required = _finite_required_tau_s(signal)
        selected_null_required = _finite_required_tau_s(selected_null)
        signal_count = int(len(signal))
        selected_null_count = int(len(selected_null))
        finite_signal_count = int(len(signal_required))
        finite_selected_null_count = int(len(selected_null_required))

        observed_medians = {
            "signal_observed_tau_b_median": _observed_role_median(region, "signal", "tau_b"),
            "selected_null_observed_tau_b_median": _observed_role_median(
                region, "selected_null", "tau_b"
            ),
            "signal_observed_tau_t_median": _observed_role_median(region, "signal", "tau_t"),
            "selected_null_observed_tau_t_median": _observed_role_median(
                region, "selected_null", "tau_t"
            ),
            "signal_observed_tau_s_median": _observed_role_median(region, "signal", "tau_s"),
            "selected_null_observed_tau_s_median": _observed_role_median(
                region, "selected_null", "tau_s"
            ),
            "signal_observed_h_k_median": _observed_role_median(region, "signal", "h_k"),
            "selected_null_observed_h_k_median": _observed_role_median(
                region, "selected_null", "h_k"
            ),
        }

        for signal_fraction in target_signal_fractions:
            target_fraction = float(signal_fraction)
            lower_bound = (
                float(signal_required.quantile(target_fraction))
                if finite_signal_count > 0
                else math.nan
            )
            for selected_null_fraction in max_selected_null_fractions:
                leak_fraction = float(selected_null_fraction)
                upper_bound = (
                    float(selected_null_required.quantile(leak_fraction))
                    if finite_selected_null_count > 0
                    else math.inf
                )
                status = _tau_s_range_status(
                    signal_count=signal_count,
                    finite_signal_count=finite_signal_count,
                    selected_null_count=selected_null_count,
                    finite_selected_null_count=finite_selected_null_count,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                width = (
                    float(upper_bound - lower_bound)
                    if math.isfinite(lower_bound) and math.isfinite(upper_bound)
                    else math.inf
                    if math.isfinite(lower_bound) and math.isinf(upper_bound)
                    else math.nan
                )
                midpoint = (
                    float((lower_bound + upper_bound) / 2.0)
                    if math.isfinite(lower_bound) and math.isfinite(upper_bound)
                    else math.nan
                )
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_role": STUDY_ROLE,
                        "case_id": region_key[0],
                        "method_id": region_key[1],
                        "replicate": region_key[2],
                        "target_signal_fraction": target_fraction,
                        "max_selected_null_fraction": leak_fraction,
                        "signal_direct_significant_count": signal_count,
                        "selected_null_direct_significant_count": selected_null_count,
                        "finite_signal_tau_s_count": finite_signal_count,
                        "finite_selected_null_tau_s_count": finite_selected_null_count,
                        "signal_tau_s_lower_bound": lower_bound,
                        "selected_null_tau_s_upper_bound": upper_bound,
                        "admissible_tau_s_width": width,
                        "admissible_tau_s_midpoint": midpoint,
                        **observed_medians,
                        "tau_s_range_status": status,
                    }
                )
    return pd.DataFrame.from_records(records, columns=REGION_TAU_S_RANGE_COLUMNS)


def run_pvalue_interpolation_comparison(
    *,
    rows_path: Path,
    output_dir: Path,
    alpha: float = DEFAULT_ALPHA,
    support_p_floor: float = DEFAULT_SUPPORT_P_FLOOR,
    min_support: int = DEFAULT_MIN_SUPPORT,
    fallback_tau_t: float = DEFAULT_FALLBACK_TAU_T,
    fallback_tau_s: float = DEFAULT_FALLBACK_TAU_S,
    fallback_h_k: float = DEFAULT_FALLBACK_H_K,
    candidate_only: bool = False,
) -> dict[str, Path]:
    input_rows = pd.read_csv(rows_path)
    comparison_rows = build_pvalue_interpolation_comparison_rows(
        input_rows,
        alpha=float(alpha),
        support_p_floor=float(support_p_floor),
        min_support=int(min_support),
        fallback_tau_t=float(fallback_tau_t),
        fallback_tau_s=float(fallback_tau_s),
        fallback_h_k=float(fallback_h_k),
        candidate_only=bool(candidate_only),
    )
    summary = summarize_pvalue_interpolation_comparison(comparison_rows)
    case_summary = summarize_pvalue_interpolation_cases(comparison_rows)
    tau_s_sensitivity = summarize_tau_s_sensitivity(comparison_rows)
    tau_s_range = summarize_tau_s_range(comparison_rows)
    region_bandwidths = summarize_region_bandwidths(comparison_rows)
    region_tau_s_ranges = summarize_region_tau_s_ranges(comparison_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out = output_dir / "selected_neighborhood_pvalue_interpolation_rows.csv"
    summary_out = output_dir / "selected_neighborhood_pvalue_interpolation_summary.csv"
    case_summary_out = output_dir / "selected_neighborhood_pvalue_interpolation_case_summary.csv"
    tau_s_sensitivity_out = (
        output_dir / "selected_neighborhood_pvalue_interpolation_tau_s_sensitivity.csv"
    )
    tau_s_range_out = output_dir / "selected_neighborhood_pvalue_interpolation_tau_s_range.csv"
    region_bandwidths_out = (
        output_dir / "selected_neighborhood_pvalue_interpolation_region_bandwidths.csv"
    )
    region_tau_s_ranges_out = (
        output_dir / "selected_neighborhood_pvalue_interpolation_region_tau_s_range.csv"
    )
    manifest_out = output_dir / "manifest.json"
    comparison_rows.to_csv(rows_out, index=False)
    summary.to_csv(summary_out, index=False)
    case_summary.to_csv(case_summary_out, index=False)
    tau_s_sensitivity.to_csv(tau_s_sensitivity_out, index=False)
    tau_s_range.to_csv(tau_s_range_out, index=False)
    region_bandwidths.to_csv(region_bandwidths_out, index=False)
    region_tau_s_ranges.to_csv(region_tau_s_ranges_out, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": rows_path,
        "alpha": float(alpha),
        "support_p_floor": float(support_p_floor),
        "min_support": int(min_support),
        "fallback_tau_t": float(fallback_tau_t),
        "fallback_tau_s": float(fallback_tau_s),
        "fallback_h_k": float(fallback_h_k),
        "candidate_only": bool(candidate_only),
        "input_row_count": int(len(input_rows)),
        "output_row_count": int(len(comparison_rows)),
        "outputs": {
            "rows": rows_out,
            "summary": summary_out,
            "case_summary": case_summary_out,
            "tau_s_sensitivity": tau_s_sensitivity_out,
            "tau_s_range": tau_s_range_out,
            "region_bandwidths": region_bandwidths_out,
            "region_tau_s_range": region_tau_s_ranges_out,
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    return {
        "rows": rows_out,
        "summary": summary_out,
        "case_summary": case_summary_out,
        "tau_s_sensitivity": tau_s_sensitivity_out,
        "tau_s_range": tau_s_range_out,
        "region_bandwidths": region_bandwidths_out,
        "region_tau_s_range": region_tau_s_ranges_out,
        "manifest": manifest_out,
    }


def main() -> None:
    args = parse_args()
    run_pvalue_interpolation_comparison(
        rows_path=args.rows,
        output_dir=args.output_dir,
        alpha=args.alpha,
        support_p_floor=args.support_p_floor,
        min_support=args.min_support,
        fallback_tau_t=args.fallback_tau_t,
        fallback_tau_s=args.fallback_tau_s,
        fallback_h_k=args.fallback_h_k,
        candidate_only=bool(args.candidate_only),
    )


if __name__ == "__main__":
    main()
