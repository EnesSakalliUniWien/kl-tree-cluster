"""Conditional topology traversal-law diagnostics.

This panel converts the current context-negative topology evidence into an
explicit directed-neighborhood law. It is non-cross-fit and non-permutation:
rows are scored from local income/outcome topology features only, then marked
fail-closed unless their incidence stratum has enough labeled support.

The output is diagnostic-only. It is meant to show whether the selected
neighborhood topology vector is a plausible conditioning object before any
production calibration rule is promoted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.diagnostics.calibration.overlap_context_negative_bayesian_topology_law import (
    beta_log_likelihood_ratio,
)
from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_overlap_conditional_topology_law_not_calibration"
SCHEMA_VERSION = "overlap_conditional_topology_law/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "overlap_conditional_topology_law_panel"
)

DEFAULT_PRIOR_TRUTH_PROBABILITY = 0.05
DEFAULT_CONTEXT_PENALTY_WEIGHT = 50.0
DEFAULT_SELECTED_FAMILY_WEIGHT = 0.25
DEFAULT_NEIGHBORHOOD_SCALE_WEIGHT = 1.0
DEFAULT_TOPOLOGY_NEIGHBORHOOD_WEIGHT = 1.0
DEFAULT_ROOT_LOG_ODDS_PENALTY = 1.0
DEFAULT_PASSTHROUGH_LOG_ODDS_PENALTY = 1.5
DEFAULT_MIN_TRUTH_SUPPORT_PER_STRATUM = 2
DEFAULT_GUARDED_RECOVERY_BALANCE_PRODUCT_FLOOR = 0.22
DEFAULT_GUARDED_RECOVERY_OUTGOING_EDGE_NORM_FLOOR = 0.95

REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "depth",
    "decision_class",
    "traversal_decision",
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "selected_family_log_bayes_factor_lower",
    "continuous_context_min_margin",
}

OPTIONAL_COLUMNS = {
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
    "n_children",
    "parent_id",
    "analytical_case",
    "neighborhood_scale",
    "topology_support_role",
    "topology_signal_role",
    "distance_to_stopping_edge",
    "balance_product",
    "root_stability_guard_blocked",
    "root_selective_guard_blocked",
    "selected_family_guard_blocked",
}

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "analytical_case",
    "incidence_role",
    "has_incoming_edge",
    "has_outgoing_test",
    "directed_degree",
    "pass_through_candidate",
    "depth",
    "n_parent_context",
    "n_node",
    "n_incoming_sibling",
    "n_left",
    "n_right",
    "incoming_branch_balance",
    "outgoing_balance",
    "outgoing_edge_norm_balance",
    "outgoing_fragment_risk_proxy_score",
    "anti_fragment_unit",
    "selected_family_log_bayes_factor_lower",
    "selected_family_evidence_log",
    "continuous_context_min_margin",
    "intercept_log_odds",
    "incoming_balance_log_lr",
    "outgoing_balance_log_lr",
    "outgoing_edge_norm_log_lr",
    "anti_fragment_log_lr",
    "selected_family_log_component",
    "context_log_component",
    "neighborhood_scale",
    "neighborhood_scale_log_value",
    "neighborhood_scale_log_center",
    "neighborhood_scale_log_bandwidth",
    "neighborhood_scale_log_component",
    "neighborhood_scale_support_truth_count",
    "neighborhood_scale_support_negative_count",
    "neighborhood_scale_support_status",
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
    "topology_neighborhood_distance_cache_status",
    "topology_neighborhood_support_status",
    "root_log_component",
    "passthrough_log_component",
    "topology_core_log_odds",
    "selected_context_log_odds",
    "conditional_log_odds",
    "conditional_probability",
    "conditional_rank",
    "topology_core_supported",
    "missing_topology_feature",
    "support_group",
    "support_truth_count",
    "support_negative_count",
    "support_status",
    "conditional_topology_status",
    "guarded_recovery_balance_product",
    "guarded_recovery_outgoing_edge_norm_balance",
    "guarded_recovery_balance_product_floor",
    "guarded_recovery_outgoing_edge_norm_floor",
    "guarded_recovery_guard_status",
    "guarded_recovery_support_status",
    "guarded_recovery_evidence_status",
    "guarded_recovery_status",
    "recover_internal_split",
)

COMPONENT_COLUMNS = (
    "schema_version",
    "study_role",
    "component",
    "truth_count",
    "negative_count",
    "truth_min",
    "truth_median",
    "truth_max",
    "negative_min",
    "negative_median",
    "negative_max",
    "truth_rank_best",
    "truth_rank_worst",
    "negative_above_truth_min_count",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "truth_recovery_count",
    "negative_count",
    "prior_truth_probability",
    "selected_family_weight",
    "context_penalty_weight",
    "neighborhood_scale_weight",
    "topology_neighborhood_weight",
    "min_truth_support_per_stratum",
    "truth_conditional_log_odds_min",
    "truth_conditional_log_odds_median",
    "truth_conditional_log_odds_max",
    "negative_conditional_log_odds_max",
    "truth_rank_best",
    "truth_rank_worst",
    "truth_above_all_negatives_count",
    "negative_above_truth_min_count",
    "posterior_log_odds_margin",
    "support_insufficient_row_count",
    "feature_missing_row_count",
    "neighborhood_scale_support_insufficient_row_count",
    "neighborhood_scale_unavailable_row_count",
    "topology_neighborhood_support_insufficient_row_count",
    "topology_neighborhood_unavailable_row_count",
    "guarded_recovery_candidate_count",
    "guarded_recovery_guard_blocked_row_count",
    "guarded_recovery_support_blocked_row_count",
    "guarded_recovery_evidence_blocked_row_count",
    "candidate_row_count",
    "diagnostic_status",
    "production_status",
)

BENCHMARK_COLUMNS = (
    "schema_version",
    "study_role",
    "support_group",
    "incidence_role",
    "pass_through_candidate",
    "data_role",
    "row_count",
    "truth_recovery_count",
    "negative_count",
    "candidate_row_count",
    "fail_closed_row_count",
    "truth_rank_best",
    "negative_above_truth_min_count",
    "group_status",
)


@dataclass(frozen=True)
class DirectedIncidence:
    """Directed tree incidence for a tested node."""

    incidence_role: str
    has_incoming_edge: bool
    has_outgoing_test: bool
    directed_degree: int
    pass_through_candidate: bool


@dataclass(frozen=True)
class CachedTreeDistances:
    """Cached all-pairs undirected tree distances for topology diagnostics."""

    distances: dict[tuple[str, str], float]
    nodes: tuple[str, ...]
    status: str

    @property
    def computed_pair_count(self) -> int:
        return int(len(self.distances))

    def distance(self, left: object, right: object) -> float:
        """Return the cached tree distance between two node identifiers."""
        left_id = str(left)
        right_id = str(right)
        if left_id == right_id:
            return 0.0
        key = tuple(sorted((left_id, right_id)))
        return float(self.distances.get(key, math.inf))


@dataclass(frozen=True)
class OverlapConditionalTopologyLawPanelConfig:
    """Runtime contract for conditional topology-law diagnostics."""

    topology_rows_path: Path
    output_dir: Path
    prior_truth_probability: float = DEFAULT_PRIOR_TRUTH_PROBABILITY
    selected_family_weight: float = DEFAULT_SELECTED_FAMILY_WEIGHT
    context_penalty_weight: float = DEFAULT_CONTEXT_PENALTY_WEIGHT
    neighborhood_scale_weight: float = DEFAULT_NEIGHBORHOOD_SCALE_WEIGHT
    root_log_odds_penalty: float = DEFAULT_ROOT_LOG_ODDS_PENALTY
    passthrough_log_odds_penalty: float = DEFAULT_PASSTHROUGH_LOG_ODDS_PENALTY
    min_truth_support_per_stratum: int = DEFAULT_MIN_TRUTH_SUPPORT_PER_STRATUM
    topology_neighborhood_weight: float = DEFAULT_TOPOLOGY_NEIGHBORHOOD_WEIGHT

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_conditional_topology_law_rows.csv"

    @property
    def component_summary_path(self) -> Path:
        return (
            self.output_dir
            / "overlap_conditional_topology_law_component_summary.csv"
        )

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_conditional_topology_law_summary.csv"

    @property
    def benchmark_summary_path(self) -> Path:
        return (
            self.output_dir
            / "overlap_conditional_topology_law_benchmark_summary.csv"
        )

    @property
    def analytical_cases_path(self) -> Path:
        return self.output_dir / "overlap_conditional_topology_law_analytical_cases.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate(rows: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = sorted(set(required) - set(rows.columns))
    if missing:
        raise ValueError(f"Topology rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in rows:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _string_value(row: pd.Series, column: str, default: str = "") -> str:
    value = row[column] if column in row else default
    if pd.isna(value):
        return default
    return str(value)


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _clip_unit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _balance_unit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    return _clip_unit(2.0 * np.asarray(values, dtype=float))


def _anti_fragment_unit(values: pd.Series | np.ndarray | float) -> np.ndarray:
    risk = np.maximum(np.asarray(values, dtype=float), 0.0)
    return _clip_unit(1.0 / (1.0 + risk))


def _neutralize_missing(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=-700.0, upper=700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def build_cached_tree_distances(rows: pd.DataFrame) -> CachedTreeDistances:
    """Build an all-pairs distance cache from `node_id` and `parent_id` columns."""
    if "node_id" not in rows or "parent_id" not in rows:
        return CachedTreeDistances({}, (), "tree_distance_parent_links_unavailable")

    row_node_ids = tuple(str(node) for node in rows["node_id"].dropna().astype(str))
    if not row_node_ids:
        return CachedTreeDistances({}, (), "tree_distance_no_nodes")

    auxiliary_parent_ids: list[str] = []
    if "parent_id" in rows:
        for value in rows["parent_id"]:
            if pd.isna(value):
                continue
            parent = str(value)
            if parent:
                auxiliary_parent_ids.append(parent)
    node_ids = tuple(dict.fromkeys([*row_node_ids, *auxiliary_parent_ids]))
    adjacency: dict[str, set[str]] = {node: set() for node in node_ids}
    node_set = set(node_ids)
    edge_count = 0
    for _, row in rows.iterrows():
        node = str(row["node_id"])
        parent_value = row.get("parent_id", "")
        if pd.isna(parent_value):
            continue
        parent = str(parent_value)
        if not parent or parent not in node_set or parent == node:
            continue
        adjacency.setdefault(node, set()).add(parent)
        adjacency.setdefault(parent, set()).add(node)
        edge_count += 1

    distances: dict[tuple[str, str], float] = {}
    for source in node_ids:
        seen = {source: 0}
        queue = [source]
        for current in queue:
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                seen[neighbor] = seen[current] + 1
                queue.append(neighbor)
        for target in node_ids:
            if source == target:
                continue
            key = tuple(sorted((source, target)))
            if key in distances:
                continue
            distances[key] = float(seen[target]) if target in seen else math.inf
    status = (
        "cached_all_pairs_tree_distances"
        if edge_count
        else "tree_distance_parent_edges_unavailable"
    )
    return CachedTreeDistances(distances, node_ids, status)


def _nearest_distance(
    cache: CachedTreeDistances,
    node: object,
    targets: Iterable[object],
) -> float:
    distances = [
        cache.distance(node, target)
        for target in targets
        if str(target) != str(node)
    ]
    finite = [distance for distance in distances if math.isfinite(distance)]
    return float(min(finite)) if finite else math.inf


def _safe_positive_median(values: Iterable[float], default: float = 1.0) -> float:
    finite = [
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not finite:
        return float(default)
    return max(float(np.median(finite)), 1e-12)


def _neighborhood_scale_components(
    rows: pd.DataFrame,
    *,
    roles: pd.Series,
    support_group_by_index: dict[object, str],
    min_truth_support: int,
    weight: float,
) -> dict[object, dict[str, object]]:
    """Return explicit old-style log-scale proximity components by row.

    This is diagnostic-only. It preserves the old local-in-log-scale idea as a
    visible component with support counts; insufficient scale support is
    reported fail-closed rather than used as calibration.
    """
    scale = _numeric(rows, "neighborhood_scale")
    positive = scale.where(scale > 0.0)
    log_scale = np.log(positive)
    has_any_scale = bool(log_scale.notna().any())
    support_groups = pd.Series(support_group_by_index)
    records: dict[object, dict[str, object]] = {}

    for group in sorted(set(support_group_by_index.values())):
        member_index = support_groups[support_groups.eq(group)].index
        group_logs = log_scale.loc[member_index]
        finite_logs = group_logs.dropna()
        truth_mask = roles.loc[member_index].eq("truth_recovery") & group_logs.notna()
        truth_logs = group_logs.loc[truth_mask[truth_mask].index]
        truth_count = int(truth_logs.shape[0])
        negative_count = int(
            (roles.loc[member_index].ne("truth_recovery") & group_logs.notna()).sum()
        )
        center = float(truth_logs.median()) if truth_count else math.nan
        bandwidth = (
            float(finite_logs.std(ddof=0))
            if finite_logs.shape[0] > 1
            else math.nan
        )
        if not math.isfinite(bandwidth) or bandwidth <= 1e-9:
            bandwidth = 1.0

        for idx in member_index:
            value = float(scale.loc[idx]) if math.isfinite(_finite_float(scale.loc[idx])) else math.nan
            log_value = (
                float(log_scale.loc[idx])
                if math.isfinite(_finite_float(log_scale.loc[idx]))
                else math.nan
            )
            if not has_any_scale:
                status = "neighborhood_scale_unavailable_neutral"
                component = 0.0
                row_center = math.nan
                row_bandwidth = math.nan
            elif not math.isfinite(log_value):
                status = "neighborhood_scale_missing_fail_closed"
                component = 0.0
                row_center = center
                row_bandwidth = bandwidth
            elif truth_count < int(min_truth_support):
                status = "neighborhood_scale_support_insufficient_fail_closed"
                component = 0.0
                row_center = center
                row_bandwidth = bandwidth
            else:
                status = "neighborhood_scale_support_observed_diagnostic_only"
                z_score = (log_value - center) / bandwidth
                component = -0.5 * float(weight) * float(z_score * z_score)
                row_center = center
                row_bandwidth = bandwidth

            records[idx] = {
                "neighborhood_scale": value,
                "neighborhood_scale_log_value": log_value,
                "neighborhood_scale_log_center": row_center,
                "neighborhood_scale_log_bandwidth": row_bandwidth,
                "neighborhood_scale_log_component": float(component),
                "neighborhood_scale_support_truth_count": truth_count,
                "neighborhood_scale_support_negative_count": negative_count,
                "neighborhood_scale_support_status": status,
            }
    return records


def _topology_neighborhood_components(
    rows: pd.DataFrame,
    *,
    roles: pd.Series,
    support_group_by_index: dict[object, str],
    min_support: int,
    weight: float,
) -> dict[object, dict[str, object]]:
    """Return old-style topology-neighborhood bandwidth components by row.

    The component is diagnostic-only. It borrows the old structural idea of
    tree-distance neighborhoods and log-scale bandwidths, but it never counts
    selected non-null rows as empirical-null support.
    """
    cache = build_cached_tree_distances(rows)
    scale = _numeric(rows, "neighborhood_scale")
    log_scale = np.log(scale.where(scale > 0.0))
    support_role = (
        rows["topology_support_role"].astype(str)
        if "topology_support_role" in rows
        else pd.Series("", index=rows.index, dtype=str)
    ).str.strip()
    signal_role = (
        rows["topology_signal_role"].astype(str)
        if "topology_signal_role" in rows
        else pd.Series("", index=rows.index, dtype=str)
    ).str.strip()
    distance_to_stopping = _numeric(rows, "distance_to_stopping_edge")
    support_groups = pd.Series(support_group_by_index)
    records: dict[object, dict[str, object]] = {}
    eligible_support_roles = {
        "strict_null",
        "edge_blocked",
        "stopped_or_null",
        "null_like",
        "stable",
    }
    selected_nonnull_roles = {"selected_nonnull", "selected_non_null", "nonnull"}
    has_explicit_neighborhood_evidence = bool(
        support_role.replace({"nan": "", "None": ""}).str.len().gt(0).any()
        or signal_role.replace({"nan": "", "None": ""}).str.len().gt(0).any()
    )

    for group in sorted(set(support_group_by_index.values())):
        member_index = support_groups[support_groups.eq(group)].index
        group_support_roles = support_role.loc[member_index]
        group_roles = roles.loc[member_index]
        eligible_mask = group_support_roles.isin(eligible_support_roles) | (
            group_roles.eq("null_like") & ~group_support_roles.isin(selected_nonnull_roles)
        )
        selected_nonnull_excluded = int(
            group_support_roles.isin(selected_nonnull_roles).sum()
        )
        support_index = eligible_mask[eligible_mask].index
        support_nodes = rows.loc[support_index, "node_id"].astype(str).tolist()
        signal_index = signal_role.loc[member_index].eq("signal")
        signal_nodes = (
            rows.loc[signal_index[signal_index].index, "node_id"].astype(str).tolist()
        )
        support_logs = log_scale.loc[support_index].dropna()
        scale_center = (
            float(support_logs.median()) if not support_logs.empty else math.nan
        )
        h_k = float(support_logs.std(ddof=0)) if support_logs.shape[0] > 1 else 0.0
        if not math.isfinite(h_k) or h_k <= 1e-12:
            h_k = 0.0
        tau_b = _safe_positive_median(distance_to_stopping.loc[member_index], default=1.0)
        tau_t = _safe_positive_median(
            (
                _nearest_distance(cache, rows.loc[idx, "node_id"], support_nodes)
                for idx in member_index
            ),
            default=1.0,
        )
        tau_s = _safe_positive_median(
            (
                _nearest_distance(cache, rows.loc[idx, "node_id"], signal_nodes)
                for idx in member_index
            ),
            default=1.0,
        )

        for idx in member_index:
            node_id = rows.loc[idx, "node_id"]
            nearest_stable = _nearest_distance(cache, node_id, support_nodes)
            nearest_signal = _nearest_distance(cache, node_id, signal_nodes)
            log_value = _finite_float(log_scale.loc[idx])
            if not has_explicit_neighborhood_evidence:
                status = "topology_neighborhood_unavailable_neutral"
                component = 0.0
            elif cache.status != "cached_all_pairs_tree_distances":
                status = "topology_neighborhood_unavailable_neutral"
                component = 0.0
            elif len(support_nodes) < int(min_support) or not signal_nodes:
                status = "topology_neighborhood_support_insufficient_fail_closed"
                component = 0.0
            else:
                stable_kernel = math.exp(
                    -0.5 * (nearest_stable / tau_t) ** 2
                ) if math.isfinite(nearest_stable) else 0.0
                signal_kernel = math.exp(
                    -0.5 * (nearest_signal / tau_s) ** 2
                ) if math.isfinite(nearest_signal) else 0.0
                scale_penalty = 0.0
                if math.isfinite(log_value) and math.isfinite(scale_center) and h_k > 0.0:
                    scale_penalty = 0.5 * ((log_value - scale_center) / h_k) ** 2
                status = "topology_neighborhood_support_observed_diagnostic_only"
                component = float(weight) * (signal_kernel - stable_kernel) - scale_penalty

            records[idx] = {
                "topology_neighborhood_log_component": float(component),
                "topology_neighborhood_tau_b": float(tau_b),
                "topology_neighborhood_tau_t": float(tau_t),
                "topology_neighborhood_tau_s": float(tau_s),
                "topology_neighborhood_h_k": float(h_k),
                "topology_neighborhood_nearest_stable_distance": float(nearest_stable),
                "topology_neighborhood_nearest_signal_distance": float(nearest_signal),
                "topology_neighborhood_support_count": int(len(support_nodes)),
                "topology_neighborhood_signal_count": int(len(signal_nodes)),
                "topology_neighborhood_selected_nonnull_excluded_count": int(
                    selected_nonnull_excluded
                ),
                "topology_neighborhood_distance_cache_status": cache.status,
                "topology_neighborhood_support_status": status,
            }
    return records


def infer_directed_incidence(
    *,
    depth: object = math.nan,
    decision_class: str = "",
    traversal_decision: str = "",
    n_children: object = math.nan,
    n_left: object = math.nan,
    n_right: object = math.nan,
    parent_id: str = "",
) -> DirectedIncidence:
    """Infer root/internal/leaf directed incidence from row-level metadata."""
    depth_value = _finite_float(depth)
    child_value = _finite_float(n_children)
    left_value = _finite_float(n_left)
    right_value = _finite_float(n_right)
    parent = "" if parent_id is None else str(parent_id)
    decision = str(decision_class)
    traversal = str(traversal_decision)

    is_root = (
        bool(depth_value == 0.0)
        if math.isfinite(depth_value)
        else parent == ""
    )
    if math.isfinite(child_value):
        child_count = max(int(child_value), 0)
    elif math.isfinite(left_value) or math.isfinite(right_value):
        child_count = int(math.isfinite(left_value) and left_value > 0.0) + int(
            math.isfinite(right_value) and right_value > 0.0
        )
    elif decision == "leaf_fragment":
        child_count = 0
    else:
        child_count = 2

    is_leaf = child_count == 0 or decision == "leaf_fragment"
    if is_root:
        incidence_role = "root"
    elif is_leaf:
        incidence_role = "leaf"
    else:
        incidence_role = "internal"

    has_incoming = incidence_role != "root"
    has_outgoing = incidence_role != "leaf" and child_count >= 2
    degree = int(has_incoming) + child_count
    pass_through = "pass" in traversal or decision == "unstable_passthrough_zone"
    return DirectedIncidence(
        incidence_role=incidence_role,
        has_incoming_edge=has_incoming,
        has_outgoing_test=has_outgoing,
        directed_degree=degree,
        pass_through_candidate=pass_through,
    )


def _incidence_for_row(row: pd.Series) -> DirectedIncidence:
    return infer_directed_incidence(
        depth=row.get("depth", math.nan),
        decision_class=_string_value(row, "decision_class"),
        traversal_decision=_string_value(row, "traversal_decision"),
        n_children=row.get("n_children", math.nan),
        n_left=row.get("n_left", math.nan),
        n_right=row.get("n_right", math.nan),
        parent_id=_string_value(row, "parent_id"),
    )


def _support_status(
    *,
    incidence: DirectedIncidence,
    missing_topology: bool,
    support_truth_count: int,
    min_truth_support: int,
) -> str:
    if incidence.incidence_role == "leaf":
        return "leaf_no_outgoing_test_fail_closed"
    if missing_topology:
        return "topology_features_missing_fail_closed"
    if support_truth_count < int(min_truth_support):
        return "support_insufficient_fail_closed"
    return "support_observed_diagnostic_only"


def _conditional_status(
    *,
    support_status: str,
    neighborhood_scale_support_status: str,
    topology_neighborhood_support_status: str,
    topology_core_supported: bool,
    conditional_log_odds: float,
) -> str:
    if support_status != "support_observed_diagnostic_only":
        return support_status
    if neighborhood_scale_support_status.endswith("_fail_closed"):
        return neighborhood_scale_support_status
    if topology_neighborhood_support_status.endswith("_fail_closed"):
        return topology_neighborhood_support_status
    if not topology_core_supported:
        return "selected_context_only_not_promoted"
    if conditional_log_odds > 0.0:
        return "conditional_topology_candidate_diagnostic_only"
    return "conditional_topology_below_support"


def _bool_column_value(row: pd.Series, column: str) -> bool:
    value = row[column] if column in row else False
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _guarded_recovery_fields(
    *,
    row: pd.Series,
    incidence: DirectedIncidence,
    support_status: str,
    balance_product: float,
    outgoing_edge_norm_balance: float,
    balance_product_floor: float,
    outgoing_edge_norm_floor: float,
) -> dict[str, object]:
    """Return guarded internal-recovery diagnostics for one row."""
    support_role = _string_value(row, "topology_support_role").strip()
    data_role = _string_value(row, "data_role").strip()
    root_or_family_blocked = any(
        _bool_column_value(row, column)
        for column in (
            "root_stability_guard_blocked",
            "root_selective_guard_blocked",
            "selected_family_guard_blocked",
        )
    )
    null_like_blocked = support_role in {
        "strict_null",
        "edge_blocked",
        "stopped_or_null",
        "null_like",
        "stable",
    } or data_role in {"null", "selected_null"}

    if incidence.incidence_role != "internal" or incidence.pass_through_candidate:
        guard_status = "not_internal_recovery_candidate"
    elif root_or_family_blocked or null_like_blocked:
        guard_status = "root_or_null_guard_blocked"
    else:
        guard_status = "root_null_guards_pass"

    support_ok = support_status == "support_observed_diagnostic_only"
    support_gate_status = (
        "support_sufficient"
        if support_ok
        else "support_insufficient_fail_closed"
    )
    if not math.isfinite(balance_product):
        evidence_status = "balance_product_missing"
    elif balance_product < float(balance_product_floor):
        evidence_status = "balance_product_below_floor"
    elif not math.isfinite(outgoing_edge_norm_balance):
        evidence_status = "outgoing_edge_missing"
    elif outgoing_edge_norm_balance < float(outgoing_edge_norm_floor):
        evidence_status = "outgoing_edge_below_floor"
    else:
        evidence_status = "balance_product_outgoing_edge_supported"

    if guard_status != "root_null_guards_pass":
        status = guard_status
    elif not support_ok:
        status = support_gate_status
    elif evidence_status != "balance_product_outgoing_edge_supported":
        status = evidence_status
    else:
        status = "guarded_internal_recovery_candidate_diagnostic_only"

    return {
        "guarded_recovery_balance_product": float(balance_product),
        "guarded_recovery_outgoing_edge_norm_balance": float(
            outgoing_edge_norm_balance
        ),
        "guarded_recovery_balance_product_floor": float(balance_product_floor),
        "guarded_recovery_outgoing_edge_norm_floor": float(outgoing_edge_norm_floor),
        "guarded_recovery_guard_status": guard_status,
        "guarded_recovery_support_status": support_gate_status,
        "guarded_recovery_evidence_status": evidence_status,
        "guarded_recovery_status": status,
        "recover_internal_split": bool(
            status == "guarded_internal_recovery_candidate_diagnostic_only"
        ),
    }


def build_conditional_topology_law_rows(
    topology_rows: pd.DataFrame,
    *,
    prior_truth_probability: float = DEFAULT_PRIOR_TRUTH_PROBABILITY,
    selected_family_weight: float = DEFAULT_SELECTED_FAMILY_WEIGHT,
    context_penalty_weight: float = DEFAULT_CONTEXT_PENALTY_WEIGHT,
    neighborhood_scale_weight: float = DEFAULT_NEIGHBORHOOD_SCALE_WEIGHT,
    topology_neighborhood_weight: float = DEFAULT_TOPOLOGY_NEIGHBORHOOD_WEIGHT,
    root_log_odds_penalty: float = DEFAULT_ROOT_LOG_ODDS_PENALTY,
    passthrough_log_odds_penalty: float = DEFAULT_PASSTHROUGH_LOG_ODDS_PENALTY,
    min_truth_support_per_stratum: int = DEFAULT_MIN_TRUTH_SUPPORT_PER_STRATUM,
    guarded_recovery_balance_product_floor: float = (
        DEFAULT_GUARDED_RECOVERY_BALANCE_PRODUCT_FLOOR
    ),
    guarded_recovery_outgoing_edge_norm_floor: float = (
        DEFAULT_GUARDED_RECOVERY_OUTGOING_EDGE_NORM_FLOOR
    ),
) -> pd.DataFrame:
    """Build incidence-aware conditional topology-law rows."""
    _validate(topology_rows)
    rows = topology_rows.copy()
    for column in OPTIONAL_COLUMNS - set(rows.columns):
        rows[column] = math.nan

    incoming = _numeric(rows, "incoming_branch_balance")
    outgoing = _numeric(rows, "outgoing_balance")
    edge_norm = _numeric(rows, "outgoing_edge_norm_balance")
    fragment = _numeric(rows, "outgoing_fragment_risk_proxy_score")
    selected_family = _numeric(rows, "selected_family_log_bayes_factor_lower")
    context = _numeric(rows, "continuous_context_min_margin")
    balance_product = _numeric(rows, "balance_product")

    incoming_lr = _neutralize_missing(
        beta_log_likelihood_ratio(_balance_unit(incoming))
    )
    outgoing_lr = _neutralize_missing(
        beta_log_likelihood_ratio(_balance_unit(outgoing))
    )
    edge_norm_lr = _neutralize_missing(beta_log_likelihood_ratio(_clip_unit(edge_norm)))
    anti_fragment = _anti_fragment_unit(fragment)
    anti_fragment_lr = _neutralize_missing(beta_log_likelihood_ratio(anti_fragment))
    selected_component = _neutralize_missing(
        np.log1p(np.maximum(selected_family.to_numpy(dtype=float), 0.0))
        * float(selected_family_weight)
    )
    context_component = _neutralize_missing(
        np.minimum(context.to_numpy(dtype=float), 0.0) * float(context_penalty_weight)
    )
    intercept = _logit(float(prior_truth_probability))

    incidence_by_index = {idx: _incidence_for_row(row) for idx, row in rows.iterrows()}
    support_group_by_index = {
        idx: (
            f"{incidence.incidence_role}|"
            f"pass_through={int(incidence.pass_through_candidate)}"
        )
        for idx, incidence in incidence_by_index.items()
    }
    roles = rows["guard_truth_role"].astype(str)
    support_counts: dict[str, tuple[int, int]] = {}
    for group in sorted(set(support_group_by_index.values())):
        members = pd.Series(support_group_by_index).eq(group)
        group_roles = roles.loc[members[members].index]
        truth_count = int(group_roles.eq("truth_recovery").sum())
        support_counts[group] = (truth_count, int(group_roles.ne("truth_recovery").sum()))
    neighborhood_components = _neighborhood_scale_components(
        rows,
        roles=roles,
        support_group_by_index=support_group_by_index,
        min_truth_support=int(min_truth_support_per_stratum),
        weight=float(neighborhood_scale_weight),
    )
    topology_neighborhood_components = _topology_neighborhood_components(
        rows,
        roles=roles,
        support_group_by_index=support_group_by_index,
        min_support=int(min_truth_support_per_stratum),
        weight=float(topology_neighborhood_weight),
    )

    conditional_log_odds_by_index: dict[object, float] = {}
    raw_records: list[dict[str, object]] = []
    for position, (idx, row) in enumerate(rows.iterrows()):
        incidence = incidence_by_index[idx]
        support_group = support_group_by_index[idx]
        support_truth_count, support_negative_count = support_counts[support_group]

        incoming_component = 0.0 if not incidence.has_incoming_edge else incoming_lr[position]
        outgoing_component = 0.0 if not incidence.has_outgoing_test else outgoing_lr[position]
        edge_component = 0.0 if not incidence.has_outgoing_test else edge_norm_lr[position]
        anti_component = 0.0 if not incidence.has_outgoing_test else anti_fragment_lr[position]
        root_component = (
            -float(root_log_odds_penalty)
            if incidence.incidence_role == "root"
            else 0.0
        )
        passthrough_component = (
            -float(passthrough_log_odds_penalty)
            if incidence.pass_through_candidate
            else 0.0
        )

        topology_required = [
            outgoing.loc[idx],
            edge_norm.loc[idx],
            fragment.loc[idx],
        ]
        if incidence.has_incoming_edge:
            topology_required.append(incoming.loc[idx])
        missing_topology = bool(
            incidence.has_outgoing_test
            and any(not math.isfinite(_finite_float(value)) for value in topology_required)
        )
        topology_core = (
            incoming_component + outgoing_component + edge_component + anti_component
        )
        neighborhood_component = float(
            neighborhood_components[idx]["neighborhood_scale_log_component"]
        )
        topology_neighborhood_component = float(
            topology_neighborhood_components[idx][
                "topology_neighborhood_log_component"
            ]
        )
        selected_context = selected_component[position] + context_component[position]
        conditional_log_odds = (
            intercept
            + topology_core
            + selected_context
            + neighborhood_component
            + topology_neighborhood_component
            + root_component
            + passthrough_component
        )
        conditional_log_odds_by_index[idx] = float(conditional_log_odds)
        topology_core_supported = bool(
            incidence.has_outgoing_test
            and outgoing_component > 0.0
            and edge_component > 0.0
            and anti_component > 0.0
        )
        support_status = _support_status(
            incidence=incidence,
            missing_topology=missing_topology,
            support_truth_count=support_truth_count,
            min_truth_support=int(min_truth_support_per_stratum),
        )
        guarded_recovery = _guarded_recovery_fields(
            row=row,
            incidence=incidence,
            support_status=support_status,
            balance_product=float(balance_product.loc[idx]),
            outgoing_edge_norm_balance=float(edge_norm.loc[idx]),
            balance_product_floor=float(guarded_recovery_balance_product_floor),
            outgoing_edge_norm_floor=float(guarded_recovery_outgoing_edge_norm_floor),
        )
        raw_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "replicate": int(row["replicate"]),
                "node_id": str(row["node_id"]),
                "guard_truth_role": str(row["guard_truth_role"]),
                "analytical_case": _string_value(row, "analytical_case"),
                "incidence_role": incidence.incidence_role,
                "has_incoming_edge": bool(incidence.has_incoming_edge),
                "has_outgoing_test": bool(incidence.has_outgoing_test),
                "directed_degree": int(incidence.directed_degree),
                "pass_through_candidate": bool(incidence.pass_through_candidate),
                "depth": float(_numeric(rows, "depth").loc[idx]),
                "n_parent_context": float(_numeric(rows, "n_parent_context").loc[idx]),
                "n_node": float(_numeric(rows, "n_node").loc[idx]),
                "n_incoming_sibling": float(_numeric(rows, "n_incoming_sibling").loc[idx]),
                "n_left": float(_numeric(rows, "n_left").loc[idx]),
                "n_right": float(_numeric(rows, "n_right").loc[idx]),
                "incoming_branch_balance": float(incoming.loc[idx]),
                "outgoing_balance": float(outgoing.loc[idx]),
                "outgoing_edge_norm_balance": float(edge_norm.loc[idx]),
                "outgoing_fragment_risk_proxy_score": float(fragment.loc[idx]),
                "anti_fragment_unit": float(anti_fragment[position]),
                "selected_family_log_bayes_factor_lower": float(selected_family.loc[idx]),
                "selected_family_evidence_log": float(
                    np.log1p(max(float(selected_family.loc[idx]), 0.0))
                    if math.isfinite(_finite_float(selected_family.loc[idx]))
                    else 0.0
                ),
                "continuous_context_min_margin": float(context.loc[idx]),
                "intercept_log_odds": float(intercept),
                "incoming_balance_log_lr": float(incoming_component),
                "outgoing_balance_log_lr": float(outgoing_component),
                "outgoing_edge_norm_log_lr": float(edge_component),
                "anti_fragment_log_lr": float(anti_component),
                "selected_family_log_component": float(selected_component[position]),
                "context_log_component": float(context_component[position]),
                **neighborhood_components[idx],
                **topology_neighborhood_components[idx],
                "root_log_component": float(root_component),
                "passthrough_log_component": float(passthrough_component),
                "topology_core_log_odds": float(topology_core),
                "selected_context_log_odds": float(selected_context),
                "conditional_log_odds": float(conditional_log_odds),
                "conditional_probability": math.nan,
                "conditional_rank": 0,
                "topology_core_supported": topology_core_supported,
                "missing_topology_feature": bool(missing_topology),
                "support_group": support_group,
                "support_truth_count": support_truth_count,
                "support_negative_count": support_negative_count,
                "support_status": support_status,
                "conditional_topology_status": _conditional_status(
                    support_status=support_status,
                    neighborhood_scale_support_status=str(
                        neighborhood_components[idx][
                            "neighborhood_scale_support_status"
                        ]
                    ),
                    topology_neighborhood_support_status=str(
                        topology_neighborhood_components[idx][
                            "topology_neighborhood_support_status"
                        ]
                    ),
                    topology_core_supported=topology_core_supported,
                    conditional_log_odds=float(conditional_log_odds),
                ),
                **guarded_recovery,
            }
        )

    out = pd.DataFrame.from_records(raw_records, columns=ROW_COLUMNS)
    scores = pd.Series(conditional_log_odds_by_index)
    ranked = scores.rank(method="first", ascending=False).astype(int)
    probabilities = _sigmoid(pd.Series(scores.values, index=scores.index))
    for position, idx in enumerate(scores.index):
        out.loc[position, "conditional_rank"] = int(ranked.loc[idx])
        out.loc[position, "conditional_probability"] = float(probabilities.loc[idx])
    return out


def _rank_bounds(values: pd.Series, roles: pd.Series) -> tuple[int, int, int]:
    truth = roles.eq("truth_recovery")
    if not bool(truth.any()):
        return 0, 0, 0
    ranks = values.rank(method="first", ascending=False).astype(int)
    truth_ranks = ranks[truth]
    truth_min = float(values[truth].min())
    negative_above = int((values[~truth] > truth_min).sum())
    return int(truth_ranks.min()), int(truth_ranks.max()), negative_above


def summarize_conditional_topology_components(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize component ranking against truth-recovery labels."""
    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    component_columns = (
        "incoming_balance_log_lr",
        "outgoing_balance_log_lr",
        "outgoing_edge_norm_log_lr",
        "anti_fragment_log_lr",
        "selected_family_log_component",
        "context_log_component",
        "neighborhood_scale_log_component",
        "topology_neighborhood_log_component",
        "root_log_component",
        "passthrough_log_component",
        "topology_core_log_odds",
        "selected_context_log_odds",
        "conditional_log_odds",
    )
    records: list[dict[str, object]] = []
    for component in component_columns:
        values = _numeric(rows, component)
        truth_values = values[truth]
        negative_values = values[~truth]
        rank_best, rank_worst, negative_above = _rank_bounds(values, roles)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "component": component,
                "truth_count": int(truth.sum()),
                "negative_count": int((~truth).sum()),
                "truth_min": float(truth_values.min()) if not truth_values.empty else math.nan,
                "truth_median": float(truth_values.median())
                if not truth_values.empty
                else math.nan,
                "truth_max": float(truth_values.max()) if not truth_values.empty else math.nan,
                "negative_min": float(negative_values.min())
                if not negative_values.empty
                else math.nan,
                "negative_median": float(negative_values.median())
                if not negative_values.empty
                else math.nan,
                "negative_max": float(negative_values.max())
                if not negative_values.empty
                else math.nan,
                "truth_rank_best": rank_best,
                "truth_rank_worst": rank_worst,
                "negative_above_truth_min_count": negative_above,
            }
        )
    return pd.DataFrame.from_records(records, columns=COMPONENT_COLUMNS)


def summarize_conditional_topology_law_rows(
    rows: pd.DataFrame,
    *,
    prior_truth_probability: float = DEFAULT_PRIOR_TRUTH_PROBABILITY,
    selected_family_weight: float = DEFAULT_SELECTED_FAMILY_WEIGHT,
    context_penalty_weight: float = DEFAULT_CONTEXT_PENALTY_WEIGHT,
    neighborhood_scale_weight: float = DEFAULT_NEIGHBORHOOD_SCALE_WEIGHT,
    topology_neighborhood_weight: float = DEFAULT_TOPOLOGY_NEIGHBORHOOD_WEIGHT,
    min_truth_support_per_stratum: int = DEFAULT_MIN_TRUTH_SUPPORT_PER_STRATUM,
) -> pd.DataFrame:
    """Summarize conditional topology-law row scores and support status."""
    if rows.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "truth_recovery_count": 0,
                    "negative_count": 0,
                    "prior_truth_probability": float(prior_truth_probability),
                    "selected_family_weight": float(selected_family_weight),
                    "context_penalty_weight": float(context_penalty_weight),
                    "neighborhood_scale_weight": float(neighborhood_scale_weight),
                    "topology_neighborhood_weight": float(topology_neighborhood_weight),
                    "min_truth_support_per_stratum": int(min_truth_support_per_stratum),
                    "truth_conditional_log_odds_min": math.nan,
                    "truth_conditional_log_odds_median": math.nan,
                    "truth_conditional_log_odds_max": math.nan,
                    "negative_conditional_log_odds_max": math.nan,
                    "truth_rank_best": 0,
                    "truth_rank_worst": 0,
                    "truth_above_all_negatives_count": 0,
                    "negative_above_truth_min_count": 0,
                    "posterior_log_odds_margin": math.nan,
                    "support_insufficient_row_count": 0,
                    "feature_missing_row_count": 0,
                    "neighborhood_scale_support_insufficient_row_count": 0,
                    "neighborhood_scale_unavailable_row_count": 0,
                    "topology_neighborhood_support_insufficient_row_count": 0,
                    "topology_neighborhood_unavailable_row_count": 0,
                    "guarded_recovery_candidate_count": 0,
                    "guarded_recovery_guard_blocked_row_count": 0,
                    "guarded_recovery_support_blocked_row_count": 0,
                    "guarded_recovery_evidence_blocked_row_count": 0,
                    "candidate_row_count": 0,
                    "diagnostic_status": "conditional_topology_unavailable",
                    "production_status": "fail_closed_no_rows",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )

    roles = rows["guard_truth_role"].astype(str)
    truth = roles.eq("truth_recovery")
    scores = _numeric(rows, "conditional_log_odds")
    truth_scores = scores[truth]
    negative_scores = scores[~truth]
    rank_best, rank_worst, negative_above = _rank_bounds(scores, roles)
    negative_max = float(negative_scores.max()) if not negative_scores.empty else math.nan
    truth_min = float(truth_scores.min()) if not truth_scores.empty else math.nan
    truth_above_all = int((truth_scores > negative_max).sum()) if math.isfinite(negative_max) else 0
    margin = truth_min - negative_max if math.isfinite(truth_min) and math.isfinite(negative_max) else math.nan
    support_insufficient = rows["support_status"].astype(str).eq(
        "support_insufficient_fail_closed"
    )
    feature_missing = rows["support_status"].astype(str).eq(
        "topology_features_missing_fail_closed"
    )
    neighborhood_scale_status = rows["neighborhood_scale_support_status"].astype(str)
    scale_fail_closed = neighborhood_scale_status.str.endswith("_fail_closed")
    scale_unavailable = neighborhood_scale_status.eq(
        "neighborhood_scale_unavailable_neutral"
    )
    topology_neighborhood_status = rows["topology_neighborhood_support_status"].astype(str)
    topology_neighborhood_fail_closed = topology_neighborhood_status.str.endswith(
        "_fail_closed"
    )
    topology_neighborhood_unavailable = topology_neighborhood_status.eq(
        "topology_neighborhood_unavailable_neutral"
    )
    candidate_count = int(
        rows["conditional_topology_status"]
        .astype(str)
        .eq("conditional_topology_candidate_diagnostic_only")
        .sum()
    )
    guarded_recovery_status = rows["guarded_recovery_status"].astype(str)
    guarded_recovery_candidate = guarded_recovery_status.eq(
        "guarded_internal_recovery_candidate_diagnostic_only"
    )
    guarded_recovery_guard_blocked = guarded_recovery_status.isin(
        {
            "root_or_null_guard_blocked",
            "not_internal_recovery_candidate",
        }
    )
    guarded_recovery_support_blocked = guarded_recovery_status.eq(
        "support_insufficient_fail_closed"
    )
    guarded_recovery_evidence_blocked = guarded_recovery_status.isin(
        {
            "balance_product_missing",
            "balance_product_below_floor",
            "outgoing_edge_missing",
            "outgoing_edge_below_floor",
        }
    )
    if not bool(truth.any()):
        diagnostic_status = "conditional_topology_no_truth_support"
    elif truth_above_all == int(truth.sum()) and bool(
        support_insufficient.loc[truth].any()
    ):
        diagnostic_status = "conditional_topology_signal_detected_support_insufficient"
    elif truth_above_all == int(truth.sum()):
        diagnostic_status = "conditional_topology_candidate_diagnostic_only"
    elif truth_above_all:
        diagnostic_status = "conditional_topology_partial_truth_separation"
    else:
        diagnostic_status = "conditional_topology_not_separating"
    production_status = (
        "diagnostic_only_support_insufficient_fail_closed"
        if bool(support_insufficient.any())
        or bool(feature_missing.any())
        or bool(scale_fail_closed.any())
        or bool(topology_neighborhood_fail_closed.any())
        else "diagnostic_only_not_production_calibrated"
    )
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "truth_recovery_count": int(truth.sum()),
                "negative_count": int((~truth).sum()),
                "prior_truth_probability": float(prior_truth_probability),
                "selected_family_weight": float(selected_family_weight),
                "context_penalty_weight": float(context_penalty_weight),
                "neighborhood_scale_weight": float(neighborhood_scale_weight),
                "topology_neighborhood_weight": float(topology_neighborhood_weight),
                "min_truth_support_per_stratum": int(min_truth_support_per_stratum),
                "truth_conditional_log_odds_min": truth_min,
                "truth_conditional_log_odds_median": float(truth_scores.median())
                if not truth_scores.empty
                else math.nan,
                "truth_conditional_log_odds_max": float(truth_scores.max())
                if not truth_scores.empty
                else math.nan,
                "negative_conditional_log_odds_max": negative_max,
                "truth_rank_best": rank_best,
                "truth_rank_worst": rank_worst,
                "truth_above_all_negatives_count": truth_above_all,
                "negative_above_truth_min_count": negative_above,
                "posterior_log_odds_margin": margin,
                "support_insufficient_row_count": int(support_insufficient.sum()),
                "feature_missing_row_count": int(feature_missing.sum()),
                "neighborhood_scale_support_insufficient_row_count": int(
                    scale_fail_closed.sum()
                ),
                "neighborhood_scale_unavailable_row_count": int(
                    scale_unavailable.sum()
                ),
                "topology_neighborhood_support_insufficient_row_count": int(
                    topology_neighborhood_fail_closed.sum()
                ),
                "topology_neighborhood_unavailable_row_count": int(
                    topology_neighborhood_unavailable.sum()
                ),
                "guarded_recovery_candidate_count": int(
                    guarded_recovery_candidate.sum()
                ),
                "guarded_recovery_guard_blocked_row_count": int(
                    guarded_recovery_guard_blocked.sum()
                ),
                "guarded_recovery_support_blocked_row_count": int(
                    guarded_recovery_support_blocked.sum()
                ),
                "guarded_recovery_evidence_blocked_row_count": int(
                    guarded_recovery_evidence_blocked.sum()
                ),
                "candidate_row_count": candidate_count,
                "diagnostic_status": diagnostic_status,
                "production_status": production_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def summarize_conditional_topology_benchmark(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize conditional topology behavior by incidence/pass-through group."""
    if rows.empty:
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)
    records: list[dict[str, object]] = []
    groups = rows.groupby(
        ["support_group", "incidence_role", "pass_through_candidate", "data_role"],
        dropna=False,
    )
    for (support_group, incidence_role, pass_through, data_role), group in groups:
        roles = group["guard_truth_role"].astype(str)
        truth = roles.eq("truth_recovery")
        scores = _numeric(group, "conditional_log_odds")
        _, _, negative_above = _rank_bounds(scores, roles)
        truth_rank_best = (
            int(scores.rank(method="first", ascending=False)[truth].min())
            if bool(truth.any())
            else 0
        )
        fail_closed = group["support_status"].astype(str).ne(
            "support_observed_diagnostic_only"
        )
        candidate = group["conditional_topology_status"].astype(str).eq(
            "conditional_topology_candidate_diagnostic_only"
        )
        if int(truth.sum()) == 0:
            group_status = "benchmark_no_truth_support"
        elif bool(fail_closed.loc[truth].any()):
            group_status = "benchmark_truth_support_insufficient"
        elif negative_above == 0:
            group_status = "benchmark_truth_separates_diagnostic_only"
        else:
            group_status = "benchmark_truth_leakage"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "support_group": str(support_group),
                "incidence_role": str(incidence_role),
                "pass_through_candidate": bool(pass_through),
                "data_role": str(data_role),
                "row_count": int(group.shape[0]),
                "truth_recovery_count": int(truth.sum()),
                "negative_count": int((~truth).sum()),
                "candidate_row_count": int(candidate.sum()),
                "fail_closed_row_count": int(fail_closed.sum()),
                "truth_rank_best": truth_rank_best,
                "negative_above_truth_min_count": int(negative_above),
                "group_status": group_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=BENCHMARK_COLUMNS)


def build_conditional_topology_analytical_cases() -> pd.DataFrame:
    """Return focused analytical rows for the known overlap failure modes."""

    def row(
        analytical_case: str,
        role: str,
        *,
        depth: int,
        decision_class: str,
        traversal_decision: str,
        incoming: float,
        outgoing: float,
        edge_norm: float,
        fragment: float,
        selected: float,
        context: float,
        replicate: int,
    ) -> dict[str, object]:
        return {
            "case_id": "analytical_overlap_cases",
            "data_role": "diagnostic_fixture",
            "replicate": replicate,
            "node_id": f"N{replicate}",
            "guard_truth_role": role,
            "analytical_case": analytical_case,
            "depth": depth,
            "decision_class": decision_class,
            "traversal_decision": traversal_decision,
            "n_parent_context": 200,
            "n_node": 100,
            "n_incoming_sibling": 100,
            "n_left": 50,
            "n_right": 50,
            "incoming_branch_balance": incoming,
            "outgoing_balance": outgoing,
            "outgoing_edge_norm_balance": edge_norm,
            "outgoing_fragment_risk_proxy_score": fragment,
            "selected_family_log_bayes_factor_lower": selected,
            "continuous_context_min_margin": context,
        }

    return pd.DataFrame.from_records(
        [
            row(
                "context_positive_recoverable_split",
                "truth_recovery",
                depth=1,
                decision_class="accepted_internal_split",
                traversal_decision="split",
                incoming=0.48,
                outgoing=0.49,
                edge_norm=0.96,
                fragment=0.55,
                selected=10.0,
                context=0.02,
                replicate=0,
            ),
            row(
                "context_negative_emergent_true_split",
                "truth_recovery",
                depth=1,
                decision_class="accepted_internal_split",
                traversal_decision="split",
                incoming=0.47,
                outgoing=0.49,
                edge_norm=0.97,
                fragment=0.57,
                selected=18.0,
                context=-0.006,
                replicate=1,
            ),
            row(
                "context_negative_fragment_false_split",
                "fragment_like",
                depth=1,
                decision_class="accepted_internal_split",
                traversal_decision="split",
                incoming=0.46,
                outgoing=0.23,
                edge_norm=0.48,
                fragment=1.45,
                selected=18.0,
                context=-0.004,
                replicate=2,
            ),
            row(
                "closed_root_passthrough_false_split",
                "null_like",
                depth=1,
                decision_class="unstable_passthrough_zone",
                traversal_decision="pass_through",
                incoming=0.40,
                outgoing=0.28,
                edge_norm=0.52,
                fragment=1.30,
                selected=22.0,
                context=-0.003,
                replicate=3,
            ),
            row(
                "closed_root_passthrough_true_many_cluster_split",
                "truth_recovery",
                depth=1,
                decision_class="unstable_passthrough_zone",
                traversal_decision="pass_through",
                incoming=0.38,
                outgoing=0.48,
                edge_norm=0.93,
                fragment=0.62,
                selected=22.0,
                context=-0.004,
                replicate=4,
            ),
            row(
                "weak_incoming_coherent_outgoing_transition",
                "truth_recovery",
                depth=2,
                decision_class="accepted_internal_split",
                traversal_decision="split",
                incoming=0.31,
                outgoing=0.49,
                edge_norm=0.95,
                fragment=0.58,
                selected=12.0,
                context=-0.006,
                replicate=5,
            ),
            row(
                "weak_incoming_incoherent_outgoing_artifact",
                "null_like",
                depth=2,
                decision_class="accepted_internal_split",
                traversal_decision="split",
                incoming=0.31,
                outgoing=0.27,
                edge_norm=0.50,
                fragment=1.25,
                selected=12.0,
                context=-0.006,
                replicate=6,
            ),
        ]
    )


def run_overlap_conditional_topology_law_panel(
    config: OverlapConditionalTopologyLawPanelConfig,
) -> dict[str, Path]:
    """Run conditional topology-law diagnostics and write outputs."""
    topology_rows = pd.read_csv(config.topology_rows_path)
    rows = build_conditional_topology_law_rows(
        topology_rows,
        prior_truth_probability=float(config.prior_truth_probability),
        selected_family_weight=float(config.selected_family_weight),
        context_penalty_weight=float(config.context_penalty_weight),
        neighborhood_scale_weight=float(config.neighborhood_scale_weight),
        topology_neighborhood_weight=float(config.topology_neighborhood_weight),
        root_log_odds_penalty=float(config.root_log_odds_penalty),
        passthrough_log_odds_penalty=float(config.passthrough_log_odds_penalty),
        min_truth_support_per_stratum=int(config.min_truth_support_per_stratum),
    )
    component_summary = summarize_conditional_topology_components(rows)
    summary = summarize_conditional_topology_law_rows(
        rows,
        prior_truth_probability=float(config.prior_truth_probability),
        selected_family_weight=float(config.selected_family_weight),
        context_penalty_weight=float(config.context_penalty_weight),
        neighborhood_scale_weight=float(config.neighborhood_scale_weight),
        topology_neighborhood_weight=float(config.topology_neighborhood_weight),
        min_truth_support_per_stratum=int(config.min_truth_support_per_stratum),
    )
    benchmark_summary = summarize_conditional_topology_benchmark(rows)
    analytical_cases = build_conditional_topology_law_rows(
        build_conditional_topology_analytical_cases(),
        prior_truth_probability=float(config.prior_truth_probability),
        selected_family_weight=float(config.selected_family_weight),
        context_penalty_weight=float(config.context_penalty_weight),
        neighborhood_scale_weight=float(config.neighborhood_scale_weight),
        topology_neighborhood_weight=float(config.topology_neighborhood_weight),
        root_log_odds_penalty=float(config.root_log_odds_penalty),
        passthrough_log_odds_penalty=float(config.passthrough_log_odds_penalty),
        min_truth_support_per_stratum=int(config.min_truth_support_per_stratum),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(config.rows_path, index=False)
    component_summary.to_csv(config.component_summary_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    benchmark_summary.to_csv(config.benchmark_summary_path, index=False)
    analytical_cases.to_csv(config.analytical_cases_path, index=False)
    summary_record = summary.iloc[0].to_dict() if not summary.empty else {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "topology_rows_path": str(config.topology_rows_path),
        "prior_truth_probability": float(config.prior_truth_probability),
        "selected_family_weight": float(config.selected_family_weight),
        "context_penalty_weight": float(config.context_penalty_weight),
        "neighborhood_scale_weight": float(config.neighborhood_scale_weight),
        "topology_neighborhood_weight": float(config.topology_neighborhood_weight),
        "root_log_odds_penalty": float(config.root_log_odds_penalty),
        "passthrough_log_odds_penalty": float(config.passthrough_log_odds_penalty),
        "min_truth_support_per_stratum": int(config.min_truth_support_per_stratum),
        "law": (
            "eta = logit(prior) + topology_core + selected_family_weight * "
            "log1p(selected_family) + min(context, 0) * context_penalty "
            "+ neighborhood_scale_component + topology_neighborhood_component "
            "- root_penalty * I_root - passthrough_penalty * I_passthrough"
        ),
        "outputs": {
            "rows": str(config.rows_path),
            "component_summary": str(config.component_summary_path),
            "summary": str(config.summary_path),
            "benchmark_summary": str(config.benchmark_summary_path),
            "analytical_cases": str(config.analytical_cases_path),
        },
        "diagnostic_status": str(
            summary_record.get("diagnostic_status", "diagnostic_summary_unavailable")
        ),
        "production_status": str(
            summary_record.get(
                "production_status",
                "diagnostic_only_non_permutation_conditional_law",
            )
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "component_summary": config.component_summary_path,
        "summary": config.summary_path,
        "benchmark_summary": config.benchmark_summary_path,
        "analytical_cases": config.analytical_cases_path,
        "manifest": config.manifest_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-rows-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prior-truth-probability",
        type=float,
        default=DEFAULT_PRIOR_TRUTH_PROBABILITY,
    )
    parser.add_argument(
        "--selected-family-weight",
        type=float,
        default=DEFAULT_SELECTED_FAMILY_WEIGHT,
    )
    parser.add_argument(
        "--context-penalty-weight",
        type=float,
        default=DEFAULT_CONTEXT_PENALTY_WEIGHT,
    )
    parser.add_argument(
        "--neighborhood-scale-weight",
        type=float,
        default=DEFAULT_NEIGHBORHOOD_SCALE_WEIGHT,
    )
    parser.add_argument(
        "--topology-neighborhood-weight",
        type=float,
        default=DEFAULT_TOPOLOGY_NEIGHBORHOOD_WEIGHT,
    )
    parser.add_argument(
        "--root-log-odds-penalty",
        type=float,
        default=DEFAULT_ROOT_LOG_ODDS_PENALTY,
    )
    parser.add_argument(
        "--passthrough-log-odds-penalty",
        type=float,
        default=DEFAULT_PASSTHROUGH_LOG_ODDS_PENALTY,
    )
    parser.add_argument(
        "--min-truth-support-per-stratum",
        type=int,
        default=DEFAULT_MIN_TRUTH_SUPPORT_PER_STRATUM,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_overlap_conditional_topology_law_panel(
        OverlapConditionalTopologyLawPanelConfig(
            topology_rows_path=args.topology_rows_path,
            output_dir=args.output_dir,
            prior_truth_probability=float(args.prior_truth_probability),
            selected_family_weight=float(args.selected_family_weight),
            context_penalty_weight=float(args.context_penalty_weight),
            neighborhood_scale_weight=float(args.neighborhood_scale_weight),
            topology_neighborhood_weight=float(args.topology_neighborhood_weight),
            root_log_odds_penalty=float(args.root_log_odds_penalty),
            passthrough_log_odds_penalty=float(args.passthrough_log_odds_penalty),
            min_truth_support_per_stratum=int(args.min_truth_support_per_stratum),
        )
    )


if __name__ == "__main__":
    main()
