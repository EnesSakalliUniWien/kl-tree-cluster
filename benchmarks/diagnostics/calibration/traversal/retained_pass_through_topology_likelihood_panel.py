"""Retained pass-through topology likelihood diagnostics.

This panel tests whether the topology likelihood needed for retained
pass-through walks is identifiable. It does not fit or promote a production
Bayesian rule.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "diagnostic_retained_pass_through_topology_likelihood_not_calibration"
SCHEMA_VERSION = "retained_pass_through_topology_likelihood_panel/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.traversal.retained_pass_through_topology_likelihood_panel"
)

REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "left_method_id",
    "left_traversal_state",
    "right_traversal_state",
    "left_child_parent_edge_open",
    "left_sibling_open",
    "left_sibling_p_value",
    "left_depth",
    "left_n_descendant_leaves",
    "left_neighborhood_evidence_family",
    "right_neighborhood_evidence_family",
    "left_balance_product",
    "left_outgoing_edge_norm_balance",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "traversal_decision_agrees",
}

SELECTED_NEIGHBORHOOD_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "parent_id",
    "traversal_state",
    "decision_class",
}

TRAVERSAL_NETWORK_CONTEXT_COLUMNS = (
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "distance_to_pass_through_context",
    "distance_to_downstream_accepted_split",
    "network_component_node_count",
    "traversal_network_context_status",
)

LIKELIHOOD_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "likelihood_class",
    "stop_rule_pattern",
    "left_depth",
    "left_log_descendant_leaves",
    "left_n_descendant_leaves",
    "left_sibling_p_value",
    "left_neg_log10_sibling_p_value",
    "left_child_parent_edge_open",
    "left_sibling_open",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "distance_to_pass_through_context",
    "distance_to_downstream_accepted_split",
    "network_component_node_count",
    "traversal_network_context_status",
    "traversal_only_pair",
    "old_and_current_pair",
    "left_balance_product",
    "left_outgoing_edge_norm_balance",
    "finite_balance_product",
    "finite_outgoing_edge_norm_balance",
    "finite_topology_feature_count",
    "match_eligible",
)

MATCH_ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "signal_case_id",
    "signal_replicate",
    "signal_node_id",
    "control_case_id",
    "control_replicate",
    "control_node_id",
    "signal_depth",
    "control_depth",
    "abs_depth_delta",
    "signal_log_descendant_leaves",
    "control_log_descendant_leaves",
    "abs_log_descendant_leaves_delta",
    "signal_distance_to_pass_through_context",
    "control_distance_to_pass_through_context",
    "abs_pass_through_context_delta",
    "signal_distance_to_downstream_accepted_split",
    "control_distance_to_downstream_accepted_split",
    "abs_downstream_accepted_split_delta",
    "tree_network_distance",
    "tree_network_distance_status",
    "match_distance",
    "control_has_finite_topology",
    "traversal_network_match_status",
    "match_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "stop_rule_pattern",
    "signal_row_count",
    "selected_null_control_count",
    "matched_signal_count",
    "unmatched_signal_count",
    "matched_control_finite_topology_count",
    "finite_signal_balance_product_count",
    "finite_control_balance_product_count",
    "finite_signal_outgoing_edge_norm_balance_count",
    "finite_control_outgoing_edge_norm_balance_count",
    "signal_traversal_only_count",
    "control_traversal_only_count",
    "median_signal_depth",
    "median_control_depth",
    "median_signal_log_descendant_leaves",
    "median_control_log_descendant_leaves",
    "median_match_distance",
    "finite_signal_pass_through_context_count",
    "finite_control_pass_through_context_count",
    "finite_signal_downstream_split_distance_count",
    "finite_control_downstream_split_distance_count",
    "matched_traversal_network_context_count",
    "cross_case_network_distance_unavailable_count",
    "traversal_network_context_status",
    "likelihood_identifiability_status",
    "production_action",
)


@dataclass(frozen=True)
class RetainedPassThroughTopologyLikelihoodConfig:
    """Runtime contract for retained pass-through topology diagnostics."""

    candidate_rows_path: Path
    output_dir: Path
    selected_neighborhood_rows_path: Path | None = None
    max_depth_delta: float = 4.0
    max_log_descendant_leaves_delta: float = 2.0
    max_pass_through_context_delta: float = 4.0
    max_downstream_accepted_split_delta: float = 6.0
    min_matched_signal_count: int = 2

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "retained_pass_through_topology_likelihood_rows.csv"

    @property
    def match_rows_path(self) -> Path:
        return self.output_dir / "retained_pass_through_topology_match_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "retained_pass_through_topology_likelihood_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_columns(rows: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"candidate rows are missing columns: {missing!r}")


def _validate_selected_neighborhood_columns(rows: pd.DataFrame) -> None:
    missing = sorted(SELECTED_NEIGHBORHOOD_REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"selected-neighborhood rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: object) -> float:
    return float(value) if _is_finite(value) else math.nan


def _empty_parent_id(value: object) -> bool:
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _key_from_values(
    case_id: object,
    data_role: object,
    method_id: object,
    replicate: object,
    node_id: object,
) -> tuple[str, str, str, int, str] | None:
    try:
        replicate_int = int(float(replicate))
    except (TypeError, ValueError):
        return None
    return (
        str(case_id),
        str(data_role),
        str(method_id),
        replicate_int,
        str(node_id),
    )


def _neg_log10_p(value: object) -> float:
    try:
        p_value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(p_value) or p_value <= 0.0:
        return math.nan
    return float(-math.log10(p_value))


def _distance_to_pass_through_context(
    node_id: str,
    parent_by_node: dict[str, str],
    traversal_state_by_node: dict[str, str],
) -> float:
    if traversal_state_by_node.get(node_id) == "pass_through":
        return 0.0
    current = parent_by_node.get(node_id, "")
    distance = 1
    visited = {node_id}
    while current and current not in visited:
        if traversal_state_by_node.get(current) == "pass_through":
            return float(distance)
        visited.add(current)
        current = parent_by_node.get(current, "")
        distance += 1
    return math.nan


def _distance_to_downstream_accepted_split(
    node_id: str,
    children_by_node: dict[str, list[str]],
    decision_class_by_node: dict[str, str],
    traversal_state_by_node: dict[str, str],
) -> float:
    if (
        decision_class_by_node.get(node_id) == "accepted_internal_split"
        or traversal_state_by_node.get(node_id) == "split"
    ):
        return 0.0
    queue: deque[tuple[str, int]] = deque((child, 1) for child in children_by_node[node_id])
    visited = {node_id}
    while queue:
        current, distance = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if (
            decision_class_by_node.get(current) == "accepted_internal_split"
            or traversal_state_by_node.get(current) == "split"
        ):
            return float(distance)
        queue.extend((child, distance + 1) for child in children_by_node[current])
    return math.nan


def build_traversal_network_context_rows(
    selected_neighborhood_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Compute directed traversal context from selected-neighborhood tree rows."""
    if selected_neighborhood_rows.empty:
        return pd.DataFrame(columns=TRAVERSAL_NETWORK_CONTEXT_COLUMNS)
    _validate_selected_neighborhood_columns(selected_neighborhood_rows)
    rows = selected_neighborhood_rows.copy()
    rows["replicate"] = _numeric(rows, "replicate").fillna(-1).astype(int)

    records: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "method_id", "replicate"]
    for group_key, group in rows.groupby(group_columns, sort=False):
        case_id, data_role, method_id, replicate = group_key
        node_ids = [str(node_id) for node_id in group["node_id"]]
        node_set = set(node_ids)
        parent_by_node: dict[str, str] = {}
        children_by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        traversal_state_by_node: dict[str, str] = {}
        decision_class_by_node: dict[str, str] = {}

        for _, row in group.iterrows():
            node_id = str(row["node_id"])
            parent_id = "" if _empty_parent_id(row["parent_id"]) else str(row["parent_id"])
            parent_by_node[node_id] = parent_id if parent_id in node_set else ""
            traversal_state_by_node[node_id] = str(row["traversal_state"])
            decision_class_by_node[node_id] = str(row["decision_class"])

        for node_id, parent_id in parent_by_node.items():
            if parent_id:
                children_by_node.setdefault(parent_id, []).append(node_id)

        component_count = len(node_ids)
        for node_id in node_ids:
            records.append(
                {
                    "case_id": str(case_id),
                    "data_role": str(data_role),
                    "method_id": str(method_id),
                    "replicate": int(replicate),
                    "node_id": node_id,
                    "distance_to_pass_through_context": (
                        _distance_to_pass_through_context(
                            node_id,
                            parent_by_node,
                            traversal_state_by_node,
                        )
                    ),
                    "distance_to_downstream_accepted_split": (
                        _distance_to_downstream_accepted_split(
                            node_id,
                            children_by_node,
                            decision_class_by_node,
                            traversal_state_by_node,
                        )
                    ),
                    "network_component_node_count": int(component_count),
                    "traversal_network_context_status": "network_context_observed",
                }
            )
    return pd.DataFrame.from_records(records, columns=TRAVERSAL_NETWORK_CONTEXT_COLUMNS)


def _stop_rule_pattern(row: pd.Series) -> str:
    if bool(row["traversal_decision_agrees"]):
        return "paired_candidate_agreement"
    left_pass = str(row["left_traversal_state"]) == "pass_through"
    right_pass = str(row["right_traversal_state"]) == "pass_through"
    left_desc = int(row["left_descendant_accepted_split_count"])
    right_desc = int(row["right_descendant_accepted_split_count"])
    if left_pass and not right_pass and left_desc > right_desc:
        return "left_pass_through_downstream_split_right_stops"
    return "other_candidate_pattern"


def build_retained_pass_through_topology_likelihood_rows(
    candidate_rows: pd.DataFrame,
    traversal_network_context_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return signal and selected-null rows for retained pass-through likelihoods."""
    _validate_columns(candidate_rows)
    rows = candidate_rows.copy()
    rows["left_depth"] = _numeric(rows, "left_depth")
    rows["left_n_descendant_leaves"] = _numeric(rows, "left_n_descendant_leaves")
    rows["left_balance_product"] = _numeric(rows, "left_balance_product")
    rows["left_outgoing_edge_norm_balance"] = _numeric(
        rows,
        "left_outgoing_edge_norm_balance",
    )
    rows["left_descendant_accepted_split_count"] = (
        _numeric(
            rows,
            "left_descendant_accepted_split_count",
        )
        .fillna(0)
        .astype(int)
    )
    rows["right_descendant_accepted_split_count"] = (
        _numeric(
            rows,
            "right_descendant_accepted_split_count",
        )
        .fillna(0)
        .astype(int)
    )
    rows["stop_rule_pattern"] = [_stop_rule_pattern(row) for _, row in rows.iterrows()]
    selected = rows.loc[
        rows["stop_rule_pattern"].eq("left_pass_through_downstream_split_right_stops")
        & rows["data_role"].astype(str).isin({"signal", "selected_null"})
    ].copy()

    context_by_key: dict[tuple[str, str, str, int, str], pd.Series] = {}
    if traversal_network_context_rows is not None:
        for _, context_row in traversal_network_context_rows.iterrows():
            key = _key_from_values(
                context_row["case_id"],
                context_row["data_role"],
                context_row["method_id"],
                context_row["replicate"],
                context_row["node_id"],
            )
            if key is not None:
                context_by_key[key] = context_row

    records: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        key = _key_from_values(
            row["case_id"],
            row["data_role"],
            row["left_method_id"],
            row["replicate"],
            row["node_id"],
        )
        context_row = context_by_key.get(key) if key is not None else None
        if traversal_network_context_rows is None:
            context_status = "traversal_network_context_not_provided"
            pass_through_distance = math.nan
            downstream_split_distance = math.nan
            component_count = math.nan
        elif context_row is None:
            context_status = "traversal_network_context_missing_for_candidate"
            pass_through_distance = math.nan
            downstream_split_distance = math.nan
            component_count = math.nan
        else:
            context_status = str(context_row["traversal_network_context_status"])
            pass_through_distance = _finite_float(context_row["distance_to_pass_through_context"])
            downstream_split_distance = _finite_float(
                context_row["distance_to_downstream_accepted_split"]
            )
            component_count = _finite_float(context_row["network_component_node_count"])
        finite_balance = _is_finite(row["left_balance_product"])
        finite_edge_norm = _is_finite(row["left_outgoing_edge_norm_balance"])
        log_leaves = (
            float(math.log1p(float(row["left_n_descendant_leaves"])))
            if _is_finite(row["left_n_descendant_leaves"])
            else math.nan
        )
        traversal_only_pair = (
            str(row["left_neighborhood_evidence_family"]) == "traversal_only"
            and str(row["right_neighborhood_evidence_family"]) == "traversal_only"
        )
        old_and_current_pair = (
            str(row["left_neighborhood_evidence_family"]) == "old_and_current"
            and str(row["right_neighborhood_evidence_family"]) == "old_and_current"
        )
        likelihood_class = (
            "signal_retained_pass_through"
            if str(row["data_role"]) == "signal"
            else "selected_null_pass_through_control"
        )
        finite_topology_count = int(finite_balance) + int(finite_edge_norm)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "method_id": str(row["left_method_id"]),
                "replicate": int(row["replicate"]),
                "node_id": str(row["node_id"]),
                "likelihood_class": likelihood_class,
                "stop_rule_pattern": str(row["stop_rule_pattern"]),
                "left_depth": float(row["left_depth"]),
                "left_log_descendant_leaves": log_leaves,
                "left_n_descendant_leaves": float(row["left_n_descendant_leaves"]),
                "left_sibling_p_value": float(row["left_sibling_p_value"]),
                "left_neg_log10_sibling_p_value": _neg_log10_p(row["left_sibling_p_value"]),
                "left_child_parent_edge_open": _bool_value(row["left_child_parent_edge_open"]),
                "left_sibling_open": _bool_value(row["left_sibling_open"]),
                "left_descendant_accepted_split_count": int(
                    row["left_descendant_accepted_split_count"]
                ),
                "right_descendant_accepted_split_count": int(
                    row["right_descendant_accepted_split_count"]
                ),
                "distance_to_pass_through_context": pass_through_distance,
                "distance_to_downstream_accepted_split": downstream_split_distance,
                "network_component_node_count": component_count,
                "traversal_network_context_status": context_status,
                "traversal_only_pair": traversal_only_pair,
                "old_and_current_pair": old_and_current_pair,
                "left_balance_product": row["left_balance_product"],
                "left_outgoing_edge_norm_balance": row["left_outgoing_edge_norm_balance"],
                "finite_balance_product": finite_balance,
                "finite_outgoing_edge_norm_balance": finite_edge_norm,
                "finite_topology_feature_count": finite_topology_count,
                "match_eligible": bool(_is_finite(row["left_depth"]) and math.isfinite(log_leaves)),
            }
        )
    return pd.DataFrame.from_records(records, columns=LIKELIHOOD_ROW_COLUMNS)


def build_retained_pass_through_topology_match_rows(
    likelihood_rows: pd.DataFrame,
    *,
    max_depth_delta: float,
    max_log_descendant_leaves_delta: float,
    max_pass_through_context_delta: float,
    max_downstream_accepted_split_delta: float,
) -> pd.DataFrame:
    """Match signal retained pass-through rows to selected-null controls."""
    if likelihood_rows.empty:
        return pd.DataFrame(columns=MATCH_ROW_COLUMNS)
    signal = likelihood_rows.loc[
        likelihood_rows["likelihood_class"].eq("signal_retained_pass_through")
        & likelihood_rows["match_eligible"].astype(bool)
    ]
    controls = likelihood_rows.loc[
        likelihood_rows["likelihood_class"].eq("selected_null_pass_through_control")
        & likelihood_rows["match_eligible"].astype(bool)
    ]
    records: list[dict[str, object]] = []
    for _, signal_row in signal.iterrows():
        best_control = None
        best_distance = math.inf
        best_depth_delta = math.inf
        best_log_delta = math.inf
        best_pass_delta = math.nan
        best_split_delta = math.nan
        best_context_status = "traversal_network_context_incomplete"
        for _, control_row in controls.iterrows():
            depth_delta = abs(float(signal_row["left_depth"]) - float(control_row["left_depth"]))
            log_delta = abs(
                float(signal_row["left_log_descendant_leaves"])
                - float(control_row["left_log_descendant_leaves"])
            )
            if depth_delta > float(max_depth_delta) or log_delta > float(
                max_log_descendant_leaves_delta
            ):
                continue
            distance = depth_delta / float(max_depth_delta) + log_delta / float(
                max_log_descendant_leaves_delta
            )
            pass_delta = math.nan
            if _is_finite(signal_row["distance_to_pass_through_context"]) and _is_finite(
                control_row["distance_to_pass_through_context"]
            ):
                pass_delta = abs(
                    float(signal_row["distance_to_pass_through_context"])
                    - float(control_row["distance_to_pass_through_context"])
                )
                if pass_delta > float(max_pass_through_context_delta):
                    continue
                distance += pass_delta / float(max_pass_through_context_delta)
            split_delta = math.nan
            if _is_finite(signal_row["distance_to_downstream_accepted_split"]) and _is_finite(
                control_row["distance_to_downstream_accepted_split"]
            ):
                split_delta = abs(
                    float(signal_row["distance_to_downstream_accepted_split"])
                    - float(control_row["distance_to_downstream_accepted_split"])
                )
                if split_delta > float(max_downstream_accepted_split_delta):
                    continue
                distance += split_delta / float(max_downstream_accepted_split_delta)
            context_status = (
                "matched_traversal_network_context"
                if _is_finite(pass_delta) and _is_finite(split_delta)
                else "traversal_network_context_incomplete"
            )
            if distance < best_distance:
                best_control = control_row
                best_distance = distance
                best_depth_delta = depth_delta
                best_log_delta = log_delta
                best_pass_delta = pass_delta
                best_split_delta = split_delta
                best_context_status = context_status
        if best_control is None:
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "signal_case_id": str(signal_row["case_id"]),
                    "signal_replicate": int(signal_row["replicate"]),
                    "signal_node_id": str(signal_row["node_id"]),
                    "control_case_id": "",
                    "control_replicate": -1,
                    "control_node_id": "",
                    "signal_depth": float(signal_row["left_depth"]),
                    "control_depth": math.nan,
                    "abs_depth_delta": math.nan,
                    "signal_log_descendant_leaves": float(signal_row["left_log_descendant_leaves"]),
                    "control_log_descendant_leaves": math.nan,
                    "abs_log_descendant_leaves_delta": math.nan,
                    "signal_distance_to_pass_through_context": _finite_float(
                        signal_row["distance_to_pass_through_context"]
                    ),
                    "control_distance_to_pass_through_context": math.nan,
                    "abs_pass_through_context_delta": math.nan,
                    "signal_distance_to_downstream_accepted_split": _finite_float(
                        signal_row["distance_to_downstream_accepted_split"]
                    ),
                    "control_distance_to_downstream_accepted_split": math.nan,
                    "abs_downstream_accepted_split_delta": math.nan,
                    "tree_network_distance": math.nan,
                    "tree_network_distance_status": ("no_matched_selected_null_control"),
                    "match_distance": math.nan,
                    "control_has_finite_topology": False,
                    "traversal_network_match_status": ("no_matched_selected_null_control"),
                    "match_status": "no_matched_selected_null_control",
                }
            )
            continue
        same_tree = (
            str(signal_row["case_id"]) == str(best_control["case_id"])
            and str(signal_row["data_role"]) == str(best_control["data_role"])
            and str(signal_row["method_id"]) == str(best_control["method_id"])
            and int(signal_row["replicate"]) == int(best_control["replicate"])
        )
        tree_network_distance_status = (
            "same_tree_network_distance_not_computed_v1"
            if same_tree
            else "cross_case_network_distance_unavailable"
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "signal_case_id": str(signal_row["case_id"]),
                "signal_replicate": int(signal_row["replicate"]),
                "signal_node_id": str(signal_row["node_id"]),
                "control_case_id": str(best_control["case_id"]),
                "control_replicate": int(best_control["replicate"]),
                "control_node_id": str(best_control["node_id"]),
                "signal_depth": float(signal_row["left_depth"]),
                "control_depth": float(best_control["left_depth"]),
                "abs_depth_delta": float(best_depth_delta),
                "signal_log_descendant_leaves": float(signal_row["left_log_descendant_leaves"]),
                "control_log_descendant_leaves": float(best_control["left_log_descendant_leaves"]),
                "abs_log_descendant_leaves_delta": float(best_log_delta),
                "signal_distance_to_pass_through_context": _finite_float(
                    signal_row["distance_to_pass_through_context"]
                ),
                "control_distance_to_pass_through_context": _finite_float(
                    best_control["distance_to_pass_through_context"]
                ),
                "abs_pass_through_context_delta": best_pass_delta,
                "signal_distance_to_downstream_accepted_split": _finite_float(
                    signal_row["distance_to_downstream_accepted_split"]
                ),
                "control_distance_to_downstream_accepted_split": _finite_float(
                    best_control["distance_to_downstream_accepted_split"]
                ),
                "abs_downstream_accepted_split_delta": best_split_delta,
                "tree_network_distance": math.nan,
                "tree_network_distance_status": tree_network_distance_status,
                "match_distance": float(best_distance),
                "control_has_finite_topology": bool(
                    int(best_control["finite_topology_feature_count"]) > 0
                ),
                "traversal_network_match_status": best_context_status,
                "match_status": "matched_selected_null_control",
            }
        )
    return pd.DataFrame.from_records(records, columns=MATCH_ROW_COLUMNS)


def _finite_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = finite.dropna()
    if finite.empty:
        return math.nan
    return float(finite.median())


def summarize_retained_pass_through_topology_likelihood(
    likelihood_rows: pd.DataFrame,
    match_rows: pd.DataFrame,
    *,
    min_matched_signal_count: int,
) -> pd.DataFrame:
    """Summarize whether a topology likelihood is identifiable."""
    if likelihood_rows.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "stop_rule_pattern": "left_pass_through_downstream_split_right_stops",
                    "signal_row_count": 0,
                    "selected_null_control_count": 0,
                    "matched_signal_count": 0,
                    "unmatched_signal_count": 0,
                    "matched_control_finite_topology_count": 0,
                    "finite_signal_balance_product_count": 0,
                    "finite_control_balance_product_count": 0,
                    "finite_signal_outgoing_edge_norm_balance_count": 0,
                    "finite_control_outgoing_edge_norm_balance_count": 0,
                    "signal_traversal_only_count": 0,
                    "control_traversal_only_count": 0,
                    "median_signal_depth": math.nan,
                    "median_control_depth": math.nan,
                    "median_signal_log_descendant_leaves": math.nan,
                    "median_control_log_descendant_leaves": math.nan,
                    "median_match_distance": math.nan,
                    "finite_signal_pass_through_context_count": 0,
                    "finite_control_pass_through_context_count": 0,
                    "finite_signal_downstream_split_distance_count": 0,
                    "finite_control_downstream_split_distance_count": 0,
                    "matched_traversal_network_context_count": 0,
                    "cross_case_network_distance_unavailable_count": 0,
                    "traversal_network_context_status": ("traversal_network_context_not_provided"),
                    "likelihood_identifiability_status": "no_retained_pass_through_rows",
                    "production_action": "fail_closed_until_likelihood_identifiable",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
    signal = likelihood_rows.loc[
        likelihood_rows["likelihood_class"].eq("signal_retained_pass_through")
    ]
    controls = likelihood_rows.loc[
        likelihood_rows["likelihood_class"].eq("selected_null_pass_through_control")
    ]
    matched = match_rows.loc[match_rows["match_status"].eq("matched_selected_null_control")]
    unmatched = match_rows.loc[match_rows["match_status"].eq("no_matched_selected_null_control")]
    matched_control_finite = int(matched["control_has_finite_topology"].astype(bool).sum())
    finite_signal_balance = int(signal["finite_balance_product"].astype(bool).sum())
    finite_control_balance = int(controls["finite_balance_product"].astype(bool).sum())
    finite_signal_edge = int(signal["finite_outgoing_edge_norm_balance"].astype(bool).sum())
    finite_control_edge = int(controls["finite_outgoing_edge_norm_balance"].astype(bool).sum())
    finite_signal_pass_context = int(
        pd.to_numeric(
            signal["distance_to_pass_through_context"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
    )
    finite_control_pass_context = int(
        pd.to_numeric(
            controls["distance_to_pass_through_context"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
    )
    finite_signal_downstream_split = int(
        pd.to_numeric(
            signal["distance_to_downstream_accepted_split"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
    )
    finite_control_downstream_split = int(
        pd.to_numeric(
            controls["distance_to_downstream_accepted_split"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
    )
    matched_traversal_context = int(
        matched["traversal_network_match_status"].eq("matched_traversal_network_context").sum()
    )
    cross_case_network_unavailable = int(
        matched["tree_network_distance_status"].eq("cross_case_network_distance_unavailable").sum()
    )
    context_statuses = set(likelihood_rows["traversal_network_context_status"].dropna().astype(str))
    if context_statuses == {"traversal_network_context_not_provided"}:
        traversal_network_context_status = "traversal_network_context_not_provided"
    elif "traversal_network_context_missing_for_candidate" in context_statuses:
        traversal_network_context_status = "traversal_network_context_incomplete"
    elif matched.empty:
        traversal_network_context_status = "traversal_network_context_unmatched"
    elif matched_traversal_context == int(matched.shape[0]):
        traversal_network_context_status = "matched_traversal_network_context_observed"
    else:
        traversal_network_context_status = "traversal_network_context_incomplete"
    if signal.empty or controls.empty:
        status = "class_support_missing"
    elif int(matched.shape[0]) < int(min_matched_signal_count):
        status = "matched_control_support_insufficient"
    elif finite_signal_balance == 0 and finite_signal_edge == 0:
        status = "signal_topology_likelihood_not_identifiable"
    elif finite_control_balance == 0 and finite_control_edge == 0:
        status = "selected_null_topology_likelihood_not_identifiable"
    else:
        status = "topology_likelihood_identifiable_diagnostic_only"
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "stop_rule_pattern": "left_pass_through_downstream_split_right_stops",
                "signal_row_count": int(signal.shape[0]),
                "selected_null_control_count": int(controls.shape[0]),
                "matched_signal_count": int(matched.shape[0]),
                "unmatched_signal_count": int(unmatched.shape[0]),
                "matched_control_finite_topology_count": matched_control_finite,
                "finite_signal_balance_product_count": finite_signal_balance,
                "finite_control_balance_product_count": finite_control_balance,
                "finite_signal_outgoing_edge_norm_balance_count": finite_signal_edge,
                "finite_control_outgoing_edge_norm_balance_count": finite_control_edge,
                "signal_traversal_only_count": int(
                    signal["traversal_only_pair"].astype(bool).sum()
                ),
                "control_traversal_only_count": int(
                    controls["traversal_only_pair"].astype(bool).sum()
                ),
                "median_signal_depth": _finite_median(signal["left_depth"]),
                "median_control_depth": _finite_median(controls["left_depth"]),
                "median_signal_log_descendant_leaves": _finite_median(
                    signal["left_log_descendant_leaves"]
                ),
                "median_control_log_descendant_leaves": _finite_median(
                    controls["left_log_descendant_leaves"]
                ),
                "median_match_distance": _finite_median(match_rows["match_distance"]),
                "finite_signal_pass_through_context_count": (finite_signal_pass_context),
                "finite_control_pass_through_context_count": (finite_control_pass_context),
                "finite_signal_downstream_split_distance_count": (finite_signal_downstream_split),
                "finite_control_downstream_split_distance_count": (finite_control_downstream_split),
                "matched_traversal_network_context_count": (matched_traversal_context),
                "cross_case_network_distance_unavailable_count": (cross_case_network_unavailable),
                "traversal_network_context_status": traversal_network_context_status,
                "likelihood_identifiability_status": status,
                "production_action": "fail_closed_until_likelihood_identifiable",
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_retained_pass_through_topology_likelihood_panel(
    config: RetainedPassThroughTopologyLikelihoodConfig,
) -> dict[str, Path]:
    """Run retained pass-through topology likelihood diagnostics."""
    candidate_rows = pd.read_csv(config.candidate_rows_path, keep_default_na=False)
    traversal_network_context_rows = None
    if config.selected_neighborhood_rows_path is not None:
        selected_neighborhood_rows = pd.read_csv(
            config.selected_neighborhood_rows_path,
            keep_default_na=False,
        )
        traversal_network_context_rows = build_traversal_network_context_rows(
            selected_neighborhood_rows
        )
    likelihood_rows = build_retained_pass_through_topology_likelihood_rows(
        candidate_rows,
        traversal_network_context_rows,
    )
    match_rows = build_retained_pass_through_topology_match_rows(
        likelihood_rows,
        max_depth_delta=config.max_depth_delta,
        max_log_descendant_leaves_delta=config.max_log_descendant_leaves_delta,
        max_pass_through_context_delta=config.max_pass_through_context_delta,
        max_downstream_accepted_split_delta=(config.max_downstream_accepted_split_delta),
    )
    summary = summarize_retained_pass_through_topology_likelihood(
        likelihood_rows,
        match_rows,
        min_matched_signal_count=config.min_matched_signal_count,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    likelihood_rows.to_csv(config.rows_path, index=False)
    match_rows.to_csv(config.match_rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "inputs": {
            "candidate_rows": str(config.candidate_rows_path),
            "selected_neighborhood_rows": (
                None
                if config.selected_neighborhood_rows_path is None
                else str(config.selected_neighborhood_rows_path)
            ),
        },
        "matching": {
            "max_depth_delta": float(config.max_depth_delta),
            "max_log_descendant_leaves_delta": float(config.max_log_descendant_leaves_delta),
            "max_pass_through_context_delta": float(config.max_pass_through_context_delta),
            "max_downstream_accepted_split_delta": float(
                config.max_downstream_accepted_split_delta
            ),
            "min_matched_signal_count": int(config.min_matched_signal_count),
        },
        "outputs": {
            "rows": str(config.rows_path),
            "match_rows": str(config.match_rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": str(summary["production_action"].iloc[0]),
        "interpretation": (
            "Diagnostic-only identifiability panel for retained pass-through "
            "topology likelihoods; no production rule is fitted."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": config.rows_path,
        "match_rows": config.match_rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows-path", type=Path, required=True)
    parser.add_argument("--selected-neighborhood-rows-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-depth-delta", type=float, default=4.0)
    parser.add_argument("--max-log-descendant-leaves-delta", type=float, default=2.0)
    parser.add_argument("--max-pass-through-context-delta", type=float, default=4.0)
    parser.add_argument(
        "--max-downstream-accepted-split-delta",
        type=float,
        default=6.0,
    )
    parser.add_argument("--min-matched-signal-count", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_retained_pass_through_topology_likelihood_panel(
        RetainedPassThroughTopologyLikelihoodConfig(
            candidate_rows_path=args.candidate_rows_path,
            output_dir=args.output_dir,
            selected_neighborhood_rows_path=args.selected_neighborhood_rows_path,
            max_depth_delta=float(args.max_depth_delta),
            max_log_descendant_leaves_delta=float(args.max_log_descendant_leaves_delta),
            max_pass_through_context_delta=float(args.max_pass_through_context_delta),
            max_downstream_accepted_split_delta=float(args.max_downstream_accepted_split_delta),
            min_matched_signal_count=int(args.min_matched_signal_count),
        )
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
