"""Mine selected pass-through overlap fixtures from paired traversal rows.

This diagnostic turns selected-neighborhood candidate contrasts into explicit
retained pass-through fixture rows. It does not fit a production law; it reports
whether the selected event has enough signal and selected-null topology support
to make a conditional topology likelihood identifiable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.diagnostics.calibration.traversal.retained_pass_through_topology_likelihood_panel import (
    build_traversal_network_context_rows,
)
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.statistics.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
)

STUDY_ROLE = "diagnostic_overlap_selected_pass_through_fixture_not_calibration"
SCHEMA_VERSION = "overlap_selected_pass_through_fixture_miner/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap.overlap_selected_pass_through_fixture_miner"
)

SELECTED_EVENT = "left_pass_through_downstream_split_right_stops"
DEFAULT_PARTIAL_RECOVERY_ARI_FLOOR = 0.25

CANDIDATE_REQUIRED_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "left_method_id",
    "right_method_id",
    "left_traversal_state",
    "right_traversal_state",
    "left_child_parent_edge_open",
    "left_sibling_open",
    "left_sibling_p_value",
    "left_depth",
    "left_n_descendant_leaves",
    "left_neighborhood_evidence_family",
    "right_neighborhood_evidence_family",
    "right_explicit_guard_blocked",
    "left_balance_product",
    "left_outgoing_edge_norm_balance",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "traversal_decision_agrees",
}

SELECTED_NEIGHBORHOOD_OPTIONAL_COLUMNS = (
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "parent_id",
    "decision_class",
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "support_status",
    "guarded_recovery_status",
)

STRUCTURAL_TOPOLOGY_COLUMNS = (
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "structural_n_parent_context",
    "structural_n_node",
    "structural_n_incoming_sibling",
    "structural_n_left",
    "structural_n_right",
    "structural_incoming_branch_balance",
    "structural_outgoing_balance",
    "structural_balance_product",
    "structural_topology_feature_count",
    "structural_topology_context_status",
)

TRUTH_CONTEXT_COLUMNS = (
    "case_id",
    "data_role",
    "method_id",
    "replicate",
    "node_id",
    "data_seed",
    "truth_geometry_role",
    "truth_context_status",
    "truth_node_sample_count",
    "truth_node_cluster_count",
    "truth_node_majority_fraction",
    "truth_downstream_split_node_id",
    "truth_downstream_split_distance",
    "truth_downstream_split_ari",
    "truth_downstream_child_mean_purity",
    "truth_downstream_child_majority_distinct",
)

NODE_COLUMNS = (
    "schema_version",
    "study_role",
    "selected_event",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "parent_id",
    "left_method_id",
    "right_method_id",
    "fixture_role",
    "truth_evidence_role",
    "data_seed",
    "truth_geometry_role",
    "truth_context_status",
    "truth_node_sample_count",
    "truth_node_cluster_count",
    "truth_node_majority_fraction",
    "truth_downstream_split_node_id",
    "truth_downstream_split_distance",
    "truth_downstream_split_ari",
    "truth_downstream_child_mean_purity",
    "truth_downstream_child_majority_distinct",
    "topology_feature_status",
    "left_depth",
    "left_n_descendant_leaves",
    "left_sibling_p_value",
    "left_child_parent_edge_open",
    "left_sibling_open",
    "left_descendant_accepted_split_count",
    "right_descendant_accepted_split_count",
    "right_explicit_guard_blocked",
    "left_neighborhood_evidence_family",
    "right_neighborhood_evidence_family",
    "left_balance_product",
    "left_outgoing_edge_norm_balance",
    "finite_topology_feature_count",
    "structural_n_parent_context",
    "structural_n_node",
    "structural_n_incoming_sibling",
    "structural_n_left",
    "structural_n_right",
    "structural_incoming_branch_balance",
    "structural_outgoing_balance",
    "structural_balance_product",
    "structural_topology_feature_count",
    "structural_topology_context_status",
    "completed_balance_product",
    "completed_topology_feature_count",
    "completed_topology_feature_status",
    "distance_to_pass_through_context",
    "distance_to_downstream_accepted_split",
    "network_component_node_count",
    "traversal_network_context_status",
    "decision_class",
    "guard_truth_role",
    "topology_support_role",
    "topology_signal_role",
    "support_status",
    "guarded_recovery_status",
)

CASE_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "fixture_case_role",
    "selected_event_row_count",
    "signal_candidate_count",
    "selected_null_control_count",
    "truth_labeled_signal_count",
    "truth_recovery_count",
    "partial_truth_recovery_count",
    "barycentric_mixture_count",
    "fragment_false_split_count",
    "finite_topology_row_count",
    "structural_topology_row_count",
    "completed_topology_row_count",
    "finite_traversal_context_row_count",
    "min_depth",
    "max_depth",
    "median_descendant_leaves",
    "median_sibling_p_value",
    "case_support_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "selected_event",
    "selected_event_row_count",
    "signal_candidate_count",
    "selected_null_control_count",
    "truth_labeled_signal_count",
    "truth_recovery_count",
    "partial_truth_recovery_count",
    "barycentric_mixture_count",
    "fragment_false_split_count",
    "finite_signal_topology_count",
    "finite_selected_null_topology_count",
    "finite_signal_structural_topology_count",
    "finite_selected_null_structural_topology_count",
    "finite_signal_completed_topology_count",
    "finite_selected_null_completed_topology_count",
    "finite_signal_traversal_context_count",
    "finite_selected_null_traversal_context_count",
    "case_count",
    "signal_case_count",
    "selected_null_case_count",
    "fixture_support_status",
    "completed_fixture_support_status",
    "completed_balance_product_signal_min",
    "completed_balance_product_signal_median",
    "completed_balance_product_signal_max",
    "completed_balance_product_selected_null_min",
    "completed_balance_product_selected_null_median",
    "completed_balance_product_selected_null_max",
    "completed_balance_product_separator_direction",
    "completed_balance_product_separator_threshold",
    "completed_balance_product_zero_control_signal_retention",
    "completed_balance_product_zero_control_selected_null_count",
    "structural_fallback_validation_status",
    "next_required_step",
    "production_action",
)


@dataclass(frozen=True)
class OverlapSelectedPassThroughFixtureMinerConfig:
    """Runtime contract for selected pass-through fixture mining."""

    candidate_rows_path: Path
    output_dir: Path
    selected_neighborhood_rows_path: Path | None = None
    traversal_rows_path: Path | None = None
    gene_assignments_path: Path | None = None
    truth_suite: str = "binary"
    truth_recovery_ari_floor: float = 0.50
    truth_partial_recovery_ari_floor: float = DEFAULT_PARTIAL_RECOVERY_ARI_FLOOR
    truth_child_purity_floor: float = 0.65
    min_signal_candidate_count: int = 2
    min_selected_null_control_count: int = 2
    min_finite_topology_per_side: int = 2

    @property
    def node_rows_path(self) -> Path:
        return self.output_dir / "overlap_selected_pass_through_node_rows.csv"

    @property
    def case_rows_path(self) -> Path:
        return self.output_dir / "overlap_selected_pass_through_case_rows.csv"

    @property
    def support_summary_path(self) -> Path:
        return self.output_dir / "overlap_selected_pass_through_support_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _validate_candidate_columns(rows: pd.DataFrame) -> None:
    missing = sorted(CANDIDATE_REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"candidate rows are missing columns: {missing!r}")


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _is_finite(value: object) -> bool:
    return math.isfinite(_finite_float(value))


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = finite.dropna()
    if finite.empty:
        return math.nan
    return float(finite.median())


def _finite_array(values: pd.Series) -> np.ndarray:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return finite.dropna().to_numpy(dtype=float)


def _finite_int(value: object, default: int = 0) -> int:
    number = _finite_float(value)
    if not math.isfinite(number):
        return int(default)
    return int(number)


def _selected_event_pattern(row: pd.Series) -> str:
    if _bool_value(row["traversal_decision_agrees"]):
        return "paired_candidate_agreement"
    left_pass = str(row["left_traversal_state"]) == "pass_through"
    right_pass = str(row["right_traversal_state"]) == "pass_through"
    left_desc = _finite_int(row["left_descendant_accepted_split_count"])
    right_desc = _finite_int(row["right_descendant_accepted_split_count"])
    if left_pass and not right_pass and left_desc > right_desc:
        return SELECTED_EVENT
    return "other_candidate_pattern"


def _key(
    *,
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


def _optional_context_by_key(
    selected_neighborhood_rows: pd.DataFrame | None,
) -> tuple[
    dict[tuple[str, str, str, int, str], pd.Series],
    dict[tuple[str, str, str, int, str], pd.Series],
    dict[tuple[str, str, str, int, str], pd.Series],
]:
    if selected_neighborhood_rows is None or selected_neighborhood_rows.empty:
        return {}, {}, {}

    missing = sorted(
        set(SELECTED_NEIGHBORHOOD_OPTIONAL_COLUMNS[:6]) - set(selected_neighborhood_rows.columns)
    )
    if missing:
        raise ValueError(f"selected-neighborhood rows are missing columns: {missing!r}")

    context = build_traversal_network_context_rows(selected_neighborhood_rows)
    traversal_by_key: dict[tuple[str, str, str, int, str], pd.Series] = {}
    for _, row in context.iterrows():
        key = _key(
            case_id=row["case_id"],
            data_role=row["data_role"],
            method_id=row["method_id"],
            replicate=row["replicate"],
            node_id=row["node_id"],
        )
        if key is not None:
            traversal_by_key[key] = row

    node_by_key: dict[tuple[str, str, str, int, str], pd.Series] = {}
    for _, row in selected_neighborhood_rows.iterrows():
        key = _key(
            case_id=row["case_id"],
            data_role=row["data_role"],
            method_id=row["method_id"],
            replicate=row["replicate"],
            node_id=row["node_id"],
        )
        if key is not None:
            node_by_key[key] = row

    structural = build_structural_topology_context_rows(selected_neighborhood_rows)
    structural_by_key: dict[tuple[str, str, str, int, str], pd.Series] = {}
    for _, row in structural.iterrows():
        key = _key(
            case_id=row["case_id"],
            data_role=row["data_role"],
            method_id=row["method_id"],
            replicate=row["replicate"],
            node_id=row["node_id"],
        )
        if key is not None:
            structural_by_key[key] = row
    return traversal_by_key, node_by_key, structural_by_key


def _truth_context_by_key(
    truth_context_rows: pd.DataFrame | None,
) -> dict[tuple[str, str, str, int, str], pd.Series]:
    if truth_context_rows is None or truth_context_rows.empty:
        return {}
    missing = sorted(set(TRUTH_CONTEXT_COLUMNS[:5]) - set(truth_context_rows.columns))
    if missing:
        raise ValueError(f"truth context rows are missing columns: {missing!r}")
    rows_by_key: dict[tuple[str, str, str, int, str], pd.Series] = {}
    for _, row in truth_context_rows.iterrows():
        key = _key(
            case_id=row["case_id"],
            data_role=row["data_role"],
            method_id=row["method_id"],
            replicate=row["replicate"],
            node_id=row["node_id"],
        )
        if key is not None:
            rows_by_key[key] = row
    return rows_by_key


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator <= 0.0:
        return math.nan
    return float(numerator) / float(denominator)


def build_structural_topology_context_rows(
    selected_neighborhood_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Compute topology balance terms directly from selected tree structure."""
    if selected_neighborhood_rows.empty:
        return pd.DataFrame(columns=STRUCTURAL_TOPOLOGY_COLUMNS)
    required = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "node_id",
        "parent_id",
        "n_descendant_leaves",
    }
    missing = sorted(required - set(selected_neighborhood_rows.columns))
    if missing:
        raise ValueError(f"selected-neighborhood rows are missing columns: {missing!r}")

    rows = selected_neighborhood_rows.copy()
    rows["replicate"] = _numeric(rows, "replicate").fillna(-1).astype(int)
    rows["n_descendant_leaves"] = _numeric(rows, "n_descendant_leaves")
    records: list[dict[str, object]] = []
    for group_key, group in rows.groupby(
        ["case_id", "data_role", "method_id", "replicate"],
        sort=False,
    ):
        case_id, data_role, method_id, replicate = group_key
        node_ids = [str(node_id) for node_id in group["node_id"]]
        node_set = set(node_ids)
        node_size = {
            str(row["node_id"]): _finite_float(row["n_descendant_leaves"])
            for _, row in group.iterrows()
        }
        parent_by_node: dict[str, str] = {}
        children_by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for _, row in group.iterrows():
            node_id = str(row["node_id"])
            parent_id = str(row["parent_id"]).strip()
            if parent_id.lower() in {"", "nan", "none", "null"}:
                parent_id = ""
            parent_by_node[node_id] = parent_id if parent_id in node_set else ""
        for node_id, parent_id in parent_by_node.items():
            if parent_id:
                children_by_node.setdefault(parent_id, []).append(node_id)

        for node_id in node_ids:
            n_node = node_size.get(node_id, math.nan)
            parent_id = parent_by_node.get(node_id, "")
            n_parent = node_size.get(parent_id, math.nan) if parent_id else math.nan
            incoming_siblings = [
                sibling for sibling in children_by_node.get(parent_id, []) if sibling != node_id
            ]
            n_incoming_sibling = float(
                sum(
                    node_size[sibling]
                    for sibling in incoming_siblings
                    if math.isfinite(node_size.get(sibling, math.nan))
                )
            )
            if not incoming_siblings:
                n_incoming_sibling = math.nan
            incoming_balance = _safe_ratio(
                min(n_node, n_incoming_sibling),
                n_parent,
            )

            children = children_by_node.get(node_id, [])
            child_sizes = [
                node_size[child]
                for child in children
                if math.isfinite(node_size.get(child, math.nan))
            ]
            if len(child_sizes) >= 2:
                child_sizes = sorted(child_sizes, reverse=True)
                n_left = float(child_sizes[0])
                n_right = float(child_sizes[1])
                outgoing_balance = _safe_ratio(min(n_left, n_right), n_node)
            else:
                n_left = math.nan
                n_right = math.nan
                outgoing_balance = math.nan
            structural_balance_product = (
                incoming_balance * outgoing_balance
                if math.isfinite(incoming_balance) and math.isfinite(outgoing_balance)
                else math.nan
            )
            feature_count = (
                int(math.isfinite(incoming_balance))
                + int(math.isfinite(outgoing_balance))
                + int(math.isfinite(structural_balance_product))
            )
            if feature_count == 3:
                status = "structural_topology_observed"
            elif feature_count > 0:
                status = "structural_topology_partial"
            else:
                status = "structural_topology_unavailable"
            records.append(
                {
                    "case_id": str(case_id),
                    "data_role": str(data_role),
                    "method_id": str(method_id),
                    "replicate": int(replicate),
                    "node_id": node_id,
                    "structural_n_parent_context": n_parent,
                    "structural_n_node": n_node,
                    "structural_n_incoming_sibling": n_incoming_sibling,
                    "structural_n_left": n_left,
                    "structural_n_right": n_right,
                    "structural_incoming_branch_balance": incoming_balance,
                    "structural_outgoing_balance": outgoing_balance,
                    "structural_balance_product": structural_balance_product,
                    "structural_topology_feature_count": feature_count,
                    "structural_topology_context_status": status,
                }
            )
    return pd.DataFrame.from_records(records, columns=STRUCTURAL_TOPOLOGY_COLUMNS)


def _output_data_role(data_role: object) -> str:
    return "selected_null" if str(data_role) == "null" else str(data_role)


def _input_data_role(data_role: object) -> str:
    return "null" if str(data_role) == "selected_null" else str(data_role)


def _split_path(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _purity_from_labels(labels: Sequence[int]) -> float:
    if not labels:
        return math.nan
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    if values.size == 0:
        return math.nan
    return float(np.max(counts) / np.sum(counts))


def _majority_label(labels: Sequence[int]) -> int | None:
    if not labels:
        return None
    values, counts = np.unique(np.asarray(labels, dtype=int), return_counts=True)
    if values.size == 0:
        return None
    return int(values[int(np.argmax(counts))])


def _truth_label_maps_by_run(
    traversal_rows: pd.DataFrame,
    *,
    suite: str,
) -> dict[tuple[str, str, str, int], tuple[int, dict[str, int]]]:
    required = {"case_id", "data_role", "method_id", "replicate", "data_seed"}
    missing = sorted(required - set(traversal_rows.columns))
    if missing:
        raise ValueError(f"traversal rows are missing columns: {missing!r}")
    case_names = sorted(set(traversal_rows["case_id"].astype(str)))
    cases = {
        str(case["name"]): dict(case) for case in _select_cases(suite=suite, case_names=case_names)
    }
    maps: dict[tuple[str, str, str, int], tuple[int, dict[str, int]]] = {}
    for _, row in (
        traversal_rows[list(required)]
        .drop_duplicates()
        .sort_values(["case_id", "data_role", "method_id", "replicate"])
        .iterrows()
    ):
        case_id = str(row["case_id"])
        data_role = _output_data_role(row["data_role"])
        method_id = str(row["method_id"])
        replicate = int(row["replicate"])
        data_seed = int(row["data_seed"])
        case = cases[case_id]
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
        maps[(case_id, data_role, method_id, replicate)] = (
            data_seed,
            {
                str(label): int(value)
                for label, value in zip(
                    data.index.astype(str),
                    np.asarray(truth_labels, dtype=int),
                )
            },
        )
    return maps


def _is_descendant(
    *,
    ancestor_id: str,
    node_id: str,
    parent_by_node: dict[str, str],
) -> bool:
    current = parent_by_node.get(node_id, "")
    while current:
        if current == ancestor_id:
            return True
        current = parent_by_node.get(current, "")
    return False


def _truth_role_for_node(
    *,
    data_role: str,
    node_cluster_count: int,
    split_ari: float,
    child_mean_purity: float,
    child_majority_distinct: bool,
    min_recovery_ari: float,
    min_partial_recovery_ari: float,
    min_child_purity: float,
) -> tuple[str, str]:
    if data_role == "selected_null":
        return "selected_null_truth_control", "selected_null_truth_control_observed"
    if node_cluster_count <= 0:
        return "truth_context_unavailable", "truth_context_missing_node_samples"
    if node_cluster_count <= 1:
        return "fragment_false_pass_through", "homogeneous_signal_fragment_context"
    if not math.isfinite(split_ari):
        return "signal_truth_unresolved", "downstream_split_truth_unavailable"
    if (
        child_majority_distinct
        and split_ari >= float(min_recovery_ari)
        and child_mean_purity >= float(min_child_purity)
    ):
        return (
            "truth_recovery_pass_through_positive",
            "truth_recovery_downstream_split_observed",
        )
    if (
        child_majority_distinct
        and split_ari >= float(min_partial_recovery_ari)
        and child_mean_purity >= float(min_child_purity)
    ):
        return (
            "partial_truth_recovery_pass_through_candidate",
            "partial_truth_recovery_downstream_split_observed",
        )
    if (
        not child_majority_distinct
        and split_ari >= float(min_partial_recovery_ari)
        and child_mean_purity >= float(min_child_purity)
    ):
        return (
            "barycentric_mixture_pass_through_candidate",
            "barycentric_mixture_downstream_shift_observed",
        )
    if split_ari <= 0.05 and child_mean_purity >= 0.90:
        return "fragment_false_pass_through", "misaligned_signal_fragment_context"
    return "signal_truth_unresolved", "truth_context_unresolved"


def build_selected_pass_through_truth_context_rows(
    selected_neighborhood_rows: pd.DataFrame,
    gene_assignments: pd.DataFrame,
    traversal_rows: pd.DataFrame,
    *,
    suite: str = "binary",
    min_recovery_ari: float = 0.50,
    min_partial_recovery_ari: float = DEFAULT_PARTIAL_RECOVERY_ARI_FLOOR,
    min_child_purity: float = 0.65,
) -> pd.DataFrame:
    """Attach synthetic truth geometry to selected tree nodes.

    This is oracle diagnostic metadata for benchmark fixtures only. It is used
    to find overlap positives/controls and is not a calibration rule.
    """
    if selected_neighborhood_rows.empty:
        return pd.DataFrame(columns=TRUTH_CONTEXT_COLUMNS)
    required_nodes = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "node_id",
        "parent_id",
        "depth",
        "decision_class",
    }
    required_genes = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "sample_id",
        "path_node_ids",
    }
    missing_nodes = sorted(required_nodes - set(selected_neighborhood_rows.columns))
    missing_genes = sorted(required_genes - set(gene_assignments.columns))
    if missing_nodes:
        raise ValueError(f"selected-neighborhood rows are missing columns: {missing_nodes!r}")
    if missing_genes:
        raise ValueError(f"gene assignments are missing columns: {missing_genes!r}")

    truth_by_run = _truth_label_maps_by_run(traversal_rows, suite=suite)
    node_rows = selected_neighborhood_rows.copy()
    node_rows["replicate"] = _numeric(node_rows, "replicate").fillna(-1).astype(int)
    node_rows["data_role"] = node_rows["data_role"].map(_output_data_role)
    gene_rows = gene_assignments.copy()
    gene_rows["replicate"] = _numeric(gene_rows, "replicate").fillna(-1).astype(int)
    gene_rows["data_role"] = gene_rows["data_role"].map(_output_data_role)

    records: list[dict[str, object]] = []
    group_columns = ["case_id", "data_role", "method_id", "replicate"]
    for group_key, group in node_rows.groupby(group_columns, sort=False):
        case_id, data_role, method_id, replicate = (
            str(group_key[0]),
            str(group_key[1]),
            str(group_key[2]),
            int(group_key[3]),
        )
        truth_entry = truth_by_run.get((case_id, data_role, method_id, replicate))
        if truth_entry is None:
            continue
        data_seed, truth_by_sample = truth_entry
        genes = gene_rows[
            gene_rows["case_id"].astype(str).eq(case_id)
            & gene_rows["data_role"].astype(str).eq(data_role)
            & gene_rows["method_id"].astype(str).eq(method_id)
            & gene_rows["replicate"].astype(int).eq(replicate)
        ]
        if genes.empty:
            continue

        node_ids = [str(node_id) for node_id in group["node_id"]]
        node_set = set(node_ids)
        parent_by_node: dict[str, str] = {}
        children_by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        depth_by_node: dict[str, float] = {}
        decision_by_node: dict[str, str] = {}
        for _, row in group.iterrows():
            node_id = str(row["node_id"])
            parent_id = str(row["parent_id"]).strip()
            if parent_id.lower() in {"", "nan", "none", "null"}:
                parent_id = ""
            parent_by_node[node_id] = parent_id if parent_id in node_set else ""
            depth_by_node[node_id] = _finite_float(row["depth"])
            decision_by_node[node_id] = str(row["decision_class"])
        for node_id, parent_id in parent_by_node.items():
            if parent_id:
                children_by_node.setdefault(parent_id, []).append(node_id)

        samples_by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for _, gene_row in genes.iterrows():
            sample_id = str(gene_row["sample_id"])
            for path_node in _split_path(gene_row["path_node_ids"]):
                if path_node in samples_by_node:
                    samples_by_node[path_node].append(sample_id)

        def labels_for_node(node_id: str) -> list[int]:
            return [
                truth_by_sample[sample_id]
                for sample_id in samples_by_node.get(node_id, [])
                if sample_id in truth_by_sample
            ]

        def split_metrics(split_node_id: str) -> dict[str, object] | None:
            children = children_by_node.get(split_node_id, [])
            if len(children) < 2:
                return None
            left_id, right_id = children[:2]
            left_truth = labels_for_node(left_id)
            right_truth = labels_for_node(right_id)
            if not left_truth or not right_truth:
                return None
            truth = np.asarray(left_truth + right_truth, dtype=int)
            membership = np.concatenate(
                [
                    np.zeros(len(left_truth), dtype=int),
                    np.ones(len(right_truth), dtype=int),
                ]
            )
            left_majority = _majority_label(left_truth)
            right_majority = _majority_label(right_truth)
            return {
                "split_node_id": split_node_id,
                "split_ari": float(adjusted_rand_score(truth, membership)),
                "child_mean_purity": float(
                    np.nanmean([_purity_from_labels(left_truth), _purity_from_labels(right_truth)])
                ),
                "child_majority_distinct": bool(
                    left_majority is not None
                    and right_majority is not None
                    and left_majority != right_majority
                ),
            }

        for node_id in node_ids:
            node_truth = labels_for_node(node_id)
            node_cluster_count = int(np.unique(np.asarray(node_truth, dtype=int)).size)
            node_majority = _purity_from_labels(node_truth)
            accepted_descendants: list[tuple[float, str, dict[str, object]]] = []
            for candidate_id, decision in decision_by_node.items():
                if decision != "accepted_internal_split":
                    continue
                if not _is_descendant(
                    ancestor_id=node_id,
                    node_id=candidate_id,
                    parent_by_node=parent_by_node,
                ):
                    continue
                metrics = split_metrics(candidate_id)
                if metrics is None:
                    continue
                distance = depth_by_node.get(candidate_id, math.nan) - depth_by_node.get(
                    node_id,
                    math.nan,
                )
                if not math.isfinite(distance) or distance <= 0:
                    continue
                accepted_descendants.append((float(distance), candidate_id, metrics))
            if accepted_descendants:
                accepted_descendants.sort(
                    key=lambda item: (
                        item[0],
                        -_finite_float(item[2]["split_ari"]),
                    )
                )
                split_distance, split_node_id, best = accepted_descendants[0]
                split_ari = _finite_float(best["split_ari"])
                child_mean_purity = _finite_float(best["child_mean_purity"])
                child_majority_distinct = bool(best["child_majority_distinct"])
            else:
                split_distance = math.nan
                split_node_id = ""
                split_ari = math.nan
                child_mean_purity = math.nan
                child_majority_distinct = False

            role, status = _truth_role_for_node(
                data_role=data_role,
                node_cluster_count=node_cluster_count,
                split_ari=split_ari,
                child_mean_purity=child_mean_purity,
                child_majority_distinct=child_majority_distinct,
                min_recovery_ari=min_recovery_ari,
                min_partial_recovery_ari=min_partial_recovery_ari,
                min_child_purity=min_child_purity,
            )
            records.append(
                {
                    "case_id": case_id,
                    "data_role": data_role,
                    "method_id": method_id,
                    "replicate": replicate,
                    "node_id": node_id,
                    "data_seed": int(data_seed),
                    "truth_geometry_role": role,
                    "truth_context_status": status,
                    "truth_node_sample_count": int(len(node_truth)),
                    "truth_node_cluster_count": int(node_cluster_count),
                    "truth_node_majority_fraction": node_majority,
                    "truth_downstream_split_node_id": split_node_id,
                    "truth_downstream_split_distance": split_distance,
                    "truth_downstream_split_ari": split_ari,
                    "truth_downstream_child_mean_purity": child_mean_purity,
                    "truth_downstream_child_majority_distinct": child_majority_distinct,
                }
            )
    return pd.DataFrame.from_records(records, columns=TRUTH_CONTEXT_COLUMNS)


def _fixture_role(
    *,
    data_role: str,
    guard_truth_role: str,
    topology_signal_role: str,
    truth_geometry_role: str = "",
) -> str:
    if data_role == "selected_null":
        return "selected_null_pass_through_control"
    if truth_geometry_role == "truth_recovery_pass_through_positive":
        return "truth_recovery_pass_through_positive"
    if truth_geometry_role == "partial_truth_recovery_pass_through_candidate":
        return "partial_truth_recovery_pass_through_candidate"
    if truth_geometry_role == "barycentric_mixture_pass_through_candidate":
        return "signal_barycentric_mixture_pass_through_candidate"
    if truth_geometry_role == "fragment_false_pass_through":
        return "fragment_false_pass_through"
    if guard_truth_role == "truth_recovery" or topology_signal_role == "signal":
        return "truth_recovery_pass_through_positive"
    if guard_truth_role == "fragment_like":
        return "fragment_false_pass_through"
    if data_role == "signal":
        return "signal_pass_through_candidate_unresolved"
    return "unclassified_selected_pass_through"


def _topology_feature_status(finite_topology_count: int) -> str:
    if finite_topology_count >= 2:
        return "topology_features_observed"
    if finite_topology_count == 1:
        return "topology_features_partial"
    return "topology_features_missing"


def build_overlap_selected_pass_through_node_rows(
    candidate_rows: pd.DataFrame,
    selected_neighborhood_rows: pd.DataFrame | None = None,
    truth_context_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return selected pass-through fixture nodes from candidate contrasts."""
    if candidate_rows.empty:
        return pd.DataFrame(columns=NODE_COLUMNS)
    _validate_candidate_columns(candidate_rows)
    rows = candidate_rows.copy()
    rows["left_descendant_accepted_split_count"] = _numeric(
        rows,
        "left_descendant_accepted_split_count",
    ).fillna(0)
    rows["right_descendant_accepted_split_count"] = _numeric(
        rows,
        "right_descendant_accepted_split_count",
    ).fillna(0)
    rows["selected_event"] = [_selected_event_pattern(row) for _, row in rows.iterrows()]
    selected = rows.loc[rows["selected_event"].eq(SELECTED_EVENT)].copy()
    if selected.empty:
        return pd.DataFrame(columns=NODE_COLUMNS)

    traversal_by_key, node_by_key, structural_by_key = _optional_context_by_key(
        selected_neighborhood_rows
    )
    truth_by_key = _truth_context_by_key(truth_context_rows)
    records: list[dict[str, object]] = []
    for _, row in selected.sort_values(["case_id", "data_role", "replicate", "node_id"]).iterrows():
        key = _key(
            case_id=row["case_id"],
            data_role=row["data_role"],
            method_id=row["left_method_id"],
            replicate=row["replicate"],
            node_id=row["node_id"],
        )
        traversal = traversal_by_key.get(key) if key is not None else None
        node = node_by_key.get(key) if key is not None else None
        structural = structural_by_key.get(key) if key is not None else None
        truth = truth_by_key.get(key) if key is not None else None

        guard_truth_role = "" if node is None else str(node.get("guard_truth_role", ""))
        topology_support_role = "" if node is None else str(node.get("topology_support_role", ""))
        topology_signal_role = "" if node is None else str(node.get("topology_signal_role", ""))
        support_status = "" if node is None else str(node.get("support_status", ""))
        guarded_recovery_status = (
            "" if node is None else str(node.get("guarded_recovery_status", ""))
        )
        truth_geometry_role = "" if truth is None else str(truth.get("truth_geometry_role", ""))
        truth_context_status = (
            "truth_context_not_provided"
            if truth_context_rows is None
            else (
                "truth_context_missing_for_candidate"
                if truth is None
                else str(truth.get("truth_context_status", ""))
            )
        )
        finite_topology_count = int(_is_finite(row["left_balance_product"])) + int(
            _is_finite(row["left_outgoing_edge_norm_balance"])
        )
        structural_feature_count = (
            0 if structural is None else int(structural["structural_topology_feature_count"])
        )
        structural_balance_product = (
            math.nan
            if structural is None
            else _finite_float(structural["structural_balance_product"])
        )
        completed_balance_product = (
            _finite_float(row["left_balance_product"])
            if _is_finite(row["left_balance_product"])
            else structural_balance_product
        )
        completed_topology_count = finite_topology_count
        if completed_topology_count == 0 and math.isfinite(structural_balance_product):
            completed_topology_count = 1
        if finite_topology_count > 0:
            completed_status = "direct_topology_features_observed"
        elif math.isfinite(structural_balance_product):
            completed_status = "structural_topology_fallback_observed"
        else:
            completed_status = "completed_topology_features_missing"
        fixture_role = _fixture_role(
            data_role=str(row["data_role"]),
            guard_truth_role=guard_truth_role,
            topology_signal_role=topology_signal_role,
            truth_geometry_role=truth_geometry_role,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "selected_event": SELECTED_EVENT,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "replicate": int(row["replicate"]),
                "node_id": str(row["node_id"]),
                "parent_id": "" if node is None else str(node.get("parent_id", "")),
                "left_method_id": str(row["left_method_id"]),
                "right_method_id": str(row["right_method_id"]),
                "fixture_role": fixture_role,
                "truth_evidence_role": guard_truth_role,
                "data_seed": (math.nan if truth is None else _finite_float(truth["data_seed"])),
                "truth_geometry_role": truth_geometry_role,
                "truth_context_status": truth_context_status,
                "truth_node_sample_count": (
                    0 if truth is None else _finite_int(truth["truth_node_sample_count"])
                ),
                "truth_node_cluster_count": (
                    0 if truth is None else _finite_int(truth["truth_node_cluster_count"])
                ),
                "truth_node_majority_fraction": (
                    math.nan
                    if truth is None
                    else _finite_float(truth["truth_node_majority_fraction"])
                ),
                "truth_downstream_split_node_id": (
                    "" if truth is None else str(truth["truth_downstream_split_node_id"])
                ),
                "truth_downstream_split_distance": (
                    math.nan
                    if truth is None
                    else _finite_float(truth["truth_downstream_split_distance"])
                ),
                "truth_downstream_split_ari": (
                    math.nan
                    if truth is None
                    else _finite_float(truth["truth_downstream_split_ari"])
                ),
                "truth_downstream_child_mean_purity": (
                    math.nan
                    if truth is None
                    else _finite_float(truth["truth_downstream_child_mean_purity"])
                ),
                "truth_downstream_child_majority_distinct": (
                    False
                    if truth is None
                    else _bool_value(truth["truth_downstream_child_majority_distinct"])
                ),
                "topology_feature_status": _topology_feature_status(finite_topology_count),
                "left_depth": _finite_float(row["left_depth"]),
                "left_n_descendant_leaves": _finite_float(row["left_n_descendant_leaves"]),
                "left_sibling_p_value": _finite_float(row["left_sibling_p_value"]),
                "left_child_parent_edge_open": _bool_value(row["left_child_parent_edge_open"]),
                "left_sibling_open": _bool_value(row["left_sibling_open"]),
                "left_descendant_accepted_split_count": int(
                    row["left_descendant_accepted_split_count"]
                ),
                "right_descendant_accepted_split_count": int(
                    row["right_descendant_accepted_split_count"]
                ),
                "right_explicit_guard_blocked": _bool_value(row["right_explicit_guard_blocked"]),
                "left_neighborhood_evidence_family": str(row["left_neighborhood_evidence_family"]),
                "right_neighborhood_evidence_family": str(
                    row["right_neighborhood_evidence_family"]
                ),
                "left_balance_product": _finite_float(row["left_balance_product"]),
                "left_outgoing_edge_norm_balance": _finite_float(
                    row["left_outgoing_edge_norm_balance"]
                ),
                "finite_topology_feature_count": finite_topology_count,
                "structural_n_parent_context": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_n_parent_context"])
                ),
                "structural_n_node": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_n_node"])
                ),
                "structural_n_incoming_sibling": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_n_incoming_sibling"])
                ),
                "structural_n_left": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_n_left"])
                ),
                "structural_n_right": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_n_right"])
                ),
                "structural_incoming_branch_balance": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_incoming_branch_balance"])
                ),
                "structural_outgoing_balance": (
                    math.nan
                    if structural is None
                    else _finite_float(structural["structural_outgoing_balance"])
                ),
                "structural_balance_product": structural_balance_product,
                "structural_topology_feature_count": structural_feature_count,
                "structural_topology_context_status": (
                    "structural_topology_context_not_provided"
                    if selected_neighborhood_rows is None
                    else (
                        "structural_topology_context_missing_for_candidate"
                        if structural is None
                        else str(structural["structural_topology_context_status"])
                    )
                ),
                "completed_balance_product": completed_balance_product,
                "completed_topology_feature_count": completed_topology_count,
                "completed_topology_feature_status": completed_status,
                "distance_to_pass_through_context": (
                    math.nan
                    if traversal is None
                    else _finite_float(traversal["distance_to_pass_through_context"])
                ),
                "distance_to_downstream_accepted_split": (
                    math.nan
                    if traversal is None
                    else _finite_float(traversal["distance_to_downstream_accepted_split"])
                ),
                "network_component_node_count": (
                    math.nan
                    if traversal is None
                    else _finite_float(traversal["network_component_node_count"])
                ),
                "traversal_network_context_status": (
                    "traversal_network_context_not_provided"
                    if selected_neighborhood_rows is None
                    else (
                        "traversal_network_context_missing_for_candidate"
                        if traversal is None
                        else str(traversal["traversal_network_context_status"])
                    )
                ),
                "decision_class": "" if node is None else str(node.get("decision_class", "")),
                "guard_truth_role": guard_truth_role,
                "topology_support_role": topology_support_role,
                "topology_signal_role": topology_signal_role,
                "support_status": support_status,
                "guarded_recovery_status": guarded_recovery_status,
            }
        )
    return pd.DataFrame.from_records(records, columns=NODE_COLUMNS)


def _case_support_status(group: pd.DataFrame) -> str:
    if group["fixture_role"].eq("selected_null_pass_through_control").any():
        if int(group["finite_topology_feature_count"].gt(0).sum()) == 0:
            return "selected_null_control_topology_missing"
        return "selected_null_control_with_topology_context"
    if group["fixture_role"].eq("signal_pass_through_candidate_unresolved").any():
        if int(group["finite_topology_feature_count"].gt(0).sum()) == 0:
            return "signal_candidate_topology_missing"
        return "signal_candidate_with_topology_context"
    if group["fixture_role"].eq("truth_recovery_pass_through_positive").any():
        return "truth_recovery_positive_context"
    if group["fixture_role"].eq("partial_truth_recovery_pass_through_candidate").any():
        return "partial_truth_recovery_context"
    if group["fixture_role"].eq("signal_barycentric_mixture_pass_through_candidate").any():
        return "barycentric_mixture_context"
    if group["fixture_role"].eq("fragment_false_pass_through").any():
        return "fragment_false_split_context"
    return "unclassified_context"


def build_overlap_selected_pass_through_case_rows(
    node_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected pass-through fixture nodes by case and replicate."""
    if node_rows.empty:
        return pd.DataFrame(columns=CASE_COLUMNS)
    records: list[dict[str, object]] = []
    for keys, group in node_rows.groupby(
        ["case_id", "data_role", "replicate"],
        sort=True,
    ):
        case_id, data_role, replicate = keys
        finite_context = (
            pd.to_numeric(
                group["distance_to_downstream_accepted_split"],
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .notna()
        )
        selected_null_control_count = int(
            group["fixture_role"].eq("selected_null_pass_through_control").sum()
        )
        signal_candidate_count = int(group["data_role"].astype(str).eq("signal").sum())
        truth_labeled_signal_count = int(
            group["truth_geometry_role"]
            .astype(str)
            .isin(
                {
                    "truth_recovery_pass_through_positive",
                    "partial_truth_recovery_pass_through_candidate",
                    "barycentric_mixture_pass_through_candidate",
                    "fragment_false_pass_through",
                    "signal_truth_unresolved",
                }
            )
            .sum()
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(case_id),
                "data_role": str(data_role),
                "replicate": int(replicate),
                "fixture_case_role": (
                    "selected_null_control_case"
                    if selected_null_control_count > 0
                    else "signal_retained_pass_through_case"
                    if signal_candidate_count > 0
                    else "unclassified_case"
                ),
                "selected_event_row_count": int(group.shape[0]),
                "signal_candidate_count": signal_candidate_count,
                "selected_null_control_count": selected_null_control_count,
                "truth_labeled_signal_count": truth_labeled_signal_count,
                "truth_recovery_count": int(
                    group["fixture_role"].eq("truth_recovery_pass_through_positive").sum()
                ),
                "partial_truth_recovery_count": int(
                    group["fixture_role"].eq("partial_truth_recovery_pass_through_candidate").sum()
                ),
                "barycentric_mixture_count": int(
                    group["fixture_role"]
                    .eq("signal_barycentric_mixture_pass_through_candidate")
                    .sum()
                ),
                "fragment_false_split_count": int(
                    group["fixture_role"].eq("fragment_false_pass_through").sum()
                ),
                "finite_topology_row_count": int(
                    group["finite_topology_feature_count"].gt(0).sum()
                ),
                "structural_topology_row_count": int(
                    group["structural_topology_feature_count"].gt(0).sum()
                ),
                "completed_topology_row_count": int(
                    group["completed_topology_feature_count"].gt(0).sum()
                ),
                "finite_traversal_context_row_count": int(finite_context.sum()),
                "min_depth": _finite_float(group["left_depth"].min(skipna=True)),
                "max_depth": _finite_float(group["left_depth"].max(skipna=True)),
                "median_descendant_leaves": _median(group["left_n_descendant_leaves"]),
                "median_sibling_p_value": _median(group["left_sibling_p_value"]),
                "case_support_status": _case_support_status(group),
            }
        )
    return pd.DataFrame.from_records(records, columns=CASE_COLUMNS)


def _completed_balance_product_validation(
    signal: pd.DataFrame,
    selected_null: pd.DataFrame,
) -> dict[str, object]:
    signal_values = _finite_array(signal["completed_balance_product"])
    control_values = _finite_array(selected_null["completed_balance_product"])
    if signal_values.size == 0 or control_values.size == 0:
        return {
            "completed_balance_product_signal_min": math.nan,
            "completed_balance_product_signal_median": math.nan,
            "completed_balance_product_signal_max": math.nan,
            "completed_balance_product_selected_null_min": math.nan,
            "completed_balance_product_selected_null_median": math.nan,
            "completed_balance_product_selected_null_max": math.nan,
            "completed_balance_product_separator_direction": "",
            "completed_balance_product_separator_threshold": math.nan,
            "completed_balance_product_zero_control_signal_retention": math.nan,
            "completed_balance_product_zero_control_selected_null_count": 0,
            "structural_fallback_validation_status": ("completed_balance_product_support_missing"),
        }

    signal_min = float(np.min(signal_values))
    signal_median = float(np.median(signal_values))
    signal_max = float(np.max(signal_values))
    control_min = float(np.min(control_values))
    control_median = float(np.median(control_values))
    control_max = float(np.max(control_values))

    high_threshold = signal_min
    high_signal_count = int(np.sum(signal_values >= high_threshold))
    high_control_count = int(np.sum(control_values >= high_threshold))
    low_threshold = signal_max
    low_signal_count = int(np.sum(signal_values <= low_threshold))
    low_control_count = int(np.sum(control_values <= low_threshold))
    signal_total = int(signal_values.size)
    if low_control_count < high_control_count:
        direction = "low"
        threshold = low_threshold
        signal_count = low_signal_count
        control_count = low_control_count
    else:
        direction = "high"
        threshold = high_threshold
        signal_count = high_signal_count
        control_count = high_control_count

    if control_count == 0:
        status = "structural_fallback_separates_selected_event_diagnostic_only"
    else:
        status = "structural_fallback_overlaps_selected_null_controls"

    return {
        "completed_balance_product_signal_min": signal_min,
        "completed_balance_product_signal_median": signal_median,
        "completed_balance_product_signal_max": signal_max,
        "completed_balance_product_selected_null_min": control_min,
        "completed_balance_product_selected_null_median": control_median,
        "completed_balance_product_selected_null_max": control_max,
        "completed_balance_product_separator_direction": direction,
        "completed_balance_product_separator_threshold": float(threshold),
        "completed_balance_product_zero_control_signal_retention": (
            float(signal_count / signal_total) if signal_total > 0 else math.nan
        ),
        "completed_balance_product_zero_control_selected_null_count": control_count,
        "structural_fallback_validation_status": status,
    }


def summarize_overlap_selected_pass_through_support(
    node_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    *,
    min_signal_candidate_count: int,
    min_selected_null_control_count: int,
    min_finite_topology_per_side: int,
) -> pd.DataFrame:
    """Summarize whether the selected pass-through fixture is identifiable."""
    if node_rows.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "selected_event": SELECTED_EVENT,
                    "selected_event_row_count": 0,
                    "signal_candidate_count": 0,
                    "selected_null_control_count": 0,
                    "truth_labeled_signal_count": 0,
                    "truth_recovery_count": 0,
                    "partial_truth_recovery_count": 0,
                    "barycentric_mixture_count": 0,
                    "fragment_false_split_count": 0,
                    "finite_signal_topology_count": 0,
                    "finite_selected_null_topology_count": 0,
                    "finite_signal_structural_topology_count": 0,
                    "finite_selected_null_structural_topology_count": 0,
                    "finite_signal_completed_topology_count": 0,
                    "finite_selected_null_completed_topology_count": 0,
                    "finite_signal_traversal_context_count": 0,
                    "finite_selected_null_traversal_context_count": 0,
                    "case_count": 0,
                    "signal_case_count": 0,
                    "selected_null_case_count": 0,
                    "fixture_support_status": "no_selected_pass_through_fixture_rows",
                    "completed_fixture_support_status": ("no_selected_pass_through_fixture_rows"),
                    "completed_balance_product_signal_min": math.nan,
                    "completed_balance_product_signal_median": math.nan,
                    "completed_balance_product_signal_max": math.nan,
                    "completed_balance_product_selected_null_min": math.nan,
                    "completed_balance_product_selected_null_median": math.nan,
                    "completed_balance_product_selected_null_max": math.nan,
                    "completed_balance_product_separator_direction": "",
                    "completed_balance_product_separator_threshold": math.nan,
                    "completed_balance_product_zero_control_signal_retention": math.nan,
                    "completed_balance_product_zero_control_selected_null_count": 0,
                    "structural_fallback_validation_status": (
                        "completed_balance_product_support_missing"
                    ),
                    "next_required_step": "mine_or_generate_selected_pass_through_rows",
                    "production_action": "fail_closed_until_fixture_support_observed",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )

    signal = node_rows.loc[node_rows["data_role"].eq("signal")]
    selected_null = node_rows.loc[node_rows["data_role"].eq("selected_null")]
    finite_signal_topology = int(signal["finite_topology_feature_count"].gt(0).sum())
    finite_selected_null_topology = int(selected_null["finite_topology_feature_count"].gt(0).sum())
    finite_signal_structural = int(signal["structural_topology_feature_count"].gt(0).sum())
    finite_selected_null_structural = int(
        selected_null["structural_topology_feature_count"].gt(0).sum()
    )
    finite_signal_completed = int(signal["completed_topology_feature_count"].gt(0).sum())
    finite_selected_null_completed = int(
        selected_null["completed_topology_feature_count"].gt(0).sum()
    )
    finite_signal_context = int(
        pd.to_numeric(
            signal["distance_to_downstream_accepted_split"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
    )
    finite_selected_null_context = int(
        pd.to_numeric(
            selected_null["distance_to_downstream_accepted_split"],
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .sum()
    )
    signal_candidate_count = int(signal.shape[0])
    selected_null_control_count = int(selected_null.shape[0])
    truth_labeled_signal_count = int(
        signal["truth_geometry_role"]
        .astype(str)
        .isin(
            {
                "truth_recovery_pass_through_positive",
                "partial_truth_recovery_pass_through_candidate",
                "barycentric_mixture_pass_through_candidate",
                "fragment_false_pass_through",
                "signal_truth_unresolved",
            }
        )
        .sum()
    )
    truth_recovery_count = int(
        node_rows["fixture_role"].eq("truth_recovery_pass_through_positive").sum()
    )
    partial_truth_recovery_count = int(
        node_rows["fixture_role"].eq("partial_truth_recovery_pass_through_candidate").sum()
    )
    barycentric_mixture_count = int(
        node_rows["fixture_role"].eq("signal_barycentric_mixture_pass_through_candidate").sum()
    )
    fragment_false_split_count = int(
        node_rows["fixture_role"].eq("fragment_false_pass_through").sum()
    )
    if signal_candidate_count < int(min_signal_candidate_count):
        status = "signal_selected_pass_through_support_insufficient"
        next_step = "generate_more_signal_retained_pass_through_overlap_cases"
    elif selected_null_control_count < int(min_selected_null_control_count):
        status = "selected_null_control_support_insufficient"
        next_step = "generate_more_selected_null_pass_through_controls"
    elif finite_signal_topology < int(min_finite_topology_per_side):
        status = "signal_topology_support_missing"
        next_step = "compute_topology_features_for_signal_retained_pass_through_rows"
    elif finite_selected_null_topology < int(min_finite_topology_per_side):
        status = "selected_null_topology_support_missing"
        next_step = "compute_topology_features_for_selected_null_pass_through_controls"
    else:
        status = "selected_pass_through_fixture_support_observed_diagnostic_only"
        next_step = "fit_or_validate_conditional_topology_likelihood_diagnostic"

    if signal_candidate_count < int(min_signal_candidate_count):
        completed_status = "signal_selected_pass_through_support_insufficient"
    elif selected_null_control_count < int(min_selected_null_control_count):
        completed_status = "selected_null_control_support_insufficient"
    elif finite_signal_completed < int(min_finite_topology_per_side):
        completed_status = "signal_completed_topology_support_missing"
    elif finite_selected_null_completed < int(min_finite_topology_per_side):
        completed_status = "selected_null_completed_topology_support_missing"
    else:
        completed_status = (
            "selected_pass_through_completed_topology_support_observed_diagnostic_only"
        )
    direct_support_missing = status in {
        "signal_topology_support_missing",
        "selected_null_topology_support_missing",
    }
    completed_support_observed = (
        completed_status
        == "selected_pass_through_completed_topology_support_observed_diagnostic_only"
    )
    if direct_support_missing and completed_support_observed:
        next_step = "validate_structural_topology_fallback_against_selected_null_controls"
    validation = _completed_balance_product_validation(signal, selected_null)
    if (
        validation["structural_fallback_validation_status"]
        == "structural_fallback_separates_selected_event_diagnostic_only"
    ):
        next_step = "expand_selected_pass_through_fixture_support_and_truth_labels"
    if (
        truth_labeled_signal_count > 0
        and truth_recovery_count == 0
        and partial_truth_recovery_count == 0
        and barycentric_mixture_count == 0
    ):
        next_step = "generate_or_find_truth_recovery_selected_pass_through_cases"
    if partial_truth_recovery_count > 0 and truth_recovery_count == 0:
        next_step = "generate_full_recovery_selected_pass_through_cases"
    if barycentric_mixture_count > 0 and truth_recovery_count == 0:
        next_step = "derive_barycentric_mixture_vs_branch_recovery_conditioning"
    if truth_recovery_count > 0 and fragment_false_split_count > 0:
        next_step = "derive_conditional_law_for_recovery_vs_fragment_pass_through"

    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "selected_event": SELECTED_EVENT,
                "selected_event_row_count": int(node_rows.shape[0]),
                "signal_candidate_count": signal_candidate_count,
                "selected_null_control_count": selected_null_control_count,
                "truth_labeled_signal_count": truth_labeled_signal_count,
                "truth_recovery_count": int(truth_recovery_count),
                "partial_truth_recovery_count": partial_truth_recovery_count,
                "barycentric_mixture_count": barycentric_mixture_count,
                "fragment_false_split_count": fragment_false_split_count,
                "finite_signal_topology_count": finite_signal_topology,
                "finite_selected_null_topology_count": finite_selected_null_topology,
                "finite_signal_structural_topology_count": finite_signal_structural,
                "finite_selected_null_structural_topology_count": (finite_selected_null_structural),
                "finite_signal_completed_topology_count": finite_signal_completed,
                "finite_selected_null_completed_topology_count": (finite_selected_null_completed),
                "finite_signal_traversal_context_count": finite_signal_context,
                "finite_selected_null_traversal_context_count": (finite_selected_null_context),
                "case_count": int(case_rows.shape[0]),
                "signal_case_count": int(case_rows["data_role"].eq("signal").sum()),
                "selected_null_case_count": int(case_rows["data_role"].eq("selected_null").sum()),
                "fixture_support_status": status,
                "completed_fixture_support_status": completed_status,
                **validation,
                "next_required_step": next_step,
                "production_action": "fail_closed_until_fixture_support_observed",
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_selected_pass_through_fixture_miner(
    config: OverlapSelectedPassThroughFixtureMinerConfig,
) -> dict[str, Path]:
    """Run selected pass-through overlap fixture mining."""
    candidate_rows = pd.read_csv(config.candidate_rows_path, keep_default_na=False)
    selected_neighborhood_rows = None
    if config.selected_neighborhood_rows_path is not None:
        selected_neighborhood_rows = pd.read_csv(
            config.selected_neighborhood_rows_path,
            keep_default_na=False,
        )
    truth_context_rows = None
    if config.traversal_rows_path is not None or config.gene_assignments_path is not None:
        if selected_neighborhood_rows is None:
            raise ValueError(
                "selected_neighborhood_rows_path is required when truth context "
                "inputs are provided."
            )
        if config.traversal_rows_path is None or config.gene_assignments_path is None:
            raise ValueError(
                "Both traversal_rows_path and gene_assignments_path are required "
                "to compute truth context."
            )
        traversal_rows = pd.read_csv(config.traversal_rows_path, keep_default_na=False)
        gene_assignments = pd.read_csv(
            config.gene_assignments_path,
            keep_default_na=False,
        )
        truth_context_rows = build_selected_pass_through_truth_context_rows(
            selected_neighborhood_rows,
            gene_assignments,
            traversal_rows,
            suite=str(config.truth_suite),
            min_recovery_ari=float(config.truth_recovery_ari_floor),
            min_partial_recovery_ari=float(config.truth_partial_recovery_ari_floor),
            min_child_purity=float(config.truth_child_purity_floor),
        )
    node_rows = build_overlap_selected_pass_through_node_rows(
        candidate_rows,
        selected_neighborhood_rows,
        truth_context_rows,
    )
    case_rows = build_overlap_selected_pass_through_case_rows(node_rows)
    support_summary = summarize_overlap_selected_pass_through_support(
        node_rows,
        case_rows,
        min_signal_candidate_count=config.min_signal_candidate_count,
        min_selected_null_control_count=config.min_selected_null_control_count,
        min_finite_topology_per_side=config.min_finite_topology_per_side,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    node_rows.to_csv(config.node_rows_path, index=False)
    case_rows.to_csv(config.case_rows_path, index=False)
    support_summary.to_csv(config.support_summary_path, index=False)
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
            "traversal_rows": (
                None if config.traversal_rows_path is None else str(config.traversal_rows_path)
            ),
            "gene_assignments": (
                None if config.gene_assignments_path is None else str(config.gene_assignments_path)
            ),
        },
        "truth_context": {
            "truth_suite": str(config.truth_suite),
            "truth_recovery_ari_floor": float(config.truth_recovery_ari_floor),
            "truth_partial_recovery_ari_floor": float(config.truth_partial_recovery_ari_floor),
            "truth_child_purity_floor": float(config.truth_child_purity_floor),
            "truth_context_rows": (
                0 if truth_context_rows is None else int(truth_context_rows.shape[0])
            ),
        },
        "support_thresholds": {
            "min_signal_candidate_count": int(config.min_signal_candidate_count),
            "min_selected_null_control_count": int(config.min_selected_null_control_count),
            "min_finite_topology_per_side": int(config.min_finite_topology_per_side),
        },
        "outputs": {
            "node_rows": str(config.node_rows_path),
            "case_rows": str(config.case_rows_path),
            "support_summary": str(config.support_summary_path),
        },
        "production_status": str(support_summary["production_action"].iloc[0]),
        "interpretation": (
            "Diagnostic-only selected pass-through fixture miner; production "
            "must remain fail-closed until both signal and selected-null "
            "topology support are observed in the selected event stratum."
        ),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "node_rows": config.node_rows_path,
        "case_rows": config.case_rows_path,
        "support_summary": config.support_summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows-path", type=Path, required=True)
    parser.add_argument("--selected-neighborhood-rows-path", type=Path, default=None)
    parser.add_argument("--traversal-rows-path", type=Path, default=None)
    parser.add_argument("--gene-assignments-path", type=Path, default=None)
    parser.add_argument("--truth-suite", default="binary")
    parser.add_argument("--truth-recovery-ari-floor", type=float, default=0.50)
    parser.add_argument(
        "--truth-partial-recovery-ari-floor",
        type=float,
        default=DEFAULT_PARTIAL_RECOVERY_ARI_FLOOR,
    )
    parser.add_argument("--truth-child-purity-floor", type=float, default=0.65)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-signal-candidate-count", type=int, default=2)
    parser.add_argument("--min-selected-null-control-count", type=int, default=2)
    parser.add_argument("--min-finite-topology-per-side", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_overlap_selected_pass_through_fixture_miner(
        OverlapSelectedPassThroughFixtureMinerConfig(
            candidate_rows_path=args.candidate_rows_path,
            selected_neighborhood_rows_path=args.selected_neighborhood_rows_path,
            traversal_rows_path=args.traversal_rows_path,
            gene_assignments_path=args.gene_assignments_path,
            truth_suite=str(args.truth_suite),
            truth_recovery_ari_floor=float(args.truth_recovery_ari_floor),
            truth_partial_recovery_ari_floor=float(args.truth_partial_recovery_ari_floor),
            truth_child_purity_floor=float(args.truth_child_purity_floor),
            output_dir=args.output_dir,
            min_signal_candidate_count=int(args.min_signal_candidate_count),
            min_selected_null_control_count=int(args.min_selected_null_control_count),
            min_finite_topology_per_side=int(args.min_finite_topology_per_side),
        )
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
