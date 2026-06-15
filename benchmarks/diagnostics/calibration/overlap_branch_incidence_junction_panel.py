"""Branch-incidence diagnostics for overlap traversal junctions.

This panel computes the actual directional incidence geometry for each binary
internal node:

* incoming edge: node mean minus parent mean;
* incoming selected-family contrast: node mean minus its sibling under parent;
* outgoing contrast: left child mean minus right child mean.

It is diagnostic-only. The purpose is to decide whether a context-negative
truth-recovery row is a real weak-income to coherent-outcome transition or an
artifact of using parent-level context as a proxy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_panel import (
    DEFAULT_DATA_ROLES,
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.diagnostics.calibration.overlap_structural_sibling_panel import (
    DEFAULT_OVERLAP_CASES,
    DEFAULT_PROFILE,
    _as_float_matrix,
    _descendant_labels,
    _output_data_role,
    _purity,
    _raw_node_by_id,
    _truth_for_labels,
    cosine_similarity,
    jaccard_index,
    top_coordinate_set,
)
from benchmarks.diagnostics.calibration.selected_family_traversal_panel import (
    _build_node_decisions,
)
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

STUDY_ROLE = "diagnostic_overlap_branch_incidence_junction_panel"
SCHEMA_VERSION = "overlap_branch_incidence_junction_panel/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration.overlap_branch_incidence_junction_panel"
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "data_seed",
    "profile_id",
    "node_id",
    "incoming_parent_id",
    "incoming_sibling_id",
    "outgoing_left_child_id",
    "outgoing_right_child_id",
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
    "n_features",
    "top_k",
    "incoming_branch_balance",
    "outgoing_balance",
    "incoming_edge_p_value",
    "incoming_edge_bh_p_value",
    "incoming_edge_neglog10_bh_p_value",
    "incoming_edge_tested",
    "incoming_edge_rejected",
    "incoming_sibling_edge_p_value",
    "incoming_sibling_edge_bh_p_value",
    "incoming_sibling_edge_neglog10_bh_p_value",
    "incoming_sibling_edge_tested",
    "incoming_sibling_edge_rejected",
    "outgoing_left_edge_p_value",
    "outgoing_left_edge_bh_p_value",
    "outgoing_left_edge_neglog10_bh_p_value",
    "outgoing_left_edge_tested",
    "outgoing_left_edge_rejected",
    "outgoing_right_edge_p_value",
    "outgoing_right_edge_bh_p_value",
    "outgoing_right_edge_neglog10_bh_p_value",
    "outgoing_right_edge_tested",
    "outgoing_right_edge_rejected",
    "min_outgoing_edge_bh_p_value",
    "max_outgoing_edge_bh_p_value",
    "min_outgoing_edge_neglog10_bh_p_value",
    "max_outgoing_edge_neglog10_bh_p_value",
    "outgoing_edge_neglog10_balance",
    "outgoing_edges_both_rejected",
    "incoming_edges_both_rejected",
    "incoming_edge_norm",
    "incoming_family_contrast_norm",
    "outgoing_sibling_contrast_norm",
    "incoming_edge_outgoing_cosine",
    "incoming_edge_outgoing_abs_cosine",
    "incoming_family_outgoing_cosine",
    "incoming_family_outgoing_abs_cosine",
    "incoming_edge_outgoing_jaccard_topk",
    "incoming_family_outgoing_jaccard_topk",
    "incoming_edge_family_jaccard_topk",
    "node_outgoing_edge_jaccard_topk",
    "incoming_edge_outgoing_centered_abs_cosine",
    "incoming_family_outgoing_centered_abs_cosine",
    "incoming_edge_outgoing_fisher_abs_cosine",
    "incoming_family_outgoing_fisher_abs_cosine",
    "incoming_edge_outgoing_fisher_jaccard_topk",
    "incoming_family_outgoing_fisher_jaccard_topk",
    "metric_family_alignment_score",
    "metric_family_compatibility_status",
    "incoming_truth_split_ari",
    "outgoing_truth_split_ari",
    "node_truth_purity",
    "incoming_sibling_truth_purity",
    "left_truth_purity",
    "right_truth_purity",
    "branch_incidence_geometry_status",
    "guard_truth_role",
    "conditional_bayesian_gap_status",
    "context_margin_pass",
    "soft_structure_pass",
    "default_internal_node_candidate",
    "branch_incidence_transition_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "annotated_gap_row_count",
    "truth_recovery_total",
    "truth_recovery_transition_candidate_count",
    "truth_recovery_mismatch_count",
    "negative_total",
    "negative_transition_candidate_count",
    "negative_mismatch_count",
    "median_truth_incoming_family_outgoing_jaccard",
    "median_negative_incoming_family_outgoing_jaccard",
    "median_truth_metric_family_alignment_score",
    "median_negative_metric_family_alignment_score",
    "diagnostic_status",
)

GAP_COLUMNS = (
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "guard_truth_role",
    "conditional_bayesian_gap_status",
    "context_margin_pass",
    "soft_structure_pass",
    "default_internal_node_candidate",
)


@dataclass(frozen=True)
class OverlapBranchIncidenceJunctionPanelConfig:
    """Runtime contract for branch-incidence junction diagnostics."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    data_roles: tuple[str, ...]
    sibling_alpha: float
    edge_alpha: float
    replicates: int
    base_seed: int
    profile_id: str = DEFAULT_PROFILE
    top_k: int = 12
    transfer_gap_rows_path: Path | None = None
    skip_unsupported_cases: bool = True

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_branch_incidence_junction_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_branch_incidence_junction_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def classify_branch_incidence_geometry(
    *,
    incoming_family_outgoing_jaccard_topk: float,
    incoming_family_outgoing_abs_cosine: float,
    incoming_edge_outgoing_jaccard_topk: float,
    incoming_edge_outgoing_abs_cosine: float,
    outgoing_sibling_contrast_norm: float,
    aligned_jaccard_floor: float = 0.25,
    aligned_cosine_floor: float = 0.50,
    weak_jaccard_floor: float = 0.10,
    weak_cosine_floor: float = 0.25,
) -> str:
    """Classify income/outcome compatibility from actual branch vectors."""
    if (
        not math.isfinite(outgoing_sibling_contrast_norm)
        or outgoing_sibling_contrast_norm <= 0.0
    ):
        return "degenerate_outgoing_contrast"
    family_aligned = (
        incoming_family_outgoing_jaccard_topk >= float(aligned_jaccard_floor)
        or incoming_family_outgoing_abs_cosine >= float(aligned_cosine_floor)
    )
    edge_aligned = (
        incoming_edge_outgoing_jaccard_topk >= float(aligned_jaccard_floor)
        or incoming_edge_outgoing_abs_cosine >= float(aligned_cosine_floor)
    )
    if family_aligned and edge_aligned:
        return "income_outcome_branch_aligned"
    if family_aligned:
        return "selected_family_outcome_aligned"
    if edge_aligned:
        return "incoming_edge_outcome_aligned"
    family_weak = (
        incoming_family_outgoing_jaccard_topk >= float(weak_jaccard_floor)
        or incoming_family_outgoing_abs_cosine >= float(weak_cosine_floor)
    )
    edge_weak = (
        incoming_edge_outgoing_jaccard_topk >= float(weak_jaccard_floor)
        or incoming_edge_outgoing_abs_cosine >= float(weak_cosine_floor)
    )
    if family_weak or edge_weak:
        return "weak_income_outcome_branch_alignment"
    return "coordinate_income_outcome_branch_mismatch"


def centered_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine after subtracting each vector's coordinate mean."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    return cosine_similarity(x - float(np.mean(x)), y - float(np.mean(y)))


def fisher_weighted_vector(
    vector: np.ndarray,
    context_mean: np.ndarray,
    *,
    variance_floor: float = 1e-4,
) -> np.ndarray:
    """Return a diagonal Fisher-scaled vector for Bernoulli coordinates."""
    values = np.asarray(vector, dtype=float)
    context = np.asarray(context_mean, dtype=float)
    variance = np.clip(context * (1.0 - context), float(variance_floor), None)
    return values / np.sqrt(variance)


def classify_metric_family_compatibility(
    *,
    metric_family_alignment_score: float,
    strong_floor: float = 0.50,
    weak_floor: float = 0.25,
) -> str:
    """Classify compatibility across the richer metric family."""
    if metric_family_alignment_score >= float(strong_floor):
        return "metric_family_branch_compatible"
    if metric_family_alignment_score >= float(weak_floor):
        return "metric_family_branch_weakly_compatible"
    return "metric_family_branch_mismatch"


def _transition_status(
    *,
    geometry_status: str,
    guard_truth_role: str,
    conditional_bayesian_gap_status: str,
    soft_structure_pass: bool,
    context_margin_pass: bool,
    default_candidate: bool,
) -> str:
    aligned = geometry_status in {
        "income_outcome_branch_aligned",
        "selected_family_outcome_aligned",
        "incoming_edge_outcome_aligned",
    }
    weak_aligned = geometry_status == "weak_income_outcome_branch_alignment"
    truth = guard_truth_role == "truth_recovery"
    if guard_truth_role == "unannotated":
        return "unannotated_branch_incidence"
    if truth:
        if default_candidate and aligned:
            return "truth_recovery_branch_incidence_supported"
        if (
            conditional_bayesian_gap_status
            == "context_negative_but_soft_structure_supported"
            and soft_structure_pass
            and not context_margin_pass
            and aligned
        ):
            return "truth_recovery_valid_branch_transition_candidate"
        if (
            conditional_bayesian_gap_status
            == "context_negative_but_soft_structure_supported"
            and soft_structure_pass
            and not context_margin_pass
            and weak_aligned
        ):
            return "truth_recovery_weak_branch_transition_candidate"
        if geometry_status == "coordinate_income_outcome_branch_mismatch":
            return "truth_recovery_coordinate_branch_incidence_mismatch"
        return "truth_recovery_branch_incidence_unresolved"
    if default_candidate:
        return "negative_branch_incidence_leakage"
    if aligned or weak_aligned:
        return "negative_context_blocked_despite_branch_alignment"
    return "negative_coordinate_branch_incidence_mismatch_or_blocked"


def _safe_mean(data: pd.DataFrame, labels: Sequence[str]) -> np.ndarray:
    if not labels:
        raise ValueError("Cannot compute mean for an empty label set.")
    return _as_float_matrix(data.loc[list(labels)]).mean(axis=0)


def _annotation_value(annotations: pd.DataFrame, node: object, column: str) -> object:
    if annotations is None or column not in annotations.columns:
        return np.nan
    if node in annotations.index:
        return annotations.at[node, column]
    string_node = str(node)
    if string_node in annotations.index:
        return annotations.at[string_node, column]
    return np.nan


def _finite_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _neglog10_p_value(value: object) -> float:
    p_value = _finite_float(value)
    if not math.isfinite(p_value):
        return float("nan")
    return float(-math.log10(min(max(p_value, 1e-300), 1.0)))


def _edge_record(annotations: pd.DataFrame, node: object, prefix: str) -> dict[str, object]:
    p_value = _annotation_value(annotations, node, "Child_Parent_Divergence_P_Value")
    bh_p_value = _annotation_value(
        annotations,
        node,
        "Child_Parent_Divergence_P_Value_BH",
    )
    tested = _annotation_value(annotations, node, "Child_Parent_Divergence_Tested")
    rejected = _annotation_value(
        annotations,
        node,
        "Child_Parent_Divergence_Significant",
    )
    return {
        f"{prefix}_edge_p_value": _finite_float(p_value),
        f"{prefix}_edge_bh_p_value": _finite_float(bh_p_value),
        f"{prefix}_edge_neglog10_bh_p_value": _neglog10_p_value(bh_p_value),
        f"{prefix}_edge_tested": _bool_value(tested),
        f"{prefix}_edge_rejected": _bool_value(rejected),
    }


def _binary_children(tree, node: object) -> list[object] | None:
    children = list(tree.successors(node))
    if len(children) != 2:
        return None
    return children


def _single_parent(tree, node: object) -> object | None:
    parents = list(tree.predecessors(node))
    if not parents:
        return None
    if len(parents) != 1:
        raise ValueError(f"Expected at most one parent for node {node!r}.")
    return parents[0]


def _incoming_sibling(tree, *, parent: object, node: object) -> object | None:
    siblings = [child for child in tree.successors(parent) if child != node]
    if len(siblings) != 1:
        return None
    return siblings[0]


def _branch_row_for_node(
    *,
    case_id: str,
    data_role: str,
    replicate: int,
    data_seed: int,
    profile_id: str,
    node_row: pd.Series,
    result,
    data: pd.DataFrame,
    truth_by_label: dict[str, int],
    top_k: int,
) -> dict[str, object] | None:
    tree = result.extra["tree"]
    raw_nodes = _raw_node_by_id(tree)
    raw_node = raw_nodes.get(str(node_row["node_id"]))
    if raw_node is None:
        return None
    annotations = result.extra.get("annotations", tree.annotations_df)
    outgoing_children = _binary_children(tree, raw_node)
    if outgoing_children is None:
        return None
    parent = _single_parent(tree, raw_node)
    if parent is None:
        return None
    incoming_sibling = _incoming_sibling(tree, parent=parent, node=raw_node)
    if incoming_sibling is None:
        return None

    left_child, right_child = outgoing_children
    node_labels = _descendant_labels(tree, raw_node)
    parent_labels = _descendant_labels(tree, parent)
    incoming_sibling_labels = _descendant_labels(tree, incoming_sibling)
    left_labels = _descendant_labels(tree, left_child)
    right_labels = _descendant_labels(tree, right_child)
    if not all([node_labels, parent_labels, incoming_sibling_labels, left_labels, right_labels]):
        return None

    theta_node = _safe_mean(data, node_labels)
    theta_parent = _safe_mean(data, parent_labels)
    theta_incoming_sibling = _safe_mean(data, incoming_sibling_labels)
    theta_left = _safe_mean(data, left_labels)
    theta_right = _safe_mean(data, right_labels)

    incoming_edge_record = _edge_record(annotations, raw_node, "incoming")
    incoming_sibling_edge_record = _edge_record(
        annotations,
        incoming_sibling,
        "incoming_sibling",
    )
    outgoing_left_edge_record = _edge_record(annotations, left_child, "outgoing_left")
    outgoing_right_edge_record = _edge_record(annotations, right_child, "outgoing_right")
    outgoing_edge_bh_values = [
        outgoing_left_edge_record["outgoing_left_edge_bh_p_value"],
        outgoing_right_edge_record["outgoing_right_edge_bh_p_value"],
    ]
    outgoing_edge_actions = [
        outgoing_left_edge_record["outgoing_left_edge_neglog10_bh_p_value"],
        outgoing_right_edge_record["outgoing_right_edge_neglog10_bh_p_value"],
    ]
    finite_outgoing_bh = [
        value for value in outgoing_edge_bh_values if math.isfinite(float(value))
    ]
    finite_outgoing_actions = [
        value for value in outgoing_edge_actions if math.isfinite(float(value))
    ]
    min_outgoing_edge_bh = (
        float(min(finite_outgoing_bh)) if finite_outgoing_bh else float("nan")
    )
    max_outgoing_edge_bh = (
        float(max(finite_outgoing_bh)) if finite_outgoing_bh else float("nan")
    )
    min_outgoing_edge_action = (
        float(min(finite_outgoing_actions))
        if finite_outgoing_actions
        else float("nan")
    )
    max_outgoing_edge_action = (
        float(max(finite_outgoing_actions))
        if finite_outgoing_actions
        else float("nan")
    )
    outgoing_edge_neglog10_balance = (
        float(min_outgoing_edge_action / max_outgoing_edge_action)
        if math.isfinite(min_outgoing_edge_action)
        and math.isfinite(max_outgoing_edge_action)
        and max_outgoing_edge_action > 0.0
        else float("nan")
    )

    incoming_edge = theta_node - theta_parent
    incoming_family_contrast = theta_node - theta_incoming_sibling
    outgoing_contrast = theta_left - theta_right
    left_edge = theta_left - theta_node
    right_edge = theta_right - theta_node

    incoming_edge_top = top_coordinate_set(incoming_edge, top_k=top_k)
    incoming_family_top = top_coordinate_set(incoming_family_contrast, top_k=top_k)
    outgoing_top = top_coordinate_set(outgoing_contrast, top_k=top_k)
    left_edge_top = top_coordinate_set(left_edge, top_k=top_k)
    right_edge_top = top_coordinate_set(right_edge, top_k=top_k)

    incoming_edge_outgoing_cosine = cosine_similarity(incoming_edge, outgoing_contrast)
    incoming_family_outgoing_cosine = cosine_similarity(
        incoming_family_contrast,
        outgoing_contrast,
    )
    incoming_edge_outgoing_jaccard = jaccard_index(incoming_edge_top, outgoing_top)
    incoming_family_outgoing_jaccard = jaccard_index(
        incoming_family_top,
        outgoing_top,
    )
    incoming_edge_family_jaccard = jaccard_index(
        incoming_edge_top,
        incoming_family_top,
    )
    node_outgoing_edge_jaccard = max(
        jaccard_index(outgoing_top, left_edge_top),
        jaccard_index(outgoing_top, right_edge_top),
    )
    incoming_edge_outgoing_centered_abs_cosine = abs(
        centered_cosine_similarity(incoming_edge, outgoing_contrast)
    )
    incoming_family_outgoing_centered_abs_cosine = abs(
        centered_cosine_similarity(incoming_family_contrast, outgoing_contrast)
    )
    fisher_incoming_edge = fisher_weighted_vector(incoming_edge, theta_node)
    fisher_incoming_family = fisher_weighted_vector(
        incoming_family_contrast,
        theta_node,
    )
    fisher_outgoing = fisher_weighted_vector(outgoing_contrast, theta_node)
    incoming_edge_outgoing_fisher_abs_cosine = abs(
        cosine_similarity(fisher_incoming_edge, fisher_outgoing)
    )
    incoming_family_outgoing_fisher_abs_cosine = abs(
        cosine_similarity(fisher_incoming_family, fisher_outgoing)
    )
    fisher_incoming_edge_top = top_coordinate_set(fisher_incoming_edge, top_k=top_k)
    fisher_incoming_family_top = top_coordinate_set(
        fisher_incoming_family,
        top_k=top_k,
    )
    fisher_outgoing_top = top_coordinate_set(fisher_outgoing, top_k=top_k)
    incoming_edge_outgoing_fisher_jaccard = jaccard_index(
        fisher_incoming_edge_top,
        fisher_outgoing_top,
    )
    incoming_family_outgoing_fisher_jaccard = jaccard_index(
        fisher_incoming_family_top,
        fisher_outgoing_top,
    )
    metric_family_alignment_score = max(
        abs(incoming_family_outgoing_cosine),
        abs(incoming_edge_outgoing_cosine),
        incoming_family_outgoing_centered_abs_cosine,
        incoming_edge_outgoing_centered_abs_cosine,
        incoming_family_outgoing_fisher_abs_cosine,
        incoming_edge_outgoing_fisher_abs_cosine,
        incoming_family_outgoing_jaccard,
        incoming_edge_outgoing_jaccard,
        incoming_family_outgoing_fisher_jaccard,
        incoming_edge_outgoing_fisher_jaccard,
    )
    outgoing_norm = float(np.linalg.norm(outgoing_contrast))
    geometry_status = classify_branch_incidence_geometry(
        incoming_family_outgoing_jaccard_topk=float(incoming_family_outgoing_jaccard),
        incoming_family_outgoing_abs_cosine=abs(incoming_family_outgoing_cosine),
        incoming_edge_outgoing_jaccard_topk=float(incoming_edge_outgoing_jaccard),
        incoming_edge_outgoing_abs_cosine=abs(incoming_edge_outgoing_cosine),
        outgoing_sibling_contrast_norm=outgoing_norm,
    )

    node_truth = _truth_for_labels(truth_by_label=truth_by_label, labels=node_labels)
    incoming_truth = _truth_for_labels(
        truth_by_label=truth_by_label,
        labels=node_labels + incoming_sibling_labels,
    )
    incoming_membership = np.concatenate(
        [
            np.ones(len(node_labels), dtype=int),
            np.zeros(len(incoming_sibling_labels), dtype=int),
        ]
    )
    outgoing_membership = np.concatenate(
        [
            np.zeros(len(left_labels), dtype=int),
            np.ones(len(right_labels), dtype=int),
        ]
    )
    incoming_truth_split_ari = (
        float(adjusted_rand_score(incoming_truth, incoming_membership))
        if np.unique(incoming_truth).size > 1
        else np.nan
    )
    outgoing_truth = _truth_for_labels(
        truth_by_label=truth_by_label,
        labels=left_labels + right_labels,
    )
    outgoing_truth_split_ari = (
        float(adjusted_rand_score(outgoing_truth, outgoing_membership))
        if np.unique(outgoing_truth).size > 1
        else np.nan
    )

    n_node = len(node_labels)
    n_incoming_sibling = len(incoming_sibling_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": _output_data_role(data_role),
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "profile_id": str(profile_id),
        "node_id": str(node_row["node_id"]),
        "incoming_parent_id": str(parent),
        "incoming_sibling_id": str(incoming_sibling),
        "outgoing_left_child_id": str(left_child),
        "outgoing_right_child_id": str(right_child),
        "depth": int(node_row["depth"]),
        "decision_class": str(node_row["decision_class"]),
        "traversal_decision": str(node_row["traversal_decision"]),
        "sibling_open": bool(node_row["sibling_open"]),
        "sibling_p_value": float(node_row["sibling_p_value"]),
        "selected_family_guard_blocked": bool(node_row["selected_family_guard_blocked"]),
        "selected_family_p_value": float(node_row["selected_family_p_value"]),
        "n_parent_context": int(len(parent_labels)),
        "n_node": int(n_node),
        "n_incoming_sibling": int(n_incoming_sibling),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "n_features": int(data.shape[1]),
        "top_k": int(min(top_k, data.shape[1])),
        "incoming_branch_balance": float(
            min(n_node, n_incoming_sibling) / (n_node + n_incoming_sibling)
        ),
        "outgoing_balance": float(min(n_left, n_right) / (n_left + n_right)),
        **incoming_edge_record,
        **incoming_sibling_edge_record,
        **outgoing_left_edge_record,
        **outgoing_right_edge_record,
        "min_outgoing_edge_bh_p_value": min_outgoing_edge_bh,
        "max_outgoing_edge_bh_p_value": max_outgoing_edge_bh,
        "min_outgoing_edge_neglog10_bh_p_value": min_outgoing_edge_action,
        "max_outgoing_edge_neglog10_bh_p_value": max_outgoing_edge_action,
        "outgoing_edge_neglog10_balance": outgoing_edge_neglog10_balance,
        "outgoing_edges_both_rejected": bool(
            outgoing_left_edge_record["outgoing_left_edge_rejected"]
            and outgoing_right_edge_record["outgoing_right_edge_rejected"]
        ),
        "incoming_edges_both_rejected": bool(
            incoming_edge_record["incoming_edge_rejected"]
            and incoming_sibling_edge_record["incoming_sibling_edge_rejected"]
        ),
        "incoming_edge_norm": float(np.linalg.norm(incoming_edge)),
        "incoming_family_contrast_norm": float(
            np.linalg.norm(incoming_family_contrast)
        ),
        "outgoing_sibling_contrast_norm": outgoing_norm,
        "incoming_edge_outgoing_cosine": float(incoming_edge_outgoing_cosine),
        "incoming_edge_outgoing_abs_cosine": float(abs(incoming_edge_outgoing_cosine)),
        "incoming_family_outgoing_cosine": float(incoming_family_outgoing_cosine),
        "incoming_family_outgoing_abs_cosine": float(
            abs(incoming_family_outgoing_cosine)
        ),
        "incoming_edge_outgoing_jaccard_topk": float(incoming_edge_outgoing_jaccard),
        "incoming_family_outgoing_jaccard_topk": float(
            incoming_family_outgoing_jaccard
        ),
        "incoming_edge_family_jaccard_topk": float(incoming_edge_family_jaccard),
        "node_outgoing_edge_jaccard_topk": float(node_outgoing_edge_jaccard),
        "incoming_edge_outgoing_centered_abs_cosine": float(
            incoming_edge_outgoing_centered_abs_cosine
        ),
        "incoming_family_outgoing_centered_abs_cosine": float(
            incoming_family_outgoing_centered_abs_cosine
        ),
        "incoming_edge_outgoing_fisher_abs_cosine": float(
            incoming_edge_outgoing_fisher_abs_cosine
        ),
        "incoming_family_outgoing_fisher_abs_cosine": float(
            incoming_family_outgoing_fisher_abs_cosine
        ),
        "incoming_edge_outgoing_fisher_jaccard_topk": float(
            incoming_edge_outgoing_fisher_jaccard
        ),
        "incoming_family_outgoing_fisher_jaccard_topk": float(
            incoming_family_outgoing_fisher_jaccard
        ),
        "metric_family_alignment_score": float(metric_family_alignment_score),
        "metric_family_compatibility_status": classify_metric_family_compatibility(
            metric_family_alignment_score=float(metric_family_alignment_score)
        ),
        "incoming_truth_split_ari": incoming_truth_split_ari,
        "outgoing_truth_split_ari": outgoing_truth_split_ari,
        "node_truth_purity": _purity(node_truth),
        "incoming_sibling_truth_purity": _purity(
            _truth_for_labels(
                truth_by_label=truth_by_label,
                labels=incoming_sibling_labels,
            )
        ),
        "left_truth_purity": _purity(
            _truth_for_labels(truth_by_label=truth_by_label, labels=left_labels)
        ),
        "right_truth_purity": _purity(
            _truth_for_labels(truth_by_label=truth_by_label, labels=right_labels)
        ),
        "branch_incidence_geometry_status": geometry_status,
        "guard_truth_role": "unannotated",
        "conditional_bayesian_gap_status": "unannotated",
        "context_margin_pass": False,
        "soft_structure_pass": False,
        "default_internal_node_candidate": False,
        "branch_incidence_transition_status": "unannotated_branch_incidence",
    }


def _relevant_node_rows(node_decisions: pd.DataFrame) -> pd.DataFrame:
    if node_decisions.empty:
        return node_decisions
    decision_mask = node_decisions["decision_class"].isin(
        {
            "accepted_internal_split",
            "selected_root_blocked",
            "selected_family_blocked",
            "unstable_passthrough_zone",
        }
    )
    evidence_mask = (
        node_decisions["sibling_open"].astype(bool)
        | pd.to_numeric(node_decisions["sibling_p_value"], errors="coerce").notna()
    )
    return node_decisions[
        node_decisions["visited"].astype(bool)
        & (node_decisions["n_children"].astype(int) == 2)
        & (pd.to_numeric(node_decisions["depth"], errors="coerce") > 0)
        & (decision_mask | evidence_mask)
    ].copy()


def _annotate_with_gap_rows(
    branch_rows: pd.DataFrame,
    gap_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    if gap_rows is None or gap_rows.empty or branch_rows.empty:
        return branch_rows
    missing = sorted(set(GAP_COLUMNS) - set(gap_rows.columns))
    if missing:
        raise ValueError(f"Transfer-gap rows are missing columns: {missing!r}")
    keys = ["case_id", "data_role", "replicate", "node_id"]
    gap = gap_rows[list(GAP_COLUMNS)].copy()
    annotated = branch_rows.drop(columns=list(GAP_COLUMNS[4:]), errors="ignore").merge(
        gap,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    for column in GAP_COLUMNS[4:]:
        if column in {"context_margin_pass", "soft_structure_pass", "default_internal_node_candidate"}:
            annotated[column] = annotated[column].apply(
                lambda value: bool(value) if pd.notna(value) else False
            )
        else:
            annotated[column] = annotated[column].fillna("unannotated").astype(str)
    annotated["branch_incidence_transition_status"] = [
        _transition_status(
            geometry_status=str(row["branch_incidence_geometry_status"]),
            guard_truth_role=str(row["guard_truth_role"]),
            conditional_bayesian_gap_status=str(row["conditional_bayesian_gap_status"]),
            soft_structure_pass=bool(row["soft_structure_pass"]),
            context_margin_pass=bool(row["context_margin_pass"]),
            default_candidate=bool(row["default_internal_node_candidate"]),
        )
        for _, row in annotated.iterrows()
    ]
    return annotated[list(ROW_COLUMNS)]


def _run_one(
    *,
    case: dict[str, object],
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    data_role: str,
    replicate: int,
    data_seed: int,
    config: OverlapBranchIncidenceJunctionPanelConfig,
) -> pd.DataFrame:
    data, feature_space, truth_labels, _true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role=data_role,
        seed=data_seed,
    )
    distance = pdist(data.to_numpy(dtype=float), metric="hamming")
    result = _run_kl_method(
        data,
        distance,
        sibling_significance_level=float(config.sibling_alpha),
        tree_linkage_method="average",
        edge_alpha=float(config.edge_alpha),
        feature_space=feature_space,
        sibling_gate_profile=str(config.profile_id),
    )
    node_decisions = _build_node_decisions(
        case_id=case_id,
        data_role=data_role,
        method_id=str(config.profile_id),
        replicate=replicate,
        data_seed=data_seed,
        result=result,
    )
    truth_by_label = {
        str(label): int(value)
        for label, value in zip(data.index.astype(str), np.asarray(truth_labels, dtype=int))
    }
    records: list[dict[str, object]] = []
    for _, node_row in _relevant_node_rows(node_decisions).iterrows():
        row = _branch_row_for_node(
            case_id=case_id,
            data_role=data_role,
            replicate=replicate,
            data_seed=data_seed,
            profile_id=str(config.profile_id),
            node_row=node_row,
            result=result,
            data=data,
            truth_by_label=truth_by_label,
            top_k=int(config.top_k),
        )
        if row is not None:
            records.append(row)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_branch_incidence_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize annotated branch-incidence transition evidence."""
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    annotated = rows["guard_truth_role"].astype(str).ne("unannotated")
    annotated_rows = rows[annotated].copy()
    truth = annotated_rows["guard_truth_role"].astype(str).eq("truth_recovery")
    negative = annotated_rows["guard_truth_role"].astype(str).ne("truth_recovery")
    transition_status = annotated_rows["branch_incidence_transition_status"].astype(str)
    truth_transition = transition_status.isin(
        {
            "truth_recovery_valid_branch_transition_candidate",
            "truth_recovery_weak_branch_transition_candidate",
            "truth_recovery_branch_incidence_supported",
        }
    )
    truth_mismatch = transition_status.eq(
        "truth_recovery_coordinate_branch_incidence_mismatch"
    )
    negative_transition = transition_status.eq(
        "negative_context_blocked_despite_branch_alignment"
    )
    negative_mismatch = transition_status.eq(
        "negative_coordinate_branch_incidence_mismatch_or_blocked"
    )
    truth_jaccard = pd.to_numeric(
        annotated_rows.loc[truth, "incoming_family_outgoing_jaccard_topk"],
        errors="coerce",
    )
    negative_jaccard = pd.to_numeric(
        annotated_rows.loc[negative, "incoming_family_outgoing_jaccard_topk"],
        errors="coerce",
    )
    truth_metric = pd.to_numeric(
        annotated_rows.loc[truth, "metric_family_alignment_score"],
        errors="coerce",
    )
    negative_metric = pd.to_numeric(
        annotated_rows.loc[negative, "metric_family_alignment_score"],
        errors="coerce",
    )
    if int((negative & transition_status.eq("negative_branch_incidence_leakage")).sum()):
        status = "branch_incidence_negative_leakage"
    elif int((truth & truth_transition).sum()):
        status = "branch_incidence_transition_candidate_found"
    elif int((truth & truth_mismatch).sum()):
        status = "coordinate_branch_incidence_truth_mismatch"
    else:
        status = "branch_incidence_unresolved"
    return pd.DataFrame.from_records(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "annotated_gap_row_count": int(annotated.sum()),
                "truth_recovery_total": int(truth.sum()),
                "truth_recovery_transition_candidate_count": int(
                    (truth & truth_transition).sum()
                ),
                "truth_recovery_mismatch_count": int((truth & truth_mismatch).sum()),
                "negative_total": int(negative.sum()),
                "negative_transition_candidate_count": int(
                    (negative & negative_transition).sum()
                ),
                "negative_mismatch_count": int((negative & negative_mismatch).sum()),
                "median_truth_incoming_family_outgoing_jaccard": (
                    float(truth_jaccard.median())
                    if not truth_jaccard.empty
                    else float("nan")
                ),
                "median_negative_incoming_family_outgoing_jaccard": (
                    float(negative_jaccard.median())
                    if not negative_jaccard.empty
                    else float("nan")
                ),
                "median_truth_metric_family_alignment_score": (
                    float(truth_metric.median())
                    if not truth_metric.empty
                    else float("nan")
                ),
                "median_negative_metric_family_alignment_score": (
                    float(negative_metric.median())
                    if not negative_metric.empty
                    else float("nan")
                ),
                "diagnostic_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_overlap_branch_incidence_junction_panel(
    config: OverlapBranchIncidenceJunctionPanelConfig,
) -> dict[str, Path]:
    """Run branch-incidence junction diagnostics and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if int(config.top_k) <= 0:
        raise ValueError("top_k must be positive.")
    validate_data_roles(config.data_roles)
    gap_rows = (
        pd.read_csv(config.transfer_gap_rows_path)
        if config.transfer_gap_rows_path is not None
        else None
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    row_frames: list[pd.DataFrame] = []
    skipped_cases: list[dict[str, object]] = []
    for case in _select_cases(suite=config.suite, case_names=config.case_names):
        try:
            (
                case_id,
                source_family,
                feature_representation,
                n_samples,
                n_features,
                n_categories,
            ) = _case_contract(case)
        except ValueError as exc:
            case_id = str(case.get("case_id", case.get("name", "unknown_case")))
            if not bool(config.skip_unsupported_cases):
                raise
            skipped_cases.append({"case_id": case_id, "reason": str(exc)})
            continue
        if source_family != "binary_template":
            if not bool(config.skip_unsupported_cases):
                raise ValueError(
                    f"Unsupported branch-incidence family: {source_family}"
                )
            skipped_cases.append(
                {
                    "case_id": case_id,
                    "reason": f"unsupported source_family={source_family}",
                }
            )
            continue
        for replicate in range(int(config.replicates)):
            data_seed = int(config.base_seed) + replicate
            for data_role in config.data_roles:
                row_frames.append(
                    _run_one(
                        case=case,
                        case_id=case_id,
                        source_family=source_family,
                        feature_representation=feature_representation,
                        n_samples=n_samples,
                        n_features=n_features,
                        n_categories=n_categories,
                        data_role=data_role,
                        replicate=replicate,
                        data_seed=data_seed,
                        config=config,
                    )
                )

    rows = (
        pd.concat([frame for frame in row_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in row_frames)
        else pd.DataFrame(columns=ROW_COLUMNS)
    )
    rows = _annotate_with_gap_rows(rows, gap_rows)
    summary = summarize_branch_incidence_rows(rows)
    rows.to_csv(config.rows_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "suite": config.suite,
        "case_names": list(config.case_names),
        "data_roles": list(config.data_roles),
        "replicates": int(config.replicates),
        "base_seed": int(config.base_seed),
        "profile_id": str(config.profile_id),
        "sibling_alpha": float(config.sibling_alpha),
        "edge_alpha": float(config.edge_alpha),
        "top_k": int(config.top_k),
        "transfer_gap_rows_path": (
            str(config.transfer_gap_rows_path)
            if config.transfer_gap_rows_path is not None
            else None
        ),
        "rows": int(rows.shape[0]),
        "skipped_cases": skipped_cases,
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
        "production_status": "diagnostic_only_branch_incidence_law",
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "rows": config.rows_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", default=",".join(DEFAULT_OVERLAP_CASES))
    parser.add_argument("--data-roles", default=",".join(DEFAULT_DATA_ROLES))
    parser.add_argument("--sibling-alpha", default=0.01, type=float)
    parser.add_argument("--edge-alpha", default=0.001, type=float)
    parser.add_argument("--replicates", default=1, type=int)
    parser.add_argument("--base-seed", default=20260613, type=int)
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE)
    parser.add_argument("--top-k", default=12, type=int)
    parser.add_argument("--transfer-gap-rows-path", type=Path)
    parser.add_argument(
        "--fail-on-unsupported-cases",
        action="store_true",
        help="Raise instead of recording unsupported case families.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_branch_incidence_junction_panel(
        OverlapBranchIncidenceJunctionPanelConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(str(args.case_names)),
            data_roles=tuple(
                token.strip()
                for token in str(args.data_roles).split(",")
                if token.strip()
            ),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            profile_id=str(args.profile_id),
            top_k=int(args.top_k),
            transfer_gap_rows_path=args.transfer_gap_rows_path,
            skip_unsupported_cases=not bool(args.fail_on_unsupported_cases),
        )
    )


if __name__ == "__main__":
    main()
