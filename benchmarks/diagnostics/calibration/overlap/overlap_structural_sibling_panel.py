"""Overlap-focused structural sibling diagnostics.

This panel reconstructs binary overlap cases as individual analytical cases and
checks whether a sibling decision is supported by the same structural subspace
as the local homogeneity gain. It is diagnostic-only: it does not change the
traversal gate or production admissibility contract.
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

from benchmarks.diagnostics.calibration.selected.family.selected_family_traversal_panel import (
    _build_node_decisions,
    _output_data_role,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_panel import (
    DEFAULT_DATA_ROLES,
    validate_data_roles,
)
from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.shared.runners.tbs_runner import _run_tbs_method
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.statistics.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
    parse_names,
)

STUDY_ROLE = "diagnostic_overlap_structural_sibling_not_calibration"
SCHEMA_VERSION = "overlap_structural_sibling_panel/v1"
GENERATED_BY = "benchmarks.diagnostics.calibration.overlap.overlap_structural_sibling_panel"
DEFAULT_PROFILE = "fixed_coordinate_global_passthrough_refined_v1"
DEFAULT_OVERLAP_CASES = (
    "overlap_extreme_4c",
    "overlap_mod_4c_small",
    "overlap_mod_8c_large",
    "overlap_unbal_4c_small",
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
    "parent_id",
    "depth",
    "decision_class",
    "traversal_decision",
    "sibling_open",
    "sibling_p_value",
    "selected_family_guard_blocked",
    "selected_family_p_value",
    "n_parent",
    "n_left",
    "n_right",
    "n_features",
    "top_k",
    "barycentric_balance",
    "log_barycentric_leverage",
    "sibling_contrast_norm",
    "left_edge_norm",
    "right_edge_norm",
    "edge_symmetry_cosine",
    "max_sibling_edge_alignment",
    "delta_homogeneity_jaccard_topk",
    "delta_left_edge_jaccard_topk",
    "delta_right_edge_jaccard_topk",
    "max_delta_edge_jaccard_topk",
    "left_edge_homogeneity_jaccard_topk",
    "right_edge_homogeneity_jaccard_topk",
    "max_edge_homogeneity_jaccard_topk",
    "subspace_consensus_jaccard_topk",
    "delta_heterogeneity_jaccard_topk",
    "left_edge_heterogeneity_jaccard_topk",
    "right_edge_heterogeneity_jaccard_topk",
    "max_edge_heterogeneity_jaccard_topk",
    "heterogeneity_subspace_consensus_jaccard_topk",
    "parent_pairwise_jaccard",
    "left_pairwise_jaccard",
    "right_pairwise_jaccard",
    "homogeneity_gain_left",
    "homogeneity_gain_right",
    "homogeneity_gain_min",
    "heterogeneity_gain_left",
    "heterogeneity_gain_right",
    "heterogeneity_gain_max",
    "truth_split_ari",
    "parent_truth_purity",
    "left_truth_purity",
    "right_truth_purity",
    "structural_sibling_status",
    "structural_change_mode",
    "signal_alignment_status",
)


@dataclass(frozen=True)
class OverlapStructuralSiblingPanelConfig:
    """Runtime contract for overlap structural sibling diagnostics."""

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
    max_pairwise_samples: int = 200
    skip_unsupported_cases: bool = True

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "overlap_structural_sibling_rows.csv"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "overlap_structural_sibling_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


def _as_float_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=float, copy=False)


def top_coordinate_set(vector: np.ndarray, *, top_k: int) -> frozenset[int]:
    """Return coordinates with largest absolute magnitude."""
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise ValueError("top_coordinate_set expects a one-dimensional vector.")
    finite_values = np.where(np.isfinite(values), np.abs(values), 0.0)
    positive = np.flatnonzero(finite_values > 0.0)
    if positive.size == 0:
        return frozenset()
    k = min(int(top_k), int(positive.size))
    order = np.argsort(finite_values[positive], kind="mergesort")[-k:]
    return frozenset(int(idx) for idx in positive[order])


def jaccard_index(left: frozenset[int], right: frozenset[int]) -> float:
    """Return set Jaccard index with an empty-empty convention of 1."""
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return finite cosine similarity, or 0 for a zero vector."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    value = float(np.dot(x, y) / denom)
    return float(np.clip(value, -1.0, 1.0))


def pairwise_binary_jaccard_similarity(
    matrix: np.ndarray,
    *,
    max_samples: int = 200,
    seed: int = 0,
) -> float:
    """Return mean pairwise binary Jaccard similarity for rows."""
    x = np.asarray(matrix, dtype=bool)
    if x.shape[0] < 2:
        return 1.0
    if x.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        take = rng.choice(x.shape[0], size=int(max_samples), replace=False)
        x = x[np.sort(take)]
    intersection = x[:, None, :] & x[None, :, :]
    union = x[:, None, :] | x[None, :, :]
    union_count = union.sum(axis=2)
    sim = np.divide(
        intersection.sum(axis=2),
        union_count,
        out=np.ones_like(union_count, dtype=float),
        where=union_count > 0,
    )
    tri = np.triu_indices(x.shape[0], k=1)
    return float(np.mean(sim[tri]))


def _purity(labels: np.ndarray) -> float:
    if labels.size == 0:
        return np.nan
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / labels.size)


def classify_structural_sibling_row(
    *,
    homogeneity_gain_min: float,
    heterogeneity_gain_max: float,
    delta_homogeneity_jaccard_topk: float,
    max_delta_edge_jaccard_topk: float,
    max_edge_homogeneity_jaccard_topk: float,
    heterogeneity_subspace_consensus_jaccard_topk: float,
    max_sibling_edge_alignment: float,
    sibling_contrast_norm: float,
) -> str:
    """Classify whether a sibling split is structurally supported."""
    if not math.isfinite(sibling_contrast_norm) or sibling_contrast_norm <= 0.0:
        return "degenerate_sibling_contrast"
    if not math.isfinite(homogeneity_gain_min):
        return "weak_homogeneity_gain"
    if homogeneity_gain_min < 0.02:
        if (
            math.isfinite(heterogeneity_gain_max)
            and heterogeneity_gain_max >= 0.02
            and heterogeneity_subspace_consensus_jaccard_topk >= 0.25
        ):
            return "same_subspace_heterogeneity_increase"
        if math.isfinite(heterogeneity_gain_max) and heterogeneity_gain_max >= 0.02:
            return "unrelated_subspace_heterogeneity_signal"
        return "weak_homogeneity_gain"
    if delta_homogeneity_jaccard_topk < 0.25 and max_sibling_edge_alignment < 0.50:
        return "barycentric_focus_mismatch"
    if max_edge_homogeneity_jaccard_topk < 0.25:
        return "structural_homogeneity_subspace_mismatch"
    if max_delta_edge_jaccard_topk < 0.25:
        return "structural_subspace_mismatch"
    return "structural_same_subspace_supported"


def classify_structural_change_mode(
    *,
    homogeneity_gain_min: float,
    heterogeneity_gain_max: float,
    subspace_consensus_jaccard_topk: float,
    heterogeneity_subspace_consensus_jaccard_topk: float,
    sibling_contrast_norm: float,
) -> str:
    """Classify the direction and subspace agreement of structural change."""
    if not math.isfinite(sibling_contrast_norm) or sibling_contrast_norm <= 0.0:
        return "degenerate_structural_change"
    if (
        math.isfinite(homogeneity_gain_min)
        and homogeneity_gain_min >= 0.02
        and subspace_consensus_jaccard_topk >= 0.25
    ):
        return "homogeneous_same_subspace"
    if (
        math.isfinite(heterogeneity_gain_max)
        and heterogeneity_gain_max >= 0.02
        and heterogeneity_subspace_consensus_jaccard_topk >= 0.25
    ):
        return "heterogeneous_same_subspace"
    has_directional_change = (
        math.isfinite(homogeneity_gain_min) and homogeneity_gain_min >= 0.02
    ) or (math.isfinite(heterogeneity_gain_max) and heterogeneity_gain_max >= 0.02)
    if (
        has_directional_change
        and subspace_consensus_jaccard_topk < 0.25
        and heterogeneity_subspace_consensus_jaccard_topk < 0.25
    ):
        return "unrelated_subspace_signal"
    return "weak_or_mixed_structural_change"


def _signal_alignment_status(
    *,
    data_role: str,
    truth_split_ari: float,
    structural_status: str,
) -> str:
    if str(data_role) != "signal":
        return "null_context"
    if not math.isfinite(truth_split_ari):
        return "truth_unavailable"
    if truth_split_ari >= 0.50:
        return "truth_aligned_split"
    if structural_status == "structural_same_subspace_supported":
        return "structural_but_truth_misaligned"
    return "truth_and_structure_misaligned"


def _raw_node_by_id(tree) -> dict[str, object]:
    return {str(node): node for node in tree.nodes}


def _descendant_labels(tree, node: object) -> list[str]:
    return [str(label) for label in sorted(tree.compute_descendant_sets(use_labels=True)[node])]


def _truth_for_labels(
    *,
    truth_by_label: dict[str, int],
    labels: Sequence[str],
) -> np.ndarray:
    return np.asarray([truth_by_label[str(label)] for label in labels], dtype=int)


def _analytical_rows_for_node(
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
    max_pairwise_samples: int,
) -> dict[str, object] | None:
    tree = result.extra["tree"]
    raw_nodes = _raw_node_by_id(tree)
    raw_node = raw_nodes.get(str(node_row["node_id"]))
    if raw_node is None:
        return None
    children = list(tree.successors(raw_node))
    if len(children) != 2:
        return None
    left_labels = _descendant_labels(tree, children[0])
    right_labels = _descendant_labels(tree, children[1])
    parent_labels = left_labels + right_labels
    if len(left_labels) == 0 or len(right_labels) == 0:
        return None

    parent_x = _as_float_matrix(data.loc[parent_labels])
    left_x = _as_float_matrix(data.loc[left_labels])
    right_x = _as_float_matrix(data.loc[right_labels])
    parent_mean = parent_x.mean(axis=0)
    left_mean = left_x.mean(axis=0)
    right_mean = right_x.mean(axis=0)
    delta = left_mean - right_mean
    left_edge = left_mean - parent_mean
    right_edge = right_mean - parent_mean

    parent_var = parent_mean * (1.0 - parent_mean)
    left_var = left_mean * (1.0 - left_mean)
    right_var = right_mean * (1.0 - right_mean)
    homogeneity_focus = np.maximum(parent_var - left_var, parent_var - right_var)
    heterogeneity_focus = np.maximum(left_var - parent_var, right_var - parent_var)

    delta_top = top_coordinate_set(delta, top_k=top_k)
    homogeneity_top = top_coordinate_set(homogeneity_focus, top_k=top_k)
    heterogeneity_top = top_coordinate_set(heterogeneity_focus, top_k=top_k)
    left_edge_top = top_coordinate_set(left_edge, top_k=top_k)
    right_edge_top = top_coordinate_set(right_edge, top_k=top_k)

    parent_pairwise = pairwise_binary_jaccard_similarity(
        parent_x,
        max_samples=max_pairwise_samples,
        seed=data_seed + int(node_row["depth"]) + 17,
    )
    left_pairwise = pairwise_binary_jaccard_similarity(
        left_x,
        max_samples=max_pairwise_samples,
        seed=data_seed + int(node_row["depth"]) + 31,
    )
    right_pairwise = pairwise_binary_jaccard_similarity(
        right_x,
        max_samples=max_pairwise_samples,
        seed=data_seed + int(node_row["depth"]) + 43,
    )

    n_left = len(left_labels)
    n_right = len(right_labels)
    balance = min(n_left, n_right) / (n_left + n_right)
    leverage = math.log(max(n_left, n_right) / max(1, min(n_left, n_right)))
    max_edge_alignment = max(
        abs(cosine_similarity(delta, left_edge)),
        abs(cosine_similarity(delta, right_edge)),
    )
    delta_left_jaccard = jaccard_index(delta_top, left_edge_top)
    delta_right_jaccard = jaccard_index(delta_top, right_edge_top)
    max_delta_edge_jaccard = max(delta_left_jaccard, delta_right_jaccard)
    delta_homogeneity_jaccard = jaccard_index(delta_top, homogeneity_top)
    left_edge_homogeneity_jaccard = jaccard_index(left_edge_top, homogeneity_top)
    right_edge_homogeneity_jaccard = jaccard_index(right_edge_top, homogeneity_top)
    max_edge_homogeneity_jaccard = max(
        left_edge_homogeneity_jaccard,
        right_edge_homogeneity_jaccard,
    )
    subspace_consensus_jaccard = min(
        delta_homogeneity_jaccard,
        max_delta_edge_jaccard,
        max_edge_homogeneity_jaccard,
    )
    delta_heterogeneity_jaccard = jaccard_index(delta_top, heterogeneity_top)
    left_edge_heterogeneity_jaccard = jaccard_index(left_edge_top, heterogeneity_top)
    right_edge_heterogeneity_jaccard = jaccard_index(
        right_edge_top,
        heterogeneity_top,
    )
    max_edge_heterogeneity_jaccard = max(
        left_edge_heterogeneity_jaccard,
        right_edge_heterogeneity_jaccard,
    )
    heterogeneity_subspace_consensus_jaccard = min(
        delta_heterogeneity_jaccard,
        max_delta_edge_jaccard,
        max_edge_heterogeneity_jaccard,
    )
    homogeneity_gain_left = float(left_pairwise - parent_pairwise)
    homogeneity_gain_right = float(right_pairwise - parent_pairwise)
    homogeneity_gain_min = min(homogeneity_gain_left, homogeneity_gain_right)
    heterogeneity_gain_left = float(parent_pairwise - left_pairwise)
    heterogeneity_gain_right = float(parent_pairwise - right_pairwise)
    heterogeneity_gain_max = max(heterogeneity_gain_left, heterogeneity_gain_right)
    sibling_contrast_norm = float(np.linalg.norm(delta))
    structural_status = classify_structural_sibling_row(
        homogeneity_gain_min=homogeneity_gain_min,
        heterogeneity_gain_max=heterogeneity_gain_max,
        delta_homogeneity_jaccard_topk=delta_homogeneity_jaccard,
        max_delta_edge_jaccard_topk=max_delta_edge_jaccard,
        max_edge_homogeneity_jaccard_topk=max_edge_homogeneity_jaccard,
        heterogeneity_subspace_consensus_jaccard_topk=(heterogeneity_subspace_consensus_jaccard),
        max_sibling_edge_alignment=max_edge_alignment,
        sibling_contrast_norm=sibling_contrast_norm,
    )
    structural_change_mode = classify_structural_change_mode(
        homogeneity_gain_min=homogeneity_gain_min,
        heterogeneity_gain_max=heterogeneity_gain_max,
        subspace_consensus_jaccard_topk=subspace_consensus_jaccard,
        heterogeneity_subspace_consensus_jaccard_topk=(heterogeneity_subspace_consensus_jaccard),
        sibling_contrast_norm=sibling_contrast_norm,
    )

    parent_truth = _truth_for_labels(truth_by_label=truth_by_label, labels=parent_labels)
    child_membership = np.concatenate(
        [
            np.zeros(n_left, dtype=int),
            np.ones(n_right, dtype=int),
        ]
    )
    truth_split_ari = (
        float(adjusted_rand_score(parent_truth, child_membership))
        if np.unique(parent_truth).size > 1
        else np.nan
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "case_id": case_id,
        "data_role": _output_data_role(data_role),
        "replicate": int(replicate),
        "data_seed": int(data_seed),
        "profile_id": profile_id,
        "node_id": str(node_row["node_id"]),
        "parent_id": str(node_row["parent_id"]),
        "depth": int(node_row["depth"]),
        "decision_class": str(node_row["decision_class"]),
        "traversal_decision": str(node_row["traversal_decision"]),
        "sibling_open": bool(node_row["sibling_open"]),
        "sibling_p_value": float(node_row["sibling_p_value"]),
        "selected_family_guard_blocked": bool(node_row["selected_family_guard_blocked"]),
        "selected_family_p_value": float(node_row["selected_family_p_value"]),
        "n_parent": int(n_left + n_right),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "n_features": int(data.shape[1]),
        "top_k": int(min(top_k, data.shape[1])),
        "barycentric_balance": float(balance),
        "log_barycentric_leverage": float(leverage),
        "sibling_contrast_norm": sibling_contrast_norm,
        "left_edge_norm": float(np.linalg.norm(left_edge)),
        "right_edge_norm": float(np.linalg.norm(right_edge)),
        "edge_symmetry_cosine": cosine_similarity(left_edge, right_edge),
        "max_sibling_edge_alignment": float(max_edge_alignment),
        "delta_homogeneity_jaccard_topk": float(delta_homogeneity_jaccard),
        "delta_left_edge_jaccard_topk": float(delta_left_jaccard),
        "delta_right_edge_jaccard_topk": float(delta_right_jaccard),
        "max_delta_edge_jaccard_topk": float(max_delta_edge_jaccard),
        "left_edge_homogeneity_jaccard_topk": float(left_edge_homogeneity_jaccard),
        "right_edge_homogeneity_jaccard_topk": float(right_edge_homogeneity_jaccard),
        "max_edge_homogeneity_jaccard_topk": float(max_edge_homogeneity_jaccard),
        "subspace_consensus_jaccard_topk": float(subspace_consensus_jaccard),
        "delta_heterogeneity_jaccard_topk": float(delta_heterogeneity_jaccard),
        "left_edge_heterogeneity_jaccard_topk": float(left_edge_heterogeneity_jaccard),
        "right_edge_heterogeneity_jaccard_topk": float(right_edge_heterogeneity_jaccard),
        "max_edge_heterogeneity_jaccard_topk": float(max_edge_heterogeneity_jaccard),
        "heterogeneity_subspace_consensus_jaccard_topk": float(
            heterogeneity_subspace_consensus_jaccard
        ),
        "parent_pairwise_jaccard": float(parent_pairwise),
        "left_pairwise_jaccard": float(left_pairwise),
        "right_pairwise_jaccard": float(right_pairwise),
        "homogeneity_gain_left": float(homogeneity_gain_left),
        "homogeneity_gain_right": float(homogeneity_gain_right),
        "homogeneity_gain_min": float(homogeneity_gain_min),
        "heterogeneity_gain_left": float(heterogeneity_gain_left),
        "heterogeneity_gain_right": float(heterogeneity_gain_right),
        "heterogeneity_gain_max": float(heterogeneity_gain_max),
        "truth_split_ari": truth_split_ari,
        "parent_truth_purity": _purity(parent_truth),
        "left_truth_purity": _purity(
            _truth_for_labels(truth_by_label=truth_by_label, labels=left_labels)
        ),
        "right_truth_purity": _purity(
            _truth_for_labels(truth_by_label=truth_by_label, labels=right_labels)
        ),
        "structural_sibling_status": structural_status,
        "structural_change_mode": structural_change_mode,
        "signal_alignment_status": _signal_alignment_status(
            data_role=data_role,
            truth_split_ari=truth_split_ari,
            structural_status=structural_status,
        ),
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
        & (decision_mask | evidence_mask)
    ].copy()


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
    config: OverlapStructuralSiblingPanelConfig,
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
    result = _run_tbs_method(
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
        row = _analytical_rows_for_node(
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
            max_pairwise_samples=int(config.max_pairwise_samples),
        )
        if row is not None:
            records.append(row)
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def _summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "data_role",
                "replicate",
                "n_nodes",
                "accepted_internal_split_count",
                "selected_family_blocked_count",
                "weak_or_mismatch_count",
                "mean_delta_homogeneity_jaccard_topk",
                "mean_max_edge_homogeneity_jaccard_topk",
                "mean_subspace_consensus_jaccard_topk",
                "mean_heterogeneity_subspace_consensus_jaccard_topk",
                "mean_homogeneity_gain_min",
                "mean_heterogeneity_gain_max",
                "median_truth_split_ari",
                "structural_status_counts",
                "structural_change_mode_counts",
                "signal_alignment_status_counts",
                "study_role",
            ]
        )
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(["case_id", "data_role", "replicate"], sort=True):
        case_id, data_role, replicate = keys
        status_counts = group["structural_sibling_status"].value_counts().to_dict()
        change_counts = group["structural_change_mode"].value_counts().to_dict()
        signal_counts = group["signal_alignment_status"].value_counts().to_dict()
        weak_or_mismatch = group["structural_sibling_status"].isin(
            {
                "weak_homogeneity_gain",
                "barycentric_focus_mismatch",
                "structural_subspace_mismatch",
                "structural_homogeneity_subspace_mismatch",
                "same_subspace_heterogeneity_increase",
                "unrelated_subspace_heterogeneity_signal",
            }
        )
        records.append(
            {
                "case_id": case_id,
                "data_role": data_role,
                "replicate": int(replicate),
                "n_nodes": int(group.shape[0]),
                "accepted_internal_split_count": int(
                    group["decision_class"].eq("accepted_internal_split").sum()
                ),
                "selected_family_blocked_count": int(
                    group["decision_class"].eq("selected_family_blocked").sum()
                ),
                "weak_or_mismatch_count": int(weak_or_mismatch.sum()),
                "mean_delta_homogeneity_jaccard_topk": float(
                    group["delta_homogeneity_jaccard_topk"].mean()
                ),
                "mean_max_edge_homogeneity_jaccard_topk": float(
                    group["max_edge_homogeneity_jaccard_topk"].mean()
                ),
                "mean_subspace_consensus_jaccard_topk": float(
                    group["subspace_consensus_jaccard_topk"].mean()
                ),
                "mean_heterogeneity_subspace_consensus_jaccard_topk": float(
                    group["heterogeneity_subspace_consensus_jaccard_topk"].mean()
                ),
                "mean_homogeneity_gain_min": float(group["homogeneity_gain_min"].mean()),
                "mean_heterogeneity_gain_max": float(group["heterogeneity_gain_max"].mean()),
                "median_truth_split_ari": float(group["truth_split_ari"].median()),
                "structural_status_counts": json.dumps(status_counts, sort_keys=True),
                "structural_change_mode_counts": json.dumps(
                    change_counts,
                    sort_keys=True,
                ),
                "signal_alignment_status_counts": json.dumps(signal_counts, sort_keys=True),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(records)


def run_overlap_structural_sibling_panel(
    config: OverlapStructuralSiblingPanelConfig,
) -> dict[str, Path]:
    """Run overlap structural sibling diagnostics and write outputs."""
    if int(config.replicates) <= 0:
        raise ValueError("replicates must be positive.")
    if int(config.top_k) <= 0:
        raise ValueError("top_k must be positive.")
    validate_data_roles(config.data_roles)
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
                raise ValueError(f"Unsupported overlap structural family: {source_family}")
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
    summary = _summarize_rows(rows)
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
        "max_pairwise_samples": int(config.max_pairwise_samples),
        "rows": int(rows.shape[0]),
        "skipped_cases": skipped_cases,
        "outputs": {
            "rows": str(config.rows_path),
            "summary": str(config.summary_path),
        },
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
    parser.add_argument("--suite", default="full")
    parser.add_argument(
        "--case-names",
        default=",".join(DEFAULT_OVERLAP_CASES),
        help="Comma-separated case names. Defaults to the known refined-profile false-split overlaps.",
    )
    parser.add_argument("--data-roles", default=",".join(DEFAULT_DATA_ROLES))
    parser.add_argument("--sibling-alpha", default=0.01, type=float)
    parser.add_argument("--edge-alpha", default=0.001, type=float)
    parser.add_argument("--replicates", default=1, type=int)
    parser.add_argument("--base-seed", default=20260613, type=int)
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE)
    parser.add_argument("--top-k", default=12, type=int)
    parser.add_argument("--max-pairwise-samples", default=200, type=int)
    parser.add_argument("--fail-on-unsupported-cases", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    run_overlap_structural_sibling_panel(
        OverlapStructuralSiblingPanelConfig(
            output_dir=args.output_dir,
            suite=str(args.suite),
            case_names=parse_names(args.case_names),
            data_roles=parse_names(args.data_roles),
            sibling_alpha=float(args.sibling_alpha),
            edge_alpha=float(args.edge_alpha),
            replicates=int(args.replicates),
            base_seed=int(args.base_seed),
            profile_id=str(args.profile_id),
            top_k=int(args.top_k),
            max_pairwise_samples=int(args.max_pairwise_samples),
            skip_unsupported_cases=not bool(args.fail_on_unsupported_cases),
        )
    )


if __name__ == "__main__":
    main()
