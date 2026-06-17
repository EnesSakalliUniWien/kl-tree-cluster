"""Selected pass-through branch-recovery conditioning diagnostics.

This diagnostic is intentionally not a production rule. It separates the
truth-geometry target for retained pass-through walks into branch recovery,
barycentric mixture, fragments, and selected-null controls. Observable topology
metrics are reported separately from oracle truth metrics so that benchmark
truth labels cannot leak into calibration.
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

from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.diagnostics.calibration.overlap_structural_sibling_panel import (
    jaccard_index,
    pairwise_binary_jaccard_similarity,
    top_coordinate_set,
)
from benchmarks.shared.util.time import format_timestamp_utc
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
)

STUDY_ROLE = "diagnostic_selected_pass_through_branch_recovery_not_calibration"
SCHEMA_VERSION = "selected_pass_through_branch_recovery_conditioning/v1"
GENERATED_BY = (
    "benchmarks.diagnostics.calibration."
    "selected_pass_through_branch_recovery_conditioning"
)
GENERATED_SUPPORT_METHOD_ID = "generated_selected_pass_through_branch_support_fixture"

REQUIRED_NODE_COLUMNS = {
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "fixture_role",
    "truth_geometry_role",
    "truth_node_cluster_count",
    "truth_node_majority_fraction",
    "truth_downstream_split_ari",
    "truth_downstream_child_mean_purity",
    "truth_downstream_child_majority_distinct",
    "completed_balance_product",
    "structural_incoming_branch_balance",
    "structural_outgoing_balance",
    "structural_balance_product",
    "distance_to_downstream_accepted_split",
}

OBSERVABLE_METRICS = (
    "completed_balance_product",
    "structural_balance_product",
    "structural_incoming_branch_balance",
    "structural_outgoing_balance",
    "distance_to_downstream_accepted_split",
    "feature_branch_geometry_score",
    "feature_homogeneity_gain_min",
    "feature_subspace_consensus_jaccard_topk",
    "feature_heterogeneity_subspace_consensus_jaccard_topk",
)

ORACLE_METRICS = (
    "truth_downstream_split_ari",
    "truth_downstream_child_mean_purity",
    "truth_downstream_child_majority_distinct_numeric",
    "branch_recovery_oracle_score",
    "barycentric_mixture_oracle_score",
)

ROW_COLUMNS = (
    "schema_version",
    "study_role",
    "case_id",
    "data_role",
    "replicate",
    "node_id",
    "fixture_role",
    "truth_geometry_role",
    "truth_geometry_class",
    "branch_recovery_target",
    "full_branch_recovery_target",
    "partial_branch_recovery_target",
    "barycentric_mixture_target",
    "fragment_target",
    "selected_null_control_target",
    "unresolved_signal_target",
    "truth_node_cluster_count",
    "truth_node_majority_fraction",
    "truth_downstream_split_ari",
    "truth_downstream_child_mean_purity",
    "truth_downstream_child_majority_distinct",
    "truth_downstream_child_majority_distinct_numeric",
    "completed_balance_product",
    "structural_incoming_branch_balance",
    "structural_outgoing_balance",
    "structural_balance_product",
    "distance_to_downstream_accepted_split",
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
    "branch_recovery_oracle_score",
    "barycentric_mixture_oracle_score",
    "conditioning_status",
)

METRIC_SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "metric",
    "metric_kind",
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
    "zero_negative_direction",
    "zero_negative_threshold",
    "zero_negative_branch_count",
    "zero_negative_branch_retention",
    "zero_negative_negative_count",
    "zero_negative_margin",
    "zero_negative_status",
)

SUMMARY_COLUMNS = (
    "schema_version",
    "study_role",
    "row_count",
    "full_branch_recovery_count",
    "partial_branch_recovery_count",
    "branch_recovery_count",
    "barycentric_mixture_count",
    "fragment_count",
    "selected_null_control_count",
    "unresolved_signal_count",
    "best_observable_metric",
    "best_observable_auc",
    "best_observable_zero_negative_status",
    "best_oracle_metric",
    "best_oracle_auc",
    "best_oracle_zero_negative_status",
    "diagnostic_status",
    "next_required_step",
    "production_action",
)


@dataclass(frozen=True)
class SelectedPassThroughBranchRecoveryConditioningConfig:
    """Runtime contract for selected pass-through branch diagnostics."""

    output_dir: Path
    node_rows_path: Path | None = None
    gene_assignments_path: Path | None = None
    use_focused_fixture: bool = False
    use_generated_support_fixture: bool = False
    min_full_branch_recovery_count: int = 2
    truth_suite: str = "binary"
    top_k: int = 12
    max_pairwise_samples: int = 200

    @property
    def rows_path(self) -> Path:
        return self.output_dir / "selected_pass_through_branch_conditioning_rows.csv"

    @property
    def metric_summary_path(self) -> Path:
        return (
            self.output_dir
            / "selected_pass_through_branch_conditioning_metric_summary.csv"
        )

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "selected_pass_through_branch_conditioning_summary.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def generated_support_node_rows_path(self) -> Path:
        return self.output_dir / "generated_branch_support_node_rows.csv"

    @property
    def generated_support_gene_assignments_path(self) -> Path:
        return self.output_dir / "generated_branch_support_gene_assignments.csv"


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


def _finite(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.dropna().to_numpy(dtype=float)


def _input_data_role(data_role: object) -> str:
    return "null" if str(data_role) == "selected_null" else str(data_role)


def _split_path(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _path_contains(path: list[str], node_id: str) -> bool:
    return str(node_id) in path


def _next_path_node(path: list[str], node_id: str) -> str:
    node = str(node_id)
    try:
        index = path.index(node)
    except ValueError:
        return ""
    next_index = index + 1
    return path[next_index] if next_index < len(path) else ""


def _feature_geometry_empty_row(
    row: pd.Series,
    *,
    status: str,
) -> dict[str, object]:
    return {
        "case_id": str(row.get("case_id", "")),
        "data_role": str(row.get("data_role", "")),
        "replicate": int(_finite_float(row.get("replicate", -1))),
        "node_id": str(row.get("node_id", "")),
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


def _case_data_by_key(
    node_rows: pd.DataFrame,
    *,
    suite: str,
) -> dict[tuple[str, str, int], pd.DataFrame]:
    case_names = sorted(set(node_rows["case_id"].dropna().astype(str)))
    cases = {
        str(case["name"]): dict(case)
        for case in _select_cases(suite=suite, case_names=case_names)
    }
    data_by_key: dict[tuple[str, str, int], pd.DataFrame] = {}
    for _, row in (
        node_rows[["case_id", "data_role", "replicate", "data_seed"]]
        .dropna(subset=["case_id", "data_role", "replicate", "data_seed"])
        .drop_duplicates()
        .iterrows()
    ):
        case_id = str(row["case_id"])
        data_role = str(row["data_role"])
        seed = int(float(row["data_seed"]))
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
        data, _feature_space, _truth_labels, _true_clusters = _generate_data_with_truth(
            case=case,
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            data_role=_input_data_role(data_role),
            seed=seed,
        )
        data_by_key[(case_id, data_role, int(float(row["replicate"])))] = data
    return data_by_key


def _assignment_groups(
    gene_assignments: pd.DataFrame,
) -> dict[tuple[str, str, str, int], pd.DataFrame]:
    required = {
        "case_id",
        "data_role",
        "method_id",
        "replicate",
        "sample_id",
        "path_node_ids",
    }
    missing = sorted(required - set(gene_assignments.columns))
    if missing:
        raise ValueError(f"gene assignments are missing columns: {missing!r}")
    rows = gene_assignments.copy()
    rows["replicate"] = pd.to_numeric(rows["replicate"], errors="coerce").fillna(-1).astype(int)
    groups: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    for key, group in rows.groupby(
        ["case_id", "data_role", "method_id", "replicate"],
        sort=False,
    ):
        case_id, data_role, method_id, replicate = key
        groups[(str(case_id), str(data_role), str(method_id), int(replicate))] = group
    return groups


def _fixture_role_for_truth_geometry(
    *,
    data_role: str,
    truth_geometry_role: str,
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
    return "signal_pass_through_candidate_unresolved"


def _majority_fraction(labels: np.ndarray) -> float:
    if labels.size == 0:
        return math.nan
    _values, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / labels.size)


def _majority_label(labels: np.ndarray) -> int | None:
    if labels.size == 0:
        return None
    values, counts = np.unique(labels, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _child_membership_for_scheme(
    *,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    scheme: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    keep = np.ones(labels.shape[0], dtype=bool)
    if scheme == "label_lt2":
        child = np.where(labels < 2, "left", "right")
    elif scheme == "label01_by_label":
        keep = labels < 2
        child = np.where(labels == 0, "left", "right").astype(object)
    elif scheme == "label_even":
        child = np.where(labels % 2 == 0, "left", "right")
    elif scheme == "label_0_vs_rest":
        child = np.where(labels == 0, "left", "right")
    elif scheme == "label01_random":
        keep = labels < 2
        child = np.full(labels.shape[0], "", dtype=object)
        kept_indices = np.flatnonzero(keep)
        shuffled = kept_indices.copy()
        rng.shuffle(shuffled)
        half = max(1, len(shuffled) // 2)
        child[shuffled[:half]] = "left"
        child[shuffled[half:]] = "right"
    elif scheme == "label01_same_majority_random":
        child = np.full(labels.shape[0], "", dtype=object)
        label0 = np.flatnonzero(labels == 0)
        label1 = np.flatnonzero(labels == 1)
        label1_take = min(max(2, len(label0) // 5), len(label1))
        kept_indices = np.concatenate(
            [
                label0,
                rng.choice(label1, size=label1_take, replace=False)
                if label1_take
                else np.asarray([], dtype=int),
            ]
        )
        keep = np.zeros(labels.shape[0], dtype=bool)
        keep[kept_indices] = True
        shuffled = kept_indices.copy()
        rng.shuffle(shuffled)
        half = max(1, len(shuffled) // 2)
        child[shuffled[:half]] = "left"
        child[shuffled[half:]] = "right"
    elif scheme == "label0_random":
        keep = labels == 0
        child = np.full(labels.shape[0], "", dtype=object)
        kept_indices = np.flatnonzero(keep)
        shuffled = kept_indices.copy()
        rng.shuffle(shuffled)
        half = max(1, len(shuffled) // 2)
        child[shuffled[:half]] = "left"
        child[shuffled[half:]] = "right"
    elif scheme == "random_all":
        child = np.full(labels.shape[0], "", dtype=object)
        indices = np.arange(labels.shape[0])
        rng.shuffle(indices)
        half = max(1, len(indices) // 2)
        child[indices[:half]] = "left"
        child[indices[half:]] = "right"
    else:
        raise ValueError(f"unknown generated support split scheme: {scheme!r}")
    keep = keep & np.isin(child, ["left", "right"])
    return sample_ids[keep], child[keep]


def _truth_metrics_for_children(
    *,
    labels_by_sample: dict[str, int],
    selected_sample_ids: np.ndarray,
    child_by_sample: dict[str, str],
) -> dict[str, object]:
    selected_labels = np.asarray(
        [labels_by_sample[str(sample)] for sample in selected_sample_ids],
        dtype=int,
    )
    membership = np.asarray(
        [0 if child_by_sample[str(sample)] == "left" else 1 for sample in selected_sample_ids],
        dtype=int,
    )
    left_labels = selected_labels[membership == 0]
    right_labels = selected_labels[membership == 1]
    child_majorities = (_majority_label(left_labels), _majority_label(right_labels))
    child_purities = [
        _majority_fraction(left_labels),
        _majority_fraction(right_labels),
    ]
    if np.unique(selected_labels).size > 1:
        split_ari = float(adjusted_rand_score(selected_labels, membership))
    else:
        split_ari = 0.0
    return {
        "truth_node_cluster_count": int(np.unique(selected_labels).size),
        "truth_node_majority_fraction": _majority_fraction(selected_labels),
        "truth_downstream_split_ari": split_ari,
        "truth_downstream_child_mean_purity": float(np.nanmean(child_purities)),
        "truth_downstream_child_majority_distinct": (
            child_majorities[0] is not None
            and child_majorities[1] is not None
            and child_majorities[0] != child_majorities[1]
        ),
    }


def build_generated_branch_positive_support_fixture(
    *,
    suite: str = "binary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate branch-positive selected-pass-through support fixture inputs.

    The fixture is diagnostic-only. It uses existing benchmark data generators
    plus synthetic selected-tree paths so the non-oracle feature-geometry
    builder is exercised on actual binary feature matrices.
    """
    specs = [
        {
            "case_id": "binary_perfect_4c",
            "data_role": "signal",
            "seed": 20263642,
            "scheme": "label01_by_label",
            "truth_geometry_role": "truth_recovery_pass_through_positive",
        },
        {
            "case_id": "binary_low_noise_4c",
            "data_role": "signal",
            "seed": 20264651,
            "scheme": "label01_by_label",
            "truth_geometry_role": "truth_recovery_pass_through_positive",
        },
        {
            "case_id": "binary_moderate_4c",
            "data_role": "signal",
            "seed": 20261624,
            "scheme": "label01_by_label",
            "truth_geometry_role": "truth_recovery_pass_through_positive",
        },
        {
            "case_id": "overlap_part_4c_small",
            "data_role": "signal",
            "seed": 20261624,
            "scheme": "label_lt2",
            "truth_geometry_role": "partial_truth_recovery_pass_through_candidate",
        },
        {
            "case_id": "overlap_unbal_4c_small",
            "data_role": "signal",
            "seed": 20260615,
            "scheme": "label01_same_majority_random",
            "truth_geometry_role": "barycentric_mixture_pass_through_candidate",
        },
        {
            "case_id": "binary_perfect_4c",
            "data_role": "signal",
            "seed": 20260615,
            "scheme": "label0_random",
            "truth_geometry_role": "fragment_false_pass_through",
        },
        {
            "case_id": "binary_perfect_4c",
            "data_role": "selected_null",
            "seed": 20260615,
            "scheme": "random_all",
            "truth_geometry_role": "selected_null_truth_control",
        },
        {
            "case_id": "binary_low_noise_4c",
            "data_role": "selected_null",
            "seed": 20261624,
            "scheme": "random_all",
            "truth_geometry_role": "selected_null_truth_control",
        },
        {
            "case_id": "binary_moderate_4c",
            "data_role": "selected_null",
            "seed": 20262633,
            "scheme": "random_all",
            "truth_geometry_role": "selected_null_truth_control",
        },
    ]
    cases = {
        str(case["name"]): dict(case)
        for case in _select_cases(
            suite=suite,
            case_names=sorted({str(spec["case_id"]) for spec in specs}),
        )
    }
    node_records: list[dict[str, object]] = []
    assignment_records: list[dict[str, object]] = []
    for replicate, spec in enumerate(specs):
        case_id = str(spec["case_id"])
        data_role = str(spec["data_role"])
        seed = int(spec["seed"])
        node_id = f"generated_pass_{replicate}"
        split_node_id = f"generated_split_{replicate}"
        case = cases[case_id]
        (
            _case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        data, _feature_space, labels, _true_clusters = _generate_data_with_truth(
            case=case,
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            data_role=_input_data_role(data_role),
            seed=seed,
        )
        sample_ids = data.index.astype(str).to_numpy()
        labels = np.asarray(labels, dtype=int)
        selected_sample_ids, child_values = _child_membership_for_scheme(
            sample_ids=sample_ids,
            labels=labels,
            scheme=str(spec["scheme"]),
            seed=seed + replicate,
        )
        child_by_sample = {
            str(sample): str(child)
            for sample, child in zip(selected_sample_ids, child_values, strict=True)
        }
        labels_by_sample = {
            str(sample): int(label)
            for sample, label in zip(sample_ids, labels, strict=True)
        }
        truth_metrics = _truth_metrics_for_children(
            labels_by_sample=labels_by_sample,
            selected_sample_ids=selected_sample_ids,
            child_by_sample=child_by_sample,
        )
        n_selected = int(len(selected_sample_ids))
        n_left = int(np.sum(child_values == "left"))
        n_right = int(np.sum(child_values == "right"))
        outgoing_balance = min(n_left, n_right) / max(n_selected, 1)
        incoming_balance = 0.45 if data_role == "signal" else 0.40
        structural_product = incoming_balance * outgoing_balance
        truth_geometry_role = str(spec["truth_geometry_role"])
        node_records.append(
            {
                "case_id": case_id,
                "data_role": data_role,
                "replicate": int(replicate),
                "node_id": node_id,
                "fixture_role": _fixture_role_for_truth_geometry(
                    data_role=data_role,
                    truth_geometry_role=truth_geometry_role,
                ),
                "truth_geometry_role": truth_geometry_role,
                **truth_metrics,
                "truth_downstream_split_node_id": split_node_id,
                "completed_balance_product": float(structural_product),
                "structural_incoming_branch_balance": float(incoming_balance),
                "structural_outgoing_balance": float(outgoing_balance),
                "structural_balance_product": float(structural_product),
                "distance_to_downstream_accepted_split": 1.0,
                "left_method_id": GENERATED_SUPPORT_METHOD_ID,
                "data_seed": int(seed),
            }
        )
        for sample_id in selected_sample_ids:
            child = child_by_sample[str(sample_id)]
            assignment_records.append(
                {
                    "case_id": case_id,
                    "data_role": data_role,
                    "method_id": GENERATED_SUPPORT_METHOD_ID,
                    "replicate": int(replicate),
                    "sample_id": str(sample_id),
                    "path_node_ids": (
                        f"{node_id};{split_node_id};"
                        f"{split_node_id}_{child};leaf_{sample_id}"
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(node_records),
        pd.DataFrame.from_records(assignment_records),
    )


def _feature_geometry_for_partition(
    *,
    data: pd.DataFrame,
    parent_sample_ids: list[str],
    left_sample_ids: list[str],
    right_sample_ids: list[str],
    top_k: int,
    max_pairwise_samples: int,
    seed: int,
) -> dict[str, object]:
    parent_x = data.loc[parent_sample_ids].to_numpy(dtype=float, copy=False)
    left_x = data.loc[left_sample_ids].to_numpy(dtype=float, copy=False)
    right_x = data.loc[right_sample_ids].to_numpy(dtype=float, copy=False)
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

    bounded_top_k = min(int(top_k), int(data.shape[1]))
    delta_top = top_coordinate_set(delta, top_k=bounded_top_k)
    left_edge_top = top_coordinate_set(left_edge, top_k=bounded_top_k)
    right_edge_top = top_coordinate_set(right_edge, top_k=bounded_top_k)
    homogeneity_top = top_coordinate_set(homogeneity_focus, top_k=bounded_top_k)
    heterogeneity_top = top_coordinate_set(heterogeneity_focus, top_k=bounded_top_k)

    parent_pairwise = pairwise_binary_jaccard_similarity(
        parent_x,
        max_samples=max_pairwise_samples,
        seed=seed + 11,
    )
    left_pairwise = pairwise_binary_jaccard_similarity(
        left_x,
        max_samples=max_pairwise_samples,
        seed=seed + 23,
    )
    right_pairwise = pairwise_binary_jaccard_similarity(
        right_x,
        max_samples=max_pairwise_samples,
        seed=seed + 37,
    )
    homogeneity_gain_min = min(
        float(left_pairwise - parent_pairwise),
        float(right_pairwise - parent_pairwise),
    )
    heterogeneity_gain_max = max(
        float(parent_pairwise - left_pairwise),
        float(parent_pairwise - right_pairwise),
    )

    max_delta_edge_jaccard = max(
        jaccard_index(delta_top, left_edge_top),
        jaccard_index(delta_top, right_edge_top),
    )
    max_edge_homogeneity_jaccard = max(
        jaccard_index(left_edge_top, homogeneity_top),
        jaccard_index(right_edge_top, homogeneity_top),
    )
    subspace_consensus = min(
        jaccard_index(delta_top, homogeneity_top),
        max_delta_edge_jaccard,
        max_edge_homogeneity_jaccard,
    )
    max_edge_heterogeneity_jaccard = max(
        jaccard_index(left_edge_top, heterogeneity_top),
        jaccard_index(right_edge_top, heterogeneity_top),
    )
    heterogeneity_consensus = min(
        jaccard_index(delta_top, heterogeneity_top),
        max_delta_edge_jaccard,
        max_edge_heterogeneity_jaccard,
    )
    contrast_norm = float(np.linalg.norm(delta))
    contrast_unit = contrast_norm / (1.0 + contrast_norm)
    branch_score = (
        max(homogeneity_gain_min, 0.0)
        * max(subspace_consensus, 0.0)
        * contrast_unit
    )
    barycentric_score = (
        max(heterogeneity_gain_max, 0.0)
        * max(1.0 - subspace_consensus, 0.0)
        * contrast_unit
    )
    return {
        "feature_geometry_node_sample_count": int(len(parent_sample_ids)),
        "feature_geometry_left_child_count": int(len(left_sample_ids)),
        "feature_geometry_right_child_count": int(len(right_sample_ids)),
        "feature_node_pairwise_jaccard": float(parent_pairwise),
        "feature_left_pairwise_jaccard": float(left_pairwise),
        "feature_right_pairwise_jaccard": float(right_pairwise),
        "feature_homogeneity_gain_min": float(homogeneity_gain_min),
        "feature_heterogeneity_gain_max": float(heterogeneity_gain_max),
        "feature_subspace_consensus_jaccard_topk": float(subspace_consensus),
        "feature_heterogeneity_subspace_consensus_jaccard_topk": float(
            heterogeneity_consensus
        ),
        "feature_child_contrast_norm": contrast_norm,
        "feature_branch_geometry_score": float(branch_score),
        "feature_barycentric_geometry_score": float(barycentric_score),
    }


def build_feature_geometry_rows(
    node_rows: pd.DataFrame,
    gene_assignments: pd.DataFrame,
    *,
    suite: str = "binary",
    top_k: int = 12,
    max_pairwise_samples: int = 200,
) -> pd.DataFrame:
    """Compute non-oracle feature geometry for selected pass-through rows."""
    if node_rows.empty:
        return pd.DataFrame()
    required = {
        "case_id",
        "data_role",
        "replicate",
        "node_id",
        "left_method_id",
        "data_seed",
        "truth_downstream_split_node_id",
    }
    missing = sorted(required - set(node_rows.columns))
    if missing:
        raise ValueError(f"node rows are missing feature-geometry columns: {missing!r}")
    data_by_key = _case_data_by_key(node_rows, suite=suite)
    assignments_by_key = _assignment_groups(gene_assignments)

    records: list[dict[str, object]] = []
    for _, row in node_rows.iterrows():
        base = _feature_geometry_empty_row(row, status="feature_geometry_unavailable")
        case_id = str(row["case_id"])
        data_role = str(row["data_role"])
        method_id = str(row["left_method_id"])
        replicate = int(float(row["replicate"]))
        node_id = str(row["node_id"])
        split_node_id = str(row["truth_downstream_split_node_id"])
        if not split_node_id or split_node_id.lower() in {"nan", "none", "null"}:
            base["feature_geometry_status"] = "feature_geometry_missing_downstream_split"
            records.append(base)
            continue
        data = data_by_key.get((case_id, data_role, replicate))
        assignments = assignments_by_key.get((case_id, data_role, method_id, replicate))
        if data is None:
            base["feature_geometry_status"] = "feature_geometry_missing_generated_data"
            records.append(base)
            continue
        if assignments is None or assignments.empty:
            base["feature_geometry_status"] = "feature_geometry_missing_assignments"
            records.append(base)
            continue

        parsed = assignments[["sample_id", "path_node_ids"]].copy()
        parsed["path"] = parsed["path_node_ids"].map(_split_path)
        selected = parsed.loc[parsed["path"].map(lambda path: _path_contains(path, node_id))]
        split = selected.loc[
            selected["path"].map(lambda path: _path_contains(path, split_node_id))
        ].copy()
        if split.empty:
            base["feature_geometry_status"] = "feature_geometry_split_not_in_selected_path"
            records.append(base)
            continue
        split["next_child"] = split["path"].map(
            lambda path: _next_path_node(path, split_node_id)
        )
        child_counts = (
            split.loc[split["next_child"].astype(str).ne(""), "next_child"]
            .astype(str)
            .value_counts()
        )
        if child_counts.shape[0] < 2:
            base["feature_geometry_status"] = "feature_geometry_downstream_children_missing"
            records.append(base)
            continue
        left_child, right_child = child_counts.index[:2].tolist()
        parent_sample_ids = [
            str(sample)
            for sample in split["sample_id"].astype(str)
            if str(sample) in data.index
        ]
        left_sample_ids = [
            str(sample)
            for sample in split.loc[
                split["next_child"].astype(str).eq(left_child), "sample_id"
            ].astype(str)
            if str(sample) in data.index
        ]
        right_sample_ids = [
            str(sample)
            for sample in split.loc[
                split["next_child"].astype(str).eq(right_child), "sample_id"
            ].astype(str)
            if str(sample) in data.index
        ]
        if len(parent_sample_ids) < 2 or not left_sample_ids or not right_sample_ids:
            base["feature_geometry_status"] = "feature_geometry_insufficient_samples"
            records.append(base)
            continue

        geometry = _feature_geometry_for_partition(
            data=data,
            parent_sample_ids=parent_sample_ids,
            left_sample_ids=left_sample_ids,
            right_sample_ids=right_sample_ids,
            top_k=int(top_k),
            max_pairwise_samples=int(max_pairwise_samples),
            seed=int(float(row["data_seed"])),
        )
        records.append(
            {
                **base,
                **geometry,
                "feature_geometry_status": "feature_geometry_observed",
            }
        )
    return pd.DataFrame.from_records(records)


def _auc(branch_values: np.ndarray, negative_values: np.ndarray) -> float:
    if branch_values.size == 0 or negative_values.size == 0:
        return math.nan
    greater = 0.0
    total = 0
    for branch in branch_values:
        greater += float(np.sum(branch > negative_values))
        greater += 0.5 * float(np.sum(branch == negative_values))
        total += int(negative_values.size)
    return float(greater / total) if total else math.nan


def _zero_negative(
    branch_values: np.ndarray,
    negative_values: np.ndarray,
    *,
    direction: str,
) -> tuple[float, int, float, int, float, str]:
    if branch_values.size == 0 or negative_values.size == 0:
        return math.nan, 0, math.nan, 0, math.nan, "zero_negative_undefined"
    threshold = (
        float(np.min(branch_values))
        if direction == "high"
        else float(np.max(branch_values))
    )
    if direction == "high":
        branch_selected = branch_values >= threshold
        negative_selected = negative_values >= threshold
        margin = threshold - float(np.max(negative_values))
    else:
        branch_selected = branch_values <= threshold
        negative_selected = negative_values <= threshold
        margin = float(np.min(negative_values)) - threshold
    branch_count = int(np.sum(branch_selected))
    negative_count = int(np.sum(negative_selected))
    retention = float(branch_count / branch_values.size)
    if negative_count == 0 and branch_count == int(branch_values.size):
        status = "zero_negative_separates_all_branch_recovery"
    elif negative_count == 0 and branch_count > 0:
        status = "zero_negative_partial_branch_retention"
    elif branch_count == 0:
        status = "zero_negative_no_branch_retention"
    else:
        status = "zero_negative_leakage"
    return threshold, branch_count, retention, negative_count, float(margin), status


def _truth_geometry_class(row: pd.Series) -> str:
    role = str(row.get("truth_geometry_role", ""))
    fixture_role = str(row.get("fixture_role", ""))
    data_role = str(row.get("data_role", ""))
    if role == "truth_recovery_pass_through_positive":
        return "full_branch_recovery"
    if role == "partial_truth_recovery_pass_through_candidate":
        return "partial_branch_recovery"
    if role == "barycentric_mixture_pass_through_candidate":
        return "barycentric_mixture"
    if role == "fragment_false_pass_through":
        return "false_fragment"
    if data_role == "selected_null" or fixture_role == "selected_null_pass_through_control":
        return "selected_null_control"
    if data_role == "signal":
        return "unresolved_signal"
    return "unclassified"


def _conditioning_status(truth_class: str) -> str:
    if truth_class == "full_branch_recovery":
        return "full_branch_recovery_oracle_positive"
    if truth_class == "partial_branch_recovery":
        return "partial_branch_recovery_oracle_candidate"
    if truth_class == "barycentric_mixture":
        return "barycentric_mixture_not_branch_recovery"
    if truth_class == "false_fragment":
        return "fragment_not_branch_recovery"
    if truth_class == "selected_null_control":
        return "selected_null_control_not_branch_recovery"
    if truth_class == "unresolved_signal":
        return "unresolved_signal_fail_closed"
    return "unclassified_fail_closed"


def _focused_row(
    *,
    case_id: str,
    truth_geometry_role: str,
    split_ari: float,
    child_purity: float,
    child_majority_distinct: bool,
    completed_balance_product: float,
    incoming: float,
    outgoing: float,
    node_majority: float,
    data_role: str = "signal",
    replicate: int = 0,
    feature_gain: float = math.nan,
    feature_subspace: float = math.nan,
    feature_contrast: float = math.nan,
    feature_heterogeneity: float = math.nan,
) -> dict[str, object]:
    if data_role == "selected_null":
        fixture_role = "selected_null_pass_through_control"
    elif truth_geometry_role == "truth_recovery_pass_through_positive":
        fixture_role = "truth_recovery_pass_through_positive"
    elif truth_geometry_role == "partial_truth_recovery_pass_through_candidate":
        fixture_role = "partial_truth_recovery_pass_through_candidate"
    elif truth_geometry_role == "barycentric_mixture_pass_through_candidate":
        fixture_role = "signal_barycentric_mixture_pass_through_candidate"
    elif truth_geometry_role == "fragment_false_pass_through":
        fixture_role = "fragment_false_pass_through"
    else:
        fixture_role = "signal_pass_through_candidate_unresolved"
    contrast_unit = (
        feature_contrast / (1.0 + feature_contrast)
        if math.isfinite(feature_contrast)
        else math.nan
    )
    branch_score = (
        max(feature_gain, 0.0) * max(feature_subspace, 0.0) * contrast_unit
        if math.isfinite(feature_gain)
        and math.isfinite(feature_subspace)
        and math.isfinite(contrast_unit)
        else math.nan
    )
    barycentric_score = (
        max(feature_heterogeneity, 0.0)
        * max(1.0 - feature_subspace, 0.0)
        * contrast_unit
        if math.isfinite(feature_heterogeneity)
        and math.isfinite(feature_subspace)
        and math.isfinite(contrast_unit)
        else math.nan
    )
    return {
        "case_id": case_id,
        "data_role": data_role,
        "replicate": int(replicate),
        "node_id": f"{case_id}_node",
        "fixture_role": fixture_role,
        "truth_geometry_role": truth_geometry_role,
        "truth_node_cluster_count": 4 if data_role == "signal" else 1,
        "truth_node_majority_fraction": float(node_majority),
        "truth_downstream_split_ari": float(split_ari),
        "truth_downstream_child_mean_purity": float(child_purity),
        "truth_downstream_child_majority_distinct": bool(child_majority_distinct),
        "completed_balance_product": float(completed_balance_product),
        "structural_incoming_branch_balance": float(incoming),
        "structural_outgoing_balance": float(outgoing),
        "structural_balance_product": float(incoming) * float(outgoing),
        "distance_to_downstream_accepted_split": 1.0,
        "feature_geometry_status": "focused_feature_geometry_observed",
        "feature_geometry_node_sample_count": 100,
        "feature_geometry_left_child_count": 50,
        "feature_geometry_right_child_count": 50,
        "feature_node_pairwise_jaccard": 0.22,
        "feature_left_pairwise_jaccard": 0.22 + max(feature_gain, 0.0)
        if math.isfinite(feature_gain)
        else math.nan,
        "feature_right_pairwise_jaccard": 0.22 + max(feature_gain, 0.0)
        if math.isfinite(feature_gain)
        else math.nan,
        "feature_homogeneity_gain_min": feature_gain,
        "feature_heterogeneity_gain_max": feature_heterogeneity,
        "feature_subspace_consensus_jaccard_topk": feature_subspace,
        "feature_heterogeneity_subspace_consensus_jaccard_topk": max(
            0.0,
            1.0 - feature_subspace,
        )
        if math.isfinite(feature_subspace)
        else math.nan,
        "feature_child_contrast_norm": feature_contrast,
        "feature_branch_geometry_score": branch_score,
        "feature_barycentric_geometry_score": barycentric_score,
    }


def build_focused_branch_recovery_fixture_rows() -> pd.DataFrame:
    """Return analytical branch, barycentric, fragment, and null rows."""
    return pd.DataFrame.from_records(
        [
            _focused_row(
                case_id="branch_full_balanced",
                truth_geometry_role="truth_recovery_pass_through_positive",
                split_ari=0.92,
                child_purity=0.94,
                child_majority_distinct=True,
                completed_balance_product=0.23,
                incoming=0.47,
                outgoing=0.49,
                node_majority=0.52,
                replicate=0,
                feature_gain=0.13,
                feature_subspace=0.86,
                feature_contrast=1.15,
                feature_heterogeneity=0.0,
            ),
            _focused_row(
                case_id="branch_full_unbalanced",
                truth_geometry_role="truth_recovery_pass_through_positive",
                split_ari=0.76,
                child_purity=0.86,
                child_majority_distinct=True,
                completed_balance_product=0.21,
                incoming=0.43,
                outgoing=0.49,
                node_majority=0.62,
                replicate=1,
                feature_gain=0.10,
                feature_subspace=0.78,
                feature_contrast=0.95,
                feature_heterogeneity=0.0,
            ),
            _focused_row(
                case_id="branch_full_deep",
                truth_geometry_role="truth_recovery_pass_through_positive",
                split_ari=0.68,
                child_purity=0.82,
                child_majority_distinct=True,
                completed_balance_product=0.24,
                incoming=0.49,
                outgoing=0.49,
                node_majority=0.59,
                replicate=2,
                feature_gain=0.09,
                feature_subspace=0.82,
                feature_contrast=0.90,
                feature_heterogeneity=0.0,
            ),
            _focused_row(
                case_id="branch_partial",
                truth_geometry_role="partial_truth_recovery_pass_through_candidate",
                split_ari=0.34,
                child_purity=0.74,
                child_majority_distinct=True,
                completed_balance_product=0.19,
                incoming=0.39,
                outgoing=0.49,
                node_majority=0.72,
                replicate=3,
                feature_gain=0.05,
                feature_subspace=0.62,
                feature_contrast=0.80,
                feature_heterogeneity=0.0,
            ),
            _focused_row(
                case_id="barycentric_mixture",
                truth_geometry_role="barycentric_mixture_pass_through_candidate",
                split_ari=0.36,
                child_purity=0.73,
                child_majority_distinct=False,
                completed_balance_product=0.052,
                incoming=0.32,
                outgoing=0.16,
                node_majority=0.81,
                replicate=4,
                feature_gain=-0.01,
                feature_subspace=0.12,
                feature_contrast=0.70,
                feature_heterogeneity=0.09,
            ),
            _focused_row(
                case_id="fragment_homogeneous",
                truth_geometry_role="fragment_false_pass_through",
                split_ari=0.0,
                child_purity=1.0,
                child_majority_distinct=False,
                completed_balance_product=0.005,
                incoming=0.10,
                outgoing=0.05,
                node_majority=1.0,
                replicate=5,
                feature_gain=0.0,
                feature_subspace=0.05,
                feature_contrast=0.10,
                feature_heterogeneity=0.0,
            ),
            _focused_row(
                case_id="fragment_misaligned",
                truth_geometry_role="fragment_false_pass_through",
                split_ari=0.02,
                child_purity=0.97,
                child_majority_distinct=False,
                completed_balance_product=0.026,
                incoming=0.20,
                outgoing=0.13,
                node_majority=0.94,
                replicate=6,
                feature_gain=0.01,
                feature_subspace=0.14,
                feature_contrast=0.35,
                feature_heterogeneity=0.02,
            ),
            _focused_row(
                case_id="selected_null_balanced_control",
                data_role="selected_null",
                truth_geometry_role="selected_null_truth_control",
                split_ari=0.0,
                child_purity=1.0,
                child_majority_distinct=False,
                completed_balance_product=0.22,
                incoming=0.44,
                outgoing=0.50,
                node_majority=1.0,
                replicate=7,
                feature_gain=0.01,
                feature_subspace=0.16,
                feature_contrast=0.45,
                feature_heterogeneity=0.03,
            ),
            _focused_row(
                case_id="selected_null_unbalanced_control",
                data_role="selected_null",
                truth_geometry_role="selected_null_truth_control",
                split_ari=0.0,
                child_purity=1.0,
                child_majority_distinct=False,
                completed_balance_product=0.13,
                incoming=0.36,
                outgoing=0.36,
                node_majority=1.0,
                replicate=8,
                feature_gain=0.02,
                feature_subspace=0.18,
                feature_contrast=0.40,
                feature_heterogeneity=0.02,
            ),
        ]
    )


def build_branch_recovery_conditioning_rows(node_rows: pd.DataFrame) -> pd.DataFrame:
    """Return selected pass-through rows with branch-recovery conditioning fields."""
    if node_rows.empty:
        return pd.DataFrame(columns=ROW_COLUMNS)
    missing = sorted(REQUIRED_NODE_COLUMNS - set(node_rows.columns))
    if missing:
        raise ValueError(f"node rows are missing columns: {missing!r}")

    records: list[dict[str, object]] = []
    for _, row in node_rows.iterrows():
        truth_class = _truth_geometry_class(row)
        distinct = _bool_value(row["truth_downstream_child_majority_distinct"])
        split_ari = _finite_float(row["truth_downstream_split_ari"])
        child_purity = _finite_float(row["truth_downstream_child_mean_purity"])
        branch_score = (
            split_ari * child_purity
            if math.isfinite(split_ari)
            and math.isfinite(child_purity)
            and distinct
            else math.nan
        )
        if math.isfinite(split_ari) and math.isfinite(child_purity) and not distinct:
            branch_score = 0.0
        barycentric_score = (
            split_ari * child_purity
            if math.isfinite(split_ari)
            and math.isfinite(child_purity)
            and not distinct
            else math.nan
        )
        if math.isfinite(split_ari) and math.isfinite(child_purity) and distinct:
            barycentric_score = 0.0
        feature_status = str(row.get("feature_geometry_status", "feature_geometry_not_provided"))
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "case_id": str(row["case_id"]),
                "data_role": str(row["data_role"]),
                "replicate": int(float(row["replicate"])),
                "node_id": str(row["node_id"]),
                "fixture_role": str(row["fixture_role"]),
                "truth_geometry_role": str(row["truth_geometry_role"]),
                "truth_geometry_class": truth_class,
                "branch_recovery_target": truth_class
                in {"full_branch_recovery", "partial_branch_recovery"},
                "full_branch_recovery_target": truth_class == "full_branch_recovery",
                "partial_branch_recovery_target": truth_class
                == "partial_branch_recovery",
                "barycentric_mixture_target": truth_class == "barycentric_mixture",
                "fragment_target": truth_class == "false_fragment",
                "selected_null_control_target": truth_class == "selected_null_control",
                "unresolved_signal_target": truth_class == "unresolved_signal",
                "truth_node_cluster_count": int(
                    _finite_float(row["truth_node_cluster_count"])
                ),
                "truth_node_majority_fraction": _finite_float(
                    row["truth_node_majority_fraction"]
                ),
                "truth_downstream_split_ari": split_ari,
                "truth_downstream_child_mean_purity": child_purity,
                "truth_downstream_child_majority_distinct": distinct,
                "truth_downstream_child_majority_distinct_numeric": float(
                    int(distinct)
                ),
                "completed_balance_product": _finite_float(
                    row["completed_balance_product"]
                ),
                "structural_incoming_branch_balance": _finite_float(
                    row["structural_incoming_branch_balance"]
                ),
                "structural_outgoing_balance": _finite_float(
                    row["structural_outgoing_balance"]
                ),
                "structural_balance_product": _finite_float(
                    row["structural_balance_product"]
                ),
                "distance_to_downstream_accepted_split": _finite_float(
                    row["distance_to_downstream_accepted_split"]
                ),
                "feature_geometry_status": feature_status,
                "feature_geometry_node_sample_count": int(
                    _finite_float(row.get("feature_geometry_node_sample_count", 0))
                    if math.isfinite(
                        _finite_float(row.get("feature_geometry_node_sample_count", 0))
                    )
                    else 0
                ),
                "feature_geometry_left_child_count": int(
                    _finite_float(row.get("feature_geometry_left_child_count", 0))
                    if math.isfinite(
                        _finite_float(row.get("feature_geometry_left_child_count", 0))
                    )
                    else 0
                ),
                "feature_geometry_right_child_count": int(
                    _finite_float(row.get("feature_geometry_right_child_count", 0))
                    if math.isfinite(
                        _finite_float(row.get("feature_geometry_right_child_count", 0))
                    )
                    else 0
                ),
                "feature_node_pairwise_jaccard": _finite_float(
                    row.get("feature_node_pairwise_jaccard", math.nan)
                ),
                "feature_left_pairwise_jaccard": _finite_float(
                    row.get("feature_left_pairwise_jaccard", math.nan)
                ),
                "feature_right_pairwise_jaccard": _finite_float(
                    row.get("feature_right_pairwise_jaccard", math.nan)
                ),
                "feature_homogeneity_gain_min": _finite_float(
                    row.get("feature_homogeneity_gain_min", math.nan)
                ),
                "feature_heterogeneity_gain_max": _finite_float(
                    row.get("feature_heterogeneity_gain_max", math.nan)
                ),
                "feature_subspace_consensus_jaccard_topk": _finite_float(
                    row.get("feature_subspace_consensus_jaccard_topk", math.nan)
                ),
                "feature_heterogeneity_subspace_consensus_jaccard_topk": (
                    _finite_float(
                        row.get(
                            "feature_heterogeneity_subspace_consensus_jaccard_topk",
                            math.nan,
                        )
                    )
                ),
                "feature_child_contrast_norm": _finite_float(
                    row.get("feature_child_contrast_norm", math.nan)
                ),
                "feature_branch_geometry_score": _finite_float(
                    row.get("feature_branch_geometry_score", math.nan)
                ),
                "feature_barycentric_geometry_score": _finite_float(
                    row.get("feature_barycentric_geometry_score", math.nan)
                ),
                "branch_recovery_oracle_score": branch_score,
                "barycentric_mixture_oracle_score": barycentric_score,
                "conditioning_status": _conditioning_status(truth_class),
            }
        )
    return pd.DataFrame.from_records(records, columns=ROW_COLUMNS)


def summarize_conditioning_metric(rows: pd.DataFrame, *, metric: str) -> dict[str, object]:
    """Summarize branch-vs-nonbranch separability for one metric."""
    metric_kind = "oracle" if metric in ORACLE_METRICS else "observable"
    branch_mask = rows["branch_recovery_target"].astype(bool)
    negative_mask = rows["truth_geometry_class"].isin(
        {"barycentric_mixture", "false_fragment", "selected_null_control"}
    )
    branch_values = _finite(rows.loc[branch_mask, metric])
    negative_values = _finite(rows.loc[negative_mask, metric])
    high_auc = _auc(branch_values, negative_values)
    low_auc = _auc(-branch_values, -negative_values)
    if math.isfinite(high_auc) and (
        not math.isfinite(low_auc) or high_auc >= low_auc
    ):
        best_direction = "high"
        best_auc = high_auc
    else:
        best_direction = "low"
        best_auc = low_auc

    high = _zero_negative(branch_values, negative_values, direction="high")
    low = _zero_negative(branch_values, negative_values, direction="low")
    chosen = high if high[3] <= low[3] else low
    direction = "high" if chosen is high else "low"
    threshold, branch_count, retention, negative_count, margin, status = chosen
    return {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "metric": metric,
        "metric_kind": metric_kind,
        "branch_count": int(branch_mask.sum()),
        "negative_count": int(negative_mask.sum()),
        "finite_branch_count": int(branch_values.size),
        "finite_negative_count": int(negative_values.size),
        "best_direction": best_direction,
        "best_auc": best_auc,
        "branch_min": float(np.min(branch_values)) if branch_values.size else math.nan,
        "branch_median": (
            float(np.median(branch_values)) if branch_values.size else math.nan
        ),
        "branch_max": float(np.max(branch_values)) if branch_values.size else math.nan,
        "negative_min": (
            float(np.min(negative_values)) if negative_values.size else math.nan
        ),
        "negative_median": (
            float(np.median(negative_values)) if negative_values.size else math.nan
        ),
        "negative_max": (
            float(np.max(negative_values)) if negative_values.size else math.nan
        ),
        "zero_negative_direction": direction,
        "zero_negative_threshold": threshold,
        "zero_negative_branch_count": branch_count,
        "zero_negative_branch_retention": retention,
        "zero_negative_negative_count": negative_count,
        "zero_negative_margin": margin,
        "zero_negative_status": status,
    }


def summarize_branch_recovery_conditioning_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Return metric summaries for observable and oracle conditioning terms."""
    if rows.empty:
        return pd.DataFrame(columns=METRIC_SUMMARY_COLUMNS)
    metrics = tuple(metric for metric in (*OBSERVABLE_METRICS, *ORACLE_METRICS) if metric in rows)
    return pd.DataFrame.from_records(
        [summarize_conditioning_metric(rows, metric=metric) for metric in metrics],
        columns=METRIC_SUMMARY_COLUMNS,
    )


def _best_metric(metric_summary: pd.DataFrame, metric_kind: str) -> pd.Series | None:
    subset = metric_summary.loc[metric_summary["metric_kind"].eq(metric_kind)].copy()
    if subset.empty:
        return None
    subset["separates"] = subset["zero_negative_status"].eq(
        "zero_negative_separates_all_branch_recovery"
    )
    return subset.sort_values(
        ["separates", "best_auc", "zero_negative_margin"],
        ascending=[False, False, False],
    ).iloc[0]


def summarize_branch_recovery_conditioning(
    rows: pd.DataFrame,
    metric_summary: pd.DataFrame,
    *,
    min_full_branch_recovery_count: int,
) -> pd.DataFrame:
    """Return one-row diagnostic summary for selected pass-through conditioning."""
    if rows.empty:
        return pd.DataFrame(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_role": STUDY_ROLE,
                    "row_count": 0,
                    "full_branch_recovery_count": 0,
                    "partial_branch_recovery_count": 0,
                    "branch_recovery_count": 0,
                    "barycentric_mixture_count": 0,
                    "fragment_count": 0,
                    "selected_null_control_count": 0,
                    "unresolved_signal_count": 0,
                    "best_observable_metric": "",
                    "best_observable_auc": math.nan,
                    "best_observable_zero_negative_status": "",
                    "best_oracle_metric": "",
                    "best_oracle_auc": math.nan,
                    "best_oracle_zero_negative_status": "",
                    "diagnostic_status": "no_selected_pass_through_truth_rows",
                    "next_required_step": "run_truth_labeled_selected_pass_through_miner",
                    "production_action": "fail_closed_until_branch_law_validated",
                }
            ],
            columns=SUMMARY_COLUMNS,
        )

    full_count = int(rows["full_branch_recovery_target"].astype(bool).sum())
    partial_count = int(rows["partial_branch_recovery_target"].astype(bool).sum())
    branch_count = int(rows["branch_recovery_target"].astype(bool).sum())
    barycentric_count = int(rows["barycentric_mixture_target"].astype(bool).sum())
    fragment_count = int(rows["fragment_target"].astype(bool).sum())
    selected_null_count = int(rows["selected_null_control_target"].astype(bool).sum())
    unresolved_count = int(rows["unresolved_signal_target"].astype(bool).sum())

    observable = _best_metric(metric_summary, "observable")
    oracle = _best_metric(metric_summary, "oracle")
    observable_status = "" if observable is None else str(observable["zero_negative_status"])
    oracle_status = "" if oracle is None else str(oracle["zero_negative_status"])

    if full_count < int(min_full_branch_recovery_count):
        status = "full_branch_recovery_support_missing"
        next_step = "generate_focused_full_branch_recovery_selected_pass_through_cases"
    elif observable_status != "zero_negative_separates_all_branch_recovery":
        status = "branch_recovery_observable_conditioning_gap"
        next_step = "derive_observable_branch_vs_barycentric_conditioning_variable"
    else:
        status = "branch_recovery_conditioning_fixture_observed_diagnostic_only"
        next_step = "validate_branch_conditioning_on_selected_null_controls"
    if (
        oracle_status == "zero_negative_separates_all_branch_recovery"
        and observable_status != "zero_negative_separates_all_branch_recovery"
    ):
        next_step = "replace_oracle_branch_indicator_with_observable_conditioning"

    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "study_role": STUDY_ROLE,
                "row_count": int(rows.shape[0]),
                "full_branch_recovery_count": full_count,
                "partial_branch_recovery_count": partial_count,
                "branch_recovery_count": branch_count,
                "barycentric_mixture_count": barycentric_count,
                "fragment_count": fragment_count,
                "selected_null_control_count": selected_null_count,
                "unresolved_signal_count": unresolved_count,
                "best_observable_metric": (
                    "" if observable is None else str(observable["metric"])
                ),
                "best_observable_auc": (
                    math.nan if observable is None else float(observable["best_auc"])
                ),
                "best_observable_zero_negative_status": observable_status,
                "best_oracle_metric": "" if oracle is None else str(oracle["metric"]),
                "best_oracle_auc": (
                    math.nan if oracle is None else float(oracle["best_auc"])
                ),
                "best_oracle_zero_negative_status": oracle_status,
                "diagnostic_status": status,
                "next_required_step": next_step,
                "production_action": "fail_closed_until_branch_law_validated",
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def run_selected_pass_through_branch_recovery_conditioning(
    config: SelectedPassThroughBranchRecoveryConditioningConfig,
) -> dict[str, Path]:
    """Run selected pass-through branch-recovery conditioning diagnostics."""
    if (
        config.node_rows_path is None
        and not bool(config.use_focused_fixture)
        and not bool(config.use_generated_support_fixture)
    ):
        raise ValueError("Provide node_rows_path or enable use_focused_fixture.")
    frames: list[pd.DataFrame] = []
    generated_gene_assignments: pd.DataFrame | None = None
    if config.node_rows_path is not None:
        frames.append(pd.read_csv(config.node_rows_path, keep_default_na=False))
    if bool(config.use_generated_support_fixture):
        generated_node_rows, generated_gene_assignments = (
            build_generated_branch_positive_support_fixture(
                suite=str(config.truth_suite),
            )
        )
        frames.append(generated_node_rows)
    if bool(config.use_focused_fixture):
        frames.append(build_focused_branch_recovery_fixture_rows())
    node_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    gene_assignment_frames: list[pd.DataFrame] = []
    if config.gene_assignments_path is not None:
        gene_assignment_frames.append(
            pd.read_csv(config.gene_assignments_path, low_memory=False)
        )
    if generated_gene_assignments is not None:
        gene_assignment_frames.append(generated_gene_assignments)
    if gene_assignment_frames and not node_rows.empty:
        gene_assignments = pd.concat(gene_assignment_frames, ignore_index=True)
        feature_rows = build_feature_geometry_rows(
            node_rows,
            gene_assignments,
            suite=str(config.truth_suite),
            top_k=int(config.top_k),
            max_pairwise_samples=int(config.max_pairwise_samples),
        )
        merge_keys = ["case_id", "data_role", "replicate", "node_id"]
        feature_columns = [
            column for column in feature_rows.columns if column not in merge_keys
        ]
        node_rows = node_rows.drop(
            columns=[column for column in feature_columns if column in node_rows],
            errors="ignore",
        ).merge(feature_rows, on=merge_keys, how="left")
    rows = build_branch_recovery_conditioning_rows(node_rows)
    metric_summary = summarize_branch_recovery_conditioning_metrics(rows)
    summary = summarize_branch_recovery_conditioning(
        rows,
        metric_summary,
        min_full_branch_recovery_count=int(config.min_full_branch_recovery_count),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    generated_outputs: dict[str, str] = {}
    if bool(config.use_generated_support_fixture):
        generated_source_rows = [
            frame
            for frame in frames
            if "left_method_id" in frame
            and frame["left_method_id"].astype(str).eq(GENERATED_SUPPORT_METHOD_ID).any()
        ]
        generated_node_rows = (
            pd.concat(generated_source_rows, ignore_index=True)
            if generated_source_rows
            else pd.DataFrame()
        )
        generated_node_rows.to_csv(config.generated_support_node_rows_path, index=False)
        if generated_gene_assignments is not None:
            generated_gene_assignments.to_csv(
                config.generated_support_gene_assignments_path,
                index=False,
            )
        generated_outputs = {
            "generated_support_node_rows": str(config.generated_support_node_rows_path),
            "generated_support_gene_assignments": str(
                config.generated_support_gene_assignments_path
            ),
        }
    rows.to_csv(config.rows_path, index=False)
    metric_summary.to_csv(config.metric_summary_path, index=False)
    summary.to_csv(config.summary_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_role": STUDY_ROLE,
        "generated_by": GENERATED_BY,
        "generated_at_utc": format_timestamp_utc(),
        "inputs": {
            "node_rows": None if config.node_rows_path is None else str(config.node_rows_path),
            "gene_assignments": (
                None
                if config.gene_assignments_path is None
                else str(config.gene_assignments_path)
            ),
            "use_focused_fixture": bool(config.use_focused_fixture),
            "use_generated_support_fixture": bool(config.use_generated_support_fixture),
            "truth_suite": str(config.truth_suite),
            "top_k": int(config.top_k),
            "max_pairwise_samples": int(config.max_pairwise_samples),
        },
        "outputs": {
            "rows": str(config.rows_path),
            "metric_summary": str(config.metric_summary_path),
            "summary": str(config.summary_path),
            **generated_outputs,
        },
        "production_status": str(summary["production_action"].iloc[0]),
    }
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = {
        "rows": config.rows_path,
        "metric_summary": config.metric_summary_path,
        "summary": config.summary_path,
        "manifest": config.manifest_path,
    }
    if bool(config.use_generated_support_fixture):
        outputs["generated_support_node_rows"] = config.generated_support_node_rows_path
        outputs["generated_support_gene_assignments"] = (
            config.generated_support_gene_assignments_path
        )
    return outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-rows-path", type=Path, default=None)
    parser.add_argument("--gene-assignments-path", type=Path, default=None)
    parser.add_argument("--use-focused-fixture", action="store_true")
    parser.add_argument("--use-generated-support-fixture", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-full-branch-recovery-count", type=int, default=2)
    parser.add_argument("--truth-suite", default="binary")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-pairwise-samples", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_selected_pass_through_branch_recovery_conditioning(
        SelectedPassThroughBranchRecoveryConditioningConfig(
            node_rows_path=args.node_rows_path,
            gene_assignments_path=args.gene_assignments_path,
            use_focused_fixture=bool(args.use_focused_fixture),
            use_generated_support_fixture=bool(args.use_generated_support_fixture),
            output_dir=args.output_dir,
            min_full_branch_recovery_count=int(args.min_full_branch_recovery_count),
            truth_suite=str(args.truth_suite),
            top_k=int(args.top_k),
            max_pairwise_samples=int(args.max_pairwise_samples),
        )
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
