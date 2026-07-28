"""Audit old tree-neighborhood sibling-null prior scores against strict support.

This module is diagnostic-only. It reconstructs the old null-prior
interpolation idea from the current explicit edge annotation columns and uses
it to describe cases where production calibration correctly refuses
selected-nonnull-only support. It does not install interpolated priors as a
production calibration path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.decomposition.gates.column_contracts import (
    EDGE_GATE_COLUMNS,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context
from benchmarks.shared.util.time import format_timestamp_utc

EPS = 1e-12
DISTANCE_INF = 1e12
STUDY_ROLE = "diagnostic_interpolated_sibling_null_prior_not_calibration"

DEFAULT_CASE_NAMES = (
    "binary_perfect_4c",
    "cat_highcard_20cat_4c",
    "overlap_heavy_4c_small_feat",
    "phylo_large_32taxa",
)


@dataclass(frozen=True)
class EdgeStatus:
    """Explicit child-parent edge state used by the diagnostic scorer."""

    tested: bool
    ancestor_blocked: bool
    significant: bool
    p_value_bh: float
    structural_dimension: float


@dataclass(frozen=True)
class AncestorSupport:
    """Nearest stopped ancestor edge used as support for an untested child edge."""

    node: object
    p_value_bh: float
    distance: float


@dataclass(frozen=True)
class ReferenceSets:
    """Stable and signal edge neighborhoods for diagnostic interpolation."""

    stable_nodes: tuple[object, ...]
    stable_p_values: np.ndarray
    stable_log_dimensions: np.ndarray
    dimensionless_stable_nodes: tuple[object, ...]
    signal_nodes: tuple[object, ...]
    signal_p_values: np.ndarray


@dataclass(frozen=True)
class KernelBandwidths:
    """Tree-specific kernel scales for the diagnostic interpolation score."""

    ancestor_distance: float
    tree_distance: float
    signal_distance: float
    log_dimension: float


@dataclass(frozen=True)
class ChildPriorAudit:
    """Diagnostic interpolated-prior result for one child edge."""

    prior: float
    status: str
    ancestor_support: float
    ancestor_distance: float
    stable_share: float
    signal_penalty: float


@dataclass(frozen=True)
class CaseAuditTables:
    """Per-case diagnostic audit tables."""

    case_summary: pd.DataFrame
    record_audit: pd.DataFrame


def parse_csv_list(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("case_names must contain at least one case name.")
    return values


def _selected_cases(case_names: Sequence[str]) -> list[dict[str, object]]:
    case_by_name = {str(case["name"]): case for case in get_default_test_cases()}
    missing = [case_name for case_name in case_names if case_name not in case_by_name]
    if missing:
        raise ValueError(f"Unknown benchmark case name(s): {missing!r}.")
    return [case_by_name[case_name].copy() for case_name in case_names]


def _validate_edge_table(edge_df: pd.DataFrame) -> None:
    missing = [column for column in EDGE_GATE_COLUMNS if column not in edge_df.columns]
    if missing:
        raise ValueError(f"Edge annotation table is missing columns: {missing!r}.")


def _finite_probability(value: object, *, label: str) -> float:
    p_value = float(value)
    if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ValueError(f"{label} must be a finite probability in [0, 1]; got {value!r}.")
    return p_value


def _edge_status(edge_df: pd.DataFrame, child: object) -> EdgeStatus:
    if child not in edge_df.index:
        raise KeyError(f"Missing edge annotation row for child node {child!r}.")
    p_value = float(edge_df.at[child, "Child_Parent_Divergence_P_Value_BH"])
    if np.isfinite(p_value):
        p_value = _finite_probability(
            p_value,
            label=f"Child_Parent_Divergence_P_Value_BH for child {child!r}",
        )
    structural_dimension = float(edge_df.at[child, "Child_Parent_Divergence_df"])
    return EdgeStatus(
        tested=bool(edge_df.at[child, "Child_Parent_Divergence_Tested"]),
        ancestor_blocked=bool(edge_df.at[child, "Child_Parent_Divergence_Ancestor_Blocked"]),
        significant=bool(edge_df.at[child, "Child_Parent_Divergence_Significant"]),
        p_value_bh=p_value,
        structural_dimension=structural_dimension,
    )


def _tree_distance_function(tree: nx.DiGraph) -> Callable[[object, object], float]:
    undirected_tree = tree.to_undirected(as_view=True)

    @lru_cache(maxsize=None)
    def distance(node_a: object, node_b: object) -> float:
        if node_a == node_b:
            return 0.0
        try:
            return float(nx.shortest_path_length(undirected_tree, node_a, node_b))
        except nx.NetworkXNoPath:
            return DISTANCE_INF

    return distance


def _single_parent(tree: nx.DiGraph, node: object) -> object | None:
    parents = tuple(tree.predecessors(node))
    if len(parents) > 1:
        raise ValueError(f"Expected a tree node to have at most one parent; node={node!r}.")
    return parents[0] if parents else None


def nearest_stopped_ancestor_support(
    tree: nx.DiGraph,
    edge_df: pd.DataFrame,
    child: object,
) -> AncestorSupport | None:
    """Return the nearest tested non-significant ancestor with a finite BH p-value."""
    current = child
    distance = 0.0
    while True:
        parent = _single_parent(tree, current)
        if parent is None:
            return None
        distance += 1.0
        current = parent
        if current not in edge_df.index:
            continue
        status = _edge_status(edge_df, current)
        if not status.tested or status.significant or not np.isfinite(status.p_value_bh):
            continue
        return AncestorSupport(
            node=current,
            p_value_bh=status.p_value_bh,
            distance=distance,
        )


def collect_reference_sets(edge_df: pd.DataFrame) -> ReferenceSets:
    """Collect tested stable/signal child-parent edges from explicit columns."""
    _validate_edge_table(edge_df)
    stable_nodes: list[object] = []
    stable_p_values: list[float] = []
    stable_log_dimensions: list[float] = []
    dimensionless_stable_nodes: list[object] = []
    signal_nodes: list[object] = []
    signal_p_values: list[float] = []

    for node in edge_df.index:
        status = _edge_status(edge_df, node)
        if not status.tested or not np.isfinite(status.p_value_bh):
            continue
        if status.significant:
            signal_nodes.append(node)
            signal_p_values.append(status.p_value_bh)
            continue
        if np.isfinite(status.structural_dimension) and status.structural_dimension > 0.0:
            stable_nodes.append(node)
            stable_p_values.append(status.p_value_bh)
            stable_log_dimensions.append(float(np.log(status.structural_dimension)))
        else:
            dimensionless_stable_nodes.append(node)

    return ReferenceSets(
        stable_nodes=tuple(stable_nodes),
        stable_p_values=np.asarray(stable_p_values, dtype=float),
        stable_log_dimensions=np.asarray(stable_log_dimensions, dtype=float),
        dimensionless_stable_nodes=tuple(dimensionless_stable_nodes),
        signal_nodes=tuple(signal_nodes),
        signal_p_values=np.asarray(signal_p_values, dtype=float),
    )


def _positive_median(values: Iterable[float]) -> float:
    positive_values = [float(value) for value in values if np.isfinite(value) and value > 0.0]
    if not positive_values:
        return 1.0
    return max(float(np.median(positive_values)), EPS)


def compute_kernel_bandwidths(
    *,
    tree: nx.DiGraph,
    edge_df: pd.DataFrame,
    reference_sets: ReferenceSets,
    tree_distance: Callable[[object, object], float],
) -> KernelBandwidths:
    """Infer deterministic diagnostic kernel scales from current edge evidence."""
    stopped_distances = []
    for node in edge_df.index:
        support = nearest_stopped_ancestor_support(tree, edge_df, node)
        if support is not None:
            stopped_distances.append(support.distance)

    nearest_stable_distances: list[float] = []
    if reference_sets.stable_nodes:
        for node in edge_df.index:
            nearest_stable_distances.append(
                min(tree_distance(node, stable_node) for stable_node in reference_sets.stable_nodes)
            )

    nearest_signal_distances: list[float] = []
    if reference_sets.signal_nodes:
        for node in edge_df.index:
            nearest_signal_distances.append(
                min(tree_distance(node, signal_node) for signal_node in reference_sets.signal_nodes)
            )

    log_dimension_bandwidth = (
        float(np.std(reference_sets.stable_log_dimensions))
        if len(reference_sets.stable_log_dimensions) > 1
        else 0.0
    )
    return KernelBandwidths(
        ancestor_distance=_positive_median(stopped_distances),
        tree_distance=_positive_median(nearest_stable_distances),
        signal_distance=_positive_median(nearest_signal_distances),
        log_dimension=max(log_dimension_bandwidth, 0.0),
    )


def _structural_kernel(
    source_log_dimensions: np.ndarray,
    target_log_dimension: float,
    bandwidth: float,
) -> np.ndarray:
    if bandwidth <= 0.0:
        return np.where(
            np.isclose(source_log_dimensions, target_log_dimension, atol=EPS),
            1.0,
            0.0,
        )
    return np.exp(-0.5 * ((source_log_dimensions - target_log_dimension) / bandwidth) ** 2)


def diagnostic_child_interpolated_prior(
    *,
    child: object,
    tree: nx.DiGraph,
    edge_df: pd.DataFrame,
    reference_sets: ReferenceSets,
    kernel_bandwidths: KernelBandwidths,
    tree_distance: Callable[[object, object], float],
) -> ChildPriorAudit:
    """Return the diagnostic old-style prior for one child edge."""
    status = _edge_status(edge_df, child)
    if status.tested and not status.ancestor_blocked:
        if not np.isfinite(status.p_value_bh):
            raise ValueError(
                f"Tested non-blocked child edges require finite Tree-BH p-values; child={child!r}."
            )
        return ChildPriorAudit(
            prior=status.p_value_bh,
            status="direct_tested_edge_p_value",
            ancestor_support=float("nan"),
            ancestor_distance=float("nan"),
            stable_share=0.0,
            signal_penalty=0.0,
        )

    ancestor_support = nearest_stopped_ancestor_support(tree, edge_df, child)
    if ancestor_support is None:
        return ChildPriorAudit(
            prior=float("nan"),
            status="unsupported_no_stopped_ancestor",
            ancestor_support=float("nan"),
            ancestor_distance=float("nan"),
            stable_share=float("nan"),
            signal_penalty=float("nan"),
        )
    if not np.isfinite(status.structural_dimension) or status.structural_dimension <= 0.0:
        return ChildPriorAudit(
            prior=float("nan"),
            status="unsupported_no_structural_dimension",
            ancestor_support=ancestor_support.p_value_bh,
            ancestor_distance=ancestor_support.distance,
            stable_share=float("nan"),
            signal_penalty=float("nan"),
        )

    ancestor_kernel = float(
        np.exp(-max(ancestor_support.distance - 1.0, 0.0) / kernel_bandwidths.ancestor_distance)
    )
    child_log_dimension = float(np.log(status.structural_dimension))

    stable_mass = 0.0
    stable_support = ancestor_support.p_value_bh
    if reference_sets.stable_nodes:
        stable_distances = np.asarray(
            [tree_distance(child, stable_node) for stable_node in reference_sets.stable_nodes],
            dtype=float,
        )
        tree_weights = np.exp(-stable_distances / kernel_bandwidths.tree_distance)
        structural_weights = _structural_kernel(
            reference_sets.stable_log_dimensions,
            child_log_dimension,
            kernel_bandwidths.log_dimension,
        )
        stable_weights = tree_weights * structural_weights
        stable_mass = float(np.sum(stable_weights))
        if stable_mass > 0.0:
            stable_support = float(
                np.average(reference_sets.stable_p_values, weights=stable_weights)
            )

    support_denominator = ancestor_kernel + stable_mass
    support = float(
        (ancestor_kernel * ancestor_support.p_value_bh + stable_mass * stable_support)
        / support_denominator
    )

    signal_penalty = 0.0
    if reference_sets.signal_nodes:
        signal_distances = np.asarray(
            [tree_distance(child, signal_node) for signal_node in reference_sets.signal_nodes],
            dtype=float,
        )
        signal_terms = (1.0 - reference_sets.signal_p_values) * np.exp(
            -signal_distances / kernel_bandwidths.signal_distance
        )
        signal_penalty = float(np.max(signal_terms))

    prior = float(np.clip(support * (1.0 - signal_penalty), 0.0, 1.0))
    stable_share = float(stable_mass / support_denominator)
    return ChildPriorAudit(
        prior=prior,
        status="interpolated_from_stopped_ancestor",
        ancestor_support=ancestor_support.p_value_bh,
        ancestor_distance=ancestor_support.distance,
        stable_share=stable_share,
        signal_penalty=signal_penalty,
    )


def diagnostic_pair_interpolated_prior(
    record: SiblingPairRecord,
    *,
    tree: nx.DiGraph,
    edge_df: pd.DataFrame,
    reference_sets: ReferenceSets,
    kernel_bandwidths: KernelBandwidths,
    tree_distance: Callable[[object, object], float],
) -> tuple[ChildPriorAudit, ChildPriorAudit, float, str]:
    left = diagnostic_child_interpolated_prior(
        child=record.left,
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        kernel_bandwidths=kernel_bandwidths,
        tree_distance=tree_distance,
    )
    right = diagnostic_child_interpolated_prior(
        child=record.right,
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        kernel_bandwidths=kernel_bandwidths,
        tree_distance=tree_distance,
    )
    if np.isfinite(left.prior) and np.isfinite(right.prior):
        return left, right, float(left.prior * right.prior), "ok"
    return left, right, float("nan"), "unsupported_child_prior"


def _reference_expectation(record: SiblingPairRecord) -> float:
    return float(record.reference_scale * record.degrees_of_freedom)


def _weighted_inflation_mle(
    records: Sequence[SiblingPairRecord],
    weights: np.ndarray,
) -> float:
    statistics = np.asarray([record.stat for record in records], dtype=float)
    reference_expectations = np.asarray(
        [_reference_expectation(record) for record in records],
        dtype=float,
    )
    return float(np.sum(weights * statistics) / np.sum(weights * reference_expectations))


def _candidate_inflation(records: Sequence[SiblingPairRecord], weights: np.ndarray) -> float:
    valid = np.asarray(
        [
            (
                record.degrees_of_freedom > 0.0
                and np.isfinite(record.stat)
                and np.isfinite(_reference_expectation(record))
                and _reference_expectation(record) > 0.0
                and np.isfinite(weight)
                and weight > 0.0
            )
            for record, weight in zip(records, weights, strict=True)
        ],
        dtype=bool,
    )
    if not np.any(valid):
        return float("nan")
    valid_records = [record for record, keep in zip(records, valid, strict=True) if keep]
    valid_weights = weights[valid]
    return float(max(_weighted_inflation_mle(valid_records, valid_weights), 1.0))


def _collect_records_for_case(
    case: dict[str, object],
) -> tuple[
    str,
    str,
    str,
    str,
    nx.DiGraph,
    pd.DataFrame,
    tuple[SiblingPairRecord, ...],
]:
    context = build_tbs_tree_context(case, populate_node_distributions=True)
    edge_df, spectral_context = annotate_child_parent_divergence(
        context.tree,
        context.tree.annotations_df,
        significance_level_alpha=DEFAULT_EDGE_ALPHA,
        leaf_data=context.data,
        feature_space=context.feature_space,
    )
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        context.tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    records, _non_binary_nodes = collect_sibling_pair_records(
        context.tree,
        edge_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=context.feature_space,
    )
    feature_family = (
        "bernoulli" if context.feature_space is None else context.feature_space.family_label
    )
    return (
        str(context.metadata["name"]),
        str(context.metadata["generator"]),
        context.tree_distance_metric,
        context.tree_distance_source,
        context.tree,
        edge_df,
        tuple(records),
        feature_family,
    )


def _record_row(
    *,
    case_id: str,
    generator: str,
    feature_family: str,
    tree_distance_metric: str,
    tree_distance_source: str,
    record: SiblingPairRecord,
    left_prior: ChildPriorAudit,
    right_prior: ChildPriorAudit,
    pair_prior: float,
    interpolation_status: str,
) -> dict[str, object]:
    strict_supported = bool(record.is_null_like or record.is_edge_blocked)
    selected_nonnull_positive_weight = bool(
        record.degrees_of_freedom > 0.0
        and record.sibling_null_weight > 0.0
        and not strict_supported
    )
    return {
        "case_id": case_id,
        "generator": generator,
        "feature_family": feature_family,
        "tree_distance_metric": tree_distance_metric,
        "tree_distance_source": tree_distance_source,
        "parent": record.parent,
        "left": record.left,
        "right": record.right,
        "stat": record.stat,
        "reference_expectation": _reference_expectation(record),
        "degrees_of_freedom": record.degrees_of_freedom,
        "raw_sibling_p_value": record.p_value,
        "current_sibling_null_weight": record.sibling_null_weight,
        "strict_internal_support": strict_supported,
        "is_null_like": record.is_null_like,
        "is_edge_blocked": record.is_edge_blocked,
        "selected_nonnull_positive_weight": selected_nonnull_positive_weight,
        "diagnostic_interpolated_pair_prior": pair_prior,
        "interpolation_status": interpolation_status,
        "left_interpolated_prior": left_prior.prior,
        "right_interpolated_prior": right_prior.prior,
        "left_prior_status": left_prior.status,
        "right_prior_status": right_prior.status,
        "left_ancestor_support": left_prior.ancestor_support,
        "right_ancestor_support": right_prior.ancestor_support,
        "left_ancestor_distance": left_prior.ancestor_distance,
        "right_ancestor_distance": right_prior.ancestor_distance,
        "left_stable_share": left_prior.stable_share,
        "right_stable_share": right_prior.stable_share,
        "left_signal_penalty": left_prior.signal_penalty,
        "right_signal_penalty": right_prior.signal_penalty,
    }


def audit_case(case: dict[str, object]) -> CaseAuditTables:
    """Run the diagnostic interpolation audit for one benchmark case."""
    (
        case_id,
        generator,
        tree_distance_metric,
        tree_distance_source,
        tree,
        edge_df,
        records,
        feature_family,
    ) = _collect_records_for_case(case)
    _validate_edge_table(edge_df)
    tree_distance = _tree_distance_function(tree)
    reference_sets = collect_reference_sets(edge_df)
    kernel_bandwidths = compute_kernel_bandwidths(
        tree=tree,
        edge_df=edge_df,
        reference_sets=reference_sets,
        tree_distance=tree_distance,
    )

    record_rows: list[dict[str, object]] = []
    interpolated_pair_priors: list[float] = []
    for record in records:
        left_prior, right_prior, pair_prior, interpolation_status = (
            diagnostic_pair_interpolated_prior(
                record,
                tree=tree,
                edge_df=edge_df,
                reference_sets=reference_sets,
                kernel_bandwidths=kernel_bandwidths,
                tree_distance=tree_distance,
            )
        )
        interpolated_pair_priors.append(pair_prior)
        record_rows.append(
            _record_row(
                case_id=case_id,
                generator=generator,
                feature_family=feature_family,
                tree_distance_metric=tree_distance_metric,
                tree_distance_source=tree_distance_source,
                record=record,
                left_prior=left_prior,
                right_prior=right_prior,
                pair_prior=pair_prior,
                interpolation_status=interpolation_status,
            )
        )

    record_audit = pd.DataFrame(record_rows)
    weights = np.asarray(interpolated_pair_priors, dtype=float)
    current_weights = np.asarray([record.sibling_null_weight for record in records], dtype=float)
    strict_supported_mask = np.asarray(
        [
            record.degrees_of_freedom > 0.0
            and record.sibling_null_weight > 0.0
            and (record.is_null_like or record.is_edge_blocked)
            for record in records
        ],
        dtype=bool,
    )
    selected_nonnull_positive_mask = np.asarray(
        [
            record.degrees_of_freedom > 0.0
            and record.sibling_null_weight > 0.0
            and not (record.is_null_like or record.is_edge_blocked)
            for record in records
        ],
        dtype=bool,
    )
    strict_records = [
        record for record, keep in zip(records, strict_supported_mask, strict=True) if keep
    ]
    strict_weights = current_weights[strict_supported_mask]

    summary_row = {
        "case_id": case_id,
        "generator": generator,
        "feature_family": feature_family,
        "tree_distance_metric": tree_distance_metric,
        "tree_distance_source": tree_distance_source,
        "status": "ok",
        "error_message": "",
        "n_records": len(records),
        "n_positive_df_records": int(sum(record.degrees_of_freedom > 0.0 for record in records)),
        "n_current_positive_weight_records": int(
            sum(
                record.degrees_of_freedom > 0.0 and record.sibling_null_weight > 0.0
                for record in records
            )
        ),
        "n_strict_supported_records": int(np.sum(strict_supported_mask)),
        "n_selected_nonnull_positive_weight_records": int(np.sum(selected_nonnull_positive_mask)),
        "n_interpolated_positive_weight_records": int(
            sum(np.isfinite(weight) and weight > 0.0 for weight in weights)
        ),
        "n_selected_nonnull_with_interpolated_weight": int(
            np.sum(selected_nonnull_positive_mask & np.isfinite(weights) & (weights > 0.0))
        ),
        "strict_supported_c_hat": _candidate_inflation(strict_records, strict_weights)
        if strict_records
        else float("nan"),
        "diagnostic_interpolated_c_hat": _candidate_inflation(records, weights),
        "strict_support_status": "supported"
        if np.any(strict_supported_mask)
        else "no_strict_internal_support",
        "diagnostic_role": STUDY_ROLE,
        "n_stable_reference_edges": len(reference_sets.stable_nodes),
        "n_dimensionless_stable_reference_edges": len(reference_sets.dimensionless_stable_nodes),
        "n_signal_reference_edges": len(reference_sets.signal_nodes),
        "kernel_ancestor_distance": kernel_bandwidths.ancestor_distance,
        "kernel_tree_distance": kernel_bandwidths.tree_distance,
        "kernel_signal_distance": kernel_bandwidths.signal_distance,
        "kernel_log_dimension": kernel_bandwidths.log_dimension,
    }
    return CaseAuditTables(
        case_summary=pd.DataFrame([summary_row]),
        record_audit=record_audit,
    )


def run_sibling_null_prior_interpolation_audit(
    *,
    case_names: Sequence[str],
    output_dir: Path,
) -> CaseAuditTables:
    """Run the diagnostic audit and write CSV/manifest outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    summary_frames: list[pd.DataFrame] = []
    record_frames: list[pd.DataFrame] = []
    for case in _selected_cases(case_names):
        case_id = str(case["name"])
        try:
            tables = audit_case(case)
        except Exception as exc:
            summary_frames.append(
                pd.DataFrame(
                    [
                        {
                            "case_id": case_id,
                            "generator": str(case.get("generator", "")),
                            "feature_family": "",
                            "tree_distance_metric": "",
                            "tree_distance_source": "",
                            "status": "error",
                            "error_message": str(exc),
                            "diagnostic_role": STUDY_ROLE,
                        }
                    ]
                )
            )
            continue
        summary_frames.append(tables.case_summary)
        record_frames.append(tables.record_audit)

    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    records = (
        pd.concat(record_frames, ignore_index=True, sort=False) if record_frames else pd.DataFrame()
    )
    summary_path = output_dir / "case_summary.csv"
    records_path = output_dir / "record_interpolation_audit.csv"
    manifest_path = output_dir / "manifest.json"
    summary.to_csv(summary_path, index=False)
    records.to_csv(records_path, index=False)
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "case_names": list(case_names),
        "case_summary_csv": str(summary_path),
        "record_interpolation_audit_csv": str(records_path),
        "elapsed_seconds": perf_counter() - started,
        "production_method_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return CaseAuditTables(case_summary=summary, record_audit=records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict sibling calibration support with diagnostic old-style "
            "tree-neighborhood interpolated null-prior scores."
        )
    )
    parser.add_argument("--case-names", default=",".join(DEFAULT_CASE_NAMES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/sibling_null_prior_interpolation_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tables = run_sibling_null_prior_interpolation_audit(
        case_names=parse_csv_list(str(args.case_names)),
        output_dir=Path(args.output_dir),
    )
    print(tables.case_summary.to_string(index=False))


if __name__ == "__main__":
    main()
