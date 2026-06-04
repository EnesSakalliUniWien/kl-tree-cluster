"""Sibling empirical-null inflation diagnostics for oracle-recoverable failures."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.annotation_bundle import (
    GateAnnotationBundle,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    fit_empirical_null_inflation_model,
    predict_empirical_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.types.inflation_model import (
    EmpiricalNullInflationModel,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace
from scipy.special import logsumexp
from scipy.stats import chi2


@dataclass(frozen=True)
class SiblingInflationInputs:
    """Sibling records and fitted empirical-null inflation model for one tree."""

    records: tuple[SiblingPairRecord, ...]
    non_binary_nodes: tuple[object, ...]
    model: EmpiricalNullInflationModel | None


@dataclass(frozen=True)
class SiblingInflationDiagnosticTables:
    """Output tables for a sibling-inflation diagnostic run."""

    targets: pd.DataFrame
    contributors: pd.DataFrame
    summary: pd.DataFrame


def collect_sibling_inflation_inputs(
    tree: nx.DiGraph,
    gate_annotation_bundle: GateAnnotationBundle,
    *,
    feature_space: FeatureSpace | None,
) -> SiblingInflationInputs:
    """Collect sibling records and fit the runtime empirical-null model."""
    edge_gate_result = gate_annotation_bundle.edge_gate_result

    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=edge_gate_result.spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=edge_gate_result.spectral_context,
        )
    )
    records, non_binary_nodes = collect_sibling_pair_records(
        tree,
        edge_gate_result.annotated_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )
    n_focal = sum(not record.is_null_like for record in records)
    model = fit_empirical_null_inflation_model(records) if n_focal > 0 else None
    return SiblingInflationInputs(
        records=tuple(records),
        non_binary_nodes=tuple(non_binary_nodes),
        model=model,
    )


def _reference_expectation(record: SiblingPairRecord) -> float:
    return float(record.reference_scale * record.degrees_of_freedom)


def _weighted_inflation_mle(
    statistics: np.ndarray,
    reference_expectations: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights * statistics) / np.sum(weights * reference_expectations))


def _inflation_adjusted_p_value(record: SiblingPairRecord, inflation_factor: float) -> float:
    if record.degrees_of_freedom == 0:
        return 1.0
    adjusted_statistic = float(record.stat / (record.reference_scale * inflation_factor))
    return float(chi2.sf(adjusted_statistic, df=float(record.degrees_of_freedom)))


def _inflation_factor_at_alpha(record: SiblingPairRecord, alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    if record.degrees_of_freedom == 0 or record.stat == 0.0:
        return float("nan")
    critical_value = float(chi2.isf(alpha, df=float(record.degrees_of_freedom)))
    return float(record.stat / (record.reference_scale * critical_value))


def _effective_sample_size(weights: np.ndarray) -> float:
    positive_weights = weights[weights > 0.0]
    if positive_weights.size == 0:
        raise ValueError("Effective sample size requires positive weights.")
    log_weights = np.log(positive_weights)
    return float(np.exp(2.0 * logsumexp(log_weights) - logsumexp(2.0 * log_weights)))


def _calibration_records(records: tuple[SiblingPairRecord, ...]) -> tuple[SiblingPairRecord, ...]:
    return tuple(
        record
        for record in records
        if record.degrees_of_freedom > 0.0
        and record.sibling_null_weight > 0.0
        and (record.is_null_like or record.is_edge_blocked)
    )


def _validate_model_record_alignment(
    model: EmpiricalNullInflationModel,
    calibration_records: tuple[SiblingPairRecord, ...],
) -> None:
    if len(calibration_records) != model.n_calibration:
        raise ValueError(
            "Inflation diagnostic model does not align with calibration records: "
            f"{model.n_calibration} model rows but {len(calibration_records)} records."
        )
    for index, record in enumerate(calibration_records):
        if not np.isclose(model.sample_statistics[index], record.stat):
            raise ValueError(
                "Inflation diagnostic model statistic does not align with record "
                f"at index {index}; parent={record.parent!r}."
            )
        if not np.isclose(model.sample_degrees_of_freedom[index], record.degrees_of_freedom):
            raise ValueError(
                "Inflation diagnostic model degrees of freedom do not align with record "
                f"at index {index}; parent={record.parent!r}."
            )
        if model.sample_feature_families[index] != record.feature_family:
            raise ValueError(
                "Inflation diagnostic model feature family does not align with record "
                f"at index {index}; parent={record.parent!r}."
            )


def _family_mask(model: EmpiricalNullInflationModel, record: SiblingPairRecord) -> np.ndarray:
    mask = np.array(
        [
            feature_family == record.feature_family
            for feature_family in model.sample_feature_families
        ],
        dtype=bool,
    )
    if not np.any(mask):
        raise ValueError(
            "Sibling inflation diagnostic has no calibration records for "
            f"feature_family={record.feature_family!r}."
        )
    return mask


def _family_baseline_inflation(
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
) -> float:
    mask = _family_mask(model, record)
    reference_expectations = model.sample_reference_scales[mask] * (
        model.sample_degrees_of_freedom[mask]
    )
    return float(
        max(
            _weighted_inflation_mle(
                model.sample_statistics[mask],
                reference_expectations,
                model.sample_weights[mask],
            ),
            1.0,
        )
    )


def _local_calibration_weights(
    model: EmpiricalNullInflationModel,
    record: SiblingPairRecord,
) -> tuple[np.ndarray, np.ndarray]:
    mask = _family_mask(model, record)
    active_context_axes = np.array([True, record.feature_family != "bernoulli"])
    active_context_axes = active_context_axes & (model.context_bandwidth > 0.0)
    if not np.any(active_context_axes):
        return mask, model.sample_weights[mask]

    if record.n_parent <= 0:
        raise ValueError(
            "Sibling inflation diagnostic requires positive parent sample size; "
            f"parent={record.parent!r}."
        )
    target_context = np.array(
        [
            np.log(record.sibling_projection_dimension),
            np.log(float(record.n_parent)),
        ],
        dtype=float,
    )
    family_contexts = model.sample_contexts[mask]
    scaled_offsets = (
        family_contexts[:, active_context_axes] - target_context[active_context_axes]
    ) / model.context_bandwidth[active_context_axes]
    log_kernel_weights = -0.5 * np.sum(scaled_offsets**2, axis=1)
    log_kernel_weights = log_kernel_weights - float(np.max(log_kernel_weights))
    local_weights = model.sample_weights[mask] * np.exp(log_kernel_weights)
    if float(np.sum(local_weights)) <= 0.0:
        raise ValueError(
            "Sibling inflation diagnostic has zero local calibration weight; "
            f"parent={record.parent!r}."
        )
    return mask, local_weights


def _local_inflation_from_weights(
    model: EmpiricalNullInflationModel,
    mask: np.ndarray,
    local_weights: np.ndarray,
) -> float:
    reference_expectations = model.sample_reference_scales[mask] * (
        model.sample_degrees_of_freedom[mask]
    )
    return float(
        max(
            _weighted_inflation_mle(
                model.sample_statistics[mask],
                reference_expectations,
                local_weights,
            ),
            1.0,
        )
    )


def _variant_unavailable(prefix: str, status: str) -> dict[str, object]:
    return {
        f"{prefix}_status": status,
        f"{prefix}_inflation_factor": np.nan,
        f"{prefix}_p_value": np.nan,
        f"{prefix}_blocks_at_alpha": False,
        f"{prefix}_inflation_excess_ratio": np.nan,
        f"{prefix}_effective_sample_size": np.nan,
        f"{prefix}_calibration_records": 0,
        f"{prefix}_calibration_weight_sum": 0.0,
    }


def _variant_result(
    *,
    prefix: str,
    target_record: SiblingPairRecord,
    model: EmpiricalNullInflationModel,
    calibration_records: tuple[SiblingPairRecord, ...],
    sibling_alpha: float,
    inflation_at_alpha: float,
    include_record,
) -> dict[str, object]:
    family_mask, local_weights = _local_calibration_weights(model, target_record)
    family_records = tuple(
        record
        for record, selected in zip(calibration_records, family_mask, strict=True)
        if selected
    )
    selected_mask = np.array([bool(include_record(record)) for record in family_records])
    if not np.any(selected_mask):
        return _variant_unavailable(prefix, "no_calibration_records")

    selected_weights = local_weights[selected_mask]
    positive_weight_mask = selected_weights > 0.0
    if not np.any(positive_weight_mask):
        return _variant_unavailable(prefix, "zero_calibration_weight")

    selected_records = tuple(
        record
        for record, selected in zip(family_records, selected_mask, strict=True)
        if selected
    )
    weighted_records = tuple(
        record
        for record, positive in zip(selected_records, positive_weight_mask, strict=True)
        if positive
    )
    weights = selected_weights[positive_weight_mask]
    statistics = np.array([record.stat for record in weighted_records], dtype=float)
    reference_expectations = np.array(
        [_reference_expectation(record) for record in weighted_records],
        dtype=float,
    )
    denominator = float(np.sum(weights * reference_expectations))
    if denominator <= 0.0:
        return _variant_unavailable(prefix, "nonpositive_reference_weight")

    inflation_factor = max(
        _weighted_inflation_mle(statistics, reference_expectations, weights),
        1.0,
    )
    p_value = _inflation_adjusted_p_value(target_record, inflation_factor)
    inflation_excess_ratio = (
        float(inflation_factor / inflation_at_alpha)
        if np.isfinite(inflation_at_alpha) and inflation_at_alpha > 0.0
        else np.nan
    )
    return {
        f"{prefix}_status": "ok",
        f"{prefix}_inflation_factor": float(inflation_factor),
        f"{prefix}_p_value": p_value,
        f"{prefix}_blocks_at_alpha": p_value >= sibling_alpha,
        f"{prefix}_inflation_excess_ratio": inflation_excess_ratio,
        f"{prefix}_effective_sample_size": _effective_sample_size(weights),
        f"{prefix}_calibration_records": int(len(weighted_records)),
        f"{prefix}_calibration_weight_sum": float(np.sum(weights)),
    }


def _calibration_variant_results(
    *,
    record: SiblingPairRecord,
    model: EmpiricalNullInflationModel,
    calibration_records: tuple[SiblingPairRecord, ...],
    sibling_alpha: float,
    inflation_at_alpha: float,
) -> dict[str, object]:
    return {
        **_variant_result(
            prefix="leave_one_out",
            target_record=record,
            model=model,
            calibration_records=calibration_records,
            sibling_alpha=sibling_alpha,
            inflation_at_alpha=inflation_at_alpha,
            include_record=lambda candidate: candidate.parent != record.parent,
        ),
        **_variant_result(
            prefix="strict_null_like",
            target_record=record,
            model=model,
            calibration_records=calibration_records,
            sibling_alpha=sibling_alpha,
            inflation_at_alpha=inflation_at_alpha,
            include_record=lambda candidate: (
                candidate.parent != record.parent and candidate.is_null_like
            ),
        ),
        **_variant_result(
            prefix="edge_blocked_or_null_like",
            target_record=record,
            model=model,
            calibration_records=calibration_records,
            sibling_alpha=sibling_alpha,
            inflation_at_alpha=inflation_at_alpha,
            include_record=lambda candidate: (
                candidate.parent != record.parent
                and (candidate.is_null_like or candidate.is_edge_blocked)
            ),
        ),
    }


def _calibration_support_contract(variant_results: dict[str, object]) -> dict[str, object]:
    strict_status = str(variant_results["strict_null_like_status"])
    blocked_or_null_status = str(variant_results["edge_blocked_or_null_like_status"])

    if strict_status == "ok":
        status = "strict_empirical_null_supported"
        level = "strict"
        action = "internal_empirical_null_estimate_available"
        empirical_null_supported = True
    elif blocked_or_null_status == "ok":
        status = "stopped_or_strict_empirical_null_supported"
        level = "weak"
        action = "internal_stopped_or_null_estimate_available"
        empirical_null_supported = True
    else:
        status = "unsupported_without_empirical_null_support"
        level = "unsupported"
        action = "raise_calibration_data_error; diagnose_full_selection_conditioning"
        empirical_null_supported = False

    return {
        "calibration_support_status": status,
        "calibration_support_level": level,
        "calibration_support_action": action,
        "empirical_null_supported": empirical_null_supported,
    }


def _trace_row_by_parent(trace_df: pd.DataFrame, parent: object) -> pd.Series:
    matches = trace_df[trace_df["node_id"] == parent]
    if len(matches) != 1:
        raise ValueError(
            "Sibling inflation diagnostic requires exactly one trace row for "
            f"parent={parent!r}; found {len(matches)}."
        )
    return matches.iloc[0]


def _require_trace_columns(trace_df: pd.DataFrame) -> None:
    required = {
        "case_id",
        "failure_class",
        "node_id",
        "actual_decision",
        "trace_relation",
        "actual_boundary",
        "oracle_true_k_boundary",
        "split_prerequisites_open",
        "sibling_gate_open",
        "has_descendant_split",
        "sibling_adjusted_p_value",
        "sibling_corrected_p_value",
        "left_edge_significant",
        "right_edge_significant",
        "left_edge_p_value_bh",
        "right_edge_p_value_bh",
    }
    missing = required - set(trace_df.columns)
    if missing:
        raise ValueError(
            "Sibling inflation diagnostic trace is missing columns: "
            f"{sorted(missing)!r}."
        )


def _target_row(
    *,
    record: SiblingPairRecord,
    model: EmpiricalNullInflationModel,
    calibration_records: tuple[SiblingPairRecord, ...],
    trace_row: pd.Series,
    sibling_alpha: float,
) -> dict[str, object]:
    current_inflation = predict_empirical_inflation_factor(model, record)
    family_baseline_inflation = _family_baseline_inflation(model, record)
    mask, local_weights = _local_calibration_weights(model, record)
    local_inflation = _local_inflation_from_weights(model, mask, local_weights)
    local_effective_sample_size = _effective_sample_size(local_weights)

    raw_p_value = _inflation_adjusted_p_value(record, 1.0)
    family_baseline_p_value = _inflation_adjusted_p_value(
        record,
        family_baseline_inflation,
    )
    current_p_value = _inflation_adjusted_p_value(record, current_inflation)
    inflation_at_alpha = _inflation_factor_at_alpha(record, sibling_alpha)
    inflation_excess_ratio = (
        float(current_inflation / inflation_at_alpha)
        if np.isfinite(inflation_at_alpha) and inflation_at_alpha > 0.0
        else float("nan")
    )
    raw_rejects = raw_p_value < sibling_alpha
    current_blocks = current_p_value >= sibling_alpha
    annotation_corrected_p_value = float(trace_row["sibling_corrected_p_value"])
    corrected_blocks = annotation_corrected_p_value >= sibling_alpha
    fdr_crosses_alpha = current_p_value < sibling_alpha <= annotation_corrected_p_value
    blocker_candidate = bool(
        trace_row["trace_relation"] == "actual_stops_above_oracle_boundary"
        and trace_row["split_prerequisites_open"]
        and not trace_row["sibling_gate_open"]
    )
    if current_blocks:
        blocking_stage = "inflation_adjusted_test"
    elif fdr_crosses_alpha:
        blocking_stage = "sibling_fdr"
    elif blocker_candidate:
        blocking_stage = "gate_decision"
    else:
        blocking_stage = "not_blocked"
    variant_results = _calibration_variant_results(
        record=record,
        model=model,
        calibration_records=calibration_records,
        sibling_alpha=sibling_alpha,
        inflation_at_alpha=inflation_at_alpha,
    )

    return {
        "case_id": str(trace_row["case_id"]),
        "failure_class": str(trace_row["failure_class"]),
        "parent": record.parent,
        "left_child": record.left,
        "right_child": record.right,
        "actual_decision": str(trace_row["actual_decision"]),
        "trace_relation": str(trace_row["trace_relation"]),
        "actual_boundary": bool(trace_row["actual_boundary"]),
        "oracle_true_k_boundary": bool(trace_row["oracle_true_k_boundary"]),
        "blocker_candidate": blocker_candidate,
        "split_prerequisites_open": bool(trace_row["split_prerequisites_open"]),
        "sibling_gate_open": bool(trace_row["sibling_gate_open"]),
        "has_descendant_split": bool(trace_row["has_descendant_split"]),
        "left_edge_significant": bool(trace_row["left_edge_significant"]),
        "right_edge_significant": bool(trace_row["right_edge_significant"]),
        "left_edge_p_value_bh": float(trace_row["left_edge_p_value_bh"]),
        "right_edge_p_value_bh": float(trace_row["right_edge_p_value_bh"]),
        "raw_sibling_statistic": float(record.stat),
        "reference_scale": float(record.reference_scale),
        "degrees_of_freedom": float(record.degrees_of_freedom),
        "reference_expectation": _reference_expectation(record),
        "raw_sibling_p_value": raw_p_value,
        "annotation_adjusted_p_value": float(trace_row["sibling_adjusted_p_value"]),
        "annotation_corrected_p_value": annotation_corrected_p_value,
        "family_baseline_inflation_factor": family_baseline_inflation,
        "local_recomputed_inflation_factor": local_inflation,
        "current_empirical_inflation_factor": float(current_inflation),
        "inflation_factor_at_alpha": inflation_at_alpha,
        "inflation_excess_ratio": inflation_excess_ratio,
        "family_baseline_p_value": family_baseline_p_value,
        "current_empirical_p_value": current_p_value,
        "raw_rejects_at_alpha": raw_rejects,
        "family_baseline_blocks_at_alpha": family_baseline_p_value >= sibling_alpha,
        "current_blocks_at_alpha": current_blocks,
        "annotation_corrected_blocks_at_alpha": corrected_blocks,
        "inflation_crosses_alpha": raw_rejects and current_blocks,
        "fdr_crosses_alpha": fdr_crosses_alpha,
        "raw_to_corrected_crosses_alpha": raw_rejects and corrected_blocks,
        "blocking_stage": blocking_stage,
        "sibling_null_weight": float(record.sibling_null_weight),
        "sibling_projection_dimension": float(record.sibling_projection_dimension),
        "parent_sample_size": int(record.n_parent),
        "feature_family": record.feature_family,
        "is_edge_blocked": bool(record.is_edge_blocked),
        "is_null_like": bool(record.is_null_like),
        "local_effective_sample_size": local_effective_sample_size,
        "local_calibration_weight_sum": float(np.sum(local_weights)),
        "local_calibration_records": int(local_weights.size),
        **variant_results,
        **_calibration_support_contract(variant_results),
    }


def _contributor_rows(
    *,
    target_record: SiblingPairRecord,
    target_case_id: str,
    target_failure_class: str,
    target_blocking_stage: str,
    model: EmpiricalNullInflationModel,
    calibration_records: tuple[SiblingPairRecord, ...],
    max_contributors: int,
) -> list[dict[str, object]]:
    mask, local_weights = _local_calibration_weights(model, target_record)
    selected_records = tuple(
        record for record, selected in zip(calibration_records, mask, strict=True) if selected
    )
    selected_indices = np.flatnonzero(mask)
    selected_statistics = model.sample_statistics[mask]
    selected_references = model.sample_reference_scales[mask] * (
        model.sample_degrees_of_freedom[mask]
    )
    weighted_statistics = local_weights * selected_statistics
    weighted_references = local_weights * selected_references
    statistic_total = float(np.sum(weighted_statistics))
    reference_total = float(np.sum(weighted_references))
    weight_total = float(np.sum(local_weights))
    positive_local_weight_indices = np.flatnonzero(local_weights > 0.0)
    order = positive_local_weight_indices[np.argsort(-local_weights[positive_local_weight_indices])]
    if max_contributors > 0:
        order = order[: int(max_contributors)]

    rows: list[dict[str, object]] = []
    for rank, local_index in enumerate(order, start=1):
        record = selected_records[int(local_index)]
        global_index = int(selected_indices[int(local_index)])
        rows.append(
            {
                "target_case_id": target_case_id,
                "target_failure_class": target_failure_class,
                "target_parent": target_record.parent,
                "target_blocking_stage": target_blocking_stage,
                "contributor_rank": rank,
                "contributor_parent": record.parent,
                "contributor_left_child": record.left,
                "contributor_right_child": record.right,
                "contributor_model_index": global_index,
                "contributor_feature_family": record.feature_family,
                "contributor_statistic": float(record.stat),
                "contributor_reference_expectation": _reference_expectation(record),
                "contributor_stat_over_reference": float(
                    record.stat / _reference_expectation(record)
                ),
                "contributor_raw_p_value": float(record.p_value),
                "contributor_null_weight": float(record.sibling_null_weight),
                "contributor_projection_dimension": float(
                    record.sibling_projection_dimension
                ),
                "contributor_parent_sample_size": int(record.n_parent),
                "contributor_is_null_like": bool(record.is_null_like),
                "contributor_is_edge_blocked": bool(record.is_edge_blocked),
                "local_weight": float(local_weights[local_index]),
                "local_weight_share": float(local_weights[local_index] / weight_total),
                "weighted_statistic_contribution": float(
                    weighted_statistics[local_index]
                ),
                "weighted_reference_contribution": float(
                    weighted_references[local_index]
                ),
                "statistic_contribution_share": float(
                    weighted_statistics[local_index] / statistic_total
                ),
                "reference_contribution_share": float(
                    weighted_references[local_index] / reference_total
                ),
            }
        )
    return rows


def _summary_rows(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "failure_class",
                "n_focal_sibling_tests",
                "n_blocker_candidates",
                "n_raw_reject_current_block",
                "n_current_p_blocks",
                "n_corrected_p_blocks",
                "n_fdr_crosses_alpha",
                "n_leave_one_out_blocks",
                "n_strict_null_like_blocks",
                "n_strict_null_like_unavailable",
                "n_edge_blocked_or_null_like_blocks",
                "n_edge_blocked_or_null_like_unavailable",
                "n_strict_empirical_null_supported",
                "n_unsupported_without_empirical_null_support",
                "max_current_empirical_inflation_factor",
                "median_current_empirical_inflation_factor",
                "min_inflation_excess_ratio",
                "median_local_effective_sample_size",
            ]
        )

    rows: list[dict[str, object]] = []
    for (case_id, failure_class), group in targets.groupby(
        ["case_id", "failure_class"],
        sort=False,
    ):
        rows.append(
            {
                "case_id": case_id,
                "failure_class": failure_class,
                "n_focal_sibling_tests": int(len(group)),
                "n_blocker_candidates": int(group["blocker_candidate"].sum()),
                "n_raw_reject_current_block": int(group["inflation_crosses_alpha"].sum()),
                "n_current_p_blocks": int(group["current_blocks_at_alpha"].sum()),
                "n_corrected_p_blocks": int(
                    group["annotation_corrected_blocks_at_alpha"].sum()
                ),
                "n_fdr_crosses_alpha": int(group["fdr_crosses_alpha"].sum()),
                "n_leave_one_out_blocks": int(
                    group["leave_one_out_blocks_at_alpha"].eq(True).sum()
                ),
                "n_strict_null_like_blocks": int(
                    group["strict_null_like_blocks_at_alpha"].eq(True).sum()
                ),
                "n_strict_null_like_unavailable": int(
                    (group["strict_null_like_status"] != "ok").sum()
                ),
                "n_edge_blocked_or_null_like_blocks": int(
                    group["edge_blocked_or_null_like_blocks_at_alpha"].eq(True).sum()
                ),
                "n_edge_blocked_or_null_like_unavailable": int(
                    (group["edge_blocked_or_null_like_status"] != "ok").sum()
                ),
                "n_strict_empirical_null_supported": int(
                    (
                        group["calibration_support_status"]
                        == "strict_empirical_null_supported"
                    ).sum()
                ),
                "n_unsupported_without_empirical_null_support": int(
                    (
                        group["calibration_support_status"]
                        == "unsupported_without_empirical_null_support"
                    ).sum()
                ),
                "max_current_empirical_inflation_factor": float(
                    group["current_empirical_inflation_factor"].max()
                ),
                "median_current_empirical_inflation_factor": float(
                    group["current_empirical_inflation_factor"].median()
                ),
                "min_inflation_excess_ratio": float(
                    group["inflation_excess_ratio"].min()
                ),
                "median_local_effective_sample_size": float(
                    group["local_effective_sample_size"].median()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_sibling_inflation_diagnostic_tables(
    *,
    records: tuple[SiblingPairRecord, ...],
    model: EmpiricalNullInflationModel,
    trace_df: pd.DataFrame,
    sibling_alpha: float,
    max_contributors: int = 10,
    contributors_for_all_crossings: bool = False,
) -> SiblingInflationDiagnosticTables:
    """Build target and contributor tables explaining sibling inflation decisions."""
    _require_trace_columns(trace_df)
    calibration_records = _calibration_records(records)
    _validate_model_record_alignment(model, calibration_records)

    target_rows: list[dict[str, object]] = []
    contributor_rows: list[dict[str, object]] = []
    focal_records = [
        record
        for record in records
        if not record.is_null_like and record.degrees_of_freedom > 0.0
    ]
    for record in focal_records:
        trace_row = _trace_row_by_parent(trace_df, record.parent)
        target = _target_row(
            record=record,
            model=model,
            calibration_records=calibration_records,
            trace_row=trace_row,
            sibling_alpha=sibling_alpha,
        )
        target_rows.append(target)
        include_contributors = bool(target["blocker_candidate"]) or (
            contributors_for_all_crossings
            and (
                bool(target["inflation_crosses_alpha"])
                or bool(target["fdr_crosses_alpha"])
            )
        )
        if include_contributors:
            contributor_rows.extend(
                _contributor_rows(
                    target_record=record,
                    target_case_id=str(target["case_id"]),
                    target_failure_class=str(target["failure_class"]),
                    target_blocking_stage=str(target["blocking_stage"]),
                    model=model,
                    calibration_records=calibration_records,
                    max_contributors=max_contributors,
                )
            )

    targets = pd.DataFrame.from_records(target_rows)
    contributors = pd.DataFrame.from_records(contributor_rows)
    summary = _summary_rows(targets)
    return SiblingInflationDiagnosticTables(
        targets=targets,
        contributors=contributors,
        summary=summary,
    )


__all__ = [
    "SiblingInflationDiagnosticTables",
    "SiblingInflationInputs",
    "build_sibling_inflation_diagnostic_tables",
    "collect_sibling_inflation_inputs",
]
