#!/usr/bin/env python3
"""Compare Hamming topology branch-length sources against sibling null calibration.

This is an evidence-only diagnostic. It fixes the tree topology distance to
Hamming, then compares raw linkage branch lengths with fixed-topology NNLS
branch lengths whose squared-Euclidean target is exactly Hamming for binary
feature matrices. The outputs are intended to show whether branch-time changes
edge support, sibling empirical-null support, and inflation-adjusted sibling
p-values across a set of benchmark cases.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    CalibrationDecision,
    decide_empirical_null_calibration,
    fit_empirical_null_inflation_model,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
)

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc

SCHEMA_VERSION = "nnls_null_calibration_sweep/v1"
GENERATED_BY = "benchmarks.validation.sweeps.nnls_null_calibration_sweep"
DEFAULT_OUTPUT_DIR = Path("reports/nnls_null_calibration_sweep")
DEFAULT_CASE_NAMES = (
    "binary_low_noise_4c",
    "binary_moderate_4c",
    "binary_hard_4c",
    "binary_null_small",
)
BRANCH_SOURCE_LINKAGE = "linkage_ultrametric_diagnostic"
BRANCH_SOURCE_NNLS = "fixed_topology_nnls_hamming_target"
BRANCH_SOURCES = (BRANCH_SOURCE_LINKAGE, BRANCH_SOURCE_NNLS)
SPECTRAL_CONTEXT_LEAF_ONLY = "leaf_only"
SPECTRAL_CONTEXT_INTERNAL_EMPIRICAL_BARYCENTER = "internal_empirical_barycenter"
SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE = "internal_branch_length_state"
SPECTRAL_CONTEXTS = (
    SPECTRAL_CONTEXT_LEAF_ONLY,
    SPECTRAL_CONTEXT_INTERNAL_EMPIRICAL_BARYCENTER,
    SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE,
)
DEFAULT_SPECTRAL_CONTEXTS = (SPECTRAL_CONTEXT_LEAF_ONLY,)


def _record_row(
    *,
    branch_source: str,
    spectral_context: str,
    case_name: str,
    record: SiblingPairRecord,
    decision: CalibrationDecision | None = None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_case_id": case_name,
        "branch_source": branch_source,
        "spectral_context": spectral_context,
        "parent": str(record.parent),
        "left": str(record.left),
        "right": str(record.right),
        "stat": float(record.stat),
        "reference_scale": float(record.reference_scale),
        "degrees_of_freedom": float(record.degrees_of_freedom),
        "p_value": float(record.p_value),
        "branch_length_sum": float(record.branch_length_sum),
        "n_parent": int(record.n_parent),
        "is_null_like": bool(record.is_null_like),
        "is_edge_blocked": bool(record.is_edge_blocked),
        "sibling_null_weight": float(record.sibling_null_weight),
        "sibling_projection_dimension": float(record.sibling_projection_dimension),
        "parent_spectral_eigenvalue_count": float(record.parent_spectral_eigenvalue_count),
        "parent_positive_eigenvalue_count": float(record.parent_positive_eigenvalue_count),
        "parent_spectral_rank": float(record.parent_spectral_rank),
        "parent_eigenvalue_sum": float(record.parent_eigenvalue_sum),
        "parent_top_eigenvalue": float(record.parent_top_eigenvalue),
        "parent_top_eigenvalue_share": float(record.parent_top_eigenvalue_share),
        "parent_spectral_entropy": float(record.parent_spectral_entropy),
        "parent_effective_rank": float(record.parent_effective_rank),
        "parent_retained_eigenvalue_sum": float(record.parent_retained_eigenvalue_sum),
        "parent_retained_eigenvalue_share": float(record.parent_retained_eigenvalue_share),
        "parent_top_spectral_gap": float(record.parent_top_spectral_gap),
        "parent_eigengap_at_projection_dimension": float(
            record.parent_eigengap_at_projection_dimension
        ),
        "parent_spectral_gap_at_projection_dimension": float(
            record.parent_spectral_gap_at_projection_dimension
        ),
        "parent_spectral_pseudodeterminant": float(
            record.parent_spectral_pseudodeterminant
        ),
        "parent_spectral_log_pseudodeterminant": float(
            record.parent_spectral_log_pseudodeterminant
        ),
        "parent_spectral_geometric_mean": float(record.parent_spectral_geometric_mean),
        "feature_family": str(record.feature_family),
        "decision_status": "" if decision is None else decision.status,
        "decision_c_hat": math.nan
        if decision is None or decision.c_hat is None
        else float(decision.c_hat),
        "decision_p_value": math.nan
        if decision is None or decision.p_value is None
        else float(decision.p_value),
        "decision_estimator": "" if decision is None else decision.estimator,
    }


def _parse_names(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _is_binary_frame(data: pd.DataFrame) -> bool:
    values = data.to_numpy()
    return bool(np.isin(values, (0, 1)).all())


def hamming_squared_euclidean_embedding(data: pd.DataFrame) -> pd.DataFrame:
    """Return an embedding whose squared Euclidean distances equal Hamming."""
    if data.shape[1] < 1:
        raise ValueError("Hamming embedding requires at least one feature column.")
    if not _is_binary_frame(data):
        raise ValueError("Hamming NNLS target requires a binary/one-hot feature matrix.")
    matrix = data.to_numpy(dtype=float, copy=True) / math.sqrt(float(data.shape[1]))
    return pd.DataFrame(
        matrix,
        index=data.index,
        columns=[f"hamming_geometry_{index}" for index in range(data.shape[1])],
    )


def _finite_min(values: pd.Series | np.ndarray | list[object]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.min()) if not series.empty else math.nan


def _bool_sum(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(
        frame[column].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"}).sum()
    )


def _calibration_model_rows(
    *,
    branch_source: str,
    spectral_context: str,
    case_name: str,
    records: list[SiblingPairRecord],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        model = fit_empirical_null_inflation_model(records)
    except ValueError as exc:
        return (
            {
                "calibration_model_status": "unavailable",
                "calibration_model_error": str(exc),
                "baseline_empirical_inflation_factor": math.nan,
                "n_calibration": 0,
                "n_positive_weight_records": 0,
                "n_selected_nonnull_positive_weight_records": 0,
                "n_strict_null_calibration": 0,
                "n_edge_blocked_calibration": 0,
                "n_stopped_or_null_calibration": 0,
                "effective_sample_size": math.nan,
            },
            [
                _record_row(
                    branch_source=branch_source,
                    spectral_context=spectral_context,
                    case_name=case_name,
                    record=record,
                )
                for record in records
            ],
        )

    model_summary = {
        "calibration_model_status": "ok",
        "calibration_model_error": "",
        "baseline_empirical_inflation_factor": float(model.baseline_empirical_inflation_factor),
        "n_calibration": int(model.n_calibration),
        "n_positive_weight_records": int(model.n_positive_weight_records),
        "n_selected_nonnull_positive_weight_records": int(
            model.n_selected_nonnull_positive_weight_records
        ),
        "n_strict_null_calibration": int(model.n_strict_null_calibration),
        "n_edge_blocked_calibration": int(model.n_edge_blocked_calibration),
        "n_stopped_or_null_calibration": int(model.n_stopped_or_null_calibration),
        "effective_sample_size": float(model.effective_sample_size),
    }

    record_rows: list[dict[str, object]] = []
    for record in records:
        decision: CalibrationDecision | None = None
        if not record.is_null_like:
            decision = decide_empirical_null_calibration(model, record)
        record_rows.append(
            _record_row(
                branch_source=branch_source,
                spectral_context=spectral_context,
                case_name=case_name,
                record=record,
                decision=decision,
            )
        )
    return model_summary, record_rows


def _collect_records_from_result(
    result_extra: dict[str, object],
    *,
    feature_space: FeatureSpace | None,
) -> tuple[list[SiblingPairRecord], str]:
    tree = result_extra.get("tree")
    annotations = result_extra.get("annotations")
    gate_bundle = result_extra.get("gate_bundle")
    if tree is None or annotations is None or gate_bundle is None:
        return [], "missing_tree_annotations_or_gate_bundle"
    spectral_context = gate_bundle.edge_gate_result.spectral_context
    try:
        records, _non_binary = collect_sibling_pair_records(
            tree,
            annotations,
            sibling_projection_dimensions_from_edge_comparisons=(
                spectral_context.test_projection_dimensions_by_node
            ),
            parent_principal_component_projections=(
                spectral_context.principal_component_projections_by_node
            ),
            parent_principal_component_eigenvalues=(
                spectral_context.principal_component_eigenvalues_by_node
            ),
            feature_space=feature_space,
        )
    except ValueError as exc:
        return [], str(exc)
    return records, ""


def _parent_eigenvalue_rows_from_result(
    result_extra: dict[str, object],
    *,
    branch_source: str,
    spectral_context_name: str,
    case_name: str,
) -> list[dict[str, object]]:
    """Return long-form parent eigenvalue rows for selected-law diagnostics."""
    gate_bundle = result_extra.get("gate_bundle")
    if gate_bundle is None:
        return []
    spectral_context = gate_bundle.edge_gate_result.spectral_context
    eigenvalues_by_node = getattr(
        spectral_context,
        "principal_component_eigenvalues_by_node",
        {},
    )
    rows: list[dict[str, object]] = []
    for parent, eigenvalues in eigenvalues_by_node.items():
        values = np.asarray(eigenvalues, dtype=float).reshape(-1)
        finite_values = values[np.isfinite(values)]
        positive_values = finite_values[finite_values > 0.0]
        positive_sum = float(positive_values.sum()) if positive_values.size else 0.0
        for index, value in enumerate(values):
            finite = bool(np.isfinite(value))
            positive = bool(finite and value > 0.0)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_case_id": str(case_name),
                    "branch_source": branch_source,
                    "spectral_context": spectral_context_name,
                    "parent": str(parent),
                    "eigenvalue_index": int(index),
                    "eigenvalue": float(value) if finite else math.nan,
                    "is_finite": finite,
                    "is_positive": positive,
                    "positive_eigenvalue_sum": positive_sum,
                    "positive_eigenvalue_share": (
                        float(value / positive_sum) if positive and positive_sum > 0.0 else 0.0
                    ),
                }
            )
    return rows


def _spectral_context_kwargs(spectral_context: str) -> dict[str, object]:
    if spectral_context == SPECTRAL_CONTEXT_LEAF_ONLY:
        return {
            "spectral_include_internal_barycenters": False,
            "spectral_internal_distribution_mode": INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
        }
    if spectral_context == SPECTRAL_CONTEXT_INTERNAL_EMPIRICAL_BARYCENTER:
        return {
            "spectral_include_internal_barycenters": True,
            "spectral_internal_distribution_mode": INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
        }
    if spectral_context == SPECTRAL_CONTEXT_INTERNAL_BRANCH_LENGTH_STATE:
        return {
            "spectral_include_internal_barycenters": True,
            "spectral_internal_distribution_mode": INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
        }
    raise ValueError(f"Unsupported spectral context: {spectral_context!r}.")


def _run_branch_source(
    *,
    data: pd.DataFrame,
    labels: np.ndarray,
    feature_space: FeatureSpace | None,
    branch_source: str,
    spectral_context: str,
    edge_alpha: float,
    sibling_alpha: float,
    tree_linkage_method: str,
    pair_sample_size: int | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    distance_condensed = pdist(data.to_numpy(dtype=float), metric="hamming")
    branch_kwargs: dict[str, object]
    if branch_source == BRANCH_SOURCE_LINKAGE:
        branch_kwargs = {
            "branch_length_optimization_method": BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
            "allow_linkage_ultrametric_branch_time": True,
        }
    elif branch_source == BRANCH_SOURCE_NNLS:
        branch_kwargs = {
            "branch_length_data_df": hamming_squared_euclidean_embedding(data),
            "branch_length_optimization_method": BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
            "branch_length_optimization_target_metric": BRANCH_LENGTH_TARGET_SQUARED_EUCLIDEAN,
            "branch_length_optimization_pair_sample_size": pair_sample_size,
            "branch_length_optimization_random_state": 0,
            "branch_length_optimization_solver_tolerance": 1e-5,
            "branch_length_optimization_max_iterations": 300,
        }
    else:
        raise ValueError(f"Unsupported branch source: {branch_source!r}.")

    started = perf_counter()
    result = run_tbs_on_distance(
        data,
        distance_condensed,
        sibling_alpha,
        tree_linkage_method=tree_linkage_method,
        edge_alpha=edge_alpha,
        feature_space=feature_space,
        edge_branch_length_variance_policy=EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
        trace_level="full",
        **branch_kwargs,
        **_spectral_context_kwargs(spectral_context),
    )
    elapsed_sec = perf_counter() - started
    extra = dict(result.extra or {})
    annotations = extra.get("annotations")
    annotations_df = annotations if isinstance(annotations, pd.DataFrame) else pd.DataFrame()
    trace = pd.DataFrame(extra.get("full_edge_traversal_trace", []))
    records, record_error = _collect_records_from_result(extra, feature_space=feature_space)
    eigenvalue_rows = _parent_eigenvalue_rows_from_result(
        extra,
        branch_source=branch_source,
        spectral_context_name=spectral_context,
        case_name=str(extra.get("case_name", "")),
    )
    model_summary, record_rows = _calibration_model_rows(
        branch_source=branch_source,
        spectral_context=spectral_context,
        case_name=str(extra.get("case_name", "")),
        records=records,
    )

    predicted = np.asarray(result.labels, dtype=int) if result.labels is not None else np.array([])
    ari = (
        float(adjusted_rand_score(labels, predicted))
        if result.status == "ok" and len(predicted) == len(labels)
        else math.nan
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        "branch_source": branch_source,
        "spectral_context": spectral_context,
        "spectral_include_internal_barycenters": bool(
            extra.get("spectral_include_internal_barycenters", False)
        ),
        "spectral_internal_distribution_mode": str(
            extra.get("spectral_internal_distribution_mode", "")
        ),
        "status": result.status,
        "skip_reason": "" if result.skip_reason is None else str(result.skip_reason),
        "elapsed_sec": float(elapsed_sec),
        "tree_distance_metric": "hamming",
        "tree_linkage_method": tree_linkage_method,
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "edge_branch_length_variance_policy": EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
        "found_clusters": int(result.found_clusters),
        "ari": ari,
        "n_edge_tested": _bool_sum(annotations_df, "Child_Parent_Divergence_Tested"),
        "n_edge_significant": _bool_sum(annotations_df, "Child_Parent_Divergence_Significant"),
        "n_sibling_records": int(len(records)),
        "record_collection_error": record_error,
        "n_sibling_tested": int(
            annotations_df["Sibling_Divergence_P_Value"].notna().sum()
            if "Sibling_Divergence_P_Value" in annotations_df
            else 0
        ),
        "n_sibling_open": _bool_sum(annotations_df, "Sibling_BH_Different"),
        "n_sibling_fail_closed": int(
            annotations_df["Sibling_Gate_P_Value_Calibration"]
            .astype(str)
            .eq("undefined_no_internal_support")
            .sum()
            if "Sibling_Gate_P_Value_Calibration" in annotations_df
            else 0
        ),
        "min_sibling_p_value": _finite_min(
            annotations_df.get("Sibling_Divergence_P_Value", pd.Series(dtype=float))
        ),
        "min_sibling_p_value_corrected": _finite_min(
            annotations_df.get("Sibling_Divergence_P_Value_Corrected", pd.Series(dtype=float))
        ),
        "trace_edge_open": _bool_sum(trace, "edge_gate_open"),
        "trace_sibling_open": _bool_sum(trace, "sibling_gate_open"),
        "nnls_status": extra.get("branch_length_optimization_status", ""),
        "nnls_applied_to_tree": extra.get("branch_length_optimization_applied_to_tree", math.nan),
        "nnls_residual_rmse_to_target_mean": extra.get(
            "branch_length_optimization_residual_rmse_to_target_mean",
            math.nan,
        ),
        "nnls_design_density": extra.get("branch_length_optimization_design_density", math.nan),
        "nnls_zero_design_columns": extra.get(
            "branch_length_optimization_n_zero_design_columns",
            math.nan,
        ),
        **model_summary,
    }
    return row, record_rows, eigenvalue_rows


def _pair_rows(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    left = cells[cells["branch_source"].eq(BRANCH_SOURCE_LINKAGE)].copy()
    right = cells[cells["branch_source"].eq(BRANCH_SOURCE_NNLS)].copy()
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(
        right,
        on=[
            "source_case_id",
            "spectral_context",
            "tree_distance_metric",
            "tree_linkage_method",
        ],
        suffixes=("_linkage", "_nnls"),
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        linkage_c_hat = row["baseline_empirical_inflation_factor_linkage"]
        nnls_c_hat = row["baseline_empirical_inflation_factor_nnls"]
        c_hat_changed = (
            pd.notna(linkage_c_hat)
            and pd.notna(nnls_c_hat)
            and float(linkage_c_hat) != float(nnls_c_hat)
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_case_id": row["source_case_id"],
                "spectral_context": row["spectral_context"],
                "tree_distance_metric": row["tree_distance_metric"],
                "tree_linkage_method": row["tree_linkage_method"],
                "delta_ari_nnls_minus_linkage": (
                    float(row["ari_nnls"]) - float(row["ari_linkage"])
                    if pd.notna(row["ari_nnls"]) and pd.notna(row["ari_linkage"])
                    else math.nan
                ),
                "delta_found_clusters_nnls_minus_linkage": int(
                    row["found_clusters_nnls"] - row["found_clusters_linkage"]
                ),
                "delta_edge_significant_nnls_minus_linkage": int(
                    row["n_edge_significant_nnls"] - row["n_edge_significant_linkage"]
                ),
                "delta_sibling_open_nnls_minus_linkage": int(
                    row["n_sibling_open_nnls"] - row["n_sibling_open_linkage"]
                ),
                "delta_calibration_n_nnls_minus_linkage": int(
                    row["n_calibration_nnls"] - row["n_calibration_linkage"]
                ),
                "delta_c_hat_nnls_minus_linkage": (
                    float(nnls_c_hat) - float(linkage_c_hat)
                    if pd.notna(nnls_c_hat) and pd.notna(linkage_c_hat)
                    else math.nan
                ),
                "linkage_c_hat": linkage_c_hat,
                "nnls_c_hat": nnls_c_hat,
                "linkage_model_status": row["calibration_model_status_linkage"],
                "nnls_model_status": row["calibration_model_status_nnls"],
                "calibration_changed": bool(
                    row["n_calibration_nnls"] != row["n_calibration_linkage"]
                    or c_hat_changed
                    or row["n_edge_significant_nnls"] != row["n_edge_significant_linkage"]
                    or row["n_sibling_open_nnls"] != row["n_sibling_open_linkage"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _spectral_context_pair_rows(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty or "spectral_context" not in cells.columns:
        return pd.DataFrame()
    baseline = cells[cells["spectral_context"].eq(SPECTRAL_CONTEXT_LEAF_ONLY)].copy()
    variants = cells[~cells["spectral_context"].eq(SPECTRAL_CONTEXT_LEAF_ONLY)].copy()
    if baseline.empty or variants.empty:
        return pd.DataFrame()
    merged = baseline.merge(
        variants,
        on=["source_case_id", "branch_source", "tree_distance_metric", "tree_linkage_method"],
        suffixes=("_leaf_only", "_variant"),
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        baseline_c_hat = row["baseline_empirical_inflation_factor_leaf_only"]
        variant_c_hat = row["baseline_empirical_inflation_factor_variant"]
        c_hat_changed = (
            pd.notna(baseline_c_hat)
            and pd.notna(variant_c_hat)
            and float(baseline_c_hat) != float(variant_c_hat)
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_case_id": row["source_case_id"],
                "branch_source": row["branch_source"],
                "baseline_spectral_context": SPECTRAL_CONTEXT_LEAF_ONLY,
                "variant_spectral_context": row["spectral_context_variant"],
                "tree_distance_metric": row["tree_distance_metric"],
                "tree_linkage_method": row["tree_linkage_method"],
                "delta_ari_variant_minus_leaf_only": (
                    float(row["ari_variant"]) - float(row["ari_leaf_only"])
                    if pd.notna(row["ari_variant"]) and pd.notna(row["ari_leaf_only"])
                    else math.nan
                ),
                "delta_found_clusters_variant_minus_leaf_only": int(
                    row["found_clusters_variant"] - row["found_clusters_leaf_only"]
                ),
                "delta_edge_significant_variant_minus_leaf_only": int(
                    row["n_edge_significant_variant"] - row["n_edge_significant_leaf_only"]
                ),
                "delta_sibling_open_variant_minus_leaf_only": int(
                    row["n_sibling_open_variant"] - row["n_sibling_open_leaf_only"]
                ),
                "delta_calibration_n_variant_minus_leaf_only": int(
                    row["n_calibration_variant"] - row["n_calibration_leaf_only"]
                ),
                "delta_c_hat_variant_minus_leaf_only": (
                    float(variant_c_hat) - float(baseline_c_hat)
                    if pd.notna(variant_c_hat) and pd.notna(baseline_c_hat)
                    else math.nan
                ),
                "leaf_only_c_hat": baseline_c_hat,
                "variant_c_hat": variant_c_hat,
                "leaf_only_model_status": row["calibration_model_status_leaf_only"],
                "variant_model_status": row["calibration_model_status_variant"],
                "spectral_context_changed": bool(
                    row["found_clusters_variant"] != row["found_clusters_leaf_only"]
                    or row["n_calibration_variant"] != row["n_calibration_leaf_only"]
                    or c_hat_changed
                    or row["n_edge_significant_variant"] != row["n_edge_significant_leaf_only"]
                    or row["n_sibling_open_variant"] != row["n_sibling_open_leaf_only"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_nnls_null_calibration_sweep(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    suite: str = "binary",
    case_names: Sequence[str] = DEFAULT_CASE_NAMES,
    branch_sources: Sequence[str] = BRANCH_SOURCES,
    spectral_contexts: Sequence[str] = DEFAULT_SPECTRAL_CONTEXTS,
    tree_linkage_method: str = "average",
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    pair_sample_size: int | None = 50_000,
) -> dict[str, Path]:
    """Run the Hamming-topology branch-time/null-calibration sweep."""
    invalid_sources = sorted(set(branch_sources).difference(BRANCH_SOURCES))
    if invalid_sources:
        raise ValueError(f"Invalid branch_sources={invalid_sources!r}.")
    invalid_spectral_contexts = sorted(set(spectral_contexts).difference(SPECTRAL_CONTEXTS))
    if invalid_spectral_contexts:
        raise ValueError(f"Invalid spectral_contexts={invalid_spectral_contexts!r}.")
    cases = get_test_cases_by_suite(suite)
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown cases in suite {suite!r}: {missing!r}.")

    cell_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    eigenvalue_rows: list[dict[str, object]] = []
    for case_name in case_names:
        raw_case = by_name[str(case_name)]
        inputs = prepare_case_inputs(dict(raw_case), [])
        feature_space = inputs.metadata.get("feature_space")
        if feature_space is not None and not isinstance(feature_space, FeatureSpace):
            raise TypeError("Case feature_space metadata must be a FeatureSpace.")
        base = {
            "source_case_id": str(case_name),
            "case_generator": str(raw_case.get("generator", "")),
            "case_category": str(raw_case.get("category", "")),
            "n_samples": int(len(inputs.data)),
            "n_features": int(inputs.data.shape[1]),
            "true_clusters": int(len(np.unique(inputs.labels))),
        }
        if not _is_binary_frame(inputs.data):
            for branch_source in branch_sources:
                for spectral_context in spectral_contexts:
                    cell_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            **base,
                            "branch_source": str(branch_source),
                            "spectral_context": str(spectral_context),
                            "status": "skip",
                            "skip_reason": (
                                "hamming_nnls_null_sweep_requires_binary_or_one_hot_data"
                            ),
                        }
                    )
            continue
        for branch_source in branch_sources:
            for spectral_context in spectral_contexts:
                row, rows, eigen_rows = _run_branch_source(
                    data=inputs.data,
                    labels=inputs.labels,
                    feature_space=feature_space,
                    branch_source=str(branch_source),
                    spectral_context=str(spectral_context),
                    edge_alpha=edge_alpha,
                    sibling_alpha=sibling_alpha,
                    tree_linkage_method=tree_linkage_method,
                    pair_sample_size=pair_sample_size,
                )
                cell_rows.append({**base, **row, "source_case_id": str(case_name)})
                for record_row in rows:
                    record_rows.append({**record_row, "source_case_id": str(case_name)})
                for eigenvalue_row in eigen_rows:
                    eigenvalue_rows.append({**eigenvalue_row, "source_case_id": str(case_name)})

    output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame.from_records(cell_rows)
    records = pd.DataFrame.from_records(record_rows)
    parent_eigenvalues = pd.DataFrame.from_records(eigenvalue_rows)
    pairs = _pair_rows(cells)
    spectral_pairs = _spectral_context_pair_rows(cells)

    cells_path = output_dir / "nnls_null_calibration_cells.csv"
    records_path = output_dir / "nnls_null_calibration_records.csv"
    parent_eigenvalues_path = output_dir / "nnls_null_calibration_parent_eigenvalues.csv"
    pairs_path = output_dir / "nnls_null_calibration_pairs.csv"
    spectral_pairs_path = output_dir / "nnls_null_calibration_spectral_pairs.csv"
    manifest_path = output_dir / "nnls_null_calibration_manifest.json"
    cells.to_csv(cells_path, index=False)
    records.to_csv(records_path, index=False)
    parent_eigenvalues.to_csv(parent_eigenvalues_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    spectral_pairs.to_csv(spectral_pairs_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "generated_by": GENERATED_BY,
                "generated_at_utc": format_timestamp_utc(),
                "decision_scope": "evidence_only_no_production_promotion",
                "suite": suite,
                "case_names": list(case_names),
                "branch_sources": list(branch_sources),
                "spectral_contexts": list(spectral_contexts),
                "tree_distance_metric": "hamming",
                "tree_linkage_method": tree_linkage_method,
                "edge_alpha": float(edge_alpha),
                "sibling_alpha": float(sibling_alpha),
                "pair_sample_size": pair_sample_size,
                "observed_cells": int(len(cells)),
                "ok_cells": int(cells.get("status", pd.Series(dtype=str)).eq("ok").sum()),
                "observed_parent_eigenvalue_rows": int(len(parent_eigenvalues)),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "cells": cells_path,
        "records": records_path,
        "parent_eigenvalues": parent_eigenvalues_path,
        "pairs": pairs_path,
        "spectral_pairs": spectral_pairs_path,
        "manifest": manifest_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names", type=_parse_names, default=DEFAULT_CASE_NAMES)
    parser.add_argument("--branch-sources", type=_parse_names, default=BRANCH_SOURCES)
    parser.add_argument(
        "--spectral-contexts",
        type=_parse_names,
        default=DEFAULT_SPECTRAL_CONTEXTS,
    )
    parser.add_argument("--tree-linkage-method", default="average")
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--pair-sample-size", type=int, default=50_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_nnls_null_calibration_sweep(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=tuple(args.case_names),
        branch_sources=tuple(args.branch_sources),
        spectral_contexts=tuple(args.spectral_contexts),
        tree_linkage_method=str(args.tree_linkage_method),
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        pair_sample_size=int(args.pair_sample_size),
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
