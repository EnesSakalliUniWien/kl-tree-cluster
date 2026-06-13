"""Public pipeline for inflated projected-Wald sibling annotation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace

from ..inflation_correction.empirical_null_inflation_estimation import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    fit_empirical_null_inflation_model,
)
from ..inflation_correction.external_selected_tail_calibration import (
    ExternalSelectedTailCalibrationModel,
)
from ..inflation_correction.inflation_adjusted_sibling_tests import (
    compute_inflation_adjusted_sibling_tests,
)
from ..inflation_correction.types.inflation_model import CalibrationSupportThresholds
from ..pair_testing.collection.record_collection import collect_sibling_pair_records
from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .fdr_annotation import (
    apply_traversal_aligned_sibling_bh_results,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .projection_dimension_annotation import (
    write_record_projection_dimensions,
)


def _validate_focal_sibling_records(records: list[SiblingPairRecord]) -> None:
    """Validate focal sibling records before calibration fitting."""
    for record in records:
        if record.is_null_like:
            continue
        if not np.isfinite(record.stat):
            raise ValueError(
                "Sibling record must have a finite statistic before adjustment; "
                f"parent={record.parent!r}."
            )
        if record.degrees_of_freedom < 0:
            raise ValueError(
                "Sibling record must have non-negative degrees of freedom before "
                f"adjustment; parent={record.parent!r}."
            )


def annotate_sibling_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    sibling_projection_dimensions_from_edge_comparisons: dict[str, int],
    parent_principal_component_projections: dict[str, np.ndarray],
    parent_principal_component_eigenvalues: dict[str, np.ndarray],
    significance_level_alpha: float = DEFAULT_SIBLING_ALPHA,
    feature_space: FeatureSpace | None = None,
    enforce_support_thresholds: bool = False,
    support_thresholds: CalibrationSupportThresholds = DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
    external_selected_tail_model: ExternalSelectedTailCalibrationModel | None = None,
    external_selected_tail_context_by_parent: Mapping[
        object, Mapping[str, object]
    ] | None = None,
    stage_timings: MutableMapping[str, float] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence using context-weighted empirical-null inflation."""
    annotations_df = init_sibling_annotation_df(annotations_df)

    collection_start_sec = perf_counter()
    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        sibling_projection_dimensions_from_edge_comparisons=(
            sibling_projection_dimensions_from_edge_comparisons
        ),
        parent_principal_component_projections=parent_principal_component_projections,
        parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
        feature_space=feature_space,
    )
    if stage_timings is not None:
        stage_timings["sibling_gate_pair_record_collection_sec"] = float(
            stage_timings.get("sibling_gate_pair_record_collection_sec", 0.0)
        ) + float(perf_counter() - collection_start_sec)

    mark_non_binary_as_skipped(annotations_df, non_binary)

    if not records:
        return annotations_df

    write_record_projection_dimensions(annotations_df, records)
    _validate_focal_sibling_records(records)
    n_focal = sum(not record.is_null_like for record in records)

    skipped_parents = [record.parent for record in records if record.is_null_like]
    if n_focal == 0:
        sibling_fdr_start_sec = perf_counter()
        result_df = apply_traversal_aligned_sibling_bh_results(
            tree,
            annotations_df,
            [],
            [],
            significance_level_alpha,
            skipped_parents=skipped_parents,
        )
        if stage_timings is not None:
            stage_timings["sibling_gate_fdr_sec"] = float(
                stage_timings.get("sibling_gate_fdr_sec", 0.0)
            ) + float(perf_counter() - sibling_fdr_start_sec)
        return result_df

    inflation_fit_start_sec = perf_counter()
    try:
        model = fit_empirical_null_inflation_model(records)
    except ValueError:
        if external_selected_tail_model is None:
            raise
        model = None
    if stage_timings is not None:
        stage_timings["sibling_gate_inflation_fit_sec"] = float(
            stage_timings.get("sibling_gate_inflation_fit_sec", 0.0)
        ) + float(perf_counter() - inflation_fit_start_sec)

    adjusted_tests_start_sec = perf_counter()
    (
        tested_parent_ids,
        inflation_adjusted_test_summaries,
        inflation_adjustment_method_labels,
    ) = compute_inflation_adjusted_sibling_tests(
        records,
        model=model,
        enforce_support_thresholds=enforce_support_thresholds,
        support_thresholds=support_thresholds,
        external_selected_tail_model=external_selected_tail_model,
        external_selected_tail_context_by_parent=external_selected_tail_context_by_parent,
    )
    if stage_timings is not None:
        stage_timings["sibling_gate_adjusted_tests_sec"] = float(
            stage_timings.get("sibling_gate_adjusted_tests_sec", 0.0)
        ) + float(perf_counter() - adjusted_tests_start_sec)

    sibling_fdr_start_sec = perf_counter()
    annotations_df = apply_traversal_aligned_sibling_bh_results(
        tree,
        annotations_df,
        tested_parent_ids,
        inflation_adjusted_test_summaries,
        significance_level_alpha,
        method_labels=inflation_adjustment_method_labels,
        skipped_parents=skipped_parents,
    )
    if stage_timings is not None:
        stage_timings["sibling_gate_fdr_sec"] = float(
            stage_timings.get("sibling_gate_fdr_sec", 0.0)
        ) + float(perf_counter() - sibling_fdr_start_sec)

    return annotations_df


__all__ = ["annotate_sibling_divergence"]
