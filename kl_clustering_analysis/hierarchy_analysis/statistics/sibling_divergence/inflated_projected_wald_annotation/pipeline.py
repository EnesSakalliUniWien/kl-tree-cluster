"""Public pipeline for inflated projected-Wald sibling annotation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config

from ...branch_length_utils import compute_mean_branch_length
from ..inflation_correction.empirical_null_inflation_estimation import (
    fit_empirical_null_inflation_model,
)
from ..inflation_correction.inflation_adjusted_sibling_tests import (
    compute_inflation_adjusted_sibling_tests,
)
from ..pair_testing.collection.record_collection import collect_sibling_pair_records
from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .fdr_annotation import (
    apply_sibling_bh_results,
    early_return_if_no_records,
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
    significance_level_alpha: float = config.SIBLING_ALPHA,
) -> pd.DataFrame:
    """Test sibling divergence using context-weighted empirical-null inflation."""
    annotations_df = init_sibling_annotation_df(annotations_df)

    mean_branch_length = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length,
        sibling_projection_dimensions_from_edge_comparisons=(
            sibling_projection_dimensions_from_edge_comparisons
        ),
        parent_principal_component_projections=parent_principal_component_projections,
        parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
    )

    mark_non_binary_as_skipped(annotations_df, non_binary)

    early_annotations_df = early_return_if_no_records(annotations_df, records)
    if early_annotations_df is not None:
        return early_annotations_df

    write_record_projection_dimensions(annotations_df, records)
    _validate_focal_sibling_records(records)
    n_focal = sum(not record.is_null_like for record in records)

    skipped_parents = [record.parent for record in records if record.is_null_like]
    if n_focal == 0:
        return apply_sibling_bh_results(
            annotations_df,
            [],
            [],
            significance_level_alpha,
            skipped_parents=skipped_parents,
        )

    model = fit_empirical_null_inflation_model(records)

    (
        tested_parent_ids,
        inflation_adjusted_test_summaries,
        inflation_adjustment_method_labels,
    ) = compute_inflation_adjusted_sibling_tests(
        records,
        model=model,
        significance_level_alpha=significance_level_alpha,
    )

    annotations_df = apply_sibling_bh_results(
        annotations_df,
        tested_parent_ids,
        inflation_adjusted_test_summaries,
        significance_level_alpha,
        method_labels=inflation_adjustment_method_labels,
        skipped_parents=skipped_parents,
    )

    return annotations_df


__all__ = ["annotate_sibling_divergence"]
