"""Public pipeline for calibrated projected Wald sibling annotation."""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.legacy_methods.commit_c2ef9a69.kl_clustering_analysis import config

from ...branch_length_utils import compute_mean_branch_length
from ..inflation_correction.adjusted_sibling_tests import (
    compute_adjusted_sibling_tests,
    count_null_focal_pairs,
)
from ..inflation_correction.conditional_deflation import fit_sibling_inflation_calibrator
from ..inflation_correction.inflation_estimation import fit_inflation_model
from ..pair_testing.collection.record_collection import collect_sibling_pair_records
from ..pair_testing.sibling_null_prior_interpolation.sibling_null_prior_interpolation import (
    interpolate_sibling_null_priors,
)
from ..pair_testing.types.sibling_pair_record import SiblingPairRecord
from .calibration import _resolve_calibration
from .fdr_annotation import (
    apply_sibling_bh_results,
    early_return_if_no_records,
    init_sibling_annotation_df,
    mark_non_binary_as_skipped,
)
from .metadata import build_sibling_divergence_audit, write_record_projection_metadata

logger = logging.getLogger(__name__)


def annotate_sibling_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    sibling_projection_dimensions_from_edge_comparisons: Dict[str, int] | None = None,
    parent_principal_component_projections: Dict[str, np.ndarray] | None = None,
    parent_principal_component_eigenvalues: Dict[str, np.ndarray] | None = None,
    edge_projection_dimensions_by_node: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence using calibrated projected Wald."""
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

    mark_non_binary_as_skipped(annotations_df, non_binary, logger=logger)

    early_annotations_df = early_return_if_no_records(annotations_df, records)
    if early_annotations_df is not None:
        return early_annotations_df

    write_record_projection_metadata(annotations_df, records)
    n_null, n_focal, n_blocked = count_null_focal_pairs(records)

    if n_blocked > 0:
        records = interpolate_sibling_null_priors(
            records,
            tree,
            annotations_df,
            edge_projection_dimensions_by_node=edge_projection_dimensions_by_node,
        )

    calibration_records = list(records)
    excluded_from_calibration_records: list[SiblingPairRecord] = []

    model = fit_inflation_model(calibration_records)
    calibrator = fit_sibling_inflation_calibrator(
        calibration_records,
        model,
    )

    tested_parent_ids, adjusted_test_summaries, adjustment_method_labels = (
        compute_adjusted_sibling_tests(
            records,
            resolve_inflation_adjustment=partial(_resolve_calibration, calibrator=calibrator),
        )
    )

    skipped_parents = [record.parent for record in records if record.is_null_like]

    annotations_df = apply_sibling_bh_results(
        annotations_df,
        tested_parent_ids,
        adjusted_test_summaries,
        significance_level_alpha,
        logger=logger,
        audit_label="Calibrated projected Wald",
        method_labels=adjustment_method_labels,
        skipped_parents=skipped_parents,
    )

    annotations_df.attrs["sibling_divergence_audit"] = build_sibling_divergence_audit(
        records=records,
        calibration_records=calibration_records,
        excluded_from_calibration_records=excluded_from_calibration_records,
        model=model,
        calibrator=calibrator,
        n_null=n_null,
        n_focal=n_focal,
        n_blocked=n_blocked,
    )
    annotations_df.attrs["_calibration_model"] = model

    return annotations_df


__all__ = ["annotate_sibling_divergence"]
