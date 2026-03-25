"""FDR correction and DataFrame annotation for sibling divergence results.

Provides the shared implementation for applying Benjamini-Hochberg
correction to sibling test results and writing them onto the annotations
DataFrame. Also contains small DataFrame-level helpers for column
initialization, skipped-node marking, and early returns.
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from kl_clustering_analysis.core_utils.data_utils import initialize_sibling_divergence_columns

from ..multiple_testing import benjamini_hochberg_correction


def apply_sibling_bh_results(
    annotations_df: pd.DataFrame,
    parents: list[str],
    results: list[tuple[float, float, float]],
    alpha: float,
    *,
    logger: logging.Logger,
    audit_label: str,
    method_labels: list[str] | None = None,
    skipped_parents: list[str] | None = None,
) -> pd.DataFrame:
    """Apply sibling test results with BH correction onto the DataFrame."""
    if skipped_parents:
        annotations_df.loc[skipped_parents, "Sibling_Divergence_Skipped"] = True

    if not results:
        return annotations_df

    test_statistics = np.array([r[0] for r in results])
    degrees_of_freedom_values = np.array([r[1] for r in results])
    p_values = np.array([r[2] for r in results])

    invalid_test_flags = (
        (~np.isfinite(test_statistics))
        | (~np.isfinite(degrees_of_freedom_values))
        | (~np.isfinite(p_values))
    )

    p_values_for_correction = np.where(np.isfinite(p_values), p_values, 1.0)

    reject, corrected_p_values, _ = benjamini_hochberg_correction(
        p_values_for_correction,
        alpha=alpha,
    )
    reject = np.where(invalid_test_flags, False, reject)

    n_invalid = int(np.sum(invalid_test_flags))
    if n_invalid:
        logger.warning(
            "%s audit: total_tests=%d, invalid_tests=%d. "
            "Conservative correction path applied (p=1.0, reject=False).",
            audit_label,
            len(results),
            n_invalid,
        )

    annotations_df.loc[parents, "Sibling_Test_Statistic"] = test_statistics
    annotations_df.loc[parents, "Sibling_Degrees_of_Freedom"] = degrees_of_freedom_values
    annotations_df.loc[parents, "Sibling_Divergence_P_Value"] = p_values
    annotations_df.loc[parents, "Sibling_Divergence_P_Value_Corrected"] = corrected_p_values
    annotations_df.loc[parents, "Sibling_Divergence_Invalid"] = invalid_test_flags
    annotations_df.loc[parents, "Sibling_BH_Different"] = reject
    annotations_df.loc[parents, "Sibling_BH_Same"] = ~reject

    if method_labels is not None:
        annotations_df.loc[parents, "Sibling_Test_Method"] = method_labels

    return annotations_df


def init_sibling_annotation_df(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and initialize the standard sibling-annotation columns."""
    if len(annotations_df) == 0:
        raise ValueError("Empty dataframe")
    return initialize_sibling_divergence_columns(annotations_df.copy())


def mark_non_binary_as_skipped(
    annotations_df: pd.DataFrame,
    non_binary_nodes: Sequence[str],
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Mark non-binary or leaf nodes as skipped sibling tests."""
    if not non_binary_nodes:
        return
    annotations_df.loc[list(non_binary_nodes), "Sibling_Divergence_Skipped"] = True
    if logger is not None:
        logger.debug("Non-binary/leaf nodes marked as skipped: %d", len(non_binary_nodes))


def early_return_if_no_records(
    annotations_df: pd.DataFrame,
    records: Sequence[object],
    *,
    warning_message: str = "No eligible parent nodes for sibling tests",
) -> pd.DataFrame | None:
    """Return the DataFrame early when there are no eligible records."""
    if records:
        return None
    warnings.warn(warning_message, UserWarning)
    return annotations_df


__all__ = [
    "apply_sibling_bh_results",
    "early_return_if_no_records",
    "init_sibling_annotation_df",
    "mark_non_binary_as_skipped",
]
