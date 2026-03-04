"""Shared helpers for cousin-based sibling calibration pipelines."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis.core_utils.data_utils import (
    extract_bool_column_dict,
    extract_node_sample_size,
    initialize_sibling_divergence_columns,
)

from ..multiple_testing import benjamini_hochberg_correction
from .sibling_divergence_test import (
    _either_child_significant,
    _get_binary_children,
    _get_sibling_data,
    sibling_divergence_test,
)


@dataclass
class SiblingPairRecord:
    """Raw per-parent sibling-test record shared by cousin methods."""

    parent: str
    left: str
    right: str
    stat: float
    degrees_of_freedom: int
    p_value: float
    branch_length_sum: float
    n_parent: int
    is_null_like: bool


class DeflatableSiblingRecord(Protocol):
    """Structural type for per-node records that can be focal-deflated."""

    parent: str
    stat: float
    degrees_of_freedom: int
    is_null_like: bool


def _branch_length_sum(branch_length_left: float | None, branch_length_right: float | None) -> float:
    if branch_length_left is not None and branch_length_right is not None:
        return float(branch_length_left + branch_length_right)
    if branch_length_left is not None:
        return float(branch_length_left)
    if branch_length_right is not None:
        return float(branch_length_right)
    return 0.0


def collect_sibling_pair_records(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    mean_branch_length: float | None,
    *,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
) -> tuple[list[SiblingPairRecord], list[str]]:
    """Collect raw sibling-test records for all binary-child parent nodes."""
    if "Child_Parent_Divergence_Significant" not in annotations_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. Run child-parent test first."
        )

    edge_significance_by_node = extract_bool_column_dict(
        annotations_df,
        "Child_Parent_Divergence_Significant",
    )
    records: list[SiblingPairRecord] = []
    non_binary: list[str] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            non_binary.append(parent)
            continue

        left, right = children
        (
            left_dist,
            right_dist,
            n_left,
            n_right,
            branch_length_left,
            branch_length_right,
        ) = _get_sibling_data(tree, parent, left, right)

        spectral_k = spectral_dims.get(parent) if spectral_dims else None
        pca_projection = pca_projections.get(parent) if pca_projections else None
        pca_eigenvalue = pca_eigenvalues.get(parent) if pca_eigenvalues else None

        stat, degrees_of_freedom, p_value = sibling_divergence_test(
            left_dist,
            right_dist,
            float(n_left),
            float(n_right),
            branch_length_left=branch_length_left,
            branch_length_right=branch_length_right,
            mean_branch_length=mean_branch_length,
            test_id=f"sibling:{parent}",
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=pca_eigenvalue,
        )

        is_null = not _either_child_significant(left, right, edge_significance_by_node)
        records.append(
            SiblingPairRecord(
                parent=parent,
                left=left,
                right=right,
                stat=stat,
                degrees_of_freedom=int(degrees_of_freedom)
                if np.isfinite(degrees_of_freedom)
                else 0,
                p_value=p_value,
                branch_length_sum=_branch_length_sum(branch_length_left, branch_length_right),
                n_parent=extract_node_sample_size(tree, parent),
                is_null_like=is_null,
            )
        )

    return records, non_binary


def deflate_focal_pairs(
    records: Iterable[DeflatableSiblingRecord],
    *,
    calibration_resolver: Callable[[DeflatableSiblingRecord], tuple[float, str]],
) -> tuple[list[str], list[tuple[float, float, float]], list[str]]:
    """Deflate all focal records and return parent IDs, adjusted triples, and methods."""
    focal_parents: list[str] = []
    focal_results: list[tuple[float, float, float]] = []
    methods: list[str] = []

    for pair_record in records:
        if pair_record.is_null_like:
            continue

        if not np.isfinite(pair_record.stat) or pair_record.degrees_of_freedom <= 0:
            focal_parents.append(pair_record.parent)
            focal_results.append((np.nan, np.nan, np.nan))
            methods.append("invalid")
            continue

        inflation_factor, method = calibration_resolver(pair_record)
        t_adj = pair_record.stat / inflation_factor
        p_adj = float(chi2.sf(t_adj, df=pair_record.degrees_of_freedom))

        focal_parents.append(pair_record.parent)
        focal_results.append((t_adj, float(pair_record.degrees_of_freedom), p_adj))
        methods.append(method)

    return focal_parents, focal_results, methods


def init_sibling_annotation_df(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and initialize the standard sibling-annotation columns."""
    if len(annotations_df) == 0:
        raise ValueError("Empty dataframe")

    annotations_dataframe = annotations_df.copy()
    return initialize_sibling_divergence_columns(annotations_dataframe)


def mark_non_binary_as_skipped(
    annotations_df: pd.DataFrame,
    non_binary_nodes: Sequence[str],
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Mark non-binary/leaf nodes as skipped sibling tests."""
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
    """Return the dataframe early when there are no eligible records."""
    if records:
        return None

    warnings.warn(warning_message, UserWarning)
    return annotations_df


def count_null_focal_pairs(records: Sequence[DeflatableSiblingRecord]) -> tuple[int, int]:
    """Count (null-like, focal) record totals."""
    n_null = sum(1 for pair_record in records if pair_record.is_null_like)
    n_focal = len(records) - n_null
    return n_null, n_focal


def apply_calibrated_results(
    annotations_df: pd.DataFrame,
    focal_parents: list[str],
    focal_results: list[tuple[float, float, float]],
    calibration_methods: list[str],
    skipped_parents: list[str],
    alpha: float,
    *,
    logger: logging.Logger,
    audit_label: str,
) -> pd.DataFrame:
    """Apply calibrated focal sibling results and BH correction onto a dataframe."""
    if skipped_parents:
        annotations_df.loc[skipped_parents, "Sibling_Divergence_Skipped"] = True

    if not focal_results:
        return annotations_df

    test_statistics = np.array([r[0] for r in focal_results])
    degrees_of_freedom_values = np.array([r[1] for r in focal_results])
    p_values = np.array([r[2] for r in focal_results])

    invalid_mask = (
        (~np.isfinite(test_statistics))
        | (~np.isfinite(degrees_of_freedom_values))
        | (~np.isfinite(p_values))
    )
    p_values_for_correction = np.where(np.isfinite(p_values), p_values, 1.0)

    reject, corrected_p_values, _ = benjamini_hochberg_correction(
        p_values_for_correction,
        alpha=alpha,
    )
    reject = np.where(invalid_mask, False, reject)

    n_invalid = int(np.sum(invalid_mask))
    if n_invalid:
        logger.warning(
            "%s audit: total_tests=%d, invalid_tests=%d. "
            "Conservative correction path applied (p=1.0, reject=False).",
            audit_label,
            len(focal_results),
            n_invalid,
        )

    annotations_df.loc[focal_parents, "Sibling_Test_Statistic"] = test_statistics
    annotations_df.loc[focal_parents, "Sibling_Degrees_of_Freedom"] = degrees_of_freedom_values
    annotations_df.loc[focal_parents, "Sibling_Divergence_P_Value"] = p_values
    annotations_df.loc[focal_parents, "Sibling_Divergence_P_Value_Corrected"] = corrected_p_values
    annotations_df.loc[focal_parents, "Sibling_Divergence_Invalid"] = invalid_mask
    annotations_df.loc[focal_parents, "Sibling_BH_Different"] = reject
    annotations_df.loc[focal_parents, "Sibling_BH_Same"] = ~reject
    annotations_df.loc[focal_parents, "Sibling_Test_Method"] = calibration_methods

    return annotations_df


__all__ = [
    "DeflatableSiblingRecord",
    "SiblingPairRecord",
    "count_null_focal_pairs",
    "collect_sibling_pair_records",
    "deflate_focal_pairs",
    "early_return_if_no_records",
    "init_sibling_annotation_df",
    "mark_non_binary_as_skipped",
    "apply_calibrated_results",
]
