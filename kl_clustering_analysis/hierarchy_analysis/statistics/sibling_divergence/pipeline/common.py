"""Common collection and BH-application helpers for sibling pipelines."""

from __future__ import annotations

import logging
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.core_utils.data_utils import extract_bool_column_dict

from ...multiple_testing import benjamini_hochberg_correction


def collect_significant_sibling_pairs(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    get_binary_children: Callable[[nx.DiGraph, str], tuple[str, str] | None],
    either_child_significant: Callable[[str, str, dict[str, bool]], bool],
) -> tuple[list[str], list[tuple[str, str]], list[str], list[str]]:
    """Collect binary-child parent nodes and flag skipped/non-binary nodes."""
    if "Child_Parent_Divergence_Significant" not in annotations_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. Run child-parent test first."
        )

    edge_significance_by_node = extract_bool_column_dict(
        annotations_df,
        "Child_Parent_Divergence_Significant",
    )

    parents: list[str] = []
    child_pairs: list[tuple[str, str]] = []
    skipped: list[str] = []
    non_binary: list[str] = []

    for parent in tree.nodes:
        children = get_binary_children(tree, parent)
        if children is None:
            non_binary.append(parent)
            continue

        left, right = children
        if not either_child_significant(left, right, edge_significance_by_node):
            skipped.append(parent)
            continue

        parents.append(parent)
        child_pairs.append((left, right))

    return parents, child_pairs, skipped, non_binary


def apply_sibling_bh_results(
    annotations_df: pd.DataFrame,
    parents: list[str],
    results: list[tuple[float, float, float]],
    alpha: float,
    *,
    logger: logging.Logger,
    audit_label: str,
    method_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Apply sibling test results and BH correction onto the dataframe."""
    if not results:
        return annotations_df

    test_statistics = np.array([r[0] for r in results])
    degrees_of_freedom_values = np.array([r[1] for r in results])
    p_values = np.array([r[2] for r in results])

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
            len(results),
            n_invalid,
        )

    annotations_df.loc[parents, "Sibling_Test_Statistic"] = test_statistics
    annotations_df.loc[parents, "Sibling_Degrees_of_Freedom"] = degrees_of_freedom_values
    annotations_df.loc[parents, "Sibling_Divergence_P_Value"] = p_values
    annotations_df.loc[parents, "Sibling_Divergence_P_Value_Corrected"] = corrected_p_values
    annotations_df.loc[parents, "Sibling_Divergence_Invalid"] = invalid_mask
    annotations_df.loc[parents, "Sibling_BH_Different"] = reject
    annotations_df.loc[parents, "Sibling_BH_Same"] = ~reject

    if method_labels is not None:
        annotations_df.loc[parents, "Sibling_Test_Method"] = method_labels

    return annotations_df


__all__ = [
    "apply_sibling_bh_results",
    "collect_significant_sibling_pairs",
]
