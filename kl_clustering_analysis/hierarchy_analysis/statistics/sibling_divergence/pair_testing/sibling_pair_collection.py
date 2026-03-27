"""Tree traversal and pair collection for divergence testing.

Provides the shared infrastructure for walking a hierarchical tree,
running the sibling Wald kernel on binary child pairs, and packaging
results into ``SiblingPairRecord`` objects for the production
cousin-adjusted Wald pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis.core_utils.data_utils import (
    extract_bool_column_dict,
    extract_node_distribution,
    extract_node_sample_size,
)

from ...projection.chi2_pvalue import WhiteningMode
from .types import SiblingPairRecord
from .wald_statistic import sibling_divergence_test

AdjustedSiblingTestSummary = tuple[float, float, float]


def get_binary_children(tree: nx.DiGraph, parent: str) -> tuple[str, str] | None:
    """Return (left, right) children if parent has exactly 2, else None."""
    children = list(tree.successors(parent))
    if len(children) != 2:
        return None
    return children[0], children[1]


def get_sibling_data(
    tree: nx.DiGraph,
    parent: str,
    left: str,
    right: str,
) -> tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]:
    """Extract distributions, sample sizes, and branch lengths for a sibling pair.

    Branch lengths are extracted from the tree edges (parent → child).
    If not available, returns None for the branch lengths.
    """
    left_branch = tree.edges[parent, left].get("branch_length")
    right_branch = tree.edges[parent, right].get("branch_length")

    return (
        extract_node_distribution(tree, left),
        extract_node_distribution(tree, right),
        extract_node_sample_size(tree, left),
        extract_node_sample_size(tree, right),
        left_branch,
        right_branch,
    )


def _branch_length_sum(
    branch_length_left: float | None, branch_length_right: float | None
) -> float:
    """Sum sibling branch lengths. Returns 0.0 only when both are None."""
    if branch_length_left is not None and branch_length_right is not None:
        return float(branch_length_left + branch_length_right)
    if branch_length_left is None and branch_length_right is None:
        return 0.0
    raise ValueError(
        f"Inconsistent branch lengths: left={branch_length_left!r}, right={branch_length_right!r}. "
        "Supply either both branch lengths or neither."
    )


def _is_gate2_blocked_for_pair(
    left: str,
    right: str,
    edge_tested_by_node: dict[str, bool] | None,
    edge_ancestor_blocked_by_node: dict[str, bool] | None,
) -> bool:
    """Return whether either child lacks a usable Gate 2 edge annotation."""
    left_edge_tested = edge_tested_by_node.get(left, True) if edge_tested_by_node else True
    right_edge_tested = edge_tested_by_node.get(right, True) if edge_tested_by_node else True

    left_edge_blocked = (
        edge_ancestor_blocked_by_node.get(left, False) if edge_ancestor_blocked_by_node else False
    )
    right_edge_blocked = (
        edge_ancestor_blocked_by_node.get(right, False) if edge_ancestor_blocked_by_node else False
    )

    return (
        (not left_edge_tested) or (not right_edge_tested) or left_edge_blocked or right_edge_blocked
    )


def _resolve_structural_dimension(
    *,
    spectral_dimension: int | None,
    degrees_of_freedom: float,
) -> float:
    if spectral_dimension is not None and spectral_dimension > 0:
        return float(spectral_dimension)
    if np.isfinite(degrees_of_freedom) and degrees_of_freedom > 0:
        return float(degrees_of_freedom)
    return 0.0


def collect_sibling_pair_records(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    mean_branch_length: float | None,
    *,
    spectral_dims: dict[str, int] | None = None,
    pca_projections: dict[str, np.ndarray] | None = None,
    pca_eigenvalues: dict[str, np.ndarray] | None = None,
    whitening: WhiteningMode = "per_component",
) -> tuple[list[SiblingPairRecord], list[str]]:
    """Collect raw sibling-test records for ALL binary-child parent nodes.

    Runs the Wald kernel on every pair and tags each as null-like or focal
    based on the edge (child-parent) significance column.  Null-like pairs
    are needed for calibration (they estimate pure post-selection inflation);
    only focal pairs are subsequently tested (deflated + BH-corrected).

    Returns (records, non_binary_nodes).
    """
    if "Child_Parent_Divergence_Significant" not in annotations_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. Run child-parent test first."
        )

    edge_significance_by_node = extract_bool_column_dict(
        annotations_df,
        "Child_Parent_Divergence_Significant",
    )

    # Extract BH-corrected edge p-values for continuous calibration weights.
    # NaN values (e.g. skipped leaf nodes) default to 1.0.
    edge_p_value_column = "Child_Parent_Divergence_P_Value_BH"
    if edge_p_value_column not in annotations_df.columns:
        raise ValueError(f"Missing '{edge_p_value_column}' column. Run child-parent test first.")

    edge_p_value_series = annotations_df[edge_p_value_column].astype(float)

    edge_p_value_by_node: dict[str, float] = {
        str(node_id): (
            float(edge_p_value_series[node_id])
            if np.isfinite(edge_p_value_series[node_id])
            else 1.0
        )
        for node_id in annotations_df.index
    }

    edge_tested_by_node: dict[str, bool] | None = None
    edge_ancestor_blocked_by_node: dict[str, bool] | None = None
    if "Child_Parent_Divergence_Tested" in annotations_df.columns:
        edge_tested_by_node = extract_bool_column_dict(
            annotations_df,
            "Child_Parent_Divergence_Tested",
        )
    if "Child_Parent_Divergence_Ancestor_Blocked" in annotations_df.columns:
        edge_ancestor_blocked_by_node = extract_bool_column_dict(
            annotations_df,
            "Child_Parent_Divergence_Ancestor_Blocked",
        )

    records: list[SiblingPairRecord] = []
    non_binary_nodes: list[str] = []

    for parent in tree.nodes:
        children = get_binary_children(tree, parent)
        if children is None:
            non_binary_nodes.append(parent)
            continue

        left, right = children
        (
            left_distribution,
            right_distribution,
            left_sample_size,
            right_sample_size,
            branch_length_left,
            branch_length_right,
        ) = get_sibling_data(tree, parent, left, right)

        spectral_k = spectral_dims.get(parent) if spectral_dims else None
        pca_projection = pca_projections.get(parent) if pca_projections else None
        node_pca_eigenvalues = pca_eigenvalues.get(parent) if pca_eigenvalues else None

        test_statistic, degrees_of_freedom, p_value = sibling_divergence_test(
            left_distribution,
            right_distribution,
            float(left_sample_size),
            float(right_sample_size),
            branch_length_left=branch_length_left,
            branch_length_right=branch_length_right,
            mean_branch_length=mean_branch_length,
            test_id=f"sibling:{parent}",
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=node_pca_eigenvalues,
            whitening=whitening,
        )

        is_gate2_blocked = _is_gate2_blocked_for_pair(
            left,
            right,
            edge_tested_by_node,
            edge_ancestor_blocked_by_node,
        )

        is_null_like = not (
            edge_significance_by_node.get(left, False)
            or edge_significance_by_node.get(right, False)
        )
        # Sibling null prior: min(p_edge_left, p_edge_right).
        # High value → both children look null-like (high edge p-values).
        # Low value → at least one child has strong edge signal.
        # Blocked-edge adjustments are handled downstream by interpolate_sibling_null_priors().
        sibling_null_prior_from_edge_pvalue = min(
            edge_p_value_by_node.get(left, 1.0),
            edge_p_value_by_node.get(right, 1.0),
        )

        records.append(
            SiblingPairRecord(
                parent=parent,
                left=left,
                right=right,
                stat=test_statistic,
                degrees_of_freedom=(
                    float(degrees_of_freedom) if np.isfinite(degrees_of_freedom) else 0.0
                ),
                p_value=p_value,
                branch_length_sum=_branch_length_sum(branch_length_left, branch_length_right),
                n_parent=extract_node_sample_size(tree, parent),
                is_null_like=is_null_like,
                is_gate2_blocked=is_gate2_blocked,
                sibling_null_prior_from_edge_pvalue=sibling_null_prior_from_edge_pvalue,
                structural_dimension=_resolve_structural_dimension(
                    spectral_dimension=spectral_k,
                    degrees_of_freedom=float(degrees_of_freedom),
                ),
            )
        )

    return records, non_binary_nodes


# =============================================================================
# Deflation
# =============================================================================


def _compute_adjusted_sibling_test(
    sibling_test_record: SiblingPairRecord,
    *,
    resolve_inflation_adjustment: Callable[[SiblingPairRecord], tuple[float, str]],
) -> tuple[AdjustedSiblingTestSummary, str] | None:
    """Return one adjusted sibling-test summary, or ``None`` for null-like records."""
    if sibling_test_record.is_null_like:
        return None

    if (
        not np.isfinite(sibling_test_record.stat)
        or sibling_test_record.degrees_of_freedom <= 0
    ):
        return (np.nan, np.nan, np.nan), "invalid"

    estimated_inflation_factor, adjustment_method_label = resolve_inflation_adjustment(
        sibling_test_record
    )
    adjusted_statistic = sibling_test_record.stat / estimated_inflation_factor
    adjusted_degrees_of_freedom = float(sibling_test_record.degrees_of_freedom)
    adjusted_p_value = float(chi2.sf(adjusted_statistic, df=adjusted_degrees_of_freedom))
    return (
        adjusted_statistic,
        adjusted_degrees_of_freedom,
        adjusted_p_value,
    ), adjustment_method_label


def compute_adjusted_sibling_tests(
    sibling_test_records: Iterable[SiblingPairRecord],
    *,
    resolve_inflation_adjustment: Callable[[SiblingPairRecord], tuple[float, str]],
) -> tuple[list[str], list[AdjustedSiblingTestSummary], list[str]]:
    """Return adjusted sibling-test summaries and adjustment labels for tested parents."""
    tested_parent_ids: list[str] = []
    adjusted_test_summaries: list[AdjustedSiblingTestSummary] = []
    adjustment_method_labels: list[str] = []

    for sibling_test_record in sibling_test_records:
        adjusted_test_summary = _compute_adjusted_sibling_test(
            sibling_test_record,
            resolve_inflation_adjustment=resolve_inflation_adjustment,
        )
        if adjusted_test_summary is None:
            continue

        test_summary, adjustment_method_label = adjusted_test_summary
        tested_parent_ids.append(sibling_test_record.parent)
        adjusted_test_summaries.append(test_summary)
        adjustment_method_labels.append(adjustment_method_label)

    return tested_parent_ids, adjusted_test_summaries, adjustment_method_labels


def count_null_focal_pairs(records: Iterable[SiblingPairRecord]) -> tuple[int, int, int]:
    """Count (null-like, focal, gate2-blocked) record totals."""
    n_null = 0
    n_focal = 0
    n_blocked = 0
    for record in records:
        if record.is_null_like:
            n_null += 1
        else:
            n_focal += 1
        if record.is_gate2_blocked:
            n_blocked += 1
    return n_null, n_focal, n_blocked


__all__ = [
    "collect_sibling_pair_records",
    "compute_adjusted_sibling_tests",
    "count_null_focal_pairs",
    "get_binary_children",
    "get_sibling_data",
]
