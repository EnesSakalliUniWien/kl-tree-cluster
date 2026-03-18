"""Tree traversal and pair collection for divergence testing.

Provides the shared infrastructure for walking a hierarchical tree,
identifying eligible sibling pairs, running the Wald kernel on each,
and packaging results into ``SiblingPairRecord`` objects.

Used by both the standard Wald orchestrator (``standard_wald``) and the
cousin-adjusted Wald orchestrator (``adjusted_wald``).
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis.core_utils.data_utils import (
    extract_bool_column_dict,
    extract_node_distribution,
    extract_node_sample_size,
)

from .types import _R, DeflatableSiblingRecord, SiblingPairRecord
from .wald_statistic import sibling_divergence_test

# =============================================================================
# Tree helpers
# =============================================================================


def get_binary_children(tree: nx.DiGraph, parent: str) -> Optional[Tuple[str, str]]:
    """Return (left, right) children if parent has exactly 2, else None."""
    children = list(tree.successors(parent))
    if len(children) != 2:
        return None
    return children[0], children[1]


def either_child_significant(
    left: str,
    right: str,
    edge_significance_by_node: Dict[str, bool],
) -> bool:
    """Check if at least one child has significant child-parent divergence."""
    return edge_significance_by_node.get(left, False) or edge_significance_by_node.get(right, False)


def get_sibling_data(
    tree: nx.DiGraph,
    parent: str,
    left: str,
    right: str,
) -> Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]:
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
    """Sum sibling branch lengths, treating None as zero."""
    if branch_length_left is not None and branch_length_right is not None:
        return float(branch_length_left + branch_length_right)
    if branch_length_left is not None:
        return float(branch_length_left)
    if branch_length_right is not None:
        return float(branch_length_right)
    return 0.0


# =============================================================================
# Collection: standard Wald path
# =============================================================================


def collect_significant_sibling_pairs(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
) -> Tuple[List[str], List[Tuple[str, str]], List[str], List[str]]:
    """Collect binary-child parent nodes where at least one child is edge-significant.

    Returns (parents, child_pairs, skipped, non_binary).
    """
    if "Child_Parent_Divergence_Significant" not in annotations_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. Run child-parent test first."
        )

    edge_significance_by_node = extract_bool_column_dict(
        annotations_df,
        "Child_Parent_Divergence_Significant",
    )

    parents: List[str] = []
    child_pairs: List[Tuple[str, str]] = []
    skipped: List[str] = []
    non_binary: List[str] = []

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


# =============================================================================
# Collection: calibrated Wald path (records-based)
# =============================================================================


def collect_sibling_pair_records(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    mean_branch_length: float | None,
    *,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
    pca_eigenvalues: Dict[str, np.ndarray] | None = None,
    whitening: WhiteningMode = "per_component",
) -> Tuple[List[SiblingPairRecord], List[str]]:
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
    # NaN values (e.g. skipped leaf nodes) default to 1.0 (maximally null-like).
    edge_p_value_column = "Child_Parent_Divergence_P_Value_BH"
    if edge_p_value_column not in annotations_df.columns:
        raise ValueError(f"Missing '{edge_p_value_column}' column. Run child-parent test first.")

    edge_p_value_series = annotations_df[edge_p_value_column].astype(float)
    edge_p_value_by_node: Dict[str, float] = {
        node_id: (
            float(edge_p_value_series[node_id])
            if np.isfinite(edge_p_value_series[node_id])
            else 1.0
        )
        for node_id in annotations_df.index
    }

    records: List[SiblingPairRecord] = []
    non_binary_nodes: List[str] = []

    for parent in tree.nodes:

        children = get_binary_children(tree, parent)

        if children is None:
            non_binary_nodes.append(parent)
            continue

        left, right = children
        (
            left_distribution,
            right_distribution,
            n_left,
            n_right,
            branch_length_left,
            branch_length_right,
        ) = get_sibling_data(tree, parent, left, right)

        spectral_k = spectral_dims.get(parent) if spectral_dims else None
        pca_projection = pca_projections.get(parent) if pca_projections else None
        node_pca_eigenvalues = pca_eigenvalues.get(parent) if pca_eigenvalues else None

        test_statistic, degrees_of_freedom, p_value = sibling_divergence_test(
            left_distribution,
            right_distribution,
            float(n_left),
            float(n_right),
            branch_length_left=branch_length_left,
            branch_length_right=branch_length_right,
            mean_branch_length=mean_branch_length,
            test_id=f"sibling:{parent}",
            spectral_k=spectral_k,
            pca_projection=pca_projection,
            pca_eigenvalues=node_pca_eigenvalues,
            minimum_projection_dimension=minimum_projection_dimension,
            whitening=whitening,
        )

        is_null_like = not either_child_significant(left, right, edge_significance_by_node)
        # Continuous weight: min(p_edge_left, p_edge_right).
        # High weight → both children look null-like (high edge p-values).
        # Low weight → at least one child has strong edge signal.
        edge_calibration_weight = min(
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
                    int(degrees_of_freedom) if np.isfinite(degrees_of_freedom) else 0
                ),
                p_value=p_value,
                branch_length_sum=_branch_length_sum(branch_length_left, branch_length_right),
                n_parent=extract_node_sample_size(tree, parent),
                is_null_like=is_null_like,
                edge_weight=edge_calibration_weight,
            )
        )

    return records, non_binary_nodes


# =============================================================================
# Deflation
# =============================================================================


def deflate_focal_pairs(
    records: Iterable[_R],
    *,
    calibration_resolver: Callable[[_R], Tuple[float, str]],
) -> Tuple[List[str], List[Tuple[float, float, float]], List[str]]:
    """Deflate all focal records and return parent IDs, adjusted triples, and methods."""
    focal_parents: List[str] = []
    focal_results: List[Tuple[float, float, float]] = []
    methods: List[str] = []

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


# =============================================================================
# Counting
# =============================================================================


def count_null_focal_pairs(records: Iterable[DeflatableSiblingRecord]) -> Tuple[int, int]:
    """Count (null-like, focal) record totals."""
    records_list = list(records)
    n_null = sum(1 for r in records_list if r.is_null_like)
    n_focal = len(records_list) - n_null
    return n_null, n_focal


__all__ = [
    "DeflatableSiblingRecord",
    "SiblingPairRecord",
    "collect_significant_sibling_pairs",
    "collect_sibling_pair_records",
    "count_null_focal_pairs",
    "deflate_focal_pairs",
    "either_child_significant",
    "get_binary_children",
    "get_sibling_data",
]
