"""Standard Wald sibling divergence annotation.

Orchestrates the raw Wald χ² test across all eligible sibling pairs
in a tree, applies BH correction, and annotates the results DataFrame.

The core statistical kernel lives in ``wald_kernel.py``.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import initialize_sibling_divergence_columns

from ..branch_length_utils import compute_mean_branch_length
from .bh_application import apply_sibling_bh_results
from .tree_traversal import collect_significant_sibling_pairs, get_sibling_data
from .wald_kernel import sibling_divergence_test

logger = logging.getLogger(__name__)


# =============================================================================
# Test Collection and Execution
# =============================================================================


def _collect_test_arguments(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
) -> Tuple[
    List[str],
    List[Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]],
    List[str],
    List[str],
]:
    """Collect sibling pairs eligible for testing.

    Returns (parent_nodes, sibling_test_arguments, skipped_nodes, non_binary_nodes).
    Each tuple in sibling_test_arguments contains:
    (left_dist, right_dist, n_left, n_right, branch_left, branch_right)
    """
    parents, child_pairs, skipped, non_binary = collect_significant_sibling_pairs(
        tree,
        annotations_df,
    )

    sibling_test_arguments: List[
        Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]
    ] = []

    for parent, (left, right) in zip(parents, child_pairs, strict=False):

        (
            left_distribution,
            right_distribution,
            n_left,
            n_right,
            branch_length_left,
            branch_length_right,
        ) = get_sibling_data(tree, parent, left, right)

        sibling_test_arguments.append(
            (
                left_distribution,
                right_distribution,
                n_left,
                n_right,
                branch_length_left,
                branch_length_right,
            )
        )

    return parents, sibling_test_arguments, skipped, non_binary


def _run_tests(
    parents: List[str],
    sibling_test_arguments: List[
        Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]
    ],
    mean_branch_length: float | None = None,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> List[Tuple[float, float, float]]:
    """Execute sibling divergence tests for all collected pairs."""
    if len(parents) != len(sibling_test_arguments):
        raise ValueError(
            "parents and sibling_test_arguments length mismatch: "
            f"{len(parents)} != {len(sibling_test_arguments)}"
        )
    results = []

    for parent, (left, right, n_left, n_right, branch_length_left, branch_length_right) in zip(
        parents,
        sibling_test_arguments,
        strict=False,
    ):
        results.append(
            sibling_divergence_test(
                left,
                right,
                n_left,
                n_right,
                branch_length_left,
                branch_length_right,
                mean_branch_length,
                test_id=f"sibling:{parent}",
                spectral_k=spectral_dims.get(parent) if spectral_dims is not None else None,
                pca_projection=pca_projections.get(parent) if pca_projections is not None else None,
                minimum_projection_dimension=minimum_projection_dimension,
            )
        )

    return results


# =============================================================================
# DataFrame Updates
# =============================================================================


def _apply_results(
    annotations_df: pd.DataFrame,
    parents: List[str],
    results: List[Tuple[float, float, float]],
    alpha: float,
) -> pd.DataFrame:
    """Apply test results with BH correction to dataframe."""
    return apply_sibling_bh_results(
        annotations_df,
        parents,
        results,
        alpha,
        logger=logger,
        audit_label="Sibling divergence",
    )


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    minimum_projection_dimension: int | None = None,
    spectral_dims: Dict[str, int] | None = None,
    pca_projections: Dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Test sibling divergence and annotate results in dataframe.

    Parameters
    ----------
    tree : nx.DiGraph
        Hierarchical tree with 'distribution' attribute on nodes.
    annotations_df : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' column.
    significance_level_alpha : float
        FDR level for BH correction.

    Returns
    -------
    pd.DataFrame
        Updated with Sibling_Test_Statistic, Sibling_Degrees_of_Freedom,
        Sibling_Divergence_P_Value, Sibling_Divergence_P_Value_Corrected,
        Sibling_BH_Different, Sibling_BH_Same columns.
    """
    if len(annotations_df) == 0:
        raise ValueError("Empty dataframe")

    annotations_df = annotations_df.copy()
    annotations_df = initialize_sibling_divergence_columns(annotations_df)

    parents, sibling_test_arguments, skipped, non_binary = _collect_test_arguments(
        tree,
        annotations_df,
    )

    if not parents:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        # Still mark non-binary/leaf nodes as skipped before returning
        if non_binary:
            annotations_df.loc[non_binary, "Sibling_Divergence_Skipped"] = True
        return annotations_df

    if skipped:
        annotations_df.loc[skipped, "Sibling_Divergence_Skipped"] = True
        logger.debug(f"Skipped {len(skipped)} nodes")

    if non_binary:
        annotations_df.loc[non_binary, "Sibling_Divergence_Skipped"] = True
        logger.debug(f"Non-binary/leaf nodes marked as skipped: {len(non_binary)}")

    # Compute mean branch length from tree for Felsenstein normalization
    # using the shared sanitization policy.  Gated by config.
    mean_branch_length = compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None

    results = _run_tests(
        parents,
        sibling_test_arguments,
        mean_branch_length=mean_branch_length,
        minimum_projection_dimension=minimum_projection_dimension,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
    )

    annotations_df = _apply_results(annotations_df, parents, results, significance_level_alpha)

    sibling_invalid = int(annotations_df.loc[parents, "Sibling_Divergence_Invalid"].sum())

    annotations_df.attrs["sibling_divergence_audit"] = {
        "total_tests": int(len(parents)),
        "invalid_tests": sibling_invalid,
        "conservative_path_tests": sibling_invalid,
    }

    return annotations_df


__all__ = ["annotate_sibling_divergence"]
