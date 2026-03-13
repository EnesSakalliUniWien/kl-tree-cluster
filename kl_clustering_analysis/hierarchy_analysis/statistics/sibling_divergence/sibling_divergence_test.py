"""Sibling divergence test for hierarchical clustering.

Tests whether sibling nodes have significantly different distributions using
a Wald chi-square statistic with random projection for high dimensions.

Test statistic: T = ||R·z||² ~ χ²(k) where z is the standardized difference
and R is a random projection matrix reducing d features to k = O(log n).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hmean

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
    initialize_sibling_divergence_columns,
)

from ..branch_length_utils import compute_mean_branch_length, sanitize_positive_branch_length
from ..categorical_mahalanobis import categorical_whitened_vector
from .pipeline import apply_sibling_bh_results, collect_significant_sibling_pairs
from ..pooled_variance import _is_categorical, standardize_proportion_difference
from ...decomposition.backends.random_projection_backend import (
    compute_projection_dimension_backend as compute_projection_dimension,
    derive_projection_seed_backend as derive_projection_seed,
)
from ...decomposition.methods.projected_wald import run_projected_wald_kernel

logger = logging.getLogger(__name__)


def sibling_divergence_test(
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    n_left: float,
    n_right: float,
    branch_length_left: float | None = None,
    branch_length_right: float | None = None,
    mean_branch_length: float | None = None,
    *,
    test_id: str | None = None,
    spectral_k: int | None = None,
    pca_projection: np.ndarray | None = None,
    pca_eigenvalues: np.ndarray | None = None,
    minimum_projection_dimension: int | None = None,
) -> Tuple[float, float, float]:
    """Two-sample Wald test for sibling divergence.

    Uses random projection on standardized vectors for both binary and
    categorical inputs. For categorical data, the vector is first covariance-
    whitened via a multinomial Mahalanobis construction (drop-last basis).

    Optionally applies Felsenstein's (1985) Phylogenetic Independent Contrasts
    adjustment by scaling variance by the sum of branch lengths.

    Parameters
    ----------
    left_dist : np.ndarray
        Distribution of left sibling.
    right_dist : np.ndarray
        Distribution of right sibling.
    n_left : float
        Sample size of left sibling.
    n_right : float
        Sample size of right sibling.
    branch_length_left : float, optional
        Branch length (distance to parent) for left sibling.
    branch_length_right : float, optional
        Branch length (distance to parent) for right sibling.
    mean_branch_length : float, optional
        Mean branch length across the tree for Felsenstein normalization.

    Returns
    -------
    Tuple[float, float, float]
        (test_statistic, degrees_of_freedom, p_value).
    """
    # Compute branch length sum for Felsenstein adjustment.
    # When mean_branch_length is None (Felsenstein disabled via config),
    # skip branch-length computation entirely to avoid triggering the
    # ValueError in standardize_proportion_difference().
    branch_length_sum = None
    if mean_branch_length is not None:
        bl_left = sanitize_positive_branch_length(branch_length_left)
        bl_right = sanitize_positive_branch_length(branch_length_right)
        if bl_left is not None and bl_right is not None:
            branch_length_sum = bl_left + bl_right
            if branch_length_sum <= 0:
                logger.warning(
                    "Non-positive sibling branch length sum encountered "
                    "(left=%s, right=%s). Disabling branch-length variance adjustment "
                    "for this test.",
                    bl_left,
                    bl_right,
                )
                branch_length_sum = None

    n_eff = hmean([n_left, n_right])

    if _is_categorical(np.asarray(left_dist)):
        z = categorical_whitened_vector(
            np.asarray(left_dist, dtype=np.float64),
            np.asarray(right_dist, dtype=np.float64),
            float(n_left),
            float(n_right),
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )
    else:
        z, _ = standardize_proportion_difference(
            left_dist,
            right_dist,
            n_left,
            n_right,
            branch_length_sum=branch_length_sum,
            mean_branch_length=mean_branch_length,
        )

    # Explicit invalid-test path: never coerce non-finite z-scores.
    # Keep raw statistics as NaN and route p=1.0 only in correction step.
    if not np.isfinite(z).all():
        logger.warning(
            "Found %d non-finite z-scores in sibling test; marking test invalid "
            "(raw outputs NaN, conservative p=1.0 for correction).",
            int(np.sum(~np.isfinite(z))),
        )
        return np.nan, np.nan, np.nan
    z = z.astype(np.float64, copy=False)  # Force float64

    # Use n_left + n_right (total observations) for the fallback JL cap.
    # The difference vector spans both sibling samples.
    n_total = int(n_left + n_right)

    # Project and compute test statistic
    if test_id is None:
        test_id = (
            f"sibling:shapeL={tuple(np.shape(left_dist))}:shapeR={tuple(np.shape(right_dist))}:"
            f"nL={float(n_left):.6g}:nR={float(n_right):.6g}"
        )
    test_seed = derive_projection_seed(config.PROJECTION_RANDOM_SEED, test_id)

    stat, _k_nominal, effective_df, p_value = run_projected_wald_kernel(
        z,
        seed=test_seed,
        spectral_k=spectral_k,
        pca_projection=pca_projection,
        pca_eigenvalues=pca_eigenvalues,
        k_fallback=lambda dim: compute_projection_dimension(n_total, dim, minimum_projection_dimension=minimum_projection_dimension),
    )

    return stat, effective_df, p_value


# =============================================================================
# Tree Traversal Helpers
# =============================================================================


def _get_binary_children(tree: nx.DiGraph, parent: str) -> Optional[Tuple[str, str]]:
    """Return (left, right) children if parent has exactly 2, else None."""
    children = list(tree.successors(parent))
    if len(children) != 2:
        return None
    return children[0], children[1]


def _either_child_significant(
    left: str,
    right: str,
    edge_significance_by_node: Dict[str, bool],
) -> bool:
    """Check if at least one child has significant child-parent divergence."""
    return edge_significance_by_node.get(left, False) or edge_significance_by_node.get(right, False)


def _get_sibling_data(
    tree: nx.DiGraph,
    parent: str,
    left: str,
    right: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    float | None,
    float | None,
]:
    """Extract distributions, sample sizes, and branch lengths for sibling pair.

    Branch lengths are extracted from the tree edges (parent → child).
    If not available, returns None for the branch lengths.
    """
    # Extract branch lengths from tree edges
    left_branch = (
        tree.edges[parent, left].get("branch_length") if tree.has_edge(parent, left) else None
    )
    right_branch = (
        tree.edges[parent, right].get("branch_length") if tree.has_edge(parent, right) else None
    )

    return (
        extract_node_distribution(tree, left),
        extract_node_distribution(tree, right),
        extract_node_sample_size(tree, left),
        extract_node_sample_size(tree, right),
        left_branch,
        right_branch,
    )


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
        get_binary_children=_get_binary_children,
        either_child_significant=_either_child_significant,
    )

    sibling_test_arguments: List[
        Tuple[np.ndarray, np.ndarray, int, int, float | None, float | None]
    ] = []
    for parent, (left, right) in zip(parents, child_pairs, strict=False):
        left_dist, right_dist, n_left, n_right, bl_left, bl_right = _get_sibling_data(
            tree, parent, left, right
        )
        sibling_test_arguments.append((left_dist, right_dist, n_left, n_right, bl_left, bl_right))

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
        _spectral_k: int | None = None
        _pca_proj: np.ndarray | None = None
        if spectral_dims is not None:
            _spectral_k = spectral_dims.get(parent)
        if pca_projections is not None:
            _pca_proj = pca_projections.get(parent)
        sibling_test_kwargs: dict[str, object] = {
            "test_id": f"sibling:{parent}",
            "spectral_k": _spectral_k,
            "pca_projection": _pca_proj,
        }
        if minimum_projection_dimension is not None:
            sibling_test_kwargs["minimum_projection_dimension"] = minimum_projection_dimension
        results.append(
            sibling_divergence_test(
                left,
                right,
                n_left,
                n_right,
                branch_length_left,
                branch_length_right,
                mean_branch_length,
                **sibling_test_kwargs,
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


__all__ = ["annotate_sibling_divergence", "sibling_divergence_test"]
