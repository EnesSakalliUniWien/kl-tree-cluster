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
from scipy.stats import chi2, hmean

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    extract_node_distribution,
    extract_node_sample_size,
    initialize_sibling_divergence_columns,
)

from ..mi_feature_selection import select_informative_features
from ..multiple_testing import benjamini_hochberg_correction
from ..pooled_variance import standardize_proportion_difference
from ..random_projection import compute_projection_dimension, generate_projection_matrix

logger = logging.getLogger(__name__)


# =============================================================================
# Core Statistical Test
# =============================================================================


def _filter_informative_features(
    left: np.ndarray,
    right: np.ndarray,
    n_left: float,
    n_right: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Filter to informative features if MI filtering is enabled.
    
    Supports both binary (1D) and categorical (2D) distributions.
    """
    if not config.USE_MI_FEATURE_FILTER:
        # Return flattened length for categorical
        n_features = left.size if left.ndim == 2 else len(left)
        return left, right, n_features

    mask, _, n_kept = select_informative_features(
        left,
        right,
        n_left,
        n_right,
        quantile_threshold=config.MI_FILTER_QUANTILE,
        min_fraction=config.MI_FILTER_MIN_FRACTION,
    )
    # For categorical (2D), mask applies to features (rows), not categories
    if left.ndim == 2:
        return left[mask], right[mask], n_kept * left.shape[1]
    return left[mask], right[mask], n_kept


def _compute_chi_square_pvalue(
    projected: np.ndarray,
    df: int,
) -> Tuple[float, float, float]:
    """Compute χ²(k) test statistic and p-value from projected z-scores."""
    stat = float(np.sum(projected**2))
    return stat, float(df), float(chi2.sf(stat, df=df))


def sibling_divergence_test(
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    n_left: float,
    n_right: float,
) -> Tuple[float, float, float]:
    """Two-sample Wald test for sibling divergence with random projection.

    Supports both binary (1D) and categorical (2D) distributions.

    Returns (test_statistic, degrees_of_freedom, p_value).
    """
    n_eff = hmean([n_left, n_right])

    # Filter to informative features
    left, right, n_features = _filter_informative_features(
        left_dist, right_dist, n_left, n_right
    )

    # Standardize difference (Wald z-scores) - handles flattening for categorical
    z, _ = standardize_proportion_difference(left, right, n_left, n_right)
    
    # Use actual z-score length (may differ from n_features for categorical)
    d = len(z)

    # Compute projection dimension
    k = compute_projection_dimension(int(n_eff), d)

    # Project and compute test statistic
    R = generate_projection_matrix(d, k, config.PROJECTION_RANDOM_SEED)
    projected = R @ z

    return _compute_chi_square_pvalue(projected, k)


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
    sig_map: Dict[str, bool],
) -> bool:
    """Check if at least one child has significant child-parent divergence."""
    return sig_map.get(left, False) or sig_map.get(right, False)


def _get_sibling_data(
    tree: nx.DiGraph,
    left: str,
    right: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Extract distributions and sample sizes for sibling pair."""
    return (
        extract_node_distribution(tree, left),
        extract_node_distribution(tree, right),
        extract_node_sample_size(tree, left),
        extract_node_sample_size(tree, right),
    )


# =============================================================================
# Test Collection and Execution
# =============================================================================


def _collect_test_arguments(
    tree: nx.DiGraph,
    nodes_df: pd.DataFrame,
    min_samples: int,
) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray, int, int]], List[str]]:
    """Collect sibling pairs eligible for testing.

    Returns (parent_nodes, test_args, skipped_nodes).
    """
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column. "
            "Run child-parent test first."
        )

    sig_map = nodes_df["Child_Parent_Divergence_Significant"].to_dict()

    parents: List[str] = []
    args: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    skipped: List[str] = []

    for parent in tree.nodes:
        children = _get_binary_children(tree, parent)
        if children is None:
            continue

        left, right = children

        # Skip if neither child diverged from parent
        if not _either_child_significant(left, right, sig_map):
            skipped.append(parent)
            continue

        left_dist, right_dist, n_left, n_right = _get_sibling_data(tree, left, right)

        # Skip if insufficient samples
        if n_left < min_samples or n_right < min_samples:
            skipped.append(parent)
            continue

        parents.append(parent)
        args.append((left_dist, right_dist, n_left, n_right))

    return parents, args, skipped


def _run_tests(
    args: List[Tuple[np.ndarray, np.ndarray, int, int]],
) -> List[Tuple[float, float, float]]:
    """Execute sibling divergence tests for all collected pairs."""
    return [
        sibling_divergence_test(left, right, n_l, n_r) for left, right, n_l, n_r in args
    ]


# =============================================================================
# DataFrame Updates
# =============================================================================


def _apply_results(
    df: pd.DataFrame,
    parents: List[str],
    results: List[Tuple[float, float, float]],
    alpha: float,
) -> pd.DataFrame:
    """Apply test results with BH correction to dataframe."""
    if not results:
        return df

    stats = np.array([r[0] for r in results])
    dfs = np.array([r[1] for r in results])
    pvals = np.array([r[2] for r in results])

    reject, pvals_adj, _ = benjamini_hochberg_correction(pvals, alpha=alpha)

    df.loc[parents, "Sibling_Test_Statistic"] = stats
    df.loc[parents, "Sibling_Degrees_of_Freedom"] = dfs
    df.loc[parents, "Sibling_Divergence_P_Value"] = pvals
    df.loc[parents, "Sibling_Divergence_P_Value_Corrected"] = pvals_adj
    df.loc[parents, "Sibling_BH_Different"] = reject
    df["Sibling_BH_Same"] = ~df["Sibling_BH_Different"]

    return df


# =============================================================================
# Public API
# =============================================================================


def annotate_sibling_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIBLING_ALPHA,
    min_samples_per_sibling: int = 2,
) -> pd.DataFrame:
    """Test sibling divergence and annotate results in dataframe.

    Parameters
    ----------
    tree : nx.DiGraph
        Hierarchical tree with 'distribution' attribute on nodes.
    nodes_statistics_dataframe : pd.DataFrame
        Must contain 'Child_Parent_Divergence_Significant' column.
    significance_level_alpha : float
        FDR level for BH correction.
    min_samples_per_sibling : int
        Minimum samples required per sibling.

    Returns
    -------
    pd.DataFrame
        Updated with Sibling_Test_Statistic, Sibling_Degrees_of_Freedom,
        Sibling_Divergence_P_Value, Sibling_Divergence_P_Value_Corrected,
        Sibling_BH_Different, Sibling_BH_Same columns.
    """
    if len(nodes_statistics_dataframe) == 0:
        raise ValueError("Empty dataframe")

    df = nodes_statistics_dataframe.copy()
    df = initialize_sibling_divergence_columns(df)

    parents, args, skipped = _collect_test_arguments(tree, df, min_samples_per_sibling)

    if not parents:
        warnings.warn("No eligible parent nodes for sibling tests", UserWarning)
        return df

    if skipped:
        logger.debug(f"Skipped {len(skipped)} nodes")

    results = _run_tests(args)
    return _apply_results(df, parents, results, significance_level_alpha)


__all__ = ["annotate_sibling_divergence", "sibling_divergence_test"]
