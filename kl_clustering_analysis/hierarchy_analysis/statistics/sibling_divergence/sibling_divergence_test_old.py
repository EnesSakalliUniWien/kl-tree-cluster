"""Sibling divergence test for hierarchical clustering.

This module implements a statistical test to determine whether sibling nodes
in a hierarchical tree have significantly different distributions, which
would warrant splitting them into separate clusters.

The test uses a standardized Euclidean distance test statistic:
    T = Σ (θ_left - θ_right)² / Var[θ_left - θ_right]

where Var[Δθ] = θ_pooled(1-θ_pooled)(1/n_left + 1/n_right) for Bernoulli.

Under H₀ (siblings have same distribution), T ~ χ²(p) where p is the
number of features. This follows from the delta method / Wald test.

When dimensionality is high (d >> n), random projection is used to reduce
to k = O(log n) dimensions, making the chi-square approximation valid.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2, hmean

from ..multiple_testing import benjamini_hochberg_correction
from ..random_projection import (
    compute_projection_dimension,
    generate_projection_matrix,
)
from ..pooled_variance import standardize_proportion_difference
from ..mi_feature_selection import select_informative_features
from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import (
    initialize_sibling_divergence_columns,
    extract_node_distribution,
    extract_node_sample_size,
)


def _filter_informative_features(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    n_left: float,
    n_right: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Filter distributions to informative features if MI filtering is enabled.

    When config.USE_MI_FEATURE_FILTER is True, removes low-information features
    (those with MI below the configured quantile threshold) to focus the test
    on discriminative features and reduce noise.

    Parameters
    ----------
    left_distribution : np.ndarray
        Distribution for left sibling.
    right_distribution : np.ndarray
        Distribution for right sibling.
    n_left : float
        Sample size of left sibling.
    n_right : float
        Sample size of right sibling.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (left_filtered, right_filtered, n_features)
    """
    if not config.USE_MI_FEATURE_FILTER:
        return left_distribution, right_distribution, len(left_distribution)

    mask, _, n_informative = select_informative_features(
        left_distribution,
        right_distribution,
        n_left,
        n_right,
        quantile_threshold=config.MI_FILTER_QUANTILE,
        min_fraction=config.MI_FILTER_MIN_FRACTION,
    )

    return left_distribution[mask], right_distribution[mask], n_informative


def _compute_projected_chi_square(
    projected: np.ndarray,
    degrees_of_freedom: int,
) -> Tuple[float, float, float]:
    """Compute chi-square test statistic and p-value from projected differences.

    Under H₀, the projected standardized difference ||R·z||² follows a
    chi-square distribution with k degrees of freedom, where k is the
    projection dimension. This is guaranteed by the Johnson-Lindenstrauss
    lemma which preserves squared norms up to (1 ± ε).

    Parameters
    ----------
    projected : np.ndarray
        Projected standardized differences, shape (k,).
    degrees_of_freedom : int
        Number of projection dimensions (k).

    Returns
    -------
    Tuple[float, float, float]
        (test_statistic, degrees_of_freedom, p_value) where:
        - test_statistic: ||R·z||² = Σᵢ projected[i]²
        - degrees_of_freedom: k (projection dimension)
        - p_value: P(χ²(k) > test_statistic), right-tail probability
    """
    test_statistic = float(np.sum(projected**2))
    df = float(degrees_of_freedom)
    p_value = float(chi2.sf(test_statistic, df=df))
    return test_statistic, df, p_value


def _sibling_divergence_chi_square_test(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    n_left: float,
    n_right: float,
) -> Tuple[float, float, float]:
    """Chi-square test for sibling divergence using standardized Euclidean.

    Tests H₀: siblings have the same distribution.
    Uses Wald/standardized Euclidean test statistic:
        T = Σ (θ_left - θ_right)² / Var[θ_left - θ_right] ~ χ²(p)

    where Var[Δθ] = θ_pooled(1-θ_pooled)(1/n_left + 1/n_right).

    When dimensionality is high (d >> n), uses random projection to reduce
    to k = O(log n) dimensions for valid chi-square approximation.

    The test statistic derivation:
    - Under H₀: θ_left = θ_right, so Δθ̂ = θ̂_left - θ̂_right has
      Var[Δθ̂_j] = θ_j(1-θ_j) * (1/n_left + 1/n_right)
    - After projection by R (k×d matrix with orthonormal rows scaled by √(d/k)):
      R·Δθ̂ has components that are linear combinations
    - Under H₀, ||R·Δθ̂||² / σ² ~ χ²(k) where σ² is the average variance

    With MI filtering enabled:
    - First removes low-information features (MI < median)
    - Then projects the remaining informative features
    - This reduces noise and focuses on discriminative features

    Parameters
    ----------
    left_distribution : np.ndarray
        Pre-computed distribution for left sibling node.
    right_distribution : np.ndarray
        Pre-computed distribution for right sibling node.
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.

    Returns
    -------
    Tuple[float, float, float]
        (test_statistic, degrees_of_freedom, p_value)
    """

    # Effective sample size: harmonic mean (limited by smaller group)
    n_effective = hmean([n_left, n_right])

    # Filter to informative features if MI filtering is enabled
    left_filtered, right_filtered, n_features = _filter_informative_features(
        left_distribution, right_distribution, n_left, n_right
    )

    # Compute target dimension: k = multiplier * log(n)
    k = compute_projection_dimension(int(n_effective), n_features)

    # Standardize the difference using pooled variance (Wald test)
    standardized_diff, _ = standardize_proportion_difference(
        left_filtered, right_filtered, n_left, n_right
    )

    # Random projection (single projection suffices per JL lemma)
    R = generate_projection_matrix(
        n_features, k, random_state=config.PROJECTION_RANDOM_SEED
    )

    projected = R @ standardized_diff

    return _compute_projected_chi_square(projected, k)


def _collect_divergence_test_arguments(
    tree: nx.DiGraph,
    min_samples: int,
    nodes_df: pd.DataFrame,
) -> Tuple[
    List[str],
    List[Tuple[np.ndarray, np.ndarray, int, int]],
    List[str],
]:
    """Collect arguments for sibling divergence tests from tree structure.

    Only tests parents where at least one child has significant child-parent
    divergence (i.e., the edge is in E_sig). This implements conditional
    sibling testing as described in the manuscript.

    Uses pre-computed distributions from tree nodes (consistent with chi_square_test.py).

    Parameters
    ----------
    tree : nx.DiGraph
        The tree structure with pre-computed distributions.
    min_samples : int
        Minimum samples required per sibling to run the test.
    nodes_df : pd.DataFrame
        DataFrame with 'Child_Parent_Divergence_Significant' column.
        Required for conditional sibling testing.

    Returns
    -------
    Tuple[List[str], List[Tuple[...]], List[str]]
        - parent_nodes: List of parent node names to test
        - args_list: List of argument tuples for chi-square test
        - skipped_nodes: List of parent node names that were skipped
    """
    parent_nodes: List[str] = []
    args_list: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
    skipped_nodes: List[str] = []

    # Child-parent significance is required - sibling test is conditional on it
    if "Child_Parent_Divergence_Significant" not in nodes_df.columns:
        raise ValueError(
            "Missing 'Child_Parent_Divergence_Significant' column in nodes_df. "
            "Run child-parent edge significance test before sibling divergence."
        )

    child_parent_sig = nodes_df["Child_Parent_Divergence_Significant"].to_dict()

    for parent in tree.nodes:
        children = list(tree.successors(parent))

        # Skip leaf nodes (no children) and non-binary splits
        if len(children) != 2:
            continue

        left_child, right_child = children

        # CONDITIONAL TEST: Only test if at least one child has significant
        # divergence from parent (edge is in E_sig)
        if child_parent_sig:
            left_significant = child_parent_sig.get(left_child, False)
            right_significant = child_parent_sig.get(right_child, False)

            if not (left_significant or right_significant):
                # Neither child diverges significantly from parent
                # Skip sibling test - this parent won't split anyway
                skipped_nodes.append(parent)
                continue

        # Get pre-computed distributions for each sibling
        left_dist = extract_node_distribution(tree, left_child)
        right_dist = extract_node_distribution(tree, right_child)

        # Get sample sizes
        left_n = extract_node_sample_size(tree, left_child)
        right_n = extract_node_sample_size(tree, right_child)

        # Skip if insufficient samples
        if left_n < min_samples or right_n < min_samples:
            skipped_nodes.append(parent)
            continue

        parent_nodes.append(parent)
        args_list.append((left_dist, right_dist, left_n, right_n))

    return parent_nodes, args_list, skipped_nodes


def _execute_divergence_tests(
    args_list: List[Tuple[np.ndarray, np.ndarray, int, int]],
) -> List[Tuple[float, float, float]]:
    """Execute sibling divergence chi-square tests.

    Parameters
    ----------
    args_list : List[Tuple[...]]
        List of argument tuples for _sibling_divergence_chi_square_test.

    Returns
    -------
    List[Tuple[float, float, float]]
        List of (test_statistic, df, p_value) tuples.
    """
    results: List[Tuple[float, float, float]] = []

    for left_dist, right_dist, left_n, right_n in args_list:
        test_stat, df, p_value = _sibling_divergence_chi_square_test(
            left_distribution=left_dist,
            right_distribution=right_dist,
            n_left=float(left_n),
            n_right=float(right_n),
        )
        results.append((test_stat, df, p_value))

    return results


def _apply_results_to_dataframe(
    df: pd.DataFrame,
    parent_nodes: List[str],
    results: List[Tuple[float, float, float]],
    significance_level_alpha: float,
) -> pd.DataFrame:
    """Apply divergence test results to the dataframe with BH correction.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update.
    parent_nodes : List[str]
        List of parent node names tested.
    results : List[Tuple[float, float, float]]
        List of (test_statistic, df, p_value) tuples.
    significance_level_alpha : float
        Significance level for BH correction.
    Returns
    -------
    pd.DataFrame
        The updated dataframe with results.
    """

    if not results:
        return df

    test_stats = np.array([r[0] for r in results], dtype=float)
    df_values = np.array([r[1] for r in results], dtype=float)
    p_values = np.array([r[2] for r in results], dtype=float)

    reject, p_values_corrected, _ = benjamini_hochberg_correction(
        p_values, alpha=float(significance_level_alpha)
    )

    df.loc[parent_nodes, "Sibling_Test_Statistic"] = test_stats
    df.loc[parent_nodes, "Sibling_Degrees_of_Freedom"] = df_values
    df.loc[parent_nodes, "Sibling_Divergence_P_Value"] = p_values

    if p_values_corrected.size:
        df.loc[parent_nodes, "Sibling_Divergence_P_Value_Corrected"] = (
            p_values_corrected
        )
        # Reject H₀ means siblings ARE different → split
        df.loc[parent_nodes, "Sibling_BH_Different"] = reject
    else:
        df.loc[parent_nodes, "Sibling_BH_Different"] = False

    # Sibling_BH_Same is the complement: fail to reject means siblings are similar
    df["Sibling_BH_Same"] = ~df["Sibling_BH_Different"]

    return df


def annotate_sibling_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIGNIFICANCE_ALPHA,
    min_samples_per_sibling: int = 2,
) -> pd.DataFrame:
    """Sibling divergence test using Jensen-Shannon divergence with chi-square test.

    Tests whether sibling nodes in the tree have significantly different
    distributions. This test has intuitive polarity:
    rejecting H₀ means siblings are different → split.

    Uses pre-computed distributions from tree nodes (consistent with chi_square_test.py).

    Parameters
    ----------
    tree : nx.DiGraph
        The hierarchical tree structure with pre-computed distributions.
        Each node must have a 'distribution' attribute.
    nodes_statistics_dataframe : pd.DataFrame
        Dataframe with node statistics.
    significance_level_alpha : float, optional
        Significance level for hypothesis tests (default from config).
    min_samples_per_sibling : int, optional
        Minimum samples required per sibling to run the test.
        Nodes with fewer samples are skipped (default: 2).

    Returns
    -------
    pd.DataFrame
        Updated dataframe with sibling divergence test results:
        - Sibling_Test_Statistic: Wald chi-square test statistic
        - Sibling_Degrees_of_Freedom: Effective degrees of freedom
        - Sibling_Divergence_P_Value: Raw p-value from chi-square test
        - Sibling_Divergence_P_Value_Corrected: BH-corrected p-value
        - Sibling_BH_Different: True if siblings are significantly different (→ SPLIT)
        - Sibling_BH_Same: True if siblings are not significantly different (→ MERGE)
        - Sibling_Divergence_Skipped: True if test was skipped
    """
    df = nodes_statistics_dataframe.copy()
    if len(df) == 0:
        raise ValueError(
            "Empty nodes_statistics_dataframe; cannot annotate sibling divergence."
        )

    # Initialize output columns
    df = initialize_sibling_divergence_columns(df)

    # Collect test arguments for eligible parent nodes
    # Only tests parents where at least one child has significant child-parent divergence
    parent_nodes, args_list, skipped_nodes = _collect_divergence_test_arguments(
        tree=tree,
        min_samples=min_samples_per_sibling,
        nodes_df=df,
    )

    # Early exit if no tests to perform
    if not parent_nodes:
        # No eligible parent nodes - this can happen with very small trees
        # Just return the dataframe with initialized columns
        import warnings

        warnings.warn(
            "No eligible parent nodes for sibling divergence tests; "
            "all nodes skipped due to insufficient samples.",
            UserWarning,
        )
        return df

    # Log skipped nodes but don't error - this is expected for small subtrees
    if skipped_nodes:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Skipped {len(skipped_nodes)} nodes with insufficient samples "
            f"for sibling divergence test."
        )

    # Execute divergence tests
    results = _execute_divergence_tests(args_list)

    # Apply results with BH correction
    df = _apply_results_to_dataframe(
        df, parent_nodes, results, significance_level_alpha
    )

    return df


__all__ = [
    "annotate_sibling_divergence",
]
