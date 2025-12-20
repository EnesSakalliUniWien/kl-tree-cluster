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

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..multiple_testing import benjamini_hochberg_correction
from ..random_projection import (
    should_use_projection,
    compute_projection_dimension,
    generate_projection_matrix,
    projected_euclidean_distance_squared,
)
from ..mi_feature_selection import select_informative_features
from kl_clustering_analysis import config


def _jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    The Jensen-Shannon divergence is a symmetric measure of similarity
    between two probability distributions, defined as:

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)

    where M = 0.5 * (P + Q) is the mixture distribution.

    Parameters
    ----------
    p : np.ndarray
        First probability distribution (Bernoulli probabilities per feature).
    q : np.ndarray
        Second probability distribution (Bernoulli probabilities per feature).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        Jensen-Shannon divergence value (non-negative, 0 if identical).
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0 - eps)

    # Mixture distribution
    m = 0.5 * (p + q)

    # KL divergences for Bernoulli: p*log(p/m) + (1-p)*log((1-p)/(1-m))
    def _kl_bernoulli(a: np.ndarray, b: np.ndarray) -> float:
        kl = a * np.log(a / b) + (1.0 - a) * np.log((1.0 - a) / (1.0 - b))
        return float(np.sum(kl))

    jsd = 0.5 * _kl_bernoulli(p, m) + 0.5 * _kl_bernoulli(q, m)
    return max(0.0, jsd)  # Ensure non-negative due to numerical precision


def _sibling_divergence_chi_square_test(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    left_sample_size: int,
    right_sample_size: int,
) -> Tuple[float, float, float, float]:
    """Chi-square test for sibling divergence using standardized Euclidean.

    Tests H₀: siblings have the same distribution.
    Uses Wald/standardized Euclidean test statistic:
        T = Σ (θ_left - θ_right)² / Var[θ_left - θ_right] ~ χ²(p)

    where Var[Δθ] = θ_pooled(1-θ_pooled)(1/n_left + 1/n_right).

    When dimensionality is high (d >> n), uses random projection to reduce
    to k = O(log n) dimensions for valid chi-square approximation.

    Parameters
    ----------
    left_distribution : np.ndarray
        Pre-computed distribution for left sibling node.
    right_distribution : np.ndarray
        Pre-computed distribution for right sibling node.
    left_sample_size : int
        Number of leaves under left sibling.
    right_sample_size : int
        Number of leaves under right sibling.
    Returns
    -------
    Tuple[float, float, float, float]
        (jsd_for_info, test_statistic, degrees_of_freedom, p_value)
    """
    n_left = float(left_sample_size)
    n_right = float(right_sample_size)
    n_effective = (2.0 * n_left * n_right) / (n_left + n_right)

    n_features = len(left_distribution)

    # Check if random projection should be used
    if should_use_projection(n_features, int(n_effective)):
        # Use random projection for high-dimensional case
        return _sibling_divergence_projected_test(
            left_distribution,
            right_distribution,
            n_effective,
            n_left,
            n_right,
        )

    # Compute JSD for informational purposes.
    jsd = _jensen_shannon_divergence(left_distribution, right_distribution)

    # Standardized Euclidean test statistic (Wald test)
    # Under H₀: θ_left = θ_right, the difference Δθ = θ̂_left - θ̂_right
    # has variance Var[Δθ_j] = θ_j(1-θ_j)(1/n_left + 1/n_right)
    eps = 1e-10
    left_arr = np.asarray(left_distribution, dtype=np.float64)
    right_arr = np.asarray(right_distribution, dtype=np.float64)
    diff = left_arr - right_arr

    # Pooled estimate for variance (under H₀)
    pooled = 0.5 * (left_arr + right_arr)
    pooled = np.clip(pooled, eps, 1.0 - eps)

    # Variance of the difference
    inverse_n_sum = 1.0 / n_left + 1.0 / n_right
    var_diff = pooled * (1.0 - pooled) * inverse_n_sum
    var_diff = np.maximum(var_diff, eps)  # Avoid division by zero

    # Test statistic: sum of squared standardized differences
    test_statistic = float(np.sum(diff**2 / var_diff))

    # Degrees of freedom = number of features
    degrees_of_freedom = float(n_features)
    degrees_of_freedom = max(1.0, degrees_of_freedom)

    # Calculate right-tail p-value
    p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))

    return jsd, test_statistic, degrees_of_freedom, p_value


def _sibling_divergence_projected_test(
    left_distribution: np.ndarray,
    right_distribution: np.ndarray,
    n_effective: float,
    n_left: float,
    n_right: float,
) -> Tuple[float, float, float, float]:
    """Standardized Euclidean test using random projection for high dimensions.

    When d >> n, optionally filters by MI then projects to k = O(log n)
    dimensions and uses standardized Euclidean distance.

    The test statistic is derived as follows:
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
    n_effective : float
        Effective sample size (harmonic mean).
    n_left : float
        Number of samples in left sibling.
    n_right : float
        Number of samples in right sibling.
    Returns
    -------
    Tuple[float, float, float, float]
        (avg_projected_distance, test_statistic, degrees_of_freedom, p_value)
    """
    n_features_original = len(left_distribution)
    eps = 1e-10

    # Step 1: MI-based feature filtering (optional)
    if config.USE_MI_FEATURE_FILTER:
        mask, mi_values, n_informative = select_informative_features(
            left_distribution,
            right_distribution,
            n_left,
            n_right,
            quantile_threshold=config.MI_FILTER_QUANTILE,
            min_fraction=config.MI_FILTER_MIN_FRACTION,
        )

        # Filter to informative features
        left_filtered = left_distribution[mask]
        right_filtered = right_distribution[mask]
        n_features = n_informative
    else:
        # No filtering - use all features
        left_filtered = left_distribution
        right_filtered = right_distribution
        n_features = n_features_original

    # Step 2: Random projection
    # Compute target dimension: k = multiplier * log(n)
    k = compute_projection_dimension(int(n_effective), n_features)

    # Compute pooled variance for standardization
    pooled = 0.5 * (left_filtered + right_filtered)
    pooled = np.clip(pooled, eps, 1.0 - eps)
    inverse_n_sum = 1.0 / n_left + 1.0 / n_right
    var_diff = pooled * (1.0 - pooled) * inverse_n_sum
    var_diff = np.maximum(var_diff, eps)

    # Standardize the difference before projection
    diff = left_filtered - right_filtered
    standardized_diff = diff / np.sqrt(var_diff)

    # Average over multiple random projections for stability
    n_trials = config.PROJECTION_N_TRIALS
    base_seed = config.PROJECTION_RANDOM_SEED

    test_stats = []
    for trial in range(n_trials):
        seed = base_seed + trial if base_seed is not None else None
        R = generate_projection_matrix(n_features, k, random_state=seed)
        projected = R @ standardized_diff
        # Test statistic: ||R·z||² where z is standardized
        # Under H₀, this is approximately χ²(k) due to JL lemma
        stat = float(np.sum(projected**2))
        test_stats.append(stat)

    test_statistic = float(np.mean(test_stats))

    # Degrees of freedom = projection dimension
    degrees_of_freedom = float(k)

    # Calculate right-tail p-value
    p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))

    # Compute JSD for informational purposes (backward compatibility)
    jsd = _jensen_shannon_divergence(left_distribution, right_distribution)

    return jsd, test_statistic, degrees_of_freedom, p_value


def _initialize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize the output columns in the dataframe with default values.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to initialize columns in.

    Returns
    -------
    pd.DataFrame
        The dataframe with initialized columns.
    """
    df["Sibling_Divergence_Skipped"] = False
    df["Sibling_JSD"] = np.nan
    df["Sibling_Test_Statistic"] = np.nan
    df["Sibling_Degrees_of_Freedom"] = np.nan
    df["Sibling_Divergence_P_Value"] = np.nan
    df["Sibling_Divergence_P_Value_Corrected"] = np.nan
    df["Sibling_BH_Different"] = False  # Reject H₀: siblings are different
    df["Sibling_BH_Same"] = False  # Fail to reject: siblings are similar
    return df


def _get_node_distribution(
    tree: nx.DiGraph,
    node: str,
) -> Optional[np.ndarray]:
    """Get pre-computed distribution for a node.

    Parameters
    ----------
    tree : nx.DiGraph
        The hierarchical tree with pre-computed distributions.
    node : str
        Node to get distribution for.

    Returns
    -------
    np.ndarray | None
        Pre-computed distribution array, or None if not available.
    """
    node_data = tree.nodes.get(node, {})
    distribution = node_data.get("distribution")

    if distribution is None:
        return None

    return np.asarray(distribution, dtype=np.float64)


def _get_node_sample_size(
    tree: nx.DiGraph,
    node: str,
) -> int:
    """Get the number of leaves under a node.

    Parameters
    ----------
    tree : nx.DiGraph
        The hierarchical tree.
    node : str
        Node to count leaves for.

    Returns
    -------
    int
        Number of leaf descendants (or 1 if node is a leaf).
    """
    node_data = tree.nodes.get(node, {})

    # Check if sample size is stored directly
    if "sample_size" in node_data:
        return int(node_data["sample_size"])

    if "n_leaves" in node_data:
        return int(node_data["n_leaves"])

    # Check if node is a leaf
    if node_data.get("is_leaf", False) or tree.out_degree(node) == 0:
        return 1

    # Count leaf descendants
    descendants = list(nx.descendants(tree, node))
    count = 0
    for desc in descendants:
        desc_data = tree.nodes.get(desc, {})
        if desc_data.get("is_leaf", False) or tree.out_degree(desc) == 0:
            count += 1

    return max(1, count)


def _collect_divergence_test_arguments(
    tree: nx.DiGraph,
    min_samples: int,
    nodes_df: pd.DataFrame = None,
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
    nodes_df : pd.DataFrame, optional
        DataFrame with 'Child_Parent_Divergence_Significant' column.
        If provided, only tests parents where at least one child has
        significant divergence from parent.

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

    # Extract child-parent significance if available
    child_parent_sig = {}
    if (
        nodes_df is not None
        and "Child_Parent_Divergence_Significant" in nodes_df.columns
    ):
        child_parent_sig = nodes_df["Child_Parent_Divergence_Significant"].to_dict()

    for parent in tree.nodes:
        children = list(tree.successors(parent))
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
        left_dist = _get_node_distribution(tree, left_child)
        right_dist = _get_node_distribution(tree, right_child)

        # Skip if distributions are not available
        if left_dist is None or right_dist is None:
            skipped_nodes.append(parent)
            continue

        # Get sample sizes
        left_n = _get_node_sample_size(tree, left_child)
        right_n = _get_node_sample_size(tree, right_child)

        # Skip if insufficient samples
        if left_n < min_samples or right_n < min_samples:
            skipped_nodes.append(parent)
            continue

        parent_nodes.append(parent)
        args_list.append((left_dist, right_dist, left_n, right_n))

    return parent_nodes, args_list, skipped_nodes


def _execute_divergence_tests(
    args_list: List[Tuple[np.ndarray, np.ndarray, int, int]],
) -> List[Tuple[float, float, float, float]]:
    """Execute sibling divergence chi-square tests.

    Parameters
    ----------
    args_list : List[Tuple[...]]
        List of argument tuples for _sibling_divergence_chi_square_test.

    Returns
    -------
    List[Tuple[float, float, float, float]]
        List of (jsd, test_statistic, df, p_value) tuples.
    """
    results: List[Tuple[float, float, float, float]] = []

    for left_dist, right_dist, left_n, right_n in args_list:
        jsd, test_stat, df, p_value = _sibling_divergence_chi_square_test(
            left_distribution=left_dist,
            right_distribution=right_dist,
            left_sample_size=left_n,
            right_sample_size=right_n,
        )
        results.append((jsd, test_stat, df, p_value))

    return results


def _apply_results_to_dataframe(
    df: pd.DataFrame,
    parent_nodes: List[str],
    results: List[Tuple[float, float, float, float]],
    significance_level_alpha: float,
) -> pd.DataFrame:
    """Apply divergence test results to the dataframe with BH correction.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update.
    parent_nodes : List[str]
        List of parent node names tested.
    results : List[Tuple[float, float, float, float]]
        List of (jsd, test_statistic, df, p_value) tuples.
    significance_level_alpha : float
        Significance level for BH correction.
    Returns
    -------
    pd.DataFrame
        The updated dataframe with results.
    """

    if not results:
        return df

    jsd_values = np.array([r[0] for r in results], dtype=float)
    test_stats = np.array([r[1] for r in results], dtype=float)
    df_values = np.array([r[2] for r in results], dtype=float)
    p_values = np.array([r[3] for r in results], dtype=float)

    reject, p_values_corrected, _ = benjamini_hochberg_correction(
        p_values, alpha=float(significance_level_alpha)
    )

    df.loc[parent_nodes, "Sibling_JSD"] = jsd_values
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


def _finalize_skipped_nodes(
    df: pd.DataFrame,
    skipped_nodes: List[str],
) -> pd.DataFrame:
    """Mark skipped nodes in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update.
    skipped_nodes : List[str]
        List of node names that were skipped.

    Returns
    -------
    pd.DataFrame
        The updated dataframe.
    """
    if not skipped_nodes:
        return df

    df.loc[skipped_nodes, "Sibling_Divergence_Skipped"] = True
    # Skipped nodes default to "same" (conservative: don't split if unsure)
    df.loc[skipped_nodes, "Sibling_BH_Same"] = True
    df.loc[skipped_nodes, "Sibling_BH_Different"] = False

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
        - Sibling_JSD: Jensen-Shannon divergence between siblings
        - Sibling_Test_Statistic: Chi-square test statistic (2*N*JSD)
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
    df = _initialize_dataframe_columns(df)

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

    # Extract sample sizes for potential future use
    sample_sizes = [(args[2], args[3]) for args in args_list]  # (left_n, right_n)

    # Apply results with BH correction
    df = _apply_results_to_dataframe(
        df, parent_nodes, results, significance_level_alpha
    )

    missing_mask = df.loc[parent_nodes, "Sibling_Divergence_P_Value"].isna()
    if missing_mask.any():
        missing = list(pd.Index(parent_nodes)[missing_mask])
        preview = ", ".join(map(repr, missing[:5]))
        raise ValueError(f"Sibling divergence results missing for nodes: {preview}.")

    return df


__all__ = [
    "annotate_sibling_divergence",
]
