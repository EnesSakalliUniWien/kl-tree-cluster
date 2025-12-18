"""Sibling divergence test for hierarchical clustering.

This module implements a statistical test to determine whether sibling nodes
in a hierarchical tree have significantly different distributions, which
would warrant splitting them into separate clusters.

The test uses Jensen-Shannon divergence with a chi-square approximation
for statistical significance, consistent with the KL divergence chi-square test.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..multiple_testing import benjamini_hochberg_correction
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
    parent_distribution: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Chi-square test for sibling divergence.

    Tests H₀: siblings have the same distribution.
    Uses asymptotic chi-square approximation: 2*N*JSD ~ χ²(df)

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
    parent_distribution : np.ndarray, optional
        Parent node distribution for variance-weighted df calculation.

    Returns
    -------
    Tuple[float, float, float, float]
        (jsd, test_statistic, degrees_of_freedom, p_value)
    """
    # Compute Jensen-Shannon divergence using pre-computed distributions
    jsd = _jensen_shannon_divergence(left_distribution, right_distribution)

    # Effective sample size (harmonic mean for two-sample comparison)
    n_left = float(left_sample_size)
    n_right = float(right_sample_size)
    n_effective = (2.0 * n_left * n_right) / (n_left + n_right)

    # Chi-square test statistic: 2*N*JSD
    test_statistic = 2.0 * n_effective * jsd

    # Calculate degrees of freedom (variance-weighted if parent available)
    if parent_distribution is not None:
        # Variance-weighted effective df (consistent with chi_square_test.py)
        parent_theta = np.asarray(parent_distribution, dtype=np.float64)
        variance_weights = 4.0 * parent_theta * (1.0 - parent_theta)
        degrees_of_freedom = float(np.sum(variance_weights))
    else:
        # Fallback: use number of features
        degrees_of_freedom = float(len(left_distribution))

    # Ensure df is at least 1 to avoid chi2.sf issues
    degrees_of_freedom = max(1.0, degrees_of_freedom)

    # Calculate right-tail p-value
    p_value = float(chi2.sf(test_statistic, df=degrees_of_freedom))

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
) -> Tuple[
    List[str],
    List[Tuple[np.ndarray, np.ndarray, int, int, Optional[np.ndarray]]],
    List[str],
]:
    """Collect arguments for sibling divergence tests from tree structure.

    Uses pre-computed distributions from tree nodes (consistent with chi_square_test.py).

    Parameters
    ----------
    tree : nx.DiGraph
        The tree structure with pre-computed distributions.
    min_samples : int
        Minimum samples required per sibling to run the test.

    Returns
    -------
    Tuple[List[str], List[Tuple[...]], List[str]]
        - parent_nodes: List of parent node names to test
        - args_list: List of argument tuples for chi-square test
        - skipped_nodes: List of parent node names that were skipped
    """
    parent_nodes: List[str] = []
    args_list: List[Tuple[np.ndarray, np.ndarray, int, int, Optional[np.ndarray]]] = []
    skipped_nodes: List[str] = []

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        left_child, right_child = children

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

        # Get parent distribution for variance-weighted df (if available)
        parent_dist = _get_node_distribution(tree, parent)

        parent_nodes.append(parent)
        args_list.append((left_dist, right_dist, left_n, right_n, parent_dist))

    return parent_nodes, args_list, skipped_nodes


def _execute_divergence_tests(
    args_list: List[Tuple[np.ndarray, np.ndarray, int, int, Optional[np.ndarray]]],
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

    for left_dist, right_dist, left_n, right_n, parent_dist in args_list:
        jsd, test_stat, df, p_value = _sibling_divergence_chi_square_test(
            left_distribution=left_dist,
            right_distribution=right_dist,
            left_sample_size=left_n,
            right_sample_size=right_n,
            parent_distribution=parent_dist,
        )
        results.append((jsd, test_stat, df, p_value))

    return results


def _apply_results_to_dataframe(
    df: pd.DataFrame,
    parent_nodes: List[str],
    results: List[Tuple[float, float, float, float]],
    significance_level_alpha: float,
    sample_sizes: Optional[List[Tuple[int, int]]] = None,
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
    sample_sizes : List[Tuple[int, int]], optional
        List of (left_n, right_n) sample sizes for adaptive α.

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

    # Compute adaptive α if enabled and sample sizes provided
    if config.USE_ADAPTIVE_ALPHA and sample_sizes is not None:
        # Use median effective sample size and df for global adaptive α
        n_effective_list = [
            (2.0 * left_n * right_n) / (left_n + right_n)
            for left_n, right_n in sample_sizes
        ]
        median_n = np.median(n_effective_list)
        median_df = np.median(df_values)
        effective_alpha = config.compute_adaptive_alpha(
            significance_level_alpha, median_n, median_df
        )
    else:
        effective_alpha = significance_level_alpha

    reject, p_values_corrected, _ = benjamini_hochberg_correction(
        p_values, alpha=float(effective_alpha)
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
    distributions. Unlike the CMI test, this test has intuitive polarity:
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
        return df

    # Initialize output columns
    df = _initialize_dataframe_columns(df)

    # Collect test arguments for all eligible parent nodes
    parent_nodes, args_list, skipped_nodes = _collect_divergence_test_arguments(
        tree=tree,
        min_samples=min_samples_per_sibling,
    )

    # Early exit if no tests to perform
    if not parent_nodes:
        df = _finalize_skipped_nodes(df, skipped_nodes)
        return df

    # Execute divergence tests
    results = _execute_divergence_tests(args_list)

    # Extract sample sizes for adaptive α
    sample_sizes = [(args[2], args[3]) for args in args_list]  # (left_n, right_n)

    # Apply results with BH correction (using adaptive α if enabled)
    df = _apply_results_to_dataframe(
        df, parent_nodes, results, significance_level_alpha, sample_sizes
    )

    # Handle skipped nodes
    df = _finalize_skipped_nodes(df, skipped_nodes)

    return df


__all__ = [
    "annotate_sibling_divergence",
]
