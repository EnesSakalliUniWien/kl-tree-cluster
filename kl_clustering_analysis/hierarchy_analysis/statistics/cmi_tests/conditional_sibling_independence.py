from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from kl_clustering_analysis.information_metrics import _cmi_perm_from_args
from ..multiple_testing import benjamini_hochberg_correction
from kl_clustering_analysis.threshold import binary_threshold
from kl_clustering_analysis import config


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
    df["Sibling_CMI_Skipped"] = False
    df["Sibling_CMI"] = np.nan
    df["Sibling_CMI_P_Value"] = np.nan
    df["Sibling_CMI_P_Value_Corrected"] = np.nan
    df["Sibling_BH_Dependent"] = False
    df["Sibling_BH_Independent"] = False
    return df


def _extract_local_significance_map(df: pd.DataFrame) -> dict[str, bool] | None:
    """Extract local significance information from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing local significance columns.

    Returns
    -------
    dict[str, bool] | None
        A dictionary mapping node names to their local significance status,
        or None if no local significance columns are present.
    """
    if "Local_BH_Significant" in df.columns:
        return df["Local_BH_Significant"].fillna(False).astype(bool).to_dict()
    elif "Local_Are_Features_Dependent" in df.columns:
        return df["Local_Are_Features_Dependent"].fillna(False).astype(bool).to_dict()
    return None


def _prepare_binary_distributions(
    df: pd.DataFrame,
    binarization_threshold: float,
) -> dict[str, np.ndarray]:
    """Extract and binarize node distributions from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing node distributions in the 'distribution' column.
    binarization_threshold : float
        The threshold value for binarization.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary mapping node names to their binarized distributions.
    """
    dist_series = df.get("distribution", pd.Series(index=df.index, dtype=object))
    dist_dict = dist_series.to_dict()
    return {
        node: binary_threshold(dist, thr=binarization_threshold)
        for node, dist in dist_dict.items()
        if dist is not None
    }


def _should_skip_parent(
    children: List[str],
    local_sig_map: dict[str, bool] | None,
) -> bool:
    """Determine whether a parent node should be skipped based on child significance.

    Parameters
    ----------
    children : List[str]
        List of child node names (should have exactly 2 elements).
    local_sig_map : dict[str, bool] | None
        Mapping of node names to their local significance status.

    Returns
    -------
    bool
        True if the parent should be skipped, False otherwise.
    """
    if local_sig_map is None:
        return False
    if len(children) != 2:
        return False
    c1, c2 = children
    return not (local_sig_map.get(c1, False) and local_sig_map.get(c2, False))


def _validate_distribution_arrays(
    x: np.ndarray | None,
    y: np.ndarray | None,
    z: np.ndarray | None,
) -> bool:
    """Validate that distribution arrays are non-None, non-empty, and matching in size.

    Parameters
    ----------
    x : np.ndarray | None
        First distribution array.
    y : np.ndarray | None
        Second distribution array.
    z : np.ndarray | None
        Conditioning distribution array.

    Returns
    -------
    bool
        True if all arrays are valid and have matching sizes, False otherwise.
    """
    if x is None or y is None or z is None:
        return False
    if not (x.size and y.size and z.size):
        return False
    if not (x.size == y.size == z.size):
        return False
    return True


def _generate_random_seed(seed_sequence: np.random.SeedSequence | None) -> int | None:
    """Generate a random seed from a seed sequence.

    Parameters
    ----------
    seed_sequence : np.random.SeedSequence | None
        The seed sequence to generate from, or None.

    Returns
    -------
    int | None
        A generated seed value, or None if seed_sequence is None.
    """
    if seed_sequence is None:
        return None
    return int(np.random.SeedSequence(seed_sequence.generate_state(1)[0]).entropy)


def _collect_cmi_test_arguments(
    tree: nx.DiGraph,
    bin_dist: dict[str, np.ndarray],
    local_sig_map: dict[str, bool] | None,
    n_permutations: int,
    batch_size: int,
    random_state: int | None,
) -> Tuple[
    List[str],
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int]],
    List[str],
]:
    """Collect arguments for CMI permutation tests from tree structure.

    Parameters
    ----------
    tree : nx.DiGraph
        The tree structure.
    bin_dist : dict[str, np.ndarray]
        Binary distributions for each node.
    local_sig_map : dict[str, bool] | None
        Local significance map.
    n_permutations : int
        Number of permutations for tests.
    batch_size : int
        Batch size for vectorized permutation tests.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[str], List[Tuple[...]], List[str]]
        A tuple containing:
        - parent_nodes: List of parent node names to test
        - args_list: List of argument tuples for _cmi_perm_from_args
        - skipped_nodes: List of parent node names that were skipped
    """
    parent_nodes: List[str] = []
    args_list: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int]
    ] = []
    skipped_nodes: List[str] = []

    ss = np.random.SeedSequence(random_state) if random_state is not None else None

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        if _should_skip_parent(children, local_sig_map):
            skipped_nodes.append(parent)
            continue

        c1, c2 = children
        x = bin_dist.get(c1)
        y = bin_dist.get(c2)
        z = bin_dist.get(parent)

        if not _validate_distribution_arrays(x, y, z):
            continue

        seed = _generate_random_seed(ss)
        parent_nodes.append(parent)
        args_list.append((x, y, z, int(n_permutations), seed, int(batch_size)))

    return parent_nodes, args_list, skipped_nodes


def _check_parallel_execution_feasibility() -> bool:
    """Check if parallel execution using ProcessPoolExecutor is feasible.

    Returns
    -------
    bool
        True if parallel execution can be used, False otherwise.
    """
    import os
    import sys

    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    return bool(main_file) and os.path.exists(str(main_file))


def _execute_cmi_tests(
    args_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int]],
    parallel: bool,
    max_workers: int | None,
) -> List[Tuple[float, float]]:
    """Execute CMI permutation tests either in parallel or sequentially.

    Parameters
    ----------
    args_list : List[Tuple[...]]
        List of argument tuples for _cmi_perm_from_args.
    parallel : bool
        Whether to use parallel execution.
    max_workers : int | None
        Maximum number of parallel workers.

    Returns
    -------
    List[Tuple[float, float]]
        List of (cmi_value, p_value) tuples.
    """
    results: List[Tuple[float, float]] = []

    if parallel and len(args_list) > 1:
        import concurrent.futures as cf

        if _check_parallel_execution_feasibility():
            with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
                for res in ex.map(_cmi_perm_from_args, args_list):
                    results.append(res)
        else:
            # Fallback to sequential if parallel is not feasible
            for args in args_list:
                results.append(_cmi_perm_from_args(args))
    else:
        for args in args_list:
            results.append(_cmi_perm_from_args(args))

    return results


def _apply_results_to_dataframe(
    df: pd.DataFrame,
    parent_nodes: List[str],
    results: List[Tuple[float, float]],
    significance_level_alpha: float,
) -> pd.DataFrame:
    """Apply CMI test results to the dataframe with BH correction.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update.
    parent_nodes : List[str]
        List of parent node names tested.
    results : List[Tuple[float, float]]
        List of (cmi_value, p_value) tuples.
    significance_level_alpha : float
        Significance level for BH correction.

    Returns
    -------
    pd.DataFrame
        The updated dataframe with results.
    """
    if not results:
        return df

    cmi_vals = np.array([r[0] for r in results], dtype=float)
    pvals = np.array([r[1] for r in results], dtype=float)

    reject, p_corr, _ = benjamini_hochberg_correction(
        pvals, alpha=float(significance_level_alpha)
    )

    df.loc[parent_nodes, "Sibling_CMI"] = cmi_vals
    df.loc[parent_nodes, "Sibling_CMI_P_Value"] = pvals

    if p_corr.size:
        df.loc[parent_nodes, "Sibling_CMI_P_Value_Corrected"] = p_corr
        df.loc[parent_nodes, "Sibling_BH_Dependent"] = reject
    else:
        df.loc[parent_nodes, "Sibling_BH_Dependent"] = False

    df["Sibling_BH_Independent"] = ~df["Sibling_BH_Dependent"]

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

    df.loc[skipped_nodes, "Sibling_CMI_Skipped"] = True
    df.loc[skipped_nodes, "Sibling_BH_Independent"] = False

    return df


def _enforce_boolean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure boolean columns have proper boolean dtype without downcasting warnings.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to process.

    Returns
    -------
    pd.DataFrame
        The dataframe with boolean columns properly typed.
    """
    with pd.option_context("future.no_silent_downcasting", True):
        df["Sibling_BH_Dependent"] = (
            df["Sibling_BH_Dependent"].fillna(False).astype(bool)
        )
        df["Sibling_BH_Independent"] = (
            df["Sibling_BH_Independent"].fillna(False).astype(bool)
        )
        df["Sibling_CMI_Skipped"] = df["Sibling_CMI_Skipped"].fillna(False).astype(bool)

    return df


def annotate_sibling_independence_cmi(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = config.SIGNIFICANCE_ALPHA,
    n_permutations: int = config.N_PERMUTATIONS,
    binarization_threshold: float = 0.5,
    random_state: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """CMI-based sibling independence annotation using batched permutation tests.

    Tests whether sibling nodes in the tree are conditionally independent given
    their parent using conditional mutual information (CMI) and permutation tests.

    Parameters
    ----------
    tree : nx.DiGraph
        The hierarchical tree structure.
    nodes_statistics_dataframe : pd.DataFrame
        Dataframe with node statistics, must contain 'distribution' column.
    significance_level_alpha : float, optional
        Significance level for hypothesis tests.
    n_permutations : int, optional
        Number of permutations for permutation tests.
    binarization_threshold : float, optional
        Threshold for binarizing distributions.
    random_state : int | None, optional
        Random seed for reproducibility.
    parallel : bool, optional
        Whether to use parallel execution.
    max_workers : int | None, optional
        Maximum number of parallel workers.
    batch_size : int, optional
        Batch size for vectorized permutation tests.

    Returns
    -------
    pd.DataFrame
        Updated dataframe with sibling independence test results.
    """
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    # Initialize output columns
    df = _initialize_dataframe_columns(df)

    # Prepare binary distributions
    bin_dist = _prepare_binary_distributions(df, binarization_threshold)

    # Extract local significance information
    local_sig_map = _extract_local_significance_map(df)

    # Collect test arguments for all eligible parent nodes
    parent_nodes, args_list, skipped_nodes = _collect_cmi_test_arguments(
        tree=tree,
        bin_dist=bin_dist,
        local_sig_map=local_sig_map,
        n_permutations=n_permutations,
        batch_size=batch_size,
        random_state=random_state,
    )

    # Early exit if no tests to perform
    if not parent_nodes:
        df = _finalize_skipped_nodes(df, skipped_nodes)
        df = _enforce_boolean_types(df)
        return df

    # Execute CMI tests
    results = _execute_cmi_tests(args_list, parallel, max_workers)

    # Apply results with BH correction
    df = _apply_results_to_dataframe(
        df, parent_nodes, results, significance_level_alpha
    )

    # Handle skipped nodes
    df = _finalize_skipped_nodes(df, skipped_nodes)

    # Enforce boolean types
    df = _enforce_boolean_types(df)

    return df


__all__ = [
    "annotate_sibling_independence_cmi",
]
