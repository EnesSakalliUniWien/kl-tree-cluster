# Fast drop-in replacements with identical public APIs.

from typing import Dict, Union, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import networkx as nx

# Import the optimized MI/CMI functions from cmi.py
from .cmi import (
    _cmi_perm_from_args,
)

# Import thresholding functions
from misc.thresholding import compute_otsu_threshold, compute_li_threshold

# ======================================================================
# 1) SHARED HELPERS
# ======================================================================


def _calculate_chi_square_test(
    kl_divergence: float, number_of_leaves: int, number_of_features: int
) -> Tuple[float, int, float]:
    """Chi-square statistic, degrees of freedom, and p-value via 2n*KL ~ χ²_F."""
    chi2_statistic = 2.0 * float(number_of_leaves) * float(kl_divergence)
    dof = int(number_of_features)
    p_value = float(chi2.sf(chi2_statistic, df=dof))
    return chi2_statistic, dof, p_value


def apply_benjamini_hochberg_correction(
    p_values: np.ndarray, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float]:
    """BH/FDR control with empty-input guard."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float), float(alpha)
    reject, corrected_p_values, _, _ = multipletests(
        p_values, alpha=alpha, method="fdr_bh", is_sorted=False, returnsorted=False
    )
    return reject.astype(bool), corrected_p_values.astype(float), float(alpha)


def _binary_threshold(arr: np.ndarray, thr: Union[float, str] = 0.5) -> np.ndarray:
    """Return uint8(0/1) vector for thresholded probabilities."""
    a = np.asarray(arr, dtype=float)
    if isinstance(thr, str):
        if thr == "otsu":
            thr = compute_otsu_threshold(a)
        elif thr == "li":
            thr = compute_li_threshold(a)
        else:
            raise ValueError(f"Unknown threshold method: {thr}")
    return (a >= float(thr)).astype(np.uint8, copy=False)


# ======================================================================
# 2) VECTORIZED MI / CMI (FASTER)
#    (No large boolean masks; use dot-products for 2x2 counts)
# ======================================================================

# MI/CMI functions are now imported from cmi.py for better performance and accuracy


# ======================================================================
# 4) MAIN FUNCTIONS (same signatures, with speedups)
# ======================================================================


def test_feature_independence_conservative(
    kl_divergence_from_uniform: float,
    number_of_leaves_in_node: int,
    number_of_features: int,
    significance_level_alpha: float = 0.05,
    number_of_tests_for_correction: Optional[int] = None,
) -> Dict[str, Union[float, bool, str, int]]:
    """Conservative χ² test with Bonferroni-style alpha."""
    chi2_statistic, dof, p_value = _calculate_chi_square_test(
        kl_divergence_from_uniform, number_of_leaves_in_node, number_of_features
    )
    m = int(number_of_tests_for_correction) if number_of_tests_for_correction else 0
    alpha_used = (
        (significance_level_alpha / m) if m > 0 else float(significance_level_alpha)
    )
    are_features_dependent = bool(p_value < alpha_used)
    result = "Features Dependent" if are_features_dependent else "Features Independent"
    return {
        "independence_conservative_chi2_statistic": chi2_statistic,
        "independence_conservative_degrees_of_freedom": dof,
        "independence_conservative_p_value": p_value,
        "independence_conservative_alpha_used": alpha_used,
        "independence_conservative_are_features_dependent": are_features_dependent,
        "independence_conservative_result": result,
    }


def test_feature_independence_liberal(
    kl_divergence_from_uniform: float,
    number_of_leaves_in_node: int,
    number_of_features: int,
    significance_level_alpha: float = 0.05,
) -> Dict[str, Union[float, str, bool, int]]:
    """Liberal χ² test (no multiple testing correction)."""
    chi2_statistic, dof, p_value = _calculate_chi_square_test(
        kl_divergence_from_uniform, number_of_leaves_in_node, number_of_features
    )
    are_features_dependent = bool(p_value < significance_level_alpha)
    result = "Features Dependent" if are_features_dependent else "Features Independent"
    return {
        "independence_liberal_chi2_statistic": chi2_statistic,
        "independence_liberal_degrees_of_freedom": dof,
        "independence_liberal_p_value": p_value,
        "independence_liberal_alpha_used": float(significance_level_alpha),
        "independence_liberal_are_features_dependent": are_features_dependent,
        "independence_liberal_result": result,
    }


def kl_divergence_deviation_from_zero_test(
    kl_divergence: float,
    all_kl_divergences: np.ndarray,
    alpha: float = 0.05,
    num_std: float = 2.0,
    leave_one_out_value: Optional[float] = None,
) -> Dict[str, Union[float, bool, str]]:
    """Empirical z-score style deviation test with optional leave-one-out."""
    arr = np.asarray(all_kl_divergences, dtype=float)
    if arr.size == 0:
        return {
            "z_score": 0.0,
            "std_kl": 0.0,
            "threshold": 0.0,
            "is_significant": False,
            "result": "Not Significant",
        }
    if leave_one_out_value is not None and arr.size > 1:
        arr = arr[arr != leave_one_out_value]
    std_kl = float(np.std(arr)) if arr.size > 0 else 0.0
    z_score = float(kl_divergence / std_kl) if std_kl > 0 else 0.0
    threshold = float(num_std * std_kl)
    is_significant = bool(abs(z_score) > num_std)
    return {
        "z_score": z_score,
        "std_kl": std_kl,
        "threshold": threshold,
        "alpha": float(alpha),
        "num_std_used": float(num_std),
        "is_significant": is_significant,
        "result": "Significant" if is_significant else "Not Significant",
    }


def annotate_nodes_with_statistical_significance_tests(
    nodes_statistics_dataframe: pd.DataFrame,
    total_number_of_features: int,
    significance_level_alpha: float = 0.05,
    std_deviation_threshold: float = 2.0,
    include_deviation_test: bool = True,
) -> pd.DataFrame:
    """
    Vectorized annotation of nodes with BH/FDR-corrected significance.
    """
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    dof = int(total_number_of_features)
    alpha = float(significance_level_alpha)
    num_std = float(std_deviation_threshold)

    # Precreate output cols
    n = len(df)
    df["BH_P_Value_Uncorrected"] = np.nan
    df["BH_P_Value_Corrected"] = np.nan
    df["BH_Significant"] = False
    df["Are_Features_Dependent"] = False
    df["Independence_Conservative_P_Value"] = np.nan
    df["Independence_Conservative_Result"] = pd.Series([None] * n, dtype="object")
    df["Independence_Liberal_P_Value"] = np.nan
    df["Independence_Liberal_Result"] = pd.Series([None] * n, dtype="object")
    if include_deviation_test:
        df["Deviation_Z_Score"] = np.nan
        df["Deviation_Threshold"] = np.nan
        df["Deviation_Result"] = pd.Series([None] * n, dtype="object")

    # Vectorized chi-square on valid nodes
    kl_globals = df.get(
        "kl_divergence_global", pd.Series(index=df.index, dtype=float)
    ).to_numpy()
    leaf_counts = df.get(
        "leaf_count", pd.Series(index=df.index, dtype=float)
    ).to_numpy()
    valid = np.isfinite(kl_globals) & (leaf_counts > 0)

    if not np.any(valid):
        return df

    chi2_stats = 2.0 * leaf_counts[valid] * kl_globals[valid]
    p_values = chi2.sf(chi2_stats, df=dof)

    # BH/FDR across tested nodes
    bh_reject, bh_p_corr, _ = apply_benjamini_hochberg_correction(p_values, alpha=alpha)

    # Assign back
    idx = np.flatnonzero(valid)
    df.iloc[idx, df.columns.get_loc("BH_P_Value_Uncorrected")] = p_values
    df.iloc[idx, df.columns.get_loc("BH_P_Value_Corrected")] = bh_p_corr
    df.iloc[idx, df.columns.get_loc("BH_Significant")] = bh_reject
    df.iloc[idx, df.columns.get_loc("Are_Features_Dependent")] = bh_reject

    # Conservative/Liberal labels
    m_tests = p_values.size
    alpha_cons = (alpha / m_tests) if m_tests > 0 else alpha
    cons_labels = np.where(
        p_values < alpha_cons, "Features Dependent", "Features Independent"
    )
    lib_labels = np.where(
        p_values < alpha, "Features Dependent", "Features Independent"
    )
    df.iloc[idx, df.columns.get_loc("Independence_Conservative_P_Value")] = p_values
    df.iloc[idx, df.columns.get_loc("Independence_Conservative_Result")] = cons_labels
    df.iloc[idx, df.columns.get_loc("Independence_Liberal_P_Value")] = p_values
    df.iloc[idx, df.columns.get_loc("Independence_Liberal_Result")] = lib_labels

    # Deviation test (pooled std)
    if include_deviation_test:
        pool = (
            df.loc[~df.get("is_leaf", False), "kl_divergence_global"]
            .dropna()
            .to_numpy()
            if "is_leaf" in df.columns
            else df["kl_divergence_global"].dropna().to_numpy()
        )
        std_kl = float(np.std(pool)) if pool.size > 0 else 0.0
        threshold = num_std * std_kl
        z_scores = np.zeros(n, dtype=float) if std_kl == 0.0 else (kl_globals / std_kl)
        deviation_results = np.where(
            np.abs(z_scores) > num_std, "Significant", "Not Significant"
        )
        df["Deviation_Z_Score"] = z_scores
        df["Deviation_Threshold"] = threshold
        df["Deviation_Result"] = deviation_results

    # Normalize boolean
    for col in ("Are_Features_Dependent", "BH_Significant"):
        if col in df.columns:
            with pd.option_context("future.no_silent_downcasting", True):
                df[col] = df[col].fillna(False).astype(bool)

    return df


def annotate_local_child_parent_significance(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    total_number_of_features: int,
    significance_level_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Faster local (child vs parent) significance using vectorized arrays & edge list.
    """
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    dof = int(total_number_of_features)
    alpha = float(significance_level_alpha)

    # Edge list: parents->children; we only need child node ids
    children = [v for _, v in tree.edges()]
    if not children:
        # Initialize columns anyway
        df["Local_P_Value_Uncorrected"] = np.nan
        df["Local_P_Value_Corrected"] = np.nan
        df["Local_BH_Significant"] = False
        df["Local_Are_Features_Dependent"] = False
        return df

    kl_local = (
        df.get("kl_divergence_local", pd.Series(index=df.index, dtype=float))
        .reindex(children)
        .to_numpy()
    )
    leaf_cnt = (
        df.get("leaf_count", pd.Series(index=df.index, dtype=float))
        .reindex(children)
        .to_numpy()
    )

    mask = np.isfinite(kl_local) & (leaf_cnt > 0)
    pvals = np.full(len(children), np.nan, dtype=float)
    pvals[mask] = chi2.sf(2.0 * leaf_cnt[mask] * kl_local[mask], df=dof)

    reject, p_corr, _ = apply_benjamini_hochberg_correction(
        pvals[np.isfinite(pvals)], alpha=alpha
    )
    df["Local_P_Value_Uncorrected"] = np.nan
    df["Local_P_Value_Corrected"] = np.nan
    df["Local_BH_Significant"] = False
    df["Local_Are_Features_Dependent"] = False

    # Assign back for tested nodes
    tested_idx = np.flatnonzero(np.isfinite(pvals))
    tested_nodes = [children[i] for i in tested_idx]
    if tested_nodes:
        df.loc[tested_nodes, "Local_P_Value_Uncorrected"] = pvals[tested_idx]
        if p_corr.size:
            df.loc[tested_nodes, "Local_P_Value_Corrected"] = p_corr
            df.loc[tested_nodes, "Local_BH_Significant"] = reject
            df.loc[tested_nodes, "Local_Are_Features_Dependent"] = reject

    with pd.option_context("future.no_silent_downcasting", True):
        df["Local_BH_Significant"] = (
            df["Local_BH_Significant"].fillna(False).astype(bool)
        )
        df["Local_Are_Features_Dependent"] = (
            df["Local_Are_Features_Dependent"].fillna(False).astype(bool)
        )

    return df


def annotate_sibling_independence_cmi(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    *,
    significance_level_alpha: float = 0.05,
    permutations: int = 75,
    binarization_threshold: float = 0.5,
    random_state: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Faster CMI annotation using batched, vectorized permutation tests.
    """
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    # Pre-binarize distributions and store in dict (fast lookup, small dtype)
    dist_series = df.get("distribution", pd.Series(index=df.index, dtype=object))
    dist_dict = dist_series.to_dict()
    bin_dist = {
        node: _binary_threshold(dist, thr=binarization_threshold)
        for node, dist in dist_dict.items()
        if dist is not None
    }

    # Collect parents with exactly two children
    parent_nodes: list[str] = []
    args_list: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int]
    ] = []

    # Seeds via SeedSequence (cheap, reproducible)
    ss = np.random.SeedSequence(random_state) if random_state is not None else None

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        c1, c2 = children
        x = bin_dist.get(c1)
        y = bin_dist.get(c2)
        z = bin_dist.get(parent)
        if x is None or y is None or z is None:
            continue
        if not (x.size and y.size and z.size) or not (x.size == y.size == z.size):
            continue
        seed = None
        if ss is not None:
            seed = int(np.random.SeedSequence(ss.generate_state(1)[0]).entropy)
        parent_nodes.append(parent)
        args_list.append((x, y, z, int(permutations), seed, int(batch_size)))

    # Prepare outputs
    df["Sibling_CMI"] = np.nan
    df["Sibling_CMI_P_Value"] = np.nan
    df["Sibling_CMI_P_Value_Corrected"] = np.nan
    df["Sibling_BH_Dependent"] = False
    df["Sibling_BH_Independent"] = False

    if not parent_nodes:
        return df

    # Possibly parallel execution (safe-guard for interactive sessions)
    results: list[tuple[float, float]] = []
    if parallel and len(parent_nodes) > 1:
        import sys
        import os
        import concurrent.futures as cf  # local import to avoid overhead

        main_mod = sys.modules.get("__main__")
        main_file = getattr(main_mod, "__file__", None)
        can_spawn = bool(main_file) and os.path.exists(str(main_file))
        if can_spawn:
            with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
                for res in ex.map(_cmi_perm_from_args, args_list):
                    results.append(res)
        else:
            for a in args_list:
                results.append(_cmi_perm_from_args(a))
    else:
        for a in args_list:
            results.append(_cmi_perm_from_args(a))

    cmi_vals = np.array([r[0] for r in results], dtype=float)
    pvals = np.array([r[1] for r in results], dtype=float)

    # Multiple testing across parents
    reject, p_corr, _ = apply_benjamini_hochberg_correction(
        pvals, alpha=float(significance_level_alpha)
    )

    # Assign back
    df.loc[parent_nodes, "Sibling_CMI"] = cmi_vals
    df.loc[parent_nodes, "Sibling_CMI_P_Value"] = pvals
    if p_corr.size:
        df.loc[parent_nodes, "Sibling_CMI_P_Value_Corrected"] = p_corr
        df.loc[parent_nodes, "Sibling_BH_Dependent"] = reject
    else:
        df.loc[parent_nodes, "Sibling_BH_Dependent"] = False
    df["Sibling_BH_Independent"] = ~df["Sibling_BH_Dependent"]

    with pd.option_context("future.no_silent_downcasting", True):
        df["Sibling_BH_Dependent"] = (
            df["Sibling_BH_Dependent"].fillna(False).astype(bool)
        )
        df["Sibling_BH_Independent"] = (
            df["Sibling_BH_Independent"].fillna(False).astype(bool)
        )

    return df
