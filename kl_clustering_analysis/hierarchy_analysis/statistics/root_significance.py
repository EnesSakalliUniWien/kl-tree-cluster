from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .shared_utils import (
    apply_benjamini_hochberg_correction,
    calculate_chi_square_test,
)


def test_feature_independence_conservative(
    kl_divergence_from_uniform: float,
    number_of_leaves_in_node: int,
    number_of_features: int,
    significance_level_alpha: float = 0.05,
    number_of_tests_for_correction: Optional[int] = None,
) -> Dict[str, Union[float, bool, str, int]]:
    """Conservative χ² test with Bonferroni-style correction."""
    chi2_statistic, dof, p_value = calculate_chi_square_test(
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
    """Liberal χ² test without multiple-testing correction."""
    chi2_statistic, dof, p_value = calculate_chi_square_test(
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
    """Vectorized annotation of nodes with BH/FDR-corrected root-level significance."""
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    dof = int(total_number_of_features)
    alpha = float(significance_level_alpha)
    num_std = float(std_deviation_threshold)

    # Precreate output columns
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

    bh_reject, bh_p_corr, _ = apply_benjamini_hochberg_correction(p_values, alpha=alpha)

    idx = np.flatnonzero(valid)
    df.iloc[idx, df.columns.get_loc("BH_P_Value_Uncorrected")] = p_values
    df.iloc[idx, df.columns.get_loc("BH_P_Value_Corrected")] = bh_p_corr
    df.iloc[idx, df.columns.get_loc("BH_Significant")] = bh_reject
    df.iloc[idx, df.columns.get_loc("Are_Features_Dependent")] = bh_reject

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

    for col in ("Are_Features_Dependent", "BH_Significant"):
        if col in df.columns:
            with pd.option_context("future.no_silent_downcasting", True):
                df[col] = df[col].fillna(False).astype(bool)

    return df


__all__ = [
    "test_feature_independence_conservative",
    "test_feature_independence_liberal",
    "kl_divergence_deviation_from_zero_test",
    "annotate_nodes_with_statistical_significance_tests",
]
