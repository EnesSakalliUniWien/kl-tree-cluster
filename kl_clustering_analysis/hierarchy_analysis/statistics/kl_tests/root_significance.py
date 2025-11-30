from __future__ import annotations

from typing import Dict, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..multiple_testing import benjamini_hochberg_correction
from .utils import get_local_kl_series
from kl_clustering_analysis import config

if TYPE_CHECKING:  # pragma: no cover
    from kl_clustering_analysis.tree.poset_tree import PosetTree


def permutation_test_split_significance(
    tree: PosetTree,
    node_id: str,
    data_df: pd.DataFrame,
    n_permutations: int = config.N_PERMUTATIONS,
    alpha: float = config.SIGNIFICANCE_ALPHA,
    rng: np.random.Generator | None = None,
) -> Dict[str, Union[float, bool]]:
    """Permutation test for split significance using local KL divergence."""

    if rng is None:
        rng = np.random.default_rng()

    children = list(tree.successors(node_id))
    if len(children) != 2:
        return {
            "p_value": 1.0,
            "is_significant": False,
            "observed_stat": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
        }

    left_child, right_child = children

    def get_leaf_labels(nid: str) -> list[str]:
        if tree.out_degree(nid) == 0:
            return [tree.nodes[nid].get("label", nid)]
        labels: list[str] = []
        for child in tree.successors(nid):
            labels.extend(get_leaf_labels(child))
        return labels

    parent_leaves = get_leaf_labels(node_id)
    left_leaves = get_leaf_labels(left_child)
    right_leaves = get_leaf_labels(right_child)

    available_leaves = [leaf for leaf in parent_leaves if leaf in data_df.index]
    if len(available_leaves) < 2:
        return {
            "p_value": 1.0,
            "is_significant": False,
            "observed_stat": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
        }

    left_available = [leaf for leaf in left_leaves if leaf in data_df.index]
    right_available = [leaf for leaf in right_leaves if leaf in data_df.index]

    if len(left_available) == 0 or len(right_available) == 0:
        return {
            "p_value": 1.0,
            "is_significant": False,
            "observed_stat": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
        }

    n_total = len(available_leaves)
    n_left = len(left_available)
    n_right = len(right_available)

    def pseudobulk(sample_ids: list[str]) -> np.ndarray:
        subset = data_df.loc[sample_ids]
        return subset.mean(axis=0).values

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        eps = config.EPSILON
        p = np.clip(p, eps, 1 - eps)
        q = np.clip(q, eps, 1 - eps)
        kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        return float(np.sum(kl))

    P_parent = pseudobulk(available_leaves)
    P_left = pseudobulk(left_available)
    P_right = pseudobulk(right_available)

    observed = (n_left / n_total) * kl_divergence(P_left, P_parent) + (
        n_right / n_total
    ) * kl_divergence(P_right, P_parent)

    D_perm = np.zeros(n_permutations, dtype=float)
    for idx in range(n_permutations):
        perm_leaves = rng.permutation(available_leaves)
        perm_left = perm_leaves[:n_left]
        perm_right = perm_leaves[n_left:]
        P_perm_left = pseudobulk(list(perm_left))
        P_perm_right = pseudobulk(list(perm_right))
        D_perm[idx] = (n_left / n_total) * kl_divergence(P_perm_left, P_parent) + (
            n_right / n_total
        ) * kl_divergence(P_perm_right, P_parent)

    p_value = (1 + np.sum(D_perm >= observed)) / (n_permutations + 1)

    return {
        "p_value": float(p_value),
        "is_significant": bool(p_value < alpha),
        "observed_stat": float(observed),
        "null_mean": float(np.mean(D_perm)),
        "null_std": float(np.std(D_perm)),
    }


def annotate_root_node_significance(
    nodes_statistics_dataframe: pd.DataFrame,
    total_number_of_features: int,
    significance_level_alpha: float = config.SIGNIFICANCE_ALPHA,
    std_deviation_threshold: float = config.STD_DEVIATION_THRESHOLD,
    include_deviation_test: bool = True,
    tree: Optional["PosetTree"] = None,
    data_df: Optional[pd.DataFrame] = None,
    use_permutation_test: bool = False,
    n_permutations: int = config.N_PERMUTATIONS,
    permutation_seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Annotate nodes with root-level significance testing (deviation from uniform).

    Tests whether each node's distribution significantly deviates from uniform
    (i.e., features are dependent). Uses either permutation tests or chi-square
    tests with Benjamini-Hochberg correction for multiple testing.
    """
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

    if use_permutation_test and tree is not None and data_df is not None:
        internal_nodes = df.index[~df.get("is_leaf", False)].tolist()
        rng = (
            np.random.default_rng(permutation_seed)
            if permutation_seed is not None
            else np.random.default_rng()
        )
        p_values_list: list[float] = []
        valid_nodes: list[str] = []

        for node_id in internal_nodes:
            result = permutation_test_split_significance(
                tree=tree,
                node_id=node_id,
                data_df=data_df,
                n_permutations=n_permutations,
                alpha=alpha,
                rng=rng,
            )
            if result["p_value"] < 1.0:
                p_values_list.append(result["p_value"])
                valid_nodes.append(node_id)
                df.loc[node_id, "Permutation_P_Value"] = result["p_value"]
                df.loc[node_id, "Permutation_Observed"] = result["observed_stat"]
                df.loc[node_id, "Permutation_Null_Mean"] = result["null_mean"]
                df.loc[node_id, "Permutation_Null_Std"] = result["null_std"]

        if p_values_list:
            p_values = np.array(p_values_list, dtype=float)
            reject, p_corr, _ = benjamini_hochberg_correction(p_values, alpha=alpha)
            for idx, node_id in enumerate(valid_nodes):
                df.loc[node_id, "BH_P_Value_Uncorrected"] = p_values[idx]
                df.loc[node_id, "BH_P_Value_Corrected"] = p_corr[idx]
                df.loc[node_id, "BH_Significant"] = reject[idx]
                df.loc[node_id, "Are_Features_Dependent"] = reject[idx]
    else:
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

        bh_reject, bh_p_corr, _ = benjamini_hochberg_correction(p_values, alpha=alpha)

        idx_valid = np.flatnonzero(valid)
        df.iloc[idx_valid, df.columns.get_loc("BH_P_Value_Uncorrected")] = p_values
        df.iloc[idx_valid, df.columns.get_loc("BH_P_Value_Corrected")] = bh_p_corr
        df.iloc[idx_valid, df.columns.get_loc("BH_Significant")] = bh_reject
        df.iloc[idx_valid, df.columns.get_loc("Are_Features_Dependent")] = bh_reject

        m_tests = p_values.size
        alpha_cons = (alpha / m_tests) if m_tests > 0 else alpha
        cons_labels = np.where(
            p_values < alpha_cons, "Features Dependent", "Features Independent"
        )
        lib_labels = np.where(
            p_values < alpha, "Features Dependent", "Features Independent"
        )
        df.iloc[idx_valid, df.columns.get_loc("Independence_Conservative_P_Value")] = (
            p_values
        )
        df.iloc[idx_valid, df.columns.get_loc("Independence_Conservative_Result")] = (
            cons_labels
        )
        df.iloc[idx_valid, df.columns.get_loc("Independence_Liberal_P_Value")] = (
            p_values
        )
        df.iloc[idx_valid, df.columns.get_loc("Independence_Liberal_Result")] = (
            lib_labels
        )

    if include_deviation_test:
        pool = (
            get_local_kl_series(df).loc[~df.get("is_leaf", False)].dropna().to_numpy()
            if "is_leaf" in df.columns
            else get_local_kl_series(df).dropna().to_numpy()
        )
        std_kl = float(np.std(pool)) if pool.size > 0 else 0.0
        threshold = num_std * std_kl
        kl_values = get_local_kl_series(df).to_numpy()
        z_scores = np.zeros(n, dtype=float) if std_kl == 0.0 else (kl_values / std_kl)
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
    "permutation_test_split_significance",
    "annotate_root_node_significance",
]
