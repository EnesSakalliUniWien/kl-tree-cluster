from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from ..local_kl_utils import get_local_kl_series
from .shared_utils import apply_benjamini_hochberg_correction


def annotate_child_parent_divergence(
    tree: nx.DiGraph,
    nodes_statistics_dataframe: pd.DataFrame,
    total_number_of_features: int,
    significance_level_alpha: float = 0.05,
) -> pd.DataFrame:
    """Vectorized child-vs-parent χ² testing across edges."""
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    dof = int(total_number_of_features)
    alpha = float(significance_level_alpha)

    children = [v for _, v in tree.edges()]
    if not children:
        df["Local_P_Value_Uncorrected"] = np.nan
        df["Local_P_Value_Corrected"] = np.nan
        df["Local_BH_Significant"] = False
        df["Local_Are_Features_Dependent"] = False
        return df

    kl_local = (
        get_local_kl_series(df)
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


__all__ = [
    "annotate_child_parent_divergence",
]
