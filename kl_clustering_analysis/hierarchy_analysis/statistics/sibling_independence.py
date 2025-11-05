from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from ..cmi import _cmi_perm_from_args
from .shared_utils import apply_benjamini_hochberg_correction, binary_threshold


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
    """CMI-based sibling independence annotation using batched permutation tests."""
    df = nodes_statistics_dataframe.copy()
    if df.empty:
        return df

    df["Sibling_CMI_Skipped"] = False

    dist_series = df.get("distribution", pd.Series(index=df.index, dtype=object))
    dist_dict = dist_series.to_dict()
    bin_dist = {
        node: binary_threshold(dist, thr=binarization_threshold)
        for node, dist in dist_dict.items()
        if dist is not None
    }

    local_sig_map: dict[str, bool] | None = None
    if "Local_BH_Significant" in df.columns:
        local_sig_map = (
            df["Local_BH_Significant"].fillna(False).astype(bool).to_dict()
        )
    elif "Local_Are_Features_Dependent" in df.columns:
        local_sig_map = (
            df["Local_Are_Features_Dependent"].fillna(False).astype(bool).to_dict()
        )

    parent_nodes: List[str] = []
    args_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int | None, int]] = []
    skipped_nodes: List[str] = []

    ss = np.random.SeedSequence(random_state) if random_state is not None else None

    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        c1, c2 = children

        if local_sig_map is not None:
            if not (local_sig_map.get(c1, False) and local_sig_map.get(c2, False)):
                skipped_nodes.append(parent)
                continue

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

    df["Sibling_CMI"] = np.nan
    df["Sibling_CMI_P_Value"] = np.nan
    df["Sibling_CMI_P_Value_Corrected"] = np.nan
    df["Sibling_BH_Dependent"] = False
    df["Sibling_BH_Independent"] = False

    if not parent_nodes:
        if skipped_nodes:
            df.loc[skipped_nodes, "Sibling_CMI_Skipped"] = True
        return df

    results: List[Tuple[float, float]] = []
    if parallel and len(parent_nodes) > 1:
        import os
        import sys
        import concurrent.futures as cf

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

    reject, p_corr, _ = apply_benjamini_hochberg_correction(
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

    if skipped_nodes:
        df.loc[skipped_nodes, "Sibling_CMI_Skipped"] = True
        df.loc[skipped_nodes, "Sibling_BH_Independent"] = False

    with pd.option_context("future.no_silent_downcasting", True):
        df["Sibling_BH_Dependent"] = (
            df["Sibling_BH_Dependent"].fillna(False).astype(bool)
        )
        df["Sibling_BH_Independent"] = (
            df["Sibling_BH_Independent"].fillna(False).astype(bool)
        )
        df["Sibling_CMI_Skipped"] = (
            df["Sibling_CMI_Skipped"].fillna(False).astype(bool)
        )

    return df


__all__ = [
    "annotate_sibling_independence_cmi",
]
