"""KL method runner.

Builds a PosetTree using standard hierarchical clustering linkage and performs KL decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from benchmarks.shared.types import MethodRunResult
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from benchmarks.shared.util.decomposition import (
    _labels_from_decomposition,
    _create_report_dataframe,
)


def _run_kl_method(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    significance_level: float,
    tree_linkage_method: str = config.TREE_LINKAGE_METHOD,
) -> MethodRunResult:

    Z_t = linkage(distance_condensed, method=tree_linkage_method)

    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_df.index.tolist())
    decomp_t = tree_t.decompose(
        leaf_data=data_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )
    report_t = _create_report_dataframe(decomp_t.get("cluster_assignments", {}))
    labels = np.asarray(_labels_from_decomposition(decomp_t, data_df.index.tolist()))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomp_t.get("num_clusters", 0)),
        report_df=report_t,
        status="ok",
        skip_reason=None,
        extra={
            "tree": tree_t,
            "decomposition": decomp_t,
            "stats": tree_t.stats_df,
            "linkage_matrix": Z_t,
        },
    )
