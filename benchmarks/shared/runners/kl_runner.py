"""KL method runner.

Builds a PosetTree using standard hierarchical clustering linkage and performs KL decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import (
    _labels_and_report_from_decomposition,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage


def _run_kl_on_distance(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    significance_level: float,
    *,
    tree_linkage_method: str,
    extra: dict[str, object] | None = None,
) -> MethodRunResult:
    linkage_matrix = linkage(distance_condensed, method=tree_linkage_method)

    tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)
    decomposition = tree.decompose(
        annotations_df=tree.annotations_df,
        leaf_data=data_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )
    labels, report_df = _labels_and_report_from_decomposition(
        decomposition,
        data_df.index.tolist(),
    )
    result_extra = {
        "tree": tree,
        "decomposition": decomposition,
        "annotations": tree.annotations_df,
        "linkage_matrix": linkage_matrix,
    }
    if extra:
        result_extra.update(extra)

    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomposition["num_clusters"]),
        report_df=report_df,
        status="ok",
        skip_reason=None,
        extra=result_extra,
    )


def _run_kl_method(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray,
    significance_level: float,
    tree_linkage_method: str = config.TREE_LINKAGE_METHOD,
) -> MethodRunResult:
    return _run_kl_on_distance(
        data_df,
        distance_condensed,
        significance_level,
        tree_linkage_method=tree_linkage_method,
    )
