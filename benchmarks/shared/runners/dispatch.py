"""Shared dispatch helper to run a registered clustering method."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from kl_clustering_analysis import config


def run_clustering(
    data_df: pd.DataFrame,
    method_id: str,
    params: Dict[str, Any],
    seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], int, str]:
    """Run one benchmark method and return ``(labels, n_clusters, status)``."""
    spec = METHOD_SPECS[method_id]
    distance_condensed = None
    distance_matrix = None

    try:
        if method_id in {"kl", "kl_rogerstanimoto"}:
            distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
            result = spec.runner(
                data_df,
                distance_condensed,
                config.SIBLING_ALPHA,
                tree_linkage_method=params.get(
                    "tree_linkage_method", config.TREE_LINKAGE_METHOD
                ),
            )
        elif method_id in {"kmeans", "spectral"}:
            result = spec.runner(data_df.values, params, seed)
        elif method_id in {"leiden", "louvain"}:
            distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
            distance_matrix = squareform(distance_condensed)
            result = spec.runner(distance_matrix, params, seed)
        else:
            distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
            distance_matrix = squareform(distance_condensed)
            result = spec.runner(distance_matrix, params)

        if result.status == "ok" and result.labels is not None:
            return np.asarray(result.labels), int(result.found_clusters), "ok"

        found_clusters = int(getattr(result, "found_clusters", 0) or 0)
        return None, found_clusters, str(result.status)
    except Exception as exc:
        return None, 0, f"error: {exc}"


__all__ = ["run_clustering"]
