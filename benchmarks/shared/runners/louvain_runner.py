"""Louvain runner (moved to benchmarking.runners).

Same implementation as before; helpers are imported lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def _run_louvain_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None,
):
    """Run Louvain on a precomputed distance matrix and return a
    `MethodRunResult` (imported lazily to avoid circular imports).
    """
    from benchmarks.shared.types.method_run_result import (
        MethodRunResult,
    )
    from benchmarks.shared.utils_decomp import (
        _create_report_dataframe_from_labels,
    )
    from benchmarks.shared.utils import (
        _resolve_n_neighbors,
        _knn_edge_weights,
        _normalize_labels,
        _estimate_dbscan_eps,
    )

    n_samples = distance_matrix.shape[0]
    n_neighbors = _resolve_n_neighbors(n_samples, params.get("n_neighbors"))
    resolution = float(params.get("resolution", 1.0))
    edges = _knn_edge_weights(distance_matrix, n_neighbors)
    if not edges:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(
                labels, pd.Index(range(n_samples))
            ),
            status="ok",
            skip_reason=None,
        )

    try:
        import community
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(n_samples))
        graph.add_weighted_edges_from(edges)
        partition = community.best_partition(
            graph, weight="weight", resolution=resolution, random_state=seed
        )
        labels = _normalize_labels(
            np.array([partition.get(i, -1) for i in range(n_samples)], dtype=int)
        )
        report_df = _create_report_dataframe_from_labels(
            labels, pd.Index(range(n_samples))
        )
        return MethodRunResult(
            labels=labels,
            found_clusters=int(len({x for x in labels if x >= 0})),
            report_df=report_df,
            status="ok",
            skip_reason=None,
        )
    except Exception:
        # sklearn-only fallback for environments without python-louvain.
        # Use density clustering on the precomputed distance matrix.
        min_samples = max(2, min(n_neighbors, n_samples - 1))
        eps = _estimate_dbscan_eps(distance_matrix, min_samples)
        eps = float(eps / max(resolution, 1e-6))
        model = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
        labels = _normalize_labels(model.fit_predict(distance_matrix))
        report_df = _create_report_dataframe_from_labels(
            labels, pd.Index(range(n_samples))
        )
        return MethodRunResult(
            labels=labels,
            found_clusters=int(len({x for x in labels if x >= 0})),
            report_df=report_df,
            status="ok",
            skip_reason=None,
            extra={"fallback": "sklearn_dbscan", "fallback_eps": eps},
        )
