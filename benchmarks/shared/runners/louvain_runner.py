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
    from benchmarks.shared.types.method_run_result import MethodRunResult
    from benchmarks.shared.util.core import (
        _estimate_dbscan_eps,
        _knn_edge_weights,
        _normalize_labels,
        _resolve_n_neighbors,
    )
    from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels

    n_samples = int(distance_matrix.shape[0])
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(labels, pd.Index(range(n_samples))),
            status="ok",
            skip_reason=None,
        )

    random_state = 42 if seed is None else int(seed)
    try:
        n_neighbors = _resolve_n_neighbors(n_samples, params.get("n_neighbors"))
        resolution = float(params.get("resolution", 1.0))
        edges = _knn_edge_weights(distance_matrix, n_neighbors)
    except Exception as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"Louvain input preparation failed: {type(exc).__name__}: {exc}",
        )
    if not edges:
        labels = np.zeros(n_samples, dtype=int)
        return MethodRunResult(
            labels=labels,
            found_clusters=1 if n_samples else 0,
            report_df=_create_report_dataframe_from_labels(labels, pd.Index(range(n_samples))),
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
            graph, weight="weight", resolution=resolution, random_state=random_state
        )
        labels = _normalize_labels(
            np.array([partition.get(i, -1) for i in range(n_samples)], dtype=int)
        )
        report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
        return MethodRunResult(
            labels=labels,
            found_clusters=int(len({x for x in labels if x >= 0})),
            report_df=report_df,
            status="ok",
            skip_reason=None,
        )
    except Exception as primary_exc:
        # sklearn-only fallback for environments without python-louvain.
        # Use density clustering on the precomputed distance matrix.
        try:
            min_samples = max(2, min(n_neighbors, n_samples - 1))
            eps = _estimate_dbscan_eps(distance_matrix, min_samples)
            eps = float(eps / max(resolution, 1e-6))
            model = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
            labels = _normalize_labels(model.fit_predict(distance_matrix))
            report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
            return MethodRunResult(
                labels=labels,
                found_clusters=int(len({x for x in labels if x >= 0})),
                report_df=report_df,
                status="ok",
                skip_reason=None,
                extra={"fallback": "sklearn_dbscan", "fallback_eps": eps},
            )
        except Exception as fallback_exc:
            return MethodRunResult(
                labels=None,
                found_clusters=0,
                report_df=None,
                status="skip",
                skip_reason=(
                    "Louvain failed and fallback DBSCAN failed: "
                    f"{type(primary_exc).__name__}: {primary_exc}; "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ),
            )
