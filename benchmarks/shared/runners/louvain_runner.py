"""Louvain runner (moved to benchmarking.runners).

Same implementation as before; helpers are imported lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

    try:
        n_neighbors = _resolve_n_neighbors(n_samples, params.get("n_neighbors"))
        resolution = float(params.get("resolution", 1.0))
        random_state = 42 if seed is None else int(seed)
    except (TypeError, ValueError) as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"Louvain input preparation failed: {type(exc).__name__}: {exc}",
        )
    edges = _knn_edge_weights(distance_matrix, n_neighbors)
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
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"Louvain unavailable: {type(exc).__name__}: {exc}",
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(n_samples))
    graph.add_weighted_edges_from(edges)
    communities = louvain_communities(
        graph,
        weight="weight",
        resolution=resolution,
        seed=random_state,
    )
    labels = np.full(n_samples, -1, dtype=int)
    for cluster_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            labels[int(node)] = cluster_id
    labels = _normalize_labels(labels)
    report_df = _create_report_dataframe_from_labels(labels, pd.Index(range(n_samples)))
    return MethodRunResult(
        labels=labels,
        found_clusters=int(len({x for x in labels if x >= 0})),
        report_df=report_df,
        status="ok",
        skip_reason=None,
    )
