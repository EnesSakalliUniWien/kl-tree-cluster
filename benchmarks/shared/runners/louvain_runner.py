"""Louvain runner for precomputed distance matrices."""

from __future__ import annotations

import numpy as np

from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import (
    _knn_edge_weights,
    _normalize_labels,
    _resolve_n_neighbors,
)
from benchmarks.shared.util.decomposition import _ok_result_from_labels


def _run_louvain_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None,
) -> MethodRunResult:
    """Run Louvain on a precomputed distance matrix."""
    n_samples = int(distance_matrix.shape[0])
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    n_neighbors = _resolve_n_neighbors(n_samples, int(params["n_neighbors"]))
    resolution = float(params["resolution"])
    random_state = 42 if seed is None else int(seed)
    edges = _knn_edge_weights(distance_matrix, n_neighbors)
    if not edges:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    import networkx as nx
    from networkx.algorithms.community import louvain_communities

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
    return _ok_result_from_labels(labels, range(n_samples))
