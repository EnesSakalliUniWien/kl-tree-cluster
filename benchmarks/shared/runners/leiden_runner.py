"""Leiden runner for precomputed distance matrices."""

from __future__ import annotations

import numpy as np
from benchmarks.shared.types.method_run_result import MethodRunResult
from benchmarks.shared.util.core import (
    _knn_edge_weights,
    _normalize_labels,
    _resolve_n_neighbors,
)
from benchmarks.shared.util.decomposition import _ok_result_from_labels


def _run_leiden_method(
    distance_matrix: np.ndarray,
    params: dict[str, object],
    seed: int | None,
) -> MethodRunResult:
    """Run Leiden on a precomputed distance matrix."""
    n_samples = distance_matrix.shape[0]
    try:
        n_neighbors = _resolve_n_neighbors(n_samples, params.get("n_neighbors"))
        resolution = float(params.get("resolution", 1.0))
    except (TypeError, ValueError) as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"Leiden input preparation failed: {type(exc).__name__}: {exc}",
        )
    edges = _knn_edge_weights(distance_matrix, n_neighbors)
    if not edges:
        labels = np.zeros(n_samples, dtype=int)
        return _ok_result_from_labels(labels, range(n_samples))

    try:
        import igraph
        import leidenalg
    except ImportError as exc:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status="skip",
            skip_reason=f"Leiden unavailable: {type(exc).__name__}: {exc}",
        )

    edge_list = [(i, j) for i, j, _w in edges]
    weights = [w for _i, _j, w in edges]
    graph = igraph.Graph(n=n_samples, edges=edge_list, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = _normalize_labels(np.asarray(partition.membership))
    return _ok_result_from_labels(labels, range(n_samples))
