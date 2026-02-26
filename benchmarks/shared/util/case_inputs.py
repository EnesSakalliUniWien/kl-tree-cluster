"""Case input preparation helpers for benchmark pipelines."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config


DISTANCE_MATRIX_METHODS = {"leiden", "louvain", "dbscan", "optics", "hdbscan"}


def prepare_case_inputs(
    tc: dict[str, object],
    selected_methods: list[str],
) -> tuple[
    object,
    object,
    object,
    dict[str, object],
    np.ndarray | None,
    np.ndarray | None,
    object,
]:
    """Generate case data and resolve shared distance representations."""
    data_t, y_t, x_original, meta = generate_case_data(tc)

    needs_distance_matrix = any(
        method_id in DISTANCE_MATRIX_METHODS for method_id in selected_methods
    )
    needs_distance_condensed = any(m.startswith("kl") for m in selected_methods) or needs_distance_matrix

    distance_condensed = None
    distance_matrix = None
    precomputed_distance_condensed = meta.get("precomputed_distance_condensed")
    if precomputed_distance_condensed is not None and needs_distance_condensed:
        distance_condensed = np.asarray(precomputed_distance_condensed, dtype=float)

    precomputed_distance_matrix = meta.get("precomputed_distance_matrix")
    if precomputed_distance_matrix is not None and needs_distance_matrix:
        distance_matrix = np.asarray(precomputed_distance_matrix, dtype=float)
        np.fill_diagonal(distance_matrix, 0.0)

    if needs_distance_condensed and distance_condensed is None and distance_matrix is not None:
        distance_condensed = squareform(distance_matrix)

    if needs_distance_condensed and distance_condensed is None:
        distance_condensed = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
    if needs_distance_matrix and distance_matrix is None:
        if distance_condensed is None:
            distance_condensed = pdist(data_t.values, metric=config.TREE_DISTANCE_METRIC)
        distance_matrix = squareform(distance_condensed)

    return (
        data_t,
        y_t,
        x_original,
        meta,
        distance_condensed,
        distance_matrix,
        precomputed_distance_condensed,
    )


__all__ = ["prepare_case_inputs", "DISTANCE_MATRIX_METHODS"]
