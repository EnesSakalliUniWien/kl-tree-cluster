"""Core utility helpers for benchmark runners."""

from __future__ import annotations

import numpy as np


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Map non-negative labels to contiguous IDs while keeping noise at -1."""
    labels_arr = np.asarray(labels, dtype=int)
    unique = sorted({int(x) for x in labels_arr if x >= 0})
    mapping = {label: idx for idx, label in enumerate(unique)}
    return np.array([mapping.get(int(x), -1) for x in labels_arr], dtype=int)


def _estimate_dbscan_eps(distance_matrix: np.ndarray, min_samples: int) -> float:
    """Estimate eps via the median k-distance heuristic."""
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1:
        return 0.0
    k = min(max(int(min_samples), 1), n_samples - 1)
    kth = np.partition(distance_matrix, k, axis=1)[:, k]
    eps = float(np.median(kth))
    return eps if eps > 0 else 1e-9


def _resolve_n_neighbors(n_samples: int, n_neighbors: int | None) -> int:
    if n_samples <= 1:
        return 0
    if n_neighbors is None:
        return max(2, min(10, n_samples - 1))
    return max(1, min(int(n_neighbors), n_samples - 1))


def _knn_edge_weights(
    distance_matrix: np.ndarray, n_neighbors: int
) -> list[tuple[int, int, float]]:
    """Build undirected k-NN edge list with inverse-distance weights."""
    n_samples = distance_matrix.shape[0]
    if n_samples <= 1 or n_neighbors <= 0:
        return []

    edge_weights: dict[tuple[int, int], float] = {}
    for i in range(n_samples):
        neighbor_idx = np.argsort(distance_matrix[i])
        neighbors = neighbor_idx[1 : n_neighbors + 1]
        for j in neighbors:
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            dist = float(distance_matrix[i, j])
            weight = 1.0 / (1.0 + dist)
            if key not in edge_weights or weight > edge_weights[key]:
                edge_weights[key] = weight

    return [(i, j, w) for (i, j), w in edge_weights.items()]


__all__ = [
    "_normalize_labels",
    "_estimate_dbscan_eps",
    "_resolve_n_neighbors",
    "_knn_edge_weights",
]
