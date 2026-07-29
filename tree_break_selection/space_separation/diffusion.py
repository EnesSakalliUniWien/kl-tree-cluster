"""Diffusion-coordinate and distance methods for separated spaces."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist


@dataclass(frozen=True)
class DiffusionGeometry:
    """Coordinates, pairwise distances, and evidence from one diffusion fit."""

    coordinates: np.ndarray
    distance_condensed: np.ndarray
    metadata: dict[str, object]


PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS = 8
PYDIFFMAP_VARIABLE_BANDWIDTH_MIN_POSITIVE_NEIGHBORS = (
    PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS - 1
)


def _diffusion_geometry(
    coordinates: np.ndarray,
    *,
    metadata: dict[str, object],
) -> DiffusionGeometry:
    """Build a validated diffusion result through one deep interface."""
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Diffusion coordinates must be a non-empty two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Diffusion coordinates contain non-finite values.")
    distances = pdist(values, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("Diffusion distances contain non-finite values.")
    return DiffusionGeometry(
        coordinates=values,
        distance_condensed=distances,
        metadata=dict(metadata),
    )


def resolve_neighbor_search_k(n_samples: int, k_neighbors: int) -> int:
    """Choose valid sparse neighbor support for adaptive diffusion."""

    if n_samples <= 2:
        return 1
    return max(2, min(int(k_neighbors), n_samples - 1))


def _row_duplicate_counts(values: np.ndarray) -> tuple[int, int, int]:
    if len(values) == 0:
        return 0, 0, 0
    _unique_rows, counts = np.unique(values, axis=0, return_counts=True)
    unique_rows = int(len(counts))
    return unique_rows, int(len(values) - unique_rows), int(counts.max(initial=0))


def _resolve_pydiffmap_neighbor_search_k(
    values: np.ndarray,
    *,
    k_neighbors: int,
    bandwidth_type: str | float | None,
) -> int:
    neighbor_k = resolve_neighbor_search_k(len(values), k_neighbors)
    if bandwidth_type is None:
        return neighbor_k

    _unique_rows, _duplicate_rows, max_duplicate_count = _row_duplicate_counts(values)

    # pydiffmap's variable-bandwidth Kernel uses NNKDE(k=8), then builds
    # bandwidths from the 7 nearest nonzero distances in each sparse neighbor
    # row. Exact duplicate rows occupy zero-distance neighbor slots but are
    # removed by sparse nonzero extraction, so the search graph needs enough
    # extra neighbors to leave 7 positive distances after duplicates.
    min_positive_neighbors_for_nnkde = PYDIFFMAP_VARIABLE_BANDWIDTH_MIN_POSITIVE_NEIGHBORS
    if max_duplicate_count >= PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS:
        raise ValueError(
            "Adaptive pydiffmap variable-bandwidth diffusion cannot handle exact "
            "duplicate blocks at least as large as pydiffmap's internal NNKDE "
            f"query size ({PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS}); "
            f"max_duplicate_count={max_duplicate_count}. Such blocks produce zero "
            "local bandwidths and non-finite diffusion weights."
        )
    duplicate_aware_min_k = max_duplicate_count + min_positive_neighbors_for_nnkde
    max_valid_k = len(values)
    if duplicate_aware_min_k > max_valid_k:
        raise ValueError(
            "Adaptive pydiffmap variable-bandwidth diffusion needs at least "
            f"{min_positive_neighbors_for_nnkde} positive nonzero neighbors after "
            "exact duplicate rows are removed. "
            f"n_samples={len(values)}, max_duplicate_count={max_duplicate_count}, "
            f"required_k={duplicate_aware_min_k}, max_valid_k={max_valid_k}."
        )
    return max(neighbor_k, duplicate_aware_min_k)


def resolve_adaptive_epsilon(
    kernel_object: object,
    epsilon: str | float,
    *,
    metric: str,
) -> tuple[float, str]:
    """Resolve scalar epsilon after pydiffmap has fitted local bandwidths."""

    scaled_dists = getattr(kernel_object, "scaled_dists", None)
    if scaled_dists is None:
        raise ValueError("Adaptive diffusion kernel is missing scaled distance data.")
    scaled_sq = np.asarray(scaled_dists.data, dtype=float) ** 2
    return resolve_adaptive_epsilon_from_scaled_sq(scaled_sq, epsilon, metric=metric)


def resolve_adaptive_epsilon_from_scaled_sq(
    scaled_sq: np.ndarray,
    epsilon: str | float,
    *,
    metric: str,
) -> tuple[float, str]:
    """Resolve scalar epsilon from precomputed scaled squared distances."""

    if isinstance(epsilon, numbers.Real):
        return float(epsilon), "scalar"

    epsilon_text = str(epsilon).strip().lower()
    finite_positive = np.asarray(scaled_sq, dtype=float)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0)]
    if finite_positive.size == 0:
        raise ValueError("Adaptive diffusion epsilon requires positive finite scaled distances.")

    if epsilon_text == "median":
        return float(np.median(finite_positive)), "median"
    if epsilon_text == "mean":
        return float(np.mean(finite_positive)), "mean"
    if epsilon_text == "q75":
        return float(np.quantile(finite_positive, 0.75)), "q75"
    if epsilon_text == "bgh":
        if metric != "euclidean":
            raise ValueError(
                "Adaptive diffusion epsilon='bgh' is defined for Euclidean metrics; "
                f"got metric={metric!r}."
            )
        from pydiffmap import kernel as diffusion_kernel

        epsilon_value, _ = diffusion_kernel.choose_optimal_epsilon_BGH(finite_positive)
        return float(epsilon_value), "bgh"
    return float(epsilon_text), "scalar_text"


def compute_diffusion_coordinates(
    weights: np.ndarray,
    *,
    diffusion_time: int,
    n_components: int,
) -> np.ndarray:
    """Project a symmetric diffusion operator to diffusion coordinates."""

    matrix = np.asarray(weights, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square diffusion weight matrix.")
    if matrix.shape[0] < 2:
        raise ValueError("Diffusion coordinates require at least two samples.")
    if diffusion_time < 0:
        raise ValueError("diffusion_time must be non-negative.")
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    if not np.isfinite(matrix).all():
        raise ValueError("Diffusion weights contain non-finite values.")

    np.fill_diagonal(matrix, 0.0)
    matrix = np.maximum(matrix, matrix.T)
    degrees = matrix.sum(axis=1)
    degrees[degrees == 0] = 1e-10
    inv_sqrt_degrees = 1.0 / np.sqrt(degrees)
    transition = matrix * inv_sqrt_degrees[:, None] * inv_sqrt_degrees[None, :]

    n_comps = min(int(n_components), transition.shape[0] - 1)
    eigenvalues, eigenvectors = eigh(transition)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order][1 : n_comps + 1], 0.0)
    eigenvectors = eigenvectors[:, order][:, 1 : n_comps + 1]
    return eigenvectors * (eigenvalues[None, :] ** diffusion_time)


def hamming_knn_diffusion_geometry(
    data: pd.DataFrame | np.ndarray,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
) -> DiffusionGeometry:
    """Compute fixed-k Hamming-neighbor diffusion geometry.

    The input contract is a binary or one-hot sample-by-feature matrix. The
    method symmetrizes ``1 - Hamming`` neighbor similarities, computes the
    shared symmetric diffusion coordinates, and returns them with their
    condensed Euclidean distances.
    """

    from scipy.sparse import lil_matrix
    from sklearn.neighbors import NearestNeighbors

    values = (
        data.to_numpy(dtype=float)
        if isinstance(data, pd.DataFrame)
        else np.asarray(data, dtype=float)
    )
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Hamming diffusion requires at least two coordinate rows.")
    if not np.isfinite(values).all():
        raise ValueError("Hamming diffusion coordinates contain non-finite values.")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("Hamming diffusion requires binary or one-hot feature values.")
    if int(k_neighbors) < 1:
        raise ValueError("k_neighbors must be positive.")

    n_samples = len(values)
    neighbor_k = min(int(k_neighbors), n_samples - 1)
    neighbors = NearestNeighbors(n_neighbors=neighbor_k, metric="hamming")
    neighbors.fit(values)
    neighbor_distances, neighbor_indices = neighbors.kneighbors(values)

    weights = lil_matrix((n_samples, n_samples), dtype=float)
    for row_index in range(n_samples):
        for neighbor_position in range(neighbor_k):
            column_index = int(neighbor_indices[row_index, neighbor_position])
            similarity = max(
                1.0 - float(neighbor_distances[row_index, neighbor_position]),
                1e-10,
            )
            weights[row_index, column_index] = max(
                float(weights[row_index, column_index]), similarity
            )
            weights[column_index, row_index] = max(
                float(weights[column_index, row_index]), similarity
            )

    diffusion_coordinates = compute_diffusion_coordinates(
        weights.toarray(),
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    unique_rows, counts = np.unique(values, axis=0, return_counts=True)
    return _diffusion_geometry(
        diffusion_coordinates,
        metadata={
            "backend": "sklearn_hamming_knn",
            "metric": "hamming",
            "neighbor_search_k": int(neighbor_k),
            "diffusion_time": int(diffusion_time),
            "diffusion_components": int(diffusion_coordinates.shape[1]),
            "unique_rows": int(len(unique_rows)),
            "duplicate_rows": int(len(values) - len(unique_rows)),
            "max_duplicate_count": int(counts.max(initial=0)),
        },
    )


def adaptive_diffusion_geometry(
    data: pd.DataFrame | np.ndarray,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    bandwidth_type: str | float | None,
    epsilon: str | float,
) -> DiffusionGeometry:
    """Compute variable-bandwidth pydiffmap diffusion geometry."""

    from pydiffmap import kernel

    values = (
        data.to_numpy(dtype=float)
        if isinstance(data, pd.DataFrame)
        else np.asarray(data, dtype=float)
    )
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Adaptive diffusion requires at least two coordinate rows.")
    if not np.isfinite(values).all():
        raise ValueError("Adaptive diffusion coordinates contain non-finite values.")

    unique_rows, duplicate_rows, max_duplicate_count = _row_duplicate_counts(values)
    neighbor_k = _resolve_pydiffmap_neighbor_search_k(
        values,
        k_neighbors=k_neighbors,
        bandwidth_type=bandwidth_type,
    )
    kernel_object = kernel.Kernel(
        epsilon=1.0 if not isinstance(epsilon, numbers.Real) else float(epsilon),
        k=neighbor_k,
        metric=metric,
        neighbor_params={"n_jobs": -1},
        bandwidth_type=bandwidth_type,
    )
    kernel_object.fit(values)
    epsilon_value, epsilon_method = resolve_adaptive_epsilon(
        kernel_object,
        epsilon,
        metric=metric,
    )
    kernel_object.epsilon_fitted = epsilon_value
    adaptive_kernel = kernel_object.compute()
    weights = (
        adaptive_kernel.toarray()
        if hasattr(adaptive_kernel, "toarray")
        else np.asarray(adaptive_kernel, dtype=float)
    )
    coordinates = compute_diffusion_coordinates(
        weights,
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    return _diffusion_geometry(
        coordinates,
        metadata={
            "backend": "pydiffmap",
            "metric": metric,
            "requested_neighbor_search_k": int(k_neighbors),
            "neighbor_search_k": int(neighbor_k),
            "bandwidth_type": (
                bandwidth_type
                if bandwidth_type is None or isinstance(bandwidth_type, str)
                else float(bandwidth_type)
            ),
            "epsilon": float(epsilon_value),
            "epsilon_method": epsilon_method,
            "diffusion_time": int(diffusion_time),
            "diffusion_components": int(coordinates.shape[1]),
            "unique_rows": int(unique_rows),
            "duplicate_rows": int(duplicate_rows),
            "max_duplicate_count": int(max_duplicate_count),
        },
    )


def block_diffusion_geometry(
    coordinates: np.ndarray,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
) -> DiffusionGeometry:
    """Compute fixed k-NN Gaussian diffusion geometry in one spectral block."""

    from scipy.sparse import lil_matrix
    from sklearn.neighbors import NearestNeighbors

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Block diffusion requires at least three coordinate rows.")
    if not np.isfinite(values).all():
        raise ValueError("Block coordinates contain non-finite values.")

    n_samples = values.shape[0]
    neighbor_count = min(max(2, int(k_neighbors) + 1), n_samples)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    neighbors.fit(values)
    neighbor_distances, neighbor_indices = neighbors.kneighbors(values)

    positive_sq = neighbor_distances[neighbor_distances > 0.0] ** 2
    if positive_sq.size == 0:
        raise ValueError("Degenerate block coordinates for diffusion.")
    epsilon_value = float(np.median(positive_sq))
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("Could not resolve a positive block diffusion epsilon.")

    weights = lil_matrix((n_samples, n_samples), dtype=float)
    for row_index in range(n_samples):
        for distance, column_index in zip(
            neighbor_distances[row_index],
            neighbor_indices[row_index],
            strict=False,
        ):
            if int(column_index) == row_index:
                continue
            similarity = max(
                math.exp(-float(distance * distance) / epsilon_value),
                1e-12,
            )
            weights[row_index, int(column_index)] = max(
                float(weights[row_index, int(column_index)]), similarity
            )
            weights[int(column_index), row_index] = max(
                float(weights[int(column_index), row_index]), similarity
            )

    diffusion_coordinates = compute_diffusion_coordinates(
        weights.toarray(),
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    geometry = _diffusion_geometry(
        diffusion_coordinates,
        metadata={
            "kernel": "knn_gaussian",
            "metric": "euclidean",
            "neighbor_search_k": int(neighbor_count - 1),
            "epsilon": epsilon_value,
            "diffusion_time": int(diffusion_time),
            "diffusion_components": int(diffusion_coordinates.shape[1]),
        },
    )
    if np.allclose(geometry.distance_condensed, 0.0):
        raise ValueError("Degenerate diffusion distances for block.")
    return geometry


def block_adaptive_diffusion_geometry(
    coordinates: np.ndarray,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    bandwidth_type: str | float | None,
    epsilon: str | float,
) -> DiffusionGeometry:
    """Compute adaptive diffusion geometry in one spectral block."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Adaptive block diffusion requires at least three coordinate rows.")
    if not np.isfinite(values).all():
        raise ValueError("Block coordinates contain non-finite values.")

    geometry = adaptive_diffusion_geometry(
        values,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
        metric=metric,
        bandwidth_type=bandwidth_type,
        epsilon=epsilon,
    )
    distances = np.asarray(geometry.distance_condensed, dtype=float)
    if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
        raise ValueError("Degenerate adaptive diffusion distances for block.")
    return DiffusionGeometry(
        coordinates=geometry.coordinates,
        distance_condensed=geometry.distance_condensed,
        metadata={
            "kernel": "pydiffmap_adaptive",
            **geometry.metadata,
        },
    )
