"""Diffusion-coordinate and distance methods for separated spaces."""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist


def resolve_neighbor_search_k(n_samples: int, k_neighbors: int) -> int:
    """Choose valid sparse neighbor support for adaptive diffusion."""

    if n_samples <= 2:
        return 1
    return max(2, min(int(k_neighbors), n_samples - 1))


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
    finite_positive = finite_positive[
        np.isfinite(finite_positive) & (finite_positive > 0)
    ]
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


def hamming_knn_diffusion_distance(
    data: pd.DataFrame | np.ndarray,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
) -> np.ndarray:
    """Compute fixed-k Hamming-neighbor diffusion distance.

    The input contract is a binary or one-hot sample-by-feature matrix. The
    method symmetrizes ``1 - Hamming`` neighbor similarities, computes the
    shared symmetric diffusion coordinates, and returns their condensed
    Euclidean distance vector.
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
    return pdist(diffusion_coordinates, metric="euclidean")


def adaptive_diffusion_distance(
    data: pd.DataFrame | np.ndarray,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    bandwidth_type: str | float | None,
    epsilon: str | float,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute a variable-bandwidth pydiffmap diffusion distance."""

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

    neighbor_k = resolve_neighbor_search_k(len(values), k_neighbors)
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
    distances = pdist(coordinates, metric="euclidean")
    if not return_metadata:
        return distances

    metadata = {
        "backend": "pydiffmap",
        "metric": metric,
        "neighbor_search_k": int(neighbor_k),
        "bandwidth_type": (
            bandwidth_type
            if bandwidth_type is None or isinstance(bandwidth_type, str)
            else float(bandwidth_type)
        ),
        "epsilon": float(epsilon_value),
        "epsilon_method": epsilon_method,
    }
    return distances, metadata


def block_diffusion_distance(
    coordinates: np.ndarray,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute fixed k-NN Gaussian diffusion distance in one spectral block."""

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
    distances = pdist(diffusion_coordinates, metric="euclidean")
    if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
        raise ValueError("Degenerate diffusion distances for block.")
    return distances, {
        "kernel": "knn_gaussian",
        "metric": "euclidean",
        "neighbor_search_k": int(neighbor_count - 1),
        "epsilon": epsilon_value,
        "diffusion_time": int(diffusion_time),
        "diffusion_components": int(min(n_components, n_samples - 1)),
    }


def block_adaptive_diffusion_distance(
    coordinates: np.ndarray,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    bandwidth_type: str | float | None,
    epsilon: str | float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute adaptive diffusion distance in one spectral block."""

    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Adaptive block diffusion requires at least three coordinate rows.")
    if not np.isfinite(values).all():
        raise ValueError("Block coordinates contain non-finite values.")

    distances, metadata = adaptive_diffusion_distance(
        values,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
        metric=metric,
        bandwidth_type=bandwidth_type,
        epsilon=epsilon,
        return_metadata=True,
    )
    distances = np.asarray(distances, dtype=float)
    if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
        raise ValueError("Degenerate adaptive diffusion distances for block.")
    return distances, {
        "kernel": "pydiffmap_adaptive",
        **metadata,
        "diffusion_time": int(diffusion_time),
        "diffusion_components": int(min(n_components, values.shape[0] - 1)),
    }
