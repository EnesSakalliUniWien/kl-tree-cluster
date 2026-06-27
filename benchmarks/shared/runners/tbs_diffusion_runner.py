"""TBS method runner using diffusion distance + average linkage tree construction.

Builds a k-NN similarity graph, computes diffusion coordinates via the
transition-matrix eigendecomposition, then applies standard average linkage
on the Euclidean diffusion distance. The resulting tree is passed through
the normal TBS decomposition pipeline.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
)

from benchmarks.shared.runners.tbs_runner import _run_tbs_on_distance
from benchmarks.shared.types import MethodRunResult


def _resolve_neighbor_search_k(
    n_samples: int,
    k_neighbors: int,
) -> int:
    """Choose the sparse neighbor-search support for adaptive diffusion."""
    if n_samples <= 2:
        return 1
    return max(2, min(int(k_neighbors), n_samples - 1))


def _resolve_adaptive_epsilon(
    kernel_object,
    epsilon: str | float,
    *,
    metric: str,
) -> tuple[float, str]:
    """Resolve a scalar epsilon after pydiffmap has fitted local bandwidths."""
    if not hasattr(kernel_object, "scaled_dists") or kernel_object.scaled_dists is None:
        raise ValueError("Adaptive diffusion kernel is missing scaled distance data.")

    scaled_sq = np.asarray(kernel_object.scaled_dists.data, dtype=float) ** 2
    return _resolve_adaptive_epsilon_from_scaled_sq(
        scaled_sq,
        epsilon,
        metric=metric,
    )


def _resolve_adaptive_epsilon_from_scaled_sq(
    scaled_sq: np.ndarray,
    epsilon: str | float,
    *,
    metric: str,
) -> tuple[float, str]:
    """Resolve a scalar epsilon from precomputed scaled squared distances."""
    if isinstance(epsilon, numbers.Real):
        return float(epsilon), "scalar"

    epsilon_text = str(epsilon).strip().lower()
    scaled_sq = np.asarray(scaled_sq, dtype=float)
    scaled_sq = scaled_sq[np.isfinite(scaled_sq) & (scaled_sq > 0)]
    if scaled_sq.size == 0:
        raise ValueError("Adaptive diffusion epsilon requires positive finite scaled distances.")

    if epsilon_text == "median":
        return float(np.median(scaled_sq)), "median"
    if epsilon_text == "mean":
        return float(np.mean(scaled_sq)), "mean"
    if epsilon_text == "q75":
        return float(np.quantile(scaled_sq, 0.75)), "q75"
    if epsilon_text == "bgh":
        if metric != "euclidean":
            raise ValueError(
                "Adaptive diffusion epsilon='bgh' is defined for Euclidean metrics; "
                f"got metric={metric!r}."
            )
        from pydiffmap import kernel as diffusion_kernel

        epsilon_value, _ = diffusion_kernel.choose_optimal_epsilon_BGH(scaled_sq)
        return float(epsilon_value), "bgh"

    return float(epsilon_text), "scalar_text"


def _compute_diffusion_coordinates(
    weights: np.ndarray,
    diffusion_time: int,
    n_components: int,
) -> np.ndarray:
    """Project a symmetric diffusion operator to diffusion coordinates."""
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Expected a square diffusion weight matrix.")

    weights = np.asarray(weights, dtype=float)
    np.fill_diagonal(weights, 0.0)
    weights = np.maximum(weights, weights.T)

    degrees = weights.sum(axis=1)
    degrees[degrees == 0] = 1e-10
    inv_sqrt_degrees = 1.0 / np.sqrt(degrees)
    transition_sym = (
        weights
        * inv_sqrt_degrees[:, None]
        * inv_sqrt_degrees[None, :]
    )

    n_samples = transition_sym.shape[0]
    n_comps = min(int(n_components), n_samples - 1)
    eigenvalues, eigenvectors = eigh(transition_sym)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][1 : n_comps + 1]
    eigenvectors = eigenvectors[:, order][:, 1 : n_comps + 1]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return eigenvectors * (eigenvalues[None, :] ** diffusion_time)


def _require_hamming_diffusion_input(
    data_df: pd.DataFrame,
    feature_space: FeatureSpace | None,
) -> None:
    """Require the binary/one-hot matrix contract used by Hamming diffusion."""
    if feature_space is not None and feature_space.has_continuous_blocks:
        raise ValueError(
            "tbs_diffusion uses Hamming diffusion and requires binary or one-hot "
            "feature matrices; continuous FeatureSpace inputs are unsupported."
        )

    values = data_df.values
    if not np.isin(values, (0, 1)).all():
        raise ValueError(
            "tbs_diffusion uses Hamming diffusion and requires binary or one-hot "
            "feature values."
        )


def _build_diffusion_distance(
    data_df: pd.DataFrame,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
) -> np.ndarray:
    """Compute condensed diffusion distance from binary feature matrix."""
    from scipy.sparse import lil_matrix
    from sklearn.neighbors import NearestNeighbors

    X = data_df.values.astype(float)
    n = len(X)
    k = min(k_neighbors, n - 1)

    # k-NN graph with Hamming distance
    nn = NearestNeighbors(n_neighbors=k, metric="hamming")
    nn.fit(X)
    knn_dist, knn_idx = nn.kneighbors(X)

    # Symmetrized similarity matrix: sim = 1 - hamming
    W = lil_matrix((n, n), dtype=float)
    for i in range(n):
        for j_pos in range(k):
            j = knn_idx[i, j_pos]
            sim = max(1.0 - knn_dist[i, j_pos], 1e-10)
            W[i, j] = max(W[i, j], sim)
            W[j, i] = max(W[j, i], sim)

    W = W.toarray()
    np.fill_diagonal(W, 0)

    # Symmetric normalization: D^{-1/2} W D^{-1/2}
    diffusion_coords = _compute_diffusion_coordinates(
        W,
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    return pdist(diffusion_coords, metric="euclidean")


def _build_adaptive_diffusion_distance(
    data_df: pd.DataFrame,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    bandwidth_type: str | float | None,
    epsilon: str | float,
    return_metadata: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute diffusion distance with a variable-bandwidth kernel from pydiffmap."""
    from pydiffmap import kernel

    X = data_df.values.astype(float)
    n_samples = len(X)
    neighbor_k = _resolve_neighbor_search_k(n_samples, k_neighbors)

    kernel_object = kernel.Kernel(
        epsilon=1.0 if not isinstance(epsilon, numbers.Real) else float(epsilon),
        k=neighbor_k,
        metric=metric,
        neighbor_params={"n_jobs": -1},
        bandwidth_type=bandwidth_type,
    )
    kernel_object.fit(X)
    epsilon_value, epsilon_method = _resolve_adaptive_epsilon(
        kernel_object,
        epsilon,
        metric=metric,
    )
    kernel_object.epsilon_fitted = epsilon_value

    adaptive_kernel = kernel_object.compute()
    weights = adaptive_kernel.toarray() if hasattr(adaptive_kernel, "toarray") else np.asarray(
        adaptive_kernel,
        dtype=float,
    )
    diffusion_coords = _compute_diffusion_coordinates(
        weights,
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    distance_condensed = pdist(diffusion_coords, metric="euclidean")

    if not return_metadata:
        return distance_condensed

    metadata = {
        "backend": "pydiffmap",
        "metric": metric,
        "neighbor_search_k": int(neighbor_k),
        "bandwidth_type": (
            bandwidth_type if bandwidth_type is None or isinstance(bandwidth_type, str) else float(bandwidth_type)
        ),
        "epsilon": float(epsilon_value),
        "epsilon_method": epsilon_method,
    }
    return distance_condensed, metadata


def _build_graphtools_diffusion_distance(
    data_df: pd.DataFrame,
    *,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    decay: int | None,
    anisotropy: float,
    kernel_symm: str,
    random_state: int,
    return_metadata: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute diffusion distance from a graphtools kernel graph."""
    try:
        import graphtools
    except ImportError as exc:
        raise ImportError(
            "tbs_diffusion_graphtools requires the optional GPL dependency "
            "`graphtools`. Install with `uv sync --extra experimental-gpl`."
        ) from exc

    X = data_df.values.astype(float)
    n_samples = len(X)
    neighbor_k = _resolve_neighbor_search_k(n_samples, k_neighbors)

    graph = graphtools.Graph(
        X,
        n_pca=None,
        knn=neighbor_k,
        decay=decay,
        distance=metric,
        anisotropy=float(anisotropy),
        kernel_symm=kernel_symm,
        random_state=int(random_state),
        verbose=False,
    )
    kernel_matrix = graph.kernel
    weights = kernel_matrix.toarray() if hasattr(kernel_matrix, "toarray") else np.asarray(
        kernel_matrix,
        dtype=float,
    )
    diffusion_coords = _compute_diffusion_coordinates(
        weights,
        diffusion_time=diffusion_time,
        n_components=n_components,
    )
    distance_condensed = pdist(diffusion_coords, metric="euclidean")

    if not return_metadata:
        return distance_condensed

    metadata = {
        "backend": "graphtools",
        "graphtools_version": str(getattr(graphtools, "__version__", "unknown")),
        "metric": metric,
        "neighbor_search_k": int(neighbor_k),
        "decay": None if decay is None else int(decay),
        "anisotropy": float(anisotropy),
        "kernel_symm": str(kernel_symm),
        "random_state": int(random_state),
        "graph_class": graph.__class__.__name__,
        "kernel_nonzero_entries": int(getattr(kernel_matrix, "nnz", np.count_nonzero(weights))),
    }
    return distance_condensed, metadata


def _run_tbs_diffusion_method(
    data_df: pd.DataFrame,
    sibling_significance_level: float,
    k_neighbors: int,
    diffusion_time: int,
    *,
    feature_space: FeatureSpace | None = None,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
) -> MethodRunResult:
    """Run TBS decomposition on a Hamming nearest-neighbor diffusion tree."""
    _require_hamming_diffusion_input(data_df, feature_space)

    diff_dist = _build_diffusion_distance(
        data_df,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=30,
    )

    return _run_tbs_on_distance(
        data_df,
        diff_dist,
        sibling_significance_level,
        tree_linkage_method="average",
        feature_space=feature_space,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=(
            branch_length_optimization_pair_sample_size
        ),
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=(
            branch_length_optimization_solver_tolerance
        ),
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        extra={"diffusion_method": "hamming_nn_diffusion"},
    )


def _run_tbs_diffusion_graphtools_method(
    data_df: pd.DataFrame,
    sibling_significance_level: float,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    decay: int | None,
    anisotropy: float,
    kernel_symm: str,
    random_state: int,
    *,
    feature_space: FeatureSpace | None = None,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
) -> MethodRunResult:
    """Run TBS decomposition on a graphtools kernel diffusion tree."""
    diff_dist, graph_metadata = _build_graphtools_diffusion_distance(
        data_df,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
        metric=metric,
        decay=decay,
        anisotropy=anisotropy,
        kernel_symm=kernel_symm,
        random_state=random_state,
        return_metadata=True,
    )

    return _run_tbs_on_distance(
        data_df,
        diff_dist,
        sibling_significance_level,
        tree_linkage_method="average",
        feature_space=feature_space,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=(
            branch_length_optimization_pair_sample_size
        ),
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=(
            branch_length_optimization_solver_tolerance
        ),
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        extra={
            "diffusion_method": "graphtools_kernel_diffusion",
            "graphtools_diffusion": graph_metadata,
        },
    )


def _run_tbs_diffusion_adaptive_method(
    data_df: pd.DataFrame,
    sibling_significance_level: float,
    k_neighbors: int,
    diffusion_time: int,
    n_components: int,
    metric: str,
    bandwidth_type: str | float | None,
    epsilon: str | float,
    *,
    feature_space: FeatureSpace | None = None,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
) -> MethodRunResult:
    """Run TBS decomposition on an adaptive variable-bandwidth diffusion tree."""
    diff_dist, adaptive_metadata = _build_adaptive_diffusion_distance(
        data_df,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
        metric=metric,
        bandwidth_type=bandwidth_type,
        epsilon=epsilon,
        return_metadata=True,
    )

    return _run_tbs_on_distance(
        data_df,
        diff_dist,
        sibling_significance_level,
        tree_linkage_method="average",
        feature_space=feature_space,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=(
            branch_length_optimization_pair_sample_size
        ),
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=(
            branch_length_optimization_solver_tolerance
        ),
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        extra={
            "diffusion_method": "adaptive_pydiffmap_diffusion",
            "adaptive_diffusion": adaptive_metadata,
        },
    )
