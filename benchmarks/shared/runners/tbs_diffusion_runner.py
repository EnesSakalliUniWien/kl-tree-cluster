"""TBS method runner using diffusion distance + average linkage tree construction.

Builds a k-NN similarity graph, computes diffusion coordinates via the
transition-matrix eigendecomposition, then applies standard average linkage
on the Euclidean diffusion distance. The resulting tree is passed through
the normal TBS decomposition pipeline.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
)
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
)

from benchmarks.shared.runners.tbs_runner import _run_tbs_on_distance
from benchmarks.shared.types import MethodRunResult

GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_CONNECTIVITY_MINIMUM = "connectivity_minimum"
GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD = "fragmentation_guard"
GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FIXED = "fixed"
DEFAULT_GRAPHTOOLS_ADAPTIVE_NEIGHBOR_GRID = (5, 10, 15, 25, 40, 80, 160)


def _resolve_neighbor_search_k(
    n_samples: int,
    k_neighbors: int,
) -> int:
    """Choose the sparse neighbor-search support for adaptive diffusion."""
    if n_samples <= 2:
        return 1
    return max(2, min(int(k_neighbors), n_samples - 1))


def _normalize_graphtools_adaptive_neighbor_profile(profile: str | None) -> str:
    """Normalize the graphtools adaptive-neighbor profile name."""
    if profile is None:
        return GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FIXED

    normalized = str(profile).strip().lower()
    if normalized in {"", "none", "fixed", "fixed_k"}:
        return GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FIXED
    if normalized in {
        GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_CONNECTIVITY_MINIMUM,
        GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD,
    }:
        return normalized

    raise ValueError(f"Unknown graphtools adaptive neighbor profile: {profile!r}")


def _normalize_neighbor_k_candidates(
    *,
    n_samples: int,
    requested_k: int,
    adaptive_neighbor_grid: Sequence[int] | None,
    minimum_k: int | None = None,
) -> tuple[int, ...]:
    """Return valid unique K candidates for adaptive graphtools neighbor search."""
    if n_samples <= 2:
        return (1,)

    raw_candidates = list(
        DEFAULT_GRAPHTOOLS_ADAPTIVE_NEIGHBOR_GRID
        if adaptive_neighbor_grid is None
        else adaptive_neighbor_grid
    )
    raw_candidates.append(requested_k)
    if minimum_k is not None:
        raw_candidates.append(minimum_k)

    candidates: set[int] = set()
    for candidate in raw_candidates:
        candidate_k = int(candidate)
        if candidate_k <= 0:
            continue
        if minimum_k is not None:
            candidate_k = max(candidate_k, minimum_k)
        candidates.add(_resolve_neighbor_search_k(n_samples, candidate_k))

    if not candidates:
        candidates.add(_resolve_neighbor_search_k(n_samples, requested_k))

    return tuple(sorted(candidates))


def _row_duplicate_stats(X: np.ndarray) -> dict[str, int]:
    """Summarize exact duplicate rows for duplicate-aware kNN support selection."""
    if len(X) == 0:
        return {
            "unique_rows": 0,
            "duplicate_rows": 0,
            "max_duplicate_count": 0,
        }

    _, counts = np.unique(X, axis=0, return_counts=True)
    unique_rows = int(len(counts))
    return {
        "unique_rows": unique_rows,
        "duplicate_rows": int(len(X) - unique_rows),
        "max_duplicate_count": int(counts.max(initial=0)),
    }


def _graphtools_adaptive_fragmentation_thresholds(n_samples: int) -> tuple[int, int]:
    """Return unsupervised component-count and size guardrails for adaptive K."""
    max_components = max(12, min(32, int(np.ceil(np.sqrt(max(n_samples, 1))))))
    min_component_size = max(3, int(np.ceil(0.01 * max(n_samples, 1))))
    return max_components, min_component_size


def _knn_component_summary(
    X: np.ndarray,
    *,
    candidate_k: int,
    metric: str,
) -> tuple[int, int]:
    """Return component count and smallest component size for a symmetrized kNN graph."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    connectivity_graph = kneighbors_graph(
        X,
        n_neighbors=candidate_k,
        mode="connectivity",
        metric=metric,
        include_self=False,
        n_jobs=-1,
    )
    connectivity_graph = connectivity_graph.maximum(connectivity_graph.T)
    component_count, labels = connected_components(
        connectivity_graph,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels)
    min_component_size = int(component_sizes.min()) if component_sizes.size else 0
    return int(component_count), min_component_size


def _resolve_graphtools_neighbor_search_k(
    X: np.ndarray,
    *,
    k_neighbors: int,
    metric: str,
    adaptive_neighbor_profile: str | None = None,
    adaptive_neighbor_grid: Sequence[int] | None = None,
) -> tuple[int, dict[str, object]]:
    """Choose graphtools kNN support and explain the decision."""
    n_samples = len(X)
    requested_k = _resolve_neighbor_search_k(n_samples, k_neighbors)
    profile = _normalize_graphtools_adaptive_neighbor_profile(adaptive_neighbor_profile)
    duplicate_stats = _row_duplicate_stats(X)
    duplicate_aware_min_k = None
    if n_samples > 2 and duplicate_stats["unique_rows"] > 1:
        duplicate_aware_min_k = min(
            n_samples - 1,
            max(2, duplicate_stats["max_duplicate_count"]),
        )

    metadata: dict[str, object] = {
        "adaptive_neighbor_profile": profile,
        "adaptive_neighbor_requested_k": int(requested_k),
        "adaptive_neighbor_user_k": int(k_neighbors),
        "adaptive_neighbor_grid": (
            None
            if adaptive_neighbor_grid is None
            else [int(candidate) for candidate in adaptive_neighbor_grid]
        ),
        "adaptive_neighbor_unique_rows": int(duplicate_stats["unique_rows"]),
        "adaptive_neighbor_duplicate_rows": int(duplicate_stats["duplicate_rows"]),
        "adaptive_neighbor_max_duplicate_count": int(duplicate_stats["max_duplicate_count"]),
        "adaptive_neighbor_duplicate_aware_min_k": (
            None if duplicate_aware_min_k is None else int(duplicate_aware_min_k)
        ),
    }

    if profile == GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FIXED:
        metadata.update(
            {
                "adaptive_neighbor_selected_k": int(requested_k),
                "adaptive_neighbor_status": "fixed",
                "adaptive_neighbor_component_counts": {},
                "adaptive_neighbor_component_min_sizes": {},
            }
        )
        return requested_k, metadata

    if n_samples <= 2:
        metadata.update(
            {
                "adaptive_neighbor_selected_k": 1,
                "adaptive_neighbor_status": "small_sample",
                "adaptive_neighbor_component_counts": {},
                "adaptive_neighbor_component_min_sizes": {},
            }
        )
        return 1, metadata

    minimum_k = duplicate_aware_min_k
    if profile == GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD:
        minimum_k = max(requested_k, 1 if duplicate_aware_min_k is None else duplicate_aware_min_k)

    candidates = _normalize_neighbor_k_candidates(
        n_samples=n_samples,
        requested_k=requested_k,
        adaptive_neighbor_grid=adaptive_neighbor_grid,
        minimum_k=minimum_k,
    )
    component_counts: dict[str, int] = {}
    component_min_sizes: dict[str, int] = {}
    selected_k = candidates[-1]
    status = "max_candidate_disconnected"
    max_components, min_component_size = _graphtools_adaptive_fragmentation_thresholds(n_samples)
    for candidate_k in candidates:
        component_count, component_min_size = _knn_component_summary(
            X,
            candidate_k=candidate_k,
            metric=metric,
        )
        component_counts[str(candidate_k)] = component_count
        component_min_sizes[str(candidate_k)] = component_min_size
        if component_count == 1:
            selected_k = candidate_k
            status = "connected"
            break
        if (
            profile == GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD
            and component_count <= max_components
            and component_min_size >= min_component_size
        ):
            selected_k = candidate_k
            status = "stable_components"
            break

    if (
        profile == GRAPHTOOLS_ADAPTIVE_NEIGHBOR_PROFILE_FRAGMENTATION_GUARD
        and status == "max_candidate_disconnected"
    ):
        status = "max_candidate_fragmented"
    metadata.update(
        {
            "adaptive_neighbor_candidates": [int(candidate) for candidate in candidates],
            "adaptive_neighbor_selected_k": int(selected_k),
            "adaptive_neighbor_status": status,
            "adaptive_neighbor_component_counts": component_counts,
            "adaptive_neighbor_component_min_sizes": component_min_sizes,
            "adaptive_neighbor_fragmentation_max_components": int(max_components),
            "adaptive_neighbor_fragmentation_min_component_size": int(min_component_size),
        }
    )
    return selected_k, metadata


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
    transition_sym = weights * inv_sqrt_degrees[:, None] * inv_sqrt_degrees[None, :]

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
            "tbs_diffusion uses Hamming diffusion and requires binary or one-hot feature values."
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
    weights = (
        adaptive_kernel.toarray()
        if hasattr(adaptive_kernel, "toarray")
        else np.asarray(
            adaptive_kernel,
            dtype=float,
        )
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
            bandwidth_type
            if bandwidth_type is None or isinstance(bandwidth_type, str)
            else float(bandwidth_type)
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
    adaptive_neighbor_profile: str | None = None,
    adaptive_neighbor_grid: Sequence[int] | None = None,
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
    neighbor_k, neighbor_metadata = _resolve_graphtools_neighbor_search_k(
        X,
        k_neighbors=k_neighbors,
        metric=metric,
        adaptive_neighbor_profile=adaptive_neighbor_profile,
        adaptive_neighbor_grid=adaptive_neighbor_grid,
    )

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
    weights = (
        kernel_matrix.toarray()
        if hasattr(kernel_matrix, "toarray")
        else np.asarray(
            kernel_matrix,
            dtype=float,
        )
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
    metadata.update(neighbor_metadata)
    return distance_condensed, metadata


def _run_tbs_diffusion_method(
    data_df: pd.DataFrame,
    sibling_significance_level: float,
    k_neighbors: int,
    diffusion_time: int,
    *,
    tree_linkage_method: str = "average",
    feature_space: FeatureSpace | None = None,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
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
        tree_linkage_method=tree_linkage_method,
        feature_space=feature_space,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=(branch_length_optimization_pair_sample_size),
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=(branch_length_optimization_solver_tolerance),
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
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
    adaptive_neighbor_profile: str | None = None,
    adaptive_neighbor_grid: Sequence[int] | None = None,
    tree_builder: str = "linkage",
    tree_rooting: str = "linkage_root",
    tree_linkage_method: str = "average",
    feature_space: FeatureSpace | None = None,
    graph_data_df: pd.DataFrame | None = None,
    branch_length_data_df: pd.DataFrame | None = None,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
) -> MethodRunResult:
    """Run TBS decomposition on a graphtools kernel diffusion tree."""
    graph_data = data_df if graph_data_df is None else graph_data_df
    if not graph_data.index.equals(data_df.index):
        raise ValueError("graph_data_df index must exactly match the original data index.")
    diff_dist, graph_metadata = _build_graphtools_diffusion_distance(
        graph_data,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
        n_components=n_components,
        metric=metric,
        decay=decay,
        anisotropy=anisotropy,
        kernel_symm=kernel_symm,
        random_state=random_state,
        adaptive_neighbor_profile=adaptive_neighbor_profile,
        adaptive_neighbor_grid=adaptive_neighbor_grid,
        return_metadata=True,
    )

    return _run_tbs_on_distance(
        data_df,
        diff_dist,
        sibling_significance_level,
        tree_builder=tree_builder,
        tree_rooting=tree_rooting,
        tree_linkage_method=tree_linkage_method,
        feature_space=feature_space,
        branch_length_data_df=branch_length_data_df,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=(branch_length_optimization_pair_sample_size),
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=(branch_length_optimization_solver_tolerance),
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
        extra={
            "diffusion_method": "graphtools_kernel_diffusion",
            "graph_geometry_source": (
                "original_data" if graph_data_df is None else "aligned_geometry_embedding"
            ),
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
    tree_linkage_method: str = "average",
    feature_space: FeatureSpace | None = None,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
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
        tree_linkage_method=tree_linkage_method,
        feature_space=feature_space,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=(branch_length_optimization_pair_sample_size),
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=(branch_length_optimization_solver_tolerance),
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
        extra={
            "diffusion_method": "adaptive_pydiffmap_diffusion",
            "adaptive_diffusion": adaptive_metadata,
        },
    )
