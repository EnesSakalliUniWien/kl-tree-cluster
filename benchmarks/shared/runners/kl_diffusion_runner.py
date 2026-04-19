"""KL method runner using diffusion distance + average linkage tree construction.

Builds a k-NN similarity graph, computes diffusion coordinates via the
transition-matrix eigendecomposition, then applies standard average linkage
on the Euclidean diffusion distance. The resulting tree is passed through
the normal KL decomposition pipeline.
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.linalg import eigh
from scipy.spatial.distance import pdist

from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import (
    _create_report_dataframe,
    _labels_from_decomposition,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _resolve_neighbor_search_k(
    n_samples: int,
    k_neighbors: int | None,
) -> int:
    """Choose the sparse neighbor-search support for adaptive diffusion.

    Adaptive variable-bandwidth kernels still need a sparse neighbor graph for
    local KDE and matrix construction, but this is no longer the global graph
    scale parameter. When unset, use a mild ``O(log n)`` heuristic.
    """
    if n_samples <= 2:
        return 1
    if k_neighbors is not None:
        return max(2, min(int(k_neighbors), n_samples - 1))

    auto_k = int(math.ceil(math.log2(max(n_samples, 4))) * 2)
    return max(10, min(auto_k, n_samples - 1))


def _resolve_adaptive_epsilon(
    kernel_object,
    epsilon: str | float,
    *,
    metric: str,
) -> tuple[float, str]:
    """Resolve a scalar epsilon after pydiffmap has fitted local bandwidths."""
    if isinstance(epsilon, numbers.Real):
        return float(epsilon), "scalar"

    epsilon_text = str(epsilon).strip().lower()
    if not hasattr(kernel_object, "scaled_dists") or kernel_object.scaled_dists is None:
        raise ValueError("Adaptive diffusion kernel is missing scaled distance data.")

    scaled_sq = np.asarray(kernel_object.scaled_dists.data, dtype=float) ** 2
    scaled_sq = scaled_sq[np.isfinite(scaled_sq) & (scaled_sq > 0)]
    if scaled_sq.size == 0:
        return 1.0, "fallback_constant"

    if epsilon_text == "median":
        return float(np.median(scaled_sq)), "median"
    if epsilon_text == "mean":
        return float(np.mean(scaled_sq)), "mean"
    if epsilon_text == "q75":
        return float(np.quantile(scaled_sq, 0.75)), "q75"
    if epsilon_text == "bgh":
        if metric != "euclidean":
            warnings.warn(
                "Adaptive diffusion epsilon='bgh' is derived for Euclidean metrics; "
                "falling back to the median scaled distance for metric=%r." % metric,
                stacklevel=2,
            )
            return float(np.median(scaled_sq)), "median_fallback_from_bgh"
        from pydiffmap import kernel as diffusion_kernel

        epsilon_value, _ = diffusion_kernel.choose_optimal_epsilon_BGH(scaled_sq)
        return float(epsilon_value), "bgh"

    try:
        return float(epsilon_text), "scalar_text"
    except ValueError as exc:  # pragma: no cover - defensive input validation
        raise ValueError(
            "Unrecognized adaptive epsilon specification %r. "
            "Use a float or one of {'median', 'mean', 'q75', 'bgh'}." % epsilon
        ) from exc


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


def _build_diffusion_distance(
    data_df: pd.DataFrame,
    k_neighbors: int = 15,
    diffusion_time: int = 3,
    n_components: int = 30,
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
    k_neighbors: int | None = None,
    diffusion_time: int = 3,
    n_components: int = 30,
    metric: str = "hamming",
    bandwidth_type: str | float | None = "-1/(d+2)",
    epsilon: str | float = "median",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Compute diffusion distance with a variable-bandwidth kernel from pydiffmap."""
    try:
        from pydiffmap import kernel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Adaptive diffusion requires the 'pydiffmap' package."
        ) from exc

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


def _run_kl_diffusion_method(
    data_df: pd.DataFrame,
    significance_level: float,
    k_neighbors: int = 15,
    diffusion_time: int = 3,
) -> MethodRunResult:
    """Run KL decomposition on a diffusion-distance HAC tree."""

    diff_dist = _build_diffusion_distance(
        data_df,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
    )

    Z_t = linkage(diff_dist, method="average")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_df.index.tolist())

    decomp_t = tree_t.decompose(
        leaf_data=data_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )

    report_t = _create_report_dataframe(decomp_t.get("cluster_assignments", {}))
    labels = np.asarray(_labels_from_decomposition(decomp_t, data_df.index.tolist()))

    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomp_t.get("num_clusters", 0)),
        report_df=report_t,
        status="ok",
        skip_reason=None,
        extra={
            "tree": tree_t,
            "decomposition": decomp_t,
            "annotations": tree_t.annotations_df,
            "linkage_matrix": Z_t,
        },
    )


def _run_kl_diffusion_adaptive_method(
    data_df: pd.DataFrame,
    significance_level: float,
    k_neighbors: int | None = None,
    diffusion_time: int = 3,
    n_components: int = 30,
    metric: str = "hamming",
    bandwidth_type: str | float | None = "-1/(d+2)",
    epsilon: str | float = "median",
) -> MethodRunResult:
    """Run KL decomposition on an adaptive variable-bandwidth diffusion tree."""
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

    Z_t = linkage(diff_dist, method="average")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_df.index.tolist())

    decomp_t = tree_t.decompose(
        leaf_data=data_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )

    report_t = _create_report_dataframe(decomp_t.get("cluster_assignments", {}))
    labels = np.asarray(_labels_from_decomposition(decomp_t, data_df.index.tolist()))

    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomp_t.get("num_clusters", 0)),
        report_df=report_t,
        status="ok",
        skip_reason=None,
        extra={
            "tree": tree_t,
            "decomposition": decomp_t,
            "annotations": tree_t.annotations_df,
            "linkage_matrix": Z_t,
            "adaptive_diffusion": adaptive_metadata,
        },
    )
