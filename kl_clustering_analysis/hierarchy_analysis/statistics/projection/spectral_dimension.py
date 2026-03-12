"""Per-node spectral dimension estimation.

Computes the projection dimension for each internal node by eigendecomposing
the local **correlation** matrix of descendant leaf data. This replaces the
Johnson-Lindenstrauss-based dimension selection which is misapplied to the
single-vector projected Wald test.

Using the **correlation matrix** (not the covariance) is essential because
the projected Wald z-vector is standardised per-feature:
    z_i = (θ̂_child_i - θ̂_parent_i) / √Var_i
so Cov(z) under H₀ equals the Pearson correlation matrix C of the data.
Eigendecomposing C and whitening by its eigenvalues gives an exact χ²(k)
null: T = Σ (vᵢᵀz)² / λᵢ ~ χ²(k).

Two estimators are provided:

1. **Effective rank** (Roy & Vetterli, 2007): continuous dimensionality
   from the Shannon entropy of the normalised eigenvalue spectrum.
   ``erank(C) = exp(−Σ pᵢ log pᵢ)``  where ``pᵢ = λᵢ / Σλⱼ``.

2. **Marchenko-Pastur signal count**: number of eigenvalues exceeding the
   MP upper bound ``σ² (1 + √(d/n))²``; σ² estimated from the bulk median.

The chosen dimension ``k_v`` is used:
  - to set the degrees of freedom of the χ²(k) null for the projected Wald test,
  - to build the PCA projection that concentrates signal while discarding
    noise-only directions,
  - together with the top-k eigenvalues for whitening (exact χ² calibration).

References
----------
Roy, O. & Vetterli, M. (2007). "The effective rank: A measure of effective
    dimensionality". EUSIPCO.
Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for
    some sets of random matrices". Mathematics of the USSR-Sbornik.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple, cast

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ...decomposition.backends.eigen_backend import (
    build_pca_projection_backend as build_pca_projection,
)
from ...decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend as eigendecompose_correlation,
)
from ...decomposition.methods.k_estimators import (
    estimate_k_active_features,
    estimate_k_effective_rank,
    estimate_k_marchenko_pastur,
)
from .spectral_types import NodeSpectralResult, NodeSpectralTask
from .tree_helpers import is_leaf, precompute_descendants

logger = logging.getLogger(__name__)


# =====================================================================
# Parallelism control
# =====================================================================

# Default thread count for joblib.Parallel eigendecomposition.
# Set KL_TE_N_JOBS env var to override (e.g. "1" to disable parallelism).
_DEFAULT_MIN_NODES_FOR_PARALLEL = 8


def _get_n_jobs(n_tasks: int) -> int:
    """Resolve the number of parallel workers.

    Returns 1 (sequential) when the number of tasks is small or the user
    explicitly sets ``KL_TE_N_JOBS=1``.
    """
    configured_jobs = os.environ.get("KL_TE_N_JOBS")
    if configured_jobs is not None:
        try:
            return max(int(configured_jobs), 1)
        except ValueError:
            pass
    if n_tasks < _DEFAULT_MIN_NODES_FOR_PARALLEL:
        return 1
    return -1  # joblib: use all available cores


# =====================================================================
# Per-node workers (module-level for joblib pickling / clarity)
# =====================================================================


def _process_node(
    spectral_task: NodeSpectralTask,
    full_feature_matrix: np.ndarray,
    dimension_method: str,
    minimum_projection_dimension: int,
    feature_count: int,
    compute_eigendecomposition_outputs: bool,
) -> NodeSpectralResult:
    """Eigendecompose one node and return a typed result payload.

    Data is sliced lazily here (not pre-materialised by the caller) so that
    only O(n_threads × n_desc_max × d) bytes are live at any moment, instead
    of O(n_nodes × n_desc_avg × d) for the full pre-built dict.

    Parameters
    ----------
    spectral_task
        Per-node task payload with descendants and optional internal vectors.
    full_feature_matrix
        Full data matrix shared across threads (read-only view).
    """
    descendant_leaf_row_indices = spectral_task.row_indices
    internal_distribution_vectors = spectral_task.internal_distributions

    if len(descendant_leaf_row_indices) < 2:
        return NodeSpectralResult(
            node_id=spectral_task.node_id,
            projection_dimension=max(minimum_projection_dimension, 1),
            projection_matrix=None,
            eigenvalues=None,
        )

    # Slice on-demand: only this thread's copy is live during the call.
    descendant_leaf_feature_rows = full_feature_matrix[descendant_leaf_row_indices, :]

    if internal_distribution_vectors:
        descendant_feature_matrix = np.vstack(
            [
                descendant_leaf_feature_rows,
                np.array(internal_distribution_vectors, dtype=np.float64),
            ]
        )
    else:
        descendant_feature_matrix = descendant_leaf_feature_rows

    if dimension_method == "active_features":
        projection_dimension = estimate_k_active_features(
            descendant_feature_matrix,
            minimum_projection_dimension=minimum_projection_dimension,
        )
        projection_dimension = min(projection_dimension, feature_count)
        return NodeSpectralResult(
            node_id=spectral_task.node_id,
            projection_dimension=projection_dimension,
            projection_matrix=None,
            eigenvalues=None,
        )

    eigendecomposition_result = eigendecompose_correlation(
        descendant_feature_matrix,
        need_eigh=compute_eigendecomposition_outputs,
    )

    if eigendecomposition_result is None:
        return NodeSpectralResult(
            node_id=spectral_task.node_id,
            projection_dimension=max(minimum_projection_dimension, 1),
            projection_matrix=None,
            eigenvalues=None,
        )

    if dimension_method == "effective_rank":
        projection_dimension = estimate_k_effective_rank(
            eigendecomposition_result.eigenvalues,
            minimum_projection_dimension=minimum_projection_dimension,
            d_active=eigendecomposition_result.d_active,
        )
    else:  # marchenko_pastur
        projection_dimension = estimate_k_marchenko_pastur(
            eigendecomposition_result.eigenvalues,
            n_desc=descendant_feature_matrix.shape[0],
            d_active=eigendecomposition_result.d_active,
            minimum_projection_dimension=minimum_projection_dimension,
        )

    projection_matrix, pca_eigenvalues = None, None
    if compute_eigendecomposition_outputs:
        projection_matrix, pca_eigenvalues = build_pca_projection(
            eigendecomposition_result,
            k=projection_dimension,
            d=feature_count,
        )
        if projection_matrix is None or pca_eigenvalues is None:
            projection_matrix, pca_eigenvalues = None, None

    return NodeSpectralResult(
        node_id=spectral_task.node_id,
        projection_dimension=projection_dimension,
        projection_matrix=projection_matrix,
        eigenvalues=pca_eigenvalues,
    )


def _build_spectral_tasks(
    tree: nx.DiGraph,
    internal_node_ids: list[str],
    descendant_leaf_indices_by_node: dict[str, list[int]],
    descendant_internal_nodes_by_node: dict[str, list[str]],
    *,
    include_internal: bool,
    feature_count: int,
) -> list[NodeSpectralTask]:
    """Build per-node spectral tasks from precomputed descendant metadata."""

    internal_distributions_by_node: Dict[str, tuple[np.ndarray, ...]] = {}

    if include_internal:
        for node_id in internal_node_ids:

            node_internal_distributions: list[np.ndarray] = []

            for internal_node_id in descendant_internal_nodes_by_node.get(node_id, []):
                distribution = tree.nodes[internal_node_id].get("distribution")
                if distribution is None:
                    continue
                distribution_array = np.asarray(distribution, dtype=np.float64)
                if distribution_array.shape == (feature_count,):
                    node_internal_distributions.append(distribution_array)
            internal_distributions_by_node[node_id] = tuple(node_internal_distributions)

    return [
        NodeSpectralTask(
            node_id=node_id,
            row_indices=tuple(descendant_leaf_indices_by_node.get(node_id, [])),
            internal_distributions=internal_distributions_by_node.get(node_id, ()),
        )
        for node_id in internal_node_ids
    ]


def _run_spectral_tasks_parallel(
    node_spectral_tasks: list[NodeSpectralTask],
    full_feature_matrix: np.ndarray,
    *,
    dimension_method: str,
    minimum_projection_dimension: int,
    feature_count: int,
    compute_eigendecomposition_outputs: bool,
) -> list[NodeSpectralResult]:
    """Execute node spectral tasks in parallel (or sequentially for small workloads)."""
    n_jobs = _get_n_jobs(len(node_spectral_tasks))
    parallel_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_node)(
            task,
            full_feature_matrix,
            dimension_method,
            minimum_projection_dimension,
            feature_count,
            compute_eigendecomposition_outputs,
        )
        for task in node_spectral_tasks
    )
    return cast(list[NodeSpectralResult], parallel_results)


def _aggregate_spectral_results(
    spectral_results: list[NodeSpectralResult],
    *,
    spectral_dims: dict[str, int],
    pca_projections: dict[str, np.ndarray],
    pca_eigenvalues: dict[str, np.ndarray],
) -> None:
    """Write per-node worker outputs into decomposition result dicts."""
    for node_result in spectral_results:
        spectral_dims[node_result.node_id] = node_result.projection_dimension
        if node_result.projection_matrix is not None and node_result.eigenvalues is not None:
            pca_projections[node_result.node_id] = node_result.projection_matrix
            pca_eigenvalues[node_result.node_id] = node_result.eigenvalues


def compute_spectral_decomposition(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    method: str = "effective_rank",
    minimum_projection_dimension: int = 1,
    compute_projections: bool = True,
    include_internal: bool | None = None,
) -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute spectral dimensions, PCA projections, and eigenvalues.

    Performs exactly one eigendecomposition per internal node of the local
    **correlation** matrix, extracting the dimension estimate, the top-k
    eigenvector projection matrix, and the corresponding eigenvalues for
    whitening.

    The correlation matrix is used (not the covariance) because the Wald
    z-vector is per-feature standardised, so its covariance under H₀ equals
    the Pearson correlation C = D^{-1/2} Σ D^{-1/2}.  Only features with
    non-zero variance ("active" features) are included; constant features
    receive zero weight in the projection.

    Parameters
    ----------
    tree
        Directed hierarchy with leaf labels accessible via
        ``tree.nodes[n].get("label", n)``.
    leaf_data
        DataFrame with leaf labels as index and features as columns.
    method
        Dimension estimator: ``"effective_rank"`` (default),
        ``"marchenko_pastur"``, or ``"active_features"``.
    minimum_projection_dimension
        Floor on the returned dimension.
    compute_projections
        If True, also returns PCA projection matrices (k_v × d) and
        eigenvalue arrays for eigenvalue-based methods. For
        ``"active_features"`` these are always empty.
    include_internal
        If True, include internal node distribution vectors in the data
        matrix used for eigendecomposition.  If None, reads from
        ``config.INCLUDE_INTERNAL_IN_SPECTRAL`` (default False).
        Internal distributions are convex combinations of leaf data — they
        do NOT increase rank and typically reduce effective rank by ~30%.

    Returns
    -------
    (spectral_dims, pca_projections, pca_eigenvalues)
        spectral_dims : dict[str, int] — node_id → k_v
        pca_projections : dict[str, np.ndarray] — node_id → (k_v × d) matrix
        pca_eigenvalues : dict[str, np.ndarray] — node_id → (k_v,) eigenvalues
    """
    from kl_clustering_analysis import config as _config

    start_time = time.perf_counter()

    if include_internal is None:
        include_internal = _config.INCLUDE_INTERNAL_IN_SPECTRAL

    if method not in ("effective_rank", "marchenko_pastur", "active_features"):
        raise ValueError(
            f"Unknown spectral dimension method {method!r}. "
            f"Choose from 'effective_rank', 'marchenko_pastur', 'active_features'."
        )

    feature_count = leaf_data.shape[1]
    leaf_label_to_index = {label: i for i, label in enumerate(leaf_data.index)}
    leaf_feature_matrix = leaf_data.values.astype(np.float64)

    descendant_leaf_indices_by_node, descendant_internal_nodes_by_node = precompute_descendants(
        tree, leaf_label_to_index
    )

    compute_eigendecomposition_outputs = compute_projections and method in (
        "effective_rank",
        "marchenko_pastur",
    )

    spectral_dims: Dict[str, int] = {}
    pca_projections: Dict[str, np.ndarray] = {}
    pca_eigenvalues: Dict[str, np.ndarray] = {}

    # Separate leaves (trivial) from internal nodes (expensive).
    internal_node_ids: list[str] = []
    for node_id in tree.nodes:
        if is_leaf(tree, node_id):
            spectral_dims[node_id] = 0
        else:
            internal_node_ids.append(node_id)

    spectral_tasks = _build_spectral_tasks(
        tree,
        internal_node_ids,
        descendant_leaf_indices_by_node,
        descendant_internal_nodes_by_node,
        include_internal=bool(include_internal),
        feature_count=feature_count,
    )

    # Data is sliced lazily inside each worker (not pre-materialised here).
    # With prefer="threads" every worker shares the same X array (no copy).
    # Peak RAM is O(n_jobs × n_desc_max × d) instead of O(n_nodes × n_desc_avg × d).
    spectral_results = _run_spectral_tasks_parallel(
        spectral_tasks,
        leaf_feature_matrix,
        dimension_method=method,
        minimum_projection_dimension=minimum_projection_dimension,
        feature_count=feature_count,
        compute_eigendecomposition_outputs=compute_eigendecomposition_outputs,
    )

    _aggregate_spectral_results(
        spectral_results,
        spectral_dims=spectral_dims,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )

    elapsed = time.perf_counter() - start_time

    # Log summary statistics
    internal_projection_dimensions = [
        projection_dimension
        for node_id, projection_dimension in spectral_dims.items()
        if not is_leaf(tree, node_id)
    ]

    if internal_projection_dimensions:
        logger.info(
            "Spectral dimensions (%s): median=%d, mean=%.1f, min=%d, max=%d "
            "(across %d internal nodes, d=%d) [%.2fs]",
            method,
            int(np.median(internal_projection_dimensions)),
            float(np.mean(internal_projection_dimensions)),
            min(internal_projection_dimensions),
            max(internal_projection_dimensions),
            len(internal_projection_dimensions),
            feature_count,
            elapsed,
        )

    if compute_projections:
        logger.info(
            "Computed PCA projections for %d internal nodes [%.2fs total]",
            len(pca_projections),
            elapsed,
        )

    return spectral_dims, pca_projections, pca_eigenvalues


__all__ = [
    "effective_rank",
    "marchenko_pastur_signal_count",
    "count_active_features",
    "compute_spectral_decomposition",
]
