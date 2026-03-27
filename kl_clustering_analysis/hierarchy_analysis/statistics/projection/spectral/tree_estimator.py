r"""Tree-level spectral decomposition orchestrator.

Builds per-node spectral tasks, dispatches them in parallel (or
sequentially for small trees), and assembles results into the three
output dicts consumed by the rest of the pipeline.

The only supported estimator is Marchenko-Pastur rank selection. It uses
random matrix theory to separate signal eigenvalues from the noise bulk of
the local correlation matrix. For correlation matrices, $\sigma^2 = 1$
exactly, so the Marchenko-Pastur support is $(1 \pm \sqrt{d/n})^2$. When no
eigenvalues exceed the upper bound, the raw signal count is conceptually 0,
but the implementation floors the returned dimension to at least 1 and then
applies ``config.SPECTRAL_MINIMUM_DIMENSION``. Pure-noise nodes therefore get
a small fallback dimension and are expected to fail to reject rather than
taking a literal ``k = 0`` skip path.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, cast

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .marchenko_pastur import _get_n_jobs, _process_node
from .tree_helpers import is_leaf, precompute_descendants
from .types import NodeSpectralResult, NodeSpectralTask

logger = logging.getLogger(__name__)


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
    minimum_projection_dimension
        Floor on the returned dimension.
    compute_projections
        If True, also returns PCA projection matrices (k_v × d) and
        eigenvalue arrays.
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
    if include_internal is None:
        from kl_clustering_analysis import config

        include_internal = config.INCLUDE_INTERNAL_IN_SPECTRAL

    feature_count = leaf_data.shape[1]
    leaf_label_to_index = {label: i for i, label in enumerate(leaf_data.index)}
    leaf_feature_matrix = leaf_data.values.astype(np.float64)

    descendant_leaf_indices_by_node, descendant_internal_nodes_by_node = precompute_descendants(
        tree, leaf_label_to_index
    )

    compute_eigendecomposition_outputs = compute_projections

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
        dimension_method="marchenko_pastur",
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

    # Log summary statistics
    internal_projection_dimensions = [
        projection_dimension
        for node_id, projection_dimension in spectral_dims.items()
        if not is_leaf(tree, node_id)
    ]

    if internal_projection_dimensions:
        logger.info(
            "Spectral dimensions (marchenko_pastur): median=%d, mean=%.1f, min=%d, max=%d "
            "(across %d internal nodes, d=%d) [%.2fs]",
            int(np.median(internal_projection_dimensions)),
            float(np.mean(internal_projection_dimensions)),
            min(internal_projection_dimensions),
            max(internal_projection_dimensions),
            len(internal_projection_dimensions),
            feature_count,
        )

    if compute_projections:
        logger.info(
            "Computed PCA projections for %d internal nodes [%.2fs total]",
            len(pca_projections),
        )

    return spectral_dims, pca_projections, pca_eigenvalues


__all__ = [
    "compute_spectral_decomposition",
]
