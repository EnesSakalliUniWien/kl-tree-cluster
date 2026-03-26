"""Per-node Marchenko-Pastur spectral worker.

Handles eigendecomposition, dimension estimation, and projection matrix
construction for a single tree node.  Module-level functions are defined
here (not as inner closures) to ensure correct pickling when dispatched
via joblib ``Parallel``.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ....decomposition.backends.eigen_backend import (
    build_pca_projection_backend as build_pca_projection,
)
from ....decomposition.backends.eigen_backend import (
    eigendecompose_correlation_backend as eigendecompose_correlation,
)
from ..k_estimators import estimate_k_marchenko_pastur
from .types import NodeSpectralResult, NodeSpectralTask

logger = logging.getLogger(__name__)

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

    eigendecomposition_result = eigendecompose_correlation(
        descendant_feature_matrix,
        compute_eigenvectors=compute_eigendecomposition_outputs,
    )

    if eigendecomposition_result is None:
        return NodeSpectralResult(
            node_id=spectral_task.node_id,
            projection_dimension=max(minimum_projection_dimension, 1),
            projection_matrix=None,
            eigenvalues=None,
        )

    projection_dimension = estimate_k_marchenko_pastur(
        eigendecomposition_result.eigenvalues,
        n_samples=descendant_feature_matrix.shape[0],
        n_features=eigendecomposition_result.active_feature_count,
        minimum_projection_dimension=minimum_projection_dimension,
    )

    projection_matrix, pca_eigenvalues = None, None

    if compute_eigendecomposition_outputs:
        projection_matrix, pca_eigenvalues = build_pca_projection(
            eigendecomposition_result,
            projection_dimension=projection_dimension,
            n_features_total=feature_count,
        )

    return NodeSpectralResult(
        node_id=spectral_task.node_id,
        projection_dimension=projection_dimension,
        projection_matrix=projection_matrix,
        eigenvalues=pca_eigenvalues,
    )


__all__ = [
    "_DEFAULT_MIN_NODES_FOR_PARALLEL",
    "_get_n_jobs",
    "_process_node",
]
