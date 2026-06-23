"""Per-node Marchenko-Pastur spectral worker.

Handles eigendecomposition, dimension estimation, and projection matrix
construction for a single tree node.  Module-level functions are defined
here (not as inner closures) to ensure correct pickling when dispatched
via joblib ``Parallel``.
"""

from __future__ import annotations

import logging
import os
from time import perf_counter

import numpy as np

from ....decomposition.backends.eigen.decomposition import eigendecompose_covariance
from ....decomposition.backends.eigen.projection import build_pca_projection
from ...contrast_covariance import _build_trusted_null_whitened_tangent_matrix
from ..projection_dimension_estimation.projection_dimension_estimators import (
    estimate_marchenko_pastur_dimension,
)
from .node_spectral_result import NodeSpectralResult
from .node_spectral_task import NodeSpectralTask

logger = logging.getLogger(__name__)

# Default thread count for joblib.Parallel eigendecomposition.
# Set TBS_N_JOBS env var to override (e.g. "1" to disable parallelism).
_DEFAULT_MIN_NODES_FOR_PARALLEL = 8
MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS = "leaf_effective_rows"
MP_ROW_COUNT_LEGACY_STACKED_ROWS = "legacy_stacked_rows"


def _empty_stage_timings() -> dict[str, float]:
    """Return the per-node spectral timing keys used by benchmark aggregation."""
    return {
        "tangent_whitening_sec": 0.0,
        "eigensolve_sec": 0.0,
        "pca_projection_sec": 0.0,
    }


def _get_n_jobs(n_tasks: int) -> int:
    """Resolve the number of parallel workers.

    Returns 1 (sequential) when the number of tasks is small or the user
    explicitly sets ``TBS_N_JOBS=1``.
    """
    configured_jobs = os.environ.get("TBS_N_JOBS")
    if configured_jobs is not None:
        configured_job_count = int(configured_jobs)
        if configured_job_count < 1:
            raise ValueError(
                f"TBS_N_JOBS must be a positive integer; got {configured_jobs!r}."
            )
        return configured_job_count
    if n_tasks < _DEFAULT_MIN_NODES_FOR_PARALLEL:
        return 1
    return -1  # joblib: use all available cores


def _process_node(
    spectral_task: NodeSpectralTask,
    full_feature_matrix: np.ndarray,
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
        Per-node task payload with descendant leaf rows and, in the legacy
        diagnostic path, descendant internal barycenter rows.
    full_feature_matrix
        Full data matrix shared across threads (read-only view).
    """
    descendant_leaf_row_indices = spectral_task.row_indices
    internal_distribution_vectors = spectral_task.internal_distributions
    mp_row_count_mode = str(spectral_task.mp_row_count_mode)
    stage_timings = _empty_stage_timings()

    if len(descendant_leaf_row_indices) < 2:
        return NodeSpectralResult(
            node_id=spectral_task.node_id,
            raw_mp_signal_count=0,
            test_projection_dimension=0,
            effective_independent_rows=len(descendant_leaf_row_indices),
            mp_threshold_rows=len(descendant_leaf_row_indices),
            projection_matrix=np.zeros((0, feature_count), dtype=np.float64),
            eigenvalues=np.zeros(0, dtype=np.float64),
            full_eigenvalues=np.zeros(0, dtype=np.float64),
            active_feature_count=0,
            stage_timings=stage_timings,
        )

    # Slice on-demand: only this thread's copy is live during the call.
    descendant_leaf_feature_rows = full_feature_matrix[descendant_leaf_row_indices, :]
    whitening_start_sec = perf_counter()
    descendant_leaf_feature_rows = _build_trusted_null_whitened_tangent_matrix(
        descendant_leaf_feature_rows,
        spectral_task.null_distribution,
        spectral_task.feature_space,
        spectral_task.continuous_covariance_by_block or {},
        ridge=1e-12,
    )
    stage_timings["tangent_whitening_sec"] += float(perf_counter() - whitening_start_sec)

    descendant_feature_matrix = descendant_leaf_feature_rows
    if internal_distribution_vectors:
        internal_feature_rows = np.asarray(internal_distribution_vectors, dtype=np.float64)
        if internal_feature_rows.ndim == 3 and internal_feature_rows.shape[1] == 1:
            internal_feature_rows = internal_feature_rows[:, 0, :]
        whitening_start_sec = perf_counter()
        internal_feature_rows = _build_trusted_null_whitened_tangent_matrix(
            internal_feature_rows,
            spectral_task.null_distribution,
            spectral_task.feature_space,
            spectral_task.continuous_covariance_by_block or {},
            ridge=1e-12,
        )
        stage_timings["tangent_whitening_sec"] += float(
            perf_counter() - whitening_start_sec
        )
        descendant_feature_matrix = np.vstack(
            [descendant_leaf_feature_rows, internal_feature_rows]
        )

    eigensolve_start_sec = perf_counter()
    eigendecomposition_result = eigendecompose_covariance(
        descendant_feature_matrix,
        compute_eigenvectors=compute_eigendecomposition_outputs,
    )
    stage_timings["eigensolve_sec"] = float(perf_counter() - eigensolve_start_sec)

    if eigendecomposition_result is None:
        return NodeSpectralResult(
            node_id=spectral_task.node_id,
            raw_mp_signal_count=0,
            test_projection_dimension=0,
            effective_independent_rows=len(descendant_leaf_row_indices),
            mp_threshold_rows=descendant_feature_matrix.shape[0],
            projection_matrix=np.zeros((0, feature_count), dtype=np.float64),
            eigenvalues=np.zeros(0, dtype=np.float64),
            full_eigenvalues=np.zeros(0, dtype=np.float64),
            active_feature_count=0,
            stage_timings=stage_timings,
        )

    if mp_row_count_mode == MP_ROW_COUNT_LEGACY_STACKED_ROWS:
        mp_threshold_rows = int(descendant_feature_matrix.shape[0])
    elif mp_row_count_mode == MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS:
        mp_threshold_rows = int(len(descendant_leaf_row_indices))
    else:
        raise ValueError(
            f"Unknown MP row-count mode {mp_row_count_mode!r}; "
            f"allowed={(MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS, MP_ROW_COUNT_LEGACY_STACKED_ROWS)!r}."
        )

    dimension_estimate = estimate_marchenko_pastur_dimension(
        eigendecomposition_result.eigenvalues,
        n_samples=descendant_feature_matrix.shape[0],
        n_features=eigendecomposition_result.active_feature_count,
        effective_independent_rows=len(descendant_leaf_row_indices),
        mp_threshold_rows=mp_threshold_rows,
        minimum_projection_dimension=minimum_projection_dimension,
    )
    test_projection_dimension = dimension_estimate.test_projection_dimension

    projection_matrix, pca_eigenvalues = (
        np.zeros((0, feature_count), dtype=np.float64),
        np.zeros(0, dtype=np.float64),
    )

    if compute_eigendecomposition_outputs and test_projection_dimension > 0:
        projection_start_sec = perf_counter()
        projection_matrix, pca_eigenvalues = build_pca_projection(
            eigendecomposition_result,
            projection_dimension=test_projection_dimension,
            n_features_total=feature_count,
        )
        stage_timings["pca_projection_sec"] = float(
            perf_counter() - projection_start_sec
        )
        test_projection_dimension = int(projection_matrix.shape[0])
    elif not compute_eigendecomposition_outputs:
        projection_matrix, pca_eigenvalues = None, None

    return NodeSpectralResult(
        node_id=spectral_task.node_id,
        raw_mp_signal_count=dimension_estimate.raw_mp_signal_count,
        test_projection_dimension=test_projection_dimension,
        effective_independent_rows=dimension_estimate.effective_independent_rows,
        mp_threshold_rows=dimension_estimate.mp_threshold_rows,
        projection_matrix=projection_matrix,
        eigenvalues=pca_eigenvalues,
        full_eigenvalues=np.asarray(
            eigendecomposition_result.eigenvalues,
            dtype=np.float64,
        ),
        active_feature_count=int(eigendecomposition_result.active_feature_count),
        stage_timings=stage_timings,
    )


__all__ = [
    "MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS",
    "MP_ROW_COUNT_LEGACY_STACKED_ROWS",
    "_DEFAULT_MIN_NODES_FOR_PARALLEL",
    "_get_n_jobs",
    "_process_node",
]
