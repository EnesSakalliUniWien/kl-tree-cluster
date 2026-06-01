r"""Tree-level spectral decomposition orchestrator.

Builds per-node spectral tasks, dispatches them in parallel (or
sequentially for small trees), and assembles the explicit spectral contract
consumed by the rest of the pipeline.

The only supported estimator is Marchenko-Pastur rank selection. It uses
random matrix theory to separate signal eigenvalues from the noise bulk of
the local covariance matrix after each descendant row is mapped into the
null-whitened tangent coordinates used by the Wald statistic. In that
coordinate system the working null covariance scale is one, so the
Marchenko-Pastur support is $(1 \pm \sqrt{d/m_{\mathrm{MP}}})^2$. When no
eigenvalues exceed the upper bound, the raw signal count is 0, but the test
dimension is floored to the caller's minimum and capped by the number of
active features. The effective independent row count is recorded separately
from the MP threshold row count.
Pure-noise nodes therefore run a low-rank test and are expected to fail to
reject. Nodes with no active feature directions are represented explicitly as
zero-dimensional PCA contexts.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, cast

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from kl_clustering_analysis.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    resolve_feature_space,
    validate_feature_matrix,
)

from .marchenko_pastur import _get_n_jobs, _process_node
from .node_spectral_result import NodeSpectralResult
from .node_spectral_task import NodeSpectralTask
from .spectral_decomposition_result import SpectralDecompositionResult
from .tree_helpers import is_leaf, precompute_descendants

logger = logging.getLogger(__name__)


def _build_spectral_tasks(
    tree: nx.DiGraph,
    internal_node_ids: list[str],
    descendant_leaf_indices_by_node: dict[str, list[int]],
    *,
    feature_space: FeatureSpace,
) -> list[NodeSpectralTask]:
    """Build per-node spectral tasks from precomputed descendant metadata."""

    return [
        NodeSpectralTask(
            node_id=node_id,
            row_indices=tuple(descendant_leaf_indices_by_node[node_id]),
            null_distribution=np.asarray(tree.nodes[node_id]["distribution"], dtype=np.float64),
            feature_space=feature_space,
            continuous_covariance_by_block=require_node_continuous_covariance_by_block(
                tree,
                node_id,
                feature_space,
            ),
        )
        for node_id in internal_node_ids
    ]


def _run_spectral_tasks_parallel(
    node_spectral_tasks: list[NodeSpectralTask],
    full_feature_matrix: np.ndarray,
    *,
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
    test_projection_dimensions: dict[str, int],
    raw_mp_signal_counts: dict[str, int],
    effective_independent_rows: dict[str, int],
    mp_threshold_rows: dict[str, int],
    pca_projections: dict[str, np.ndarray],
    pca_eigenvalues: dict[str, np.ndarray],
) -> None:
    """Write per-node worker outputs into decomposition result dicts."""
    for node_result in spectral_results:
        test_projection_dimensions[node_result.node_id] = (
            node_result.test_projection_dimension
        )
        raw_mp_signal_counts[node_result.node_id] = node_result.raw_mp_signal_count
        effective_independent_rows[node_result.node_id] = (
            node_result.effective_independent_rows
        )
        mp_threshold_rows[node_result.node_id] = node_result.mp_threshold_rows
        if node_result.projection_matrix is None or node_result.eigenvalues is None:
            raise ValueError(
                "Spectral decomposition must return PCA projection and eigenvalues for "
                f"internal node {node_result.node_id!r}."
            )
        pca_projections[node_result.node_id] = node_result.projection_matrix
        pca_eigenvalues[node_result.node_id] = node_result.eigenvalues


def compute_spectral_decomposition(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    minimum_projection_dimension: int = 1,
    feature_space: FeatureSpace | None = None,
) -> SpectralDecompositionResult:
    """Compute MP dimension metadata, PCA projections, and eigenvalues.

    Performs exactly one eigendecomposition per internal node after mapping
    descendant distributions into the same null-whitened tangent coordinates
    used by the projected Wald statistic. The covariance matrix is then
    computed in that coordinate system without re-standardizing columns. Only
    active directions with non-zero empirical variance are included; constant
    directions receive zero weight in the projection.

    Parameters
    ----------
    tree
        Directed hierarchy with leaf labels stored at ``tree.nodes[n]["label"]``.
    leaf_data
        DataFrame with leaf labels as index and features as columns.
    minimum_projection_dimension
        Floor on the returned dimension.

    Returns
    -------
    SpectralDecompositionResult
        Typed per-node spectral output. ``test_projection_dimensions_by_node``
        is the dimension used by downstream projected-Wald tests; raw MP signal
        counts, effective independent row counts, and MP threshold row counts
        are exposed separately.
    """
    spectral_start_sec = perf_counter()
    active_feature_space = resolve_feature_space(tuple(leaf_data.columns), feature_space)
    feature_count = active_feature_space.contrast_dimension
    leaf_feature_matrix = validate_feature_matrix(
        leaf_data.to_numpy(dtype=np.float64, copy=False),
        active_feature_space,
        value_name="leaf_data",
    )
    leaf_label_to_index = {label: i for i, label in enumerate(leaf_data.index)}

    descendant_leaf_indices_by_node = precompute_descendants(tree, leaf_label_to_index)

    test_projection_dimensions: Dict[str, int] = {}
    raw_mp_signal_counts: Dict[str, int] = {}
    effective_independent_rows: Dict[str, int] = {}
    mp_threshold_rows: Dict[str, int] = {}
    pca_projections: Dict[str, np.ndarray] = {}
    pca_eigenvalues: Dict[str, np.ndarray] = {}

    # Separate leaves (trivial) from internal nodes (expensive).
    internal_node_ids: list[str] = []
    for node_id in tree.nodes:
        if is_leaf(tree, node_id):
            test_projection_dimensions[node_id] = 0
            raw_mp_signal_counts[node_id] = 0
            effective_independent_rows[node_id] = 1
            mp_threshold_rows[node_id] = 1
        else:
            internal_node_ids.append(node_id)

    spectral_tasks = _build_spectral_tasks(
        tree,
        internal_node_ids,
        descendant_leaf_indices_by_node,
        feature_space=active_feature_space,
    )

    # Data is sliced lazily inside each worker (not pre-materialised here).
    # With prefer="threads" every worker shares the same X array (no copy).
    # Peak RAM is O(n_jobs × n_desc_max × d) instead of O(n_nodes × n_desc_avg × d).
    spectral_results = _run_spectral_tasks_parallel(
        spectral_tasks,
        leaf_feature_matrix,
        minimum_projection_dimension=minimum_projection_dimension,
        feature_count=feature_count,
        compute_eigendecomposition_outputs=True,
    )

    _aggregate_spectral_results(
        spectral_results,
        test_projection_dimensions=test_projection_dimensions,
        raw_mp_signal_counts=raw_mp_signal_counts,
        effective_independent_rows=effective_independent_rows,
        mp_threshold_rows=mp_threshold_rows,
        pca_projections=pca_projections,
        pca_eigenvalues=pca_eigenvalues,
    )
    spectral_wall_sec = float(perf_counter() - spectral_start_sec)
    stage_timings = {
        "tangent_whitening_sec": float(
            sum(
                result.stage_timings.get("tangent_whitening_sec", 0.0)
                for result in spectral_results
            )
        ),
        "eigensolve_sec": float(
            sum(result.stage_timings.get("eigensolve_sec", 0.0) for result in spectral_results)
        ),
        "pca_projection_sec": float(
            sum(
                result.stage_timings.get("pca_projection_sec", 0.0)
                for result in spectral_results
            )
        ),
    }

    # Log summary statistics
    internal_projection_dimensions = [
        projection_dimension
        for node_id, projection_dimension in test_projection_dimensions.items()
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
            spectral_wall_sec,
        )

    logger.info(
        "Computed PCA projections for %d internal nodes [%.2fs total]",
        len(pca_projections),
        spectral_wall_sec,
    )

    return SpectralDecompositionResult(
        test_projection_dimensions_by_node=test_projection_dimensions,
        raw_mp_signal_counts_by_node=raw_mp_signal_counts,
        effective_independent_rows_by_node=effective_independent_rows,
        mp_threshold_rows_by_node=mp_threshold_rows,
        principal_component_projections_by_node=pca_projections,
        principal_component_eigenvalues_by_node=pca_eigenvalues,
        stage_timings=stage_timings,
    )


__all__ = [
    "compute_spectral_decomposition",
]
