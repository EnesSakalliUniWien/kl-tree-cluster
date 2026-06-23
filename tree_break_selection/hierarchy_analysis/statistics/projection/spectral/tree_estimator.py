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
import math
from time import perf_counter
from typing import Dict, cast

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tree_break_selection.core_utils.tree_utils import bottom_up_nodes
from tree_break_selection.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    resolve_feature_space,
    validate_feature_matrix,
)

from .marchenko_pastur import (
    MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
    MP_ROW_COUNT_LEGACY_STACKED_ROWS,
    _get_n_jobs,
    _process_node,
)
from .node_spectral_result import NodeSpectralResult
from .node_spectral_task import NodeSpectralTask
from .spectral_decomposition_result import SpectralDecompositionResult
from .tree_helpers import (
    is_leaf,
    precompute_descendant_internal_nodes,
    precompute_descendants,
)

logger = logging.getLogger(__name__)

INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER = "empirical_barycenter"
INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE = "branch_length_state"
INTERNAL_DISTRIBUTION_MODES = (
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
)
MP_ROW_COUNT_MODES = (
    MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
    MP_ROW_COUNT_LEGACY_STACKED_ROWS,
)

_EDGE_LENGTH_KEYS = ("branch_length", "length", "weight")
_BRANCH_LENGTH_STATE_RELATIVE_FLOOR = 1e-6


def _validate_internal_distribution_mode(internal_distribution_mode: str) -> str:
    mode = str(internal_distribution_mode)
    if mode not in INTERNAL_DISTRIBUTION_MODES:
        raise ValueError(
            "Unknown internal spectral distribution mode "
            f"{internal_distribution_mode!r}; allowed={INTERNAL_DISTRIBUTION_MODES!r}."
        )
    return mode


def _validate_mp_row_count_mode(mp_row_count_mode: str) -> str:
    mode = str(mp_row_count_mode)
    if mode not in MP_ROW_COUNT_MODES:
        raise ValueError(
            f"Unknown MP row-count mode {mp_row_count_mode!r}; "
            f"allowed={MP_ROW_COUNT_MODES!r}."
        )
    return mode


def _edge_length(tree: nx.DiGraph, parent_id: object, child_id: object) -> float:
    attrs = tree.edges[parent_id, child_id]
    for key in _EDGE_LENGTH_KEYS:
        if key not in attrs:
            continue
        value = float(attrs[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "Branch-length internal spectral state requires finite "
                f"non-negative edge lengths; edge {(parent_id, child_id)!r} "
                f"has {key}={attrs[key]!r}."
            )
        return value
    return 1.0


def _branch_length_floor(tree: nx.DiGraph) -> float:
    positive_lengths: list[float] = []
    for _parent_id, _child_id, attrs in tree.edges(data=True):
        for key in _EDGE_LENGTH_KEYS:
            if key not in attrs:
                continue
            value = float(attrs[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Branch-length internal spectral state requires finite "
                    f"non-negative edge lengths; got {key}={attrs[key]!r}."
                )
            if value > 0.0:
                positive_lengths.append(value)
            break
    if not positive_lengths:
        return 1.0
    median_positive_length = float(np.median(np.asarray(positive_lengths, dtype=float)))
    return max(
        median_positive_length * _BRANCH_LENGTH_STATE_RELATIVE_FLOOR,
        float(np.finfo(np.float64).eps),
    )


def _precompute_branch_length_internal_states(
    tree: nx.DiGraph,
    *,
    feature_count: int,
) -> dict[object, np.ndarray]:
    r"""Estimate branch-length-aware latent states for internal nodes.

    The stored ``distribution`` remains the empirical leaf-count subtree
    barycenter. This diagnostic state instead treats each child subtree as a
    noisy observation of the parent state with precision proportional to

        descendant_leaf_count / max(branch_length(parent, child), floor).

    Short edges therefore transmit child state more strongly; long edges are
    allowed to drift. The floor is data-derived and only prevents zero-length
    edges from becoming infinite weights.
    """
    branch_floor = _branch_length_floor(tree)
    state_by_node: dict[object, np.ndarray] = {}
    leaf_count_by_node: dict[object, int] = {}

    for node_id in bottom_up_nodes(tree):
        children = list(tree.successors(node_id))
        if not children:
            distribution = tree.nodes[node_id].get("distribution")
            if distribution is None:
                continue
            distribution_array = np.asarray(distribution, dtype=np.float64)
            if distribution_array.shape != (feature_count,):
                continue
            state_by_node[node_id] = distribution_array
            leaf_count_by_node[node_id] = 1
            continue

        weighted_state_sum = np.zeros(feature_count, dtype=np.float64)
        total_weight = 0.0
        total_leaf_count = 0
        for child_id in children:
            child_leaf_count = int(leaf_count_by_node.get(child_id, 0))
            total_leaf_count += child_leaf_count
            child_state = state_by_node.get(child_id)
            if child_state is None or child_leaf_count <= 0:
                continue
            edge_length = max(_edge_length(tree, node_id, child_id), branch_floor)
            child_precision = float(child_leaf_count) / edge_length
            weighted_state_sum += child_state * child_precision
            total_weight += child_precision

        leaf_count_by_node[node_id] = total_leaf_count
        if total_weight > 0.0:
            state_by_node[node_id] = weighted_state_sum / total_weight
            continue

        distribution = tree.nodes[node_id].get("distribution")
        if distribution is not None:
            distribution_array = np.asarray(distribution, dtype=np.float64)
            if distribution_array.shape == (feature_count,):
                state_by_node[node_id] = distribution_array

    return {
        node_id: state
        for node_id, state in state_by_node.items()
        if not is_leaf(tree, node_id)
    }


def _build_spectral_tasks(
    tree: nx.DiGraph,
    internal_node_ids: list[str],
    descendant_leaf_indices_by_node: dict[str, list[int]],
    *,
    feature_space: FeatureSpace,
    include_internal_barycenters: bool = False,
    internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
) -> list[NodeSpectralTask]:
    """Build per-node spectral tasks from precomputed descendant metadata."""
    feature_count = int(feature_space.raw_dimension)
    internal_distribution_mode = _validate_internal_distribution_mode(
        internal_distribution_mode
    )
    mp_row_count_mode = _validate_mp_row_count_mode(mp_row_count_mode)
    descendant_internal_nodes_by_node = (
        precompute_descendant_internal_nodes(tree)
        if include_internal_barycenters
        else {}
    )
    branch_length_internal_states = (
        _precompute_branch_length_internal_states(
            tree,
            feature_count=feature_count,
        )
        if include_internal_barycenters
        and internal_distribution_mode == INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE
        else {}
    )

    def _internal_distributions(node_id: str) -> tuple[np.ndarray, ...]:
        if not include_internal_barycenters:
            return ()
        rows: list[np.ndarray] = []
        for internal_node_id in descendant_internal_nodes_by_node.get(node_id, []):
            if internal_distribution_mode == INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE:
                distribution = branch_length_internal_states.get(internal_node_id)
            else:
                distribution = tree.nodes[internal_node_id].get("distribution")
            if distribution is None:
                continue
            distribution_array = np.asarray(distribution, dtype=np.float64)
            if distribution_array.shape == (feature_count,):
                rows.append(distribution_array)
        return tuple(rows)

    return [
        NodeSpectralTask(
            node_id=node_id,
            row_indices=tuple(descendant_leaf_indices_by_node[node_id]),
            internal_distributions=_internal_distributions(node_id),
            null_distribution=np.asarray(tree.nodes[node_id]["distribution"], dtype=np.float64),
            feature_space=feature_space,
            continuous_covariance_by_block=require_node_continuous_covariance_by_block(
                tree,
                node_id,
                feature_space,
            ),
            mp_row_count_mode=mp_row_count_mode,
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
    full_eigenvalues: dict[str, np.ndarray],
    active_feature_counts: dict[str, int],
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
        full_eigenvalues[node_result.node_id] = (
            node_result.full_eigenvalues
            if node_result.full_eigenvalues is not None
            else node_result.eigenvalues
        )
        active_feature_counts[node_result.node_id] = int(
            node_result.active_feature_count
        )


def compute_spectral_decomposition(
    tree: nx.DiGraph,
    leaf_data: pd.DataFrame,
    *,
    minimum_projection_dimension: int = 1,
    feature_space: FeatureSpace | None = None,
    include_internal_barycenters: bool = False,
    internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
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
    include_internal_barycenters
        Opt-in tree-filtered spectral path. When True, descendant internal node
        distributions are appended to the node-local spectral matrix before
        eigendecomposition. These rows are deterministic barycenters of the
        leaves, so the default MP row-count mode keeps the descendant leaf count
        as the threshold row count.
    internal_distribution_mode
        Which opt-in internal rows to append. ``"empirical_barycenter"`` uses
        the stored leaf-count subtree barycenters. ``"branch_length_state"``
        estimates separate branch-length-aware latent internal states with
        child precision proportional to descendant support divided by edge
        length. The mode has no effect unless ``include_internal_barycenters``
        is true.
    mp_row_count_mode
        ``"leaf_effective_rows"`` keeps MP thresholding calibrated to descendant
        leaves even when internal rows are appended. ``"legacy_stacked_rows"``
        reproduces the commit-era stacked-row threshold as an explicit
        diagnostic comparator.

    Returns
    -------
    SpectralDecompositionResult
        Typed per-node spectral output. ``test_projection_dimensions_by_node``
        is the dimension used by downstream projected-Wald tests; raw MP signal
        counts, effective independent row counts, and MP threshold row counts
        are exposed separately.
    """
    spectral_start_sec = perf_counter()
    internal_distribution_mode = _validate_internal_distribution_mode(
        internal_distribution_mode
    )
    mp_row_count_mode = _validate_mp_row_count_mode(mp_row_count_mode)
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
    full_eigenvalues: Dict[str, np.ndarray] = {}
    active_feature_counts: Dict[str, int] = {}

    # Separate leaves (trivial) from internal nodes (expensive).
    internal_node_ids: list[str] = []
    for node_id in tree.nodes:
        if is_leaf(tree, node_id):
            test_projection_dimensions[node_id] = 0
            raw_mp_signal_counts[node_id] = 0
            effective_independent_rows[node_id] = 1
            mp_threshold_rows[node_id] = 1
            full_eigenvalues[node_id] = np.zeros(0, dtype=np.float64)
            active_feature_counts[node_id] = 0
        else:
            internal_node_ids.append(node_id)

    spectral_tasks = _build_spectral_tasks(
        tree,
        internal_node_ids,
        descendant_leaf_indices_by_node,
        feature_space=active_feature_space,
        include_internal_barycenters=bool(include_internal_barycenters),
        internal_distribution_mode=internal_distribution_mode,
        mp_row_count_mode=mp_row_count_mode,
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
        full_eigenvalues=full_eigenvalues,
        active_feature_counts=active_feature_counts,
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
        full_component_eigenvalues_by_node=full_eigenvalues,
        active_feature_counts_by_node=active_feature_counts,
        stage_timings=stage_timings,
    )


__all__ = [
    "INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE",
    "INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER",
    "INTERNAL_DISTRIBUTION_MODES",
    "MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS",
    "MP_ROW_COUNT_LEGACY_STACKED_ROWS",
    "MP_ROW_COUNT_MODES",
    "compute_spectral_decomposition",
]
