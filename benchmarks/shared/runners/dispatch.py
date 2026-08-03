"""Shared dispatch helper to run a registered clustering method."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tree_break_selection.hierarchy_analysis.decomposition.gates.spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from tree_break_selection.space_separation.diffusion import (
    PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS,
)
from tree_break_selection.tree.construction import DEFAULT_BINARY_TREE_DISTANCE_METRIC
from tree_break_selection.tree.continuous_distance import (
    CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC,
    CONTINUOUS_TREE_DISTANCE_METRIC,
    continuous_time_distance_condensed,
    standardized_euclidean_distance_condensed,
)
from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
)
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
)

from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.types import BenchmarkRunStatus, MethodRunResult
from benchmarks.shared.util.decomposition import _create_report_dataframe_from_labels
from benchmarks.shared.util.execution_mode import coerce_bool_param
from benchmarks.shared.util.method_sets import (
    TBS_DISTANCE_TREE_NNLS_METHODS,
    TBS_RUNNER_METHODS,
)


def _normalize_method_result(
    result: MethodRunResult,
    sample_index: pd.Index,
) -> MethodRunResult:
    """Normalize method outputs to the stable typed runner contract."""
    if result.status is BenchmarkRunStatus.OK:
        assert result.labels is not None
        labels = np.asarray(result.labels)
        if len(labels) != len(sample_index):
            raise ValueError(
                "Runner labels must align to input samples. "
                f"Got {len(labels)} labels for {len(sample_index)} samples."
            )
        return MethodRunResult(
            labels=labels,
            found_clusters=int(result.found_clusters),
            report_df=_create_report_dataframe_from_labels(labels, sample_index),
            status="ok",
            skip_reason=None,
            extra=result.extra,
        )

    if result.status is BenchmarkRunStatus.SKIP:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status=BenchmarkRunStatus.SKIP,
            skip_reason=result.skip_reason,
            extra=result.extra,
        )

    if result.status is BenchmarkRunStatus.UNSUPPORTED:
        return MethodRunResult(
            labels=None,
            found_clusters=0,
            report_df=None,
            status=BenchmarkRunStatus.UNSUPPORTED,
            skip_reason=None,
            extra=result.extra,
            unsupported_reason=result.unsupported_reason,
        )

    raise AssertionError(f"Unhandled benchmark run status: {result.status!r}.")


def _resolve_tbs_sibling_gate_method(
    *,
    params: Dict[str, Any],
    feature_space: FeatureSpace | None,
) -> str:
    """Resolve the active TBS sibling gate for the feature-family contract."""
    default_method = str(params.get("sibling_gate_method", "projected_wald_inflation"))
    continuous_method = params.get("continuous_sibling_gate_method")
    if (
        continuous_method is not None
        and feature_space is not None
        and feature_space.has_continuous_blocks
    ):
        return str(continuous_method)
    return default_method


def _tbs_branch_length_optimization_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the shared optimized-branch-length runner contract."""
    return {
        "branch_length_optimization_method": str(
            params.get(
                "branch_length_optimization_method",
                BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
            )
        ),
        "branch_length_optimization_target_metric": str(
            params.get(
                "branch_length_optimization_target_metric",
                BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
            )
        ),
        "branch_length_optimization_pair_sample_size": (
            None
            if params.get("branch_length_optimization_pair_sample_size") is None
            else int(params["branch_length_optimization_pair_sample_size"])
        ),
        "branch_length_optimization_random_state": int(
            params.get("branch_length_optimization_random_state", 0)
        ),
        "branch_length_optimization_solver_tolerance": float(
            params.get("branch_length_optimization_solver_tolerance", 1e-6)
        ),
        "branch_length_optimization_max_iterations": (
            None
            if params.get("branch_length_optimization_max_iterations") is None
            else int(params["branch_length_optimization_max_iterations"])
        ),
        "branch_length_optimization_apply_nonconverged": coerce_bool_param(
            params.get("branch_length_optimization_apply_nonconverged", False),
            name="branch_length_optimization_apply_nonconverged",
        ),
    }


def _skip_unsupported_hamming_diffusion(
    *,
    method_id: str,
    feature_space: FeatureSpace | None,
) -> MethodRunResult | None:
    """Return an explicit benchmark skip for Hamming-only diffusion on continuous data."""
    if feature_space is None or not feature_space.has_continuous_blocks:
        return None
    return MethodRunResult(
        labels=None,
        found_clusters=0,
        report_df=None,
        status="skip",
        skip_reason=(
            f"{method_id} uses Hamming diffusion and requires binary or one-hot "
            "feature matrices; continuous FeatureSpace inputs are unsupported."
        ),
        extra={
            "method_compatibility_status": "unsupported_feature_space",
            "method_compatibility_reason": "hamming_diffusion_requires_discrete_features",
        },
    )


def _skip_degenerate_pydiffmap_variable_bandwidth(
    *,
    data_df: pd.DataFrame,
    method_id: str,
    bandwidth_type: str | float | None,
) -> MethodRunResult | None:
    """Return an explicit benchmark skip for pydiffmap's zero-bandwidth case."""
    if bandwidth_type is None:
        return None
    values = data_df.to_numpy(dtype=float)
    _unique_rows, counts = np.unique(values, axis=0, return_counts=True)
    max_duplicate_count = int(counts.max(initial=0))
    if max_duplicate_count < PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS:
        return None
    return MethodRunResult(
        labels=None,
        found_clusters=0,
        report_df=None,
        status="skip",
        skip_reason=(
            f"{method_id} uses pydiffmap variable-bandwidth diffusion, but the "
            f"largest exact duplicate block has {max_duplicate_count} rows. "
            f"pydiffmap's internal NNKDE query size is "
            f"{PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS}, which can produce "
            "zero local bandwidths and non-finite diffusion weights."
        ),
        extra={
            "method_compatibility_status": "unsupported_duplicate_geometry",
            "method_compatibility_reason": "pydiffmap_variable_bandwidth_zero_bandwidth",
            "max_duplicate_count": max_duplicate_count,
            "pydiffmap_nnkde_query_size": PYDIFFMAP_VARIABLE_BANDWIDTH_KDE_NEIGHBORS,
        },
    )


def _optional_int_sequence(value: Any) -> tuple[int, ...] | None:
    """Parse optional integer-list method parameters from registry/env inputs."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped.replace(";", ",").split(",")
    return tuple(int(item) for item in value)


def _run_tbs_distance_tree_method(
    *,
    data_df: pd.DataFrame,
    method_id: str,
    params: Dict[str, Any],
    spec: Any,
    alpha: float,
    resolved_edge_alpha: float,
    distance_condensed: Optional[np.ndarray],
    feature_space: FeatureSpace | None,
    branch_length_data_df: pd.DataFrame | None = None,
) -> MethodRunResult:
    """Construct and run a distance-tree TBS sibling-gate method.

    ``branch_length_data_df`` is an explicit, per-caller decision: distance-tree
    methods default to ``None`` (linkage_ultrametric branch times need no
    geometry), while methods that fit fixed-topology NNLS branch lengths must
    pass their own geometry rather than relying on implicit reuse of the
    distributional feature matrix.
    """
    metric = str(params["tree_distance_metric"])
    if method_id == "tbs_iqtree3":
        tbs_distance_condensed = None
    elif distance_condensed is not None:
        # Use precomputed distance (e.g. SBM modularity distance).
        tbs_distance_condensed = np.asarray(distance_condensed, dtype=float)
    elif metric == CONTINUOUS_TREE_DISTANCE_METRIC:
        if feature_space is None:
            raise ValueError(
                "mahalanobis_time TBS tree distances require a continuous feature_space."
            )
        tbs_distance_condensed = continuous_time_distance_condensed(
            data_df.values,
            feature_space,
        )
    elif metric == CONTINUOUS_STANDARDIZED_EUCLIDEAN_TREE_DISTANCE_METRIC:
        if feature_space is None:
            raise ValueError(
                "standardized_euclidean TBS tree distances require a continuous feature_space."
            )
        tbs_distance_condensed = standardized_euclidean_distance_condensed(
            data_df.values,
            feature_space,
        )
    else:
        tbs_distance_condensed = pdist(data_df.values, metric=metric)
    result = spec.runner(
        data_df,
        tbs_distance_condensed,
        alpha,
        tree_linkage_method=str(params["tree_linkage_method"]),
        tree_builder=str(params.get("tree_builder", "linkage")),
        tree_rooting=str(params.get("tree_rooting", "linkage_root")),
        iqtree_executable=str(params.get("iqtree_executable", "iqtree3")),
        iqtree_model=str(params.get("iqtree_model", "JC2")),
        iqtree_threads=int(params.get("iqtree_threads", 1)),
        iqtree_work_dir=params.get("iqtree_work_dir"),
        edge_alpha=resolved_edge_alpha,
        feature_space=feature_space,
        branch_length_data_df=branch_length_data_df,
        spectral_minimum_dimension=int(
            params.get(
                "spectral_minimum_dimension",
                EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
            )
        ),
        adaptive_projection_dimension_energy_fraction=(
            None
            if params.get("adaptive_projection_dimension_energy_fraction") is None
            else float(params["adaptive_projection_dimension_energy_fraction"])
        ),
        spectral_include_internal_barycenters=coerce_bool_param(
            params.get("spectral_include_internal_barycenters", False),
            name="spectral_include_internal_barycenters",
        ),
        spectral_internal_distribution_mode=str(
            params.get(
                "spectral_internal_distribution_mode",
                "empirical_barycenter",
            )
        ),
        continuous_covariance_policy=str(
            params.get(
                "continuous_covariance_policy",
                DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
            )
        ),
        continuous_covariance_min_child_leaf_count=int(
            params.get(
                "continuous_covariance_min_child_leaf_count",
                DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
            )
        ),
        edge_branch_length_variance_policy=str(
            params.get("edge_branch_length_variance_policy", "none")
        ),
        enforce_internal_support_thresholds=coerce_bool_param(
            params.get("enforce_internal_support_thresholds", False),
            name="enforce_internal_support_thresholds",
        ),
        sibling_gate_profile=params.get("sibling_gate_profile"),
        sibling_gate_method=_resolve_tbs_sibling_gate_method(
            params=params,
            feature_space=feature_space,
        ),
        sibling_gate_alpha_penalty=float(params.get("sibling_gate_alpha_penalty", 1.0)),
        root_stability_guard_threshold=params.get("root_stability_guard_threshold"),
        root_stability_subsample_replicates=int(
            params.get("root_stability_subsample_replicates", 0)
        ),
        root_stability_feature_fraction=float(
            params.get("root_stability_feature_fraction", 0.8)
        ),
        root_stability_seed=int(params.get("root_stability_seed", 0)),
        root_stability_tree_distance_metric=str(
            params.get("root_stability_tree_distance_metric", "hamming")
        ),
        root_stability_tree_linkage_method=params.get("root_stability_tree_linkage_method"),
        root_selective_permutation_guard_replicates=int(
            params.get("root_selective_permutation_guard_replicates", 0)
        ),
        root_selective_permutation_guard_seed=int(
            params.get("root_selective_permutation_guard_seed", 0)
        ),
        root_selective_permutation_guard_alpha=params.get(
            "root_selective_permutation_guard_alpha"
        ),
        root_selective_permutation_guard_scope=str(
            params.get("root_selective_permutation_guard_scope", "root")
        ),
        root_selective_permutation_guard_tree_distance_metric=str(
            params.get(
                "root_selective_permutation_guard_tree_distance_metric",
                "hamming",
            )
        ),
        root_selective_permutation_guard_tree_linkage_method=params.get(
            "root_selective_permutation_guard_tree_linkage_method"
        ),
        spectral_transport_passthrough_guard=coerce_bool_param(
            params.get("spectral_transport_passthrough_guard", False),
            name="spectral_transport_passthrough_guard",
        ),
        spectral_transport_max_cost=float(
            params.get(
                "spectral_transport_max_cost",
                DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
            )
        ),
        spectral_transport_require_mp_blocks=coerce_bool_param(
            params.get("spectral_transport_require_mp_blocks", True),
            name="spectral_transport_require_mp_blocks",
        ),
        spectral_transport_block_log_tolerance=float(
            params.get(
                "spectral_transport_block_log_tolerance",
                DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
            )
        ),
        spectral_transport_unmatched_mode_penalty=float(
            params.get(
                "spectral_transport_unmatched_mode_penalty",
                DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
            )
        ),
        neighborhood_bandwidth_profile=params.get("neighborhood_bandwidth_profile"),
        **_tbs_branch_length_optimization_kwargs(params),
        allow_linkage_ultrametric_branch_time=coerce_bool_param(
            params.get("allow_linkage_ultrametric_branch_time", False),
            name="allow_linkage_ultrametric_branch_time",
        ),
        passthrough=coerce_bool_param(
            params.get("passthrough", True),
            name="passthrough",
        ),
        trace_level="compact",
    )
    return _normalize_method_result(result, data_df.index)


def run_clustering_result(
    data_df: pd.DataFrame,
    method_id: str,
    params: Dict[str, Any],
    seed: Optional[int] = None,
    *,
    significance_level: float | None = None,
    edge_alpha: float | None = None,
    distance_matrix: Optional[np.ndarray] = None,
    distance_condensed: Optional[np.ndarray] = None,
    feature_space: FeatureSpace | None = None,
) -> MethodRunResult:
    """Run one benchmark method and return a normalized ``MethodRunResult``.

    This is the canonical method dispatcher used by pipeline and benchmark helpers.
    """
    spec = METHOD_SPECS[method_id]
    alpha = DEFAULT_SIBLING_ALPHA if significance_level is None else float(significance_level)
    resolved_edge_alpha = DEFAULT_EDGE_ALPHA if edge_alpha is None else float(edge_alpha)
    if method_id == "tbs_diffusion":
        skipped = _skip_unsupported_hamming_diffusion(
            method_id=method_id,
            feature_space=feature_space,
        )
        if skipped is not None:
            return skipped
        result = spec.runner(
            data_df,
            alpha,
            k_neighbors=int(params["k_neighbors"]),
            diffusion_time=int(params["diffusion_time"]),
            tree_linkage_method=str(params["tree_linkage_method"]),
            feature_space=feature_space,
            edge_branch_length_variance_policy=str(
                params.get("edge_branch_length_variance_policy", "none")
            ),
            **_tbs_branch_length_optimization_kwargs(params),
        )
        return _normalize_method_result(result, data_df.index)
    if method_id in {"tbs_diffusion_adaptive", "tbs_diffusion_adaptive_nnls"}:
        if str(params["metric"]).lower() == "hamming":
            skipped = _skip_unsupported_hamming_diffusion(
                method_id=method_id,
                feature_space=feature_space,
            )
            if skipped is not None:
                return skipped
        skipped = _skip_degenerate_pydiffmap_variable_bandwidth(
            data_df=data_df,
            method_id=method_id,
            bandwidth_type=params["bandwidth_type"],
        )
        if skipped is not None:
            return skipped
        result = spec.runner(
            data_df,
            alpha,
            k_neighbors=int(params["k_neighbors"]),
            diffusion_time=int(params["diffusion_time"]),
            n_components=int(params["n_components"]),
            metric=str(params["metric"]),
            bandwidth_type=params["bandwidth_type"],
            epsilon=params["epsilon"],
            tree_linkage_method=str(params["tree_linkage_method"]),
            feature_space=feature_space,
            edge_branch_length_variance_policy=str(
                params.get("edge_branch_length_variance_policy", "none")
            ),
            **_tbs_branch_length_optimization_kwargs(params),
        )
        return _normalize_method_result(result, data_df.index)
    if method_id in {
        "tbs_diffusion_graphtools",
        "tbs_diffusion_graphtools_nnls",
        "tbs_diffusion_graphtools_adaptive_nnls",
    }:
        result = spec.runner(
            data_df,
            alpha,
            k_neighbors=int(params["k_neighbors"]),
            diffusion_time=int(params["diffusion_time"]),
            n_components=int(params["n_components"]),
            metric=str(params["metric"]),
            decay=None if params.get("decay") is None else int(params["decay"]),
            anisotropy=float(params["anisotropy"]),
            kernel_symm=str(params["kernel_symm"]),
            random_state=int(params.get("random_state", 0)),
            adaptive_neighbor_profile=params.get("adaptive_neighbor_profile"),
            adaptive_neighbor_grid=_optional_int_sequence(params.get("adaptive_neighbor_grid")),
            tree_builder=str(params["tree_builder"]),
            tree_rooting=str(params["tree_rooting"]),
            tree_linkage_method=str(params["tree_linkage_method"]),
            feature_space=feature_space,
            branch_length_data_df=data_df,
            edge_branch_length_variance_policy=str(
                params.get("edge_branch_length_variance_policy", "none")
            ),
            **_tbs_branch_length_optimization_kwargs(params),
        )
        return _normalize_method_result(result, data_df.index)

    if method_id in TBS_DISTANCE_TREE_NNLS_METHODS:
        return _run_tbs_distance_tree_method(
            data_df=data_df,
            method_id=method_id,
            params=params,
            spec=spec,
            alpha=alpha,
            resolved_edge_alpha=resolved_edge_alpha,
            distance_condensed=distance_condensed,
            feature_space=feature_space,
            branch_length_data_df=data_df,
        )

    if method_id in TBS_RUNNER_METHODS:
        return _run_tbs_distance_tree_method(
            data_df=data_df,
            method_id=method_id,
            params=params,
            spec=spec,
            alpha=alpha,
            resolved_edge_alpha=resolved_edge_alpha,
            distance_condensed=distance_condensed,
            feature_space=feature_space,
        )

    if method_id in {"kmeans", "spectral"}:
        int(params["n_clusters"])
        result = spec.runner(data_df.values, params, seed)
        return _normalize_method_result(result, data_df.index)

    if distance_matrix is None:
        if distance_condensed is None:
            dm_condensed = pdist(data_df.values, metric=DEFAULT_BINARY_TREE_DISTANCE_METRIC)
        else:
            dm_condensed = np.asarray(distance_condensed, dtype=float)
        dm_square = squareform(dm_condensed)
    else:
        dm_square = np.asarray(distance_matrix, dtype=float)

    if method_id in {"leiden", "louvain", "optics"}:
        result = spec.runner(dm_square, params, seed)
    else:
        result = spec.runner(dm_square, params)
    return _normalize_method_result(result, data_df.index)


__all__ = ["run_clustering_result"]
