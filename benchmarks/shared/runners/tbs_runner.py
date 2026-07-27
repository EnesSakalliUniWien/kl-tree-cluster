"""TBS method runner.

Builds a PosetTree and performs TBS decomposition.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from tree_break_selection import config
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from tree_break_selection.hierarchy_analysis.statistics.distributional_action import (
    DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
    annotate_distributional_action_split_filter,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.neighborhood_bandwidth import (
    build_branch_length_distance_cache,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
)
from tree_break_selection.tree.feature_space import FeatureSpace
from tree_break_selection.tree.io import tree_from_linkage_topology
from tree_break_selection.tree.optimized_branch_lengths import (
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
    fit_fixed_topology_nnls_branch_lengths,
    validate_branch_length_optimization_method,
)
from tree_break_selection.tree.phylogenetic import (
    iqtree3_tree_from_alignment,
    neighbor_joining_tree_from_distance,
)
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import labels_and_report_from_decomposition
from benchmarks.shared.util.time import elapsed_since


def run_tbs_on_distance(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray | None,
    sibling_significance_level: float,
    *,
    tree_builder: str = "linkage",
    tree_rooting: str = "linkage_root",
    tree_linkage_method: str,
    iqtree_executable: str = "iqtree3",
    iqtree_model: str = "JC2",
    iqtree_threads: int = 1,
    iqtree_work_dir: str | None = None,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    feature_space: FeatureSpace | None = None,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    adaptive_projection_dimension_energy_fraction: float | None = None,
    spectral_include_internal_barycenters: bool = False,
    spectral_internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    spectral_mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    ),
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    enforce_internal_support_thresholds: bool = False,
    sibling_gate_profile: str | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_stability_tree_distance_metric: str = "hamming",
    root_stability_tree_linkage_method: str | None = None,
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    root_selective_permutation_guard_tree_distance_metric: str = "hamming",
    root_selective_permutation_guard_tree_linkage_method: str | None = None,
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
    neighborhood_bandwidth_profile: str | None = None,
    distributional_action_split_filter_policy: str = DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
    distributional_action_split_filter_quantile: float = 0.0,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_data_df: pd.DataFrame | None = None,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
    allow_linkage_ultrametric_branch_time: bool = False,
    passthrough: bool = config.PASSTHROUGH,
    trace_level: str = "full",
    extra: dict[str, object] | None = None,
) -> MethodRunResult:
    stage_timings: dict[str, float] = {}
    branch_length_optimization_method = validate_branch_length_optimization_method(
        branch_length_optimization_method
    )
    if (
        tree_builder == "linkage"
        and edge_branch_length_variance_policy != EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE
        and branch_length_optimization_method == BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC
        and not allow_linkage_ultrametric_branch_time
    ):
        raise ValueError(
            "Linkage branch-time variance requires recomputed fixed-topology branch "
            "lengths, for example branch_length_optimization_method='fixed_topology_nnls'. "
            "Raw linkage ultrametric heights are selected merge diagnostics, not "
            "calibrated stochastic time. Set allow_linkage_ultrametric_branch_time=True "
            "only for an explicit diagnostic/negative-control run."
        )

    tree_build_start_sec = perf_counter()
    linkage_matrix = None
    phylogenetic_rooting = None
    iqtree_metadata = None
    linkage_topology_only_branch_lengths = False
    linkage_topology_only_reason = None
    if tree_builder == "linkage":
        if distance_condensed is None:
            raise ValueError("Linkage TBS tree construction requires distance_condensed.")
        linkage_matrix = linkage(distance_condensed, method=tree_linkage_method)
        try:
            tree = PosetTree.from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
        except ValueError as exc:
            if (
                branch_length_optimization_method == BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS
                and "nondecreasing" in str(exc)
            ):
                tree = tree_from_linkage_topology(
                    linkage_matrix,
                    leaf_names=data_df.index.tolist(),
                    fallback_branch_length=1.0,
                )
                linkage_topology_only_branch_lengths = True
                linkage_topology_only_reason = str(exc)
            else:
                raise
    elif tree_builder == "neighbor_joining":
        if distance_condensed is None:
            raise ValueError("Neighbor-joining TBS tree construction requires distance_condensed.")
        tree, phylogenetic_rooting = neighbor_joining_tree_from_distance(
            distance_condensed,
            data_df.index.astype(str).tolist(),
            rooting=tree_rooting,
        )
    elif tree_builder == "iqtree3":
        tree, phylogenetic_rooting, iqtree_metadata = iqtree3_tree_from_alignment(
            data_df,
            executable=iqtree_executable,
            model=iqtree_model,
            threads=iqtree_threads,
            rooting=tree_rooting,
            work_dir=iqtree_work_dir,
        )
    else:
        raise ValueError(f"Unsupported TBS tree_builder: {tree_builder!r}.")
    stage_timings["tree_build_sec"] = elapsed_since(tree_build_start_sec)

    branch_length_optimization_metadata: dict[str, object] = {
        "branch_length_optimization_method": branch_length_optimization_method,
    }
    if branch_length_optimization_method == BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS:
        branch_data = data_df if branch_length_data_df is None else branch_length_data_df
        if not branch_data.index.equals(data_df.index):
            raise ValueError(
                "branch_length_data_df index must exactly match the original data index."
            )
        optimization_result = fit_fixed_topology_nnls_branch_lengths(
            tree,
            branch_data,
            target_metric=branch_length_optimization_target_metric,
            pair_sample_size=branch_length_optimization_pair_sample_size,
            random_state=branch_length_optimization_random_state,
            solver_tolerance=branch_length_optimization_solver_tolerance,
            max_iterations=branch_length_optimization_max_iterations,
        )
        stage_timings["branch_length_optimization_sec"] = optimization_result.elapsed_sec
        branch_length_optimization_metadata.update(
            {
                f"branch_length_optimization_{key}": value
                for key, value in optimization_result.to_dict().items()
            }
        )
        branch_length_optimization_metadata["branch_length_geometry_source"] = (
            "original_data" if branch_length_data_df is None else "aligned_geometry_embedding"
        )
    elif branch_length_optimization_method != BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC:
        raise ValueError(
            f"Unsupported branch_length_optimization_method {branch_length_optimization_method!r}."
        )

    neighborhood_bandwidth_metadata: dict[str, object] = {}
    if neighborhood_bandwidth_profile is not None:
        distance_cache = build_branch_length_distance_cache(tree)
        neighborhood_bandwidth_metadata = {
            "neighborhood_bandwidth_profile": str(neighborhood_bandwidth_profile),
            "neighborhood_distance_status": distance_cache.status,
            "neighborhood_distance_fallback_edge_length": (distance_cache.fallback_edge_length),
            "neighborhood_distance_pair_count": len(distance_cache.distances),
            "neighborhood_bandwidth_action": "support_regularizer_only_no_pvalue_rescue",
        }
    root_stability_replay_linkage = (
        tree_linkage_method
        if root_stability_tree_linkage_method is None
        else str(root_stability_tree_linkage_method)
    )
    root_selective_replay_linkage = (
        tree_linkage_method
        if root_selective_permutation_guard_tree_linkage_method is None
        else str(root_selective_permutation_guard_tree_linkage_method)
    )

    populate_start_sec = perf_counter()
    tree.populate_node_divergences(
        data_df,
        feature_space=feature_space,
    )
    stage_timings["populate_divergences_sec"] = elapsed_since(populate_start_sec)

    gate_annotation_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_significance_level,
        leaf_data=data_df,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        adaptive_projection_dimension_energy_fraction=(
            adaptive_projection_dimension_energy_fraction
        ),
        spectral_include_internal_barycenters=(spectral_include_internal_barycenters),
        spectral_internal_distribution_mode=str(spectral_internal_distribution_mode),
        spectral_mp_row_count_mode=str(spectral_mp_row_count_mode),
        continuous_covariance_policy=continuous_covariance_policy,
        continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
        enforce_internal_support_thresholds=bool(enforce_internal_support_thresholds),
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_stability_tree_distance_metric=str(root_stability_tree_distance_metric),
        root_stability_tree_linkage_method=root_stability_replay_linkage,
        root_selective_permutation_guard_replicates=(root_selective_permutation_guard_replicates),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        root_selective_permutation_guard_tree_distance_metric=str(
            root_selective_permutation_guard_tree_distance_metric
        ),
        root_selective_permutation_guard_tree_linkage_method=(root_selective_replay_linkage),
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(spectral_transport_unmatched_mode_penalty),
    )
    stage_timings.update(gate_annotation_bundle.stage_timings)
    resolved_gate_config = gate_annotation_bundle.metadata.config

    distributional_action_start_sec = perf_counter()
    filtered_annotations_df, distributional_action_metadata = (
        annotate_distributional_action_split_filter(
            tree,
            gate_annotation_bundle.annotated_df,
            data_df,
            policy=distributional_action_split_filter_policy,
            quantile=distributional_action_split_filter_quantile,
        )
    )
    stage_timings["distributional_action_split_filter_sec"] = elapsed_since(
        distributional_action_start_sec
    )
    gate_annotation_bundle = replace(
        gate_annotation_bundle,
        annotated_df=filtered_annotations_df,
        edge_gate_result=replace(
            gate_annotation_bundle.edge_gate_result,
            annotated_df=filtered_annotations_df,
        ),
    )

    decomposer = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=gate_annotation_bundle,
        leaf_data=data_df,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        adaptive_projection_dimension_energy_fraction=(
            adaptive_projection_dimension_energy_fraction
        ),
        spectral_include_internal_barycenters=(spectral_include_internal_barycenters),
        spectral_internal_distribution_mode=str(spectral_internal_distribution_mode),
        spectral_mp_row_count_mode=str(spectral_mp_row_count_mode),
        continuous_covariance_policy=continuous_covariance_policy,
        continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
        enforce_internal_support_thresholds=bool(enforce_internal_support_thresholds),
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_stability_tree_distance_metric=str(root_stability_tree_distance_metric),
        root_stability_tree_linkage_method=root_stability_replay_linkage,
        root_selective_permutation_guard_replicates=(root_selective_permutation_guard_replicates),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        root_selective_permutation_guard_tree_distance_metric=str(
            root_selective_permutation_guard_tree_distance_metric
        ),
        root_selective_permutation_guard_tree_linkage_method=(root_selective_replay_linkage),
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(spectral_transport_unmatched_mode_penalty),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_significance_level,
        passthrough=passthrough,
        trace_level=trace_level,
    )
    traversal_start_sec = perf_counter()
    decomposition = decomposer.decompose_tree()
    stage_timings["traversal_sec"] = elapsed_since(traversal_start_sec)
    tree.annotations_df = decomposer.annotations_df

    labels, report_df = labels_and_report_from_decomposition(
        decomposition,
        data_df.index.tolist(),
    )
    result_extra = {
        "tree": tree,
        "decomposition": decomposition,
        "traversal_trace": decomposition.get("traversal_trace", []),
        "full_edge_traversal_trace": decomposition.get(
            "full_edge_traversal_trace",
            [],
        ),
        "traversal_counters": decomposition.get("traversal_counters", {}),
        "annotations": tree.annotations_df,
        "gate_bundle": gate_annotation_bundle,
        "linkage_matrix": linkage_matrix,
        "tree_builder": str(tree_builder),
        "tree_rooting": str(tree_rooting),
        "linkage_topology_only_branch_lengths": bool(linkage_topology_only_branch_lengths),
        "linkage_topology_only_reason": linkage_topology_only_reason,
        "phylogenetic_rooting": phylogenetic_rooting,
        "iqtree_metadata": iqtree_metadata,
        "stage_timings": stage_timings,
        **branch_length_optimization_metadata,
        "spectral_minimum_dimension": int(resolved_gate_config.spectral_minimum_dimension),
        "adaptive_projection_dimension_energy_fraction": (
            resolved_gate_config.adaptive_projection_dimension_energy_fraction
        ),
        "spectral_include_internal_barycenters": bool(spectral_include_internal_barycenters),
        "spectral_internal_distribution_mode": str(
            resolved_gate_config.spectral_internal_distribution_mode
        ),
        "spectral_mp_row_count_mode": str(resolved_gate_config.spectral_mp_row_count_mode),
        "continuous_covariance_policy": str(resolved_gate_config.continuous_covariance_policy),
        "continuous_covariance_min_child_leaf_count": int(
            resolved_gate_config.continuous_covariance_min_child_leaf_count
        ),
        "edge_branch_length_variance_policy": str(
            resolved_gate_config.edge_branch_length_variance_policy
        ),
        "allow_linkage_ultrametric_branch_time": bool(allow_linkage_ultrametric_branch_time),
        "enforce_internal_support_thresholds": bool(
            resolved_gate_config.enforce_internal_support_thresholds
        ),
        "passthrough": bool(passthrough),
        "sibling_gate_profile": resolved_gate_config.sibling_gate_profile_id,
        "sibling_gate_method": str(resolved_gate_config.sibling_gate_method),
        "sibling_gate_alpha_penalty": float(resolved_gate_config.sibling_gate_alpha_penalty),
        "root_stability_guard_threshold": (resolved_gate_config.root_stability_guard_threshold),
        "root_stability_subsample_replicates": int(
            resolved_gate_config.root_stability_subsample_replicates
        ),
        "root_stability_feature_fraction": float(
            resolved_gate_config.root_stability_feature_fraction
        ),
        "root_stability_seed": int(resolved_gate_config.root_stability_seed),
        "root_stability_tree_distance_metric": str(
            resolved_gate_config.root_stability_tree_distance_metric
        ),
        "root_stability_tree_linkage_method": str(
            resolved_gate_config.root_stability_tree_linkage_method
        ),
        "root_selective_permutation_guard_replicates": int(
            resolved_gate_config.root_selective_permutation_guard_replicates
        ),
        "root_selective_permutation_guard_seed": int(
            resolved_gate_config.root_selective_permutation_guard_seed
        ),
        "root_selective_permutation_guard_alpha": (
            None
            if resolved_gate_config.root_selective_permutation_guard_alpha is None
            else float(resolved_gate_config.root_selective_permutation_guard_alpha)
        ),
        "root_selective_permutation_guard_scope": str(
            resolved_gate_config.root_selective_permutation_guard_scope
        ),
        "root_selective_permutation_guard_tree_distance_metric": str(
            resolved_gate_config.root_selective_permutation_guard_tree_distance_metric
        ),
        "root_selective_permutation_guard_tree_linkage_method": str(
            resolved_gate_config.root_selective_permutation_guard_tree_linkage_method
        ),
        "spectral_transport_passthrough_guard": bool(
            resolved_gate_config.spectral_transport_passthrough_guard
        ),
        "spectral_transport_max_cost": float(resolved_gate_config.spectral_transport_max_cost),
        "spectral_transport_require_mp_blocks": bool(
            resolved_gate_config.spectral_transport_require_mp_blocks
        ),
        "spectral_transport_block_log_tolerance": float(
            resolved_gate_config.spectral_transport_block_log_tolerance
        ),
        "spectral_transport_unmatched_mode_penalty": float(
            resolved_gate_config.spectral_transport_unmatched_mode_penalty
        ),
        **neighborhood_bandwidth_metadata,
        **distributional_action_metadata,
    }
    if extra:
        duplicate_extra_keys = sorted(set(result_extra).intersection(extra))
        if duplicate_extra_keys:
            raise ValueError(
                "TBS runner extra metadata must not override canonical result artifacts; "
                f"duplicate key(s): {duplicate_extra_keys!r}."
            )
        result_extra.update(extra)

    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomposition["num_clusters"]),
        report_df=report_df,
        status="ok",
        skip_reason=None,
        extra=result_extra,
    )


def _run_tbs_method(
    data_df: pd.DataFrame,
    distance_condensed: np.ndarray | None,
    sibling_significance_level: float,
    tree_linkage_method: str = config.TREE_LINKAGE_METHOD,
    *,
    tree_builder: str = "linkage",
    tree_rooting: str = "linkage_root",
    iqtree_executable: str = "iqtree3",
    iqtree_model: str = "JC2",
    iqtree_threads: int = 1,
    iqtree_work_dir: str | None = None,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    feature_space: FeatureSpace | None = None,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    adaptive_projection_dimension_energy_fraction: float | None = None,
    spectral_include_internal_barycenters: bool = False,
    spectral_internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    spectral_mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    ),
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    enforce_internal_support_thresholds: bool = False,
    sibling_gate_profile: str | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_stability_tree_distance_metric: str = "hamming",
    root_stability_tree_linkage_method: str | None = None,
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    root_selective_permutation_guard_tree_distance_metric: str = "hamming",
    root_selective_permutation_guard_tree_linkage_method: str | None = None,
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
    neighborhood_bandwidth_profile: str | None = None,
    distributional_action_split_filter_policy: str = DISTRIBUTIONAL_ACTION_SPLIT_FILTER_NONE,
    distributional_action_split_filter_quantile: float = 0.0,
    branch_length_optimization_method: str = BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    branch_length_optimization_target_metric: str = (
        BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN
    ),
    branch_length_optimization_pair_sample_size: int | None = 100_000,
    branch_length_optimization_random_state: int = 0,
    branch_length_optimization_solver_tolerance: float = 1e-6,
    branch_length_optimization_max_iterations: int | None = None,
    allow_linkage_ultrametric_branch_time: bool = False,
    passthrough: bool = config.PASSTHROUGH,
    trace_level: str = "full",
) -> MethodRunResult:
    return run_tbs_on_distance(
        data_df,
        distance_condensed,
        sibling_significance_level,
        tree_builder=tree_builder,
        tree_rooting=tree_rooting,
        tree_linkage_method=tree_linkage_method,
        iqtree_executable=iqtree_executable,
        iqtree_model=iqtree_model,
        iqtree_threads=iqtree_threads,
        iqtree_work_dir=iqtree_work_dir,
        edge_alpha=edge_alpha,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        adaptive_projection_dimension_energy_fraction=(
            adaptive_projection_dimension_energy_fraction
        ),
        spectral_include_internal_barycenters=(spectral_include_internal_barycenters),
        spectral_internal_distribution_mode=str(spectral_internal_distribution_mode),
        spectral_mp_row_count_mode=str(spectral_mp_row_count_mode),
        continuous_covariance_policy=continuous_covariance_policy,
        continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
        enforce_internal_support_thresholds=bool(enforce_internal_support_thresholds),
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_stability_tree_distance_metric=root_stability_tree_distance_metric,
        root_stability_tree_linkage_method=root_stability_tree_linkage_method,
        root_selective_permutation_guard_replicates=(root_selective_permutation_guard_replicates),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        root_selective_permutation_guard_tree_distance_metric=(
            root_selective_permutation_guard_tree_distance_metric
        ),
        root_selective_permutation_guard_tree_linkage_method=(
            root_selective_permutation_guard_tree_linkage_method
        ),
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(spectral_transport_unmatched_mode_penalty),
        neighborhood_bandwidth_profile=neighborhood_bandwidth_profile,
        distributional_action_split_filter_policy=distributional_action_split_filter_policy,
        distributional_action_split_filter_quantile=distributional_action_split_filter_quantile,
        branch_length_optimization_method=branch_length_optimization_method,
        branch_length_optimization_target_metric=branch_length_optimization_target_metric,
        branch_length_optimization_pair_sample_size=branch_length_optimization_pair_sample_size,
        branch_length_optimization_random_state=branch_length_optimization_random_state,
        branch_length_optimization_solver_tolerance=branch_length_optimization_solver_tolerance,
        branch_length_optimization_max_iterations=branch_length_optimization_max_iterations,
        allow_linkage_ultrametric_branch_time=allow_linkage_ultrametric_branch_time,
        passthrough=passthrough,
        trace_level=trace_level,
    )
