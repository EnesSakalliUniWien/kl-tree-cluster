"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    validate_edge_branch_length_variance_policy,
)
from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    validate_continuous_covariance_min_child_leaf_count,
    validate_continuous_covariance_policy,
)
from tree_break_selection.tree.feature_space import (
    FeatureSpace,
    infer_feature_space_from_columns,
)

from ...statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence,
)
from ...statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from ...statistics.projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
)
from ...statistics.sibling_divergence.fixed_subspace_annotation import (
    FIXED_SUBSPACE_SIBLING_GATE_METHODS,
    annotate_fixed_subspace_sibling_divergence,
    annotate_fixed_subspace_sibling_evidence_channels,
)
from ...statistics.sibling_divergence.inflated_projected_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)
from ...statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
)
from ...statistics.sibling_divergence.inflation_correction.external_selected_tail_calibration import (
    ExternalSelectedTailCalibrationModel,
)
from ...statistics.sibling_divergence.inflation_correction.types.inflation_model import (
    CalibrationSupportThresholds,
)
from ...statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from ...statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from .annotation_bundle import (
    EdgeGateResult,
    GateAnnotationBundle,
    GateAnnotationConfigMetadata,
    GateAnnotationMetadata,
    GateMetadata,
)
from .column_contracts import (
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
)
from .guards import (
    apply_root_selective_permutation_guard,
    apply_root_stability_guard,
    compute_root_feature_subsample_stability,
    selected_global_sibling_min_permutation_p_value,
    selected_root_permutation_p_value,
    validate_root_selective_permutation_guard_config,
    validate_root_stability_guard_config,
)
from .profiles import (
    SIBLING_GATE_PROFILES,
    SiblingGateProfile,
    resolve_sibling_gate_profile,
    resolve_sibling_gate_profile_config,
)
from .spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
    annotate_spectral_transport_passthrough_support,
)


@dataclass(frozen=True)
class _SiblingGateInputs:
    projection_dimensions_from_edge_comparisons: dict[str, int]
    parent_principal_component_projections: dict[str, np.ndarray]
    parent_principal_component_eigenvalues: dict[str, np.ndarray]


def _build_edge_metadata(
    *,
    edge_alpha: float,
) -> GateMetadata:
    """Build metadata for edge-gate output.

    Tree-BH is the only supported FDR method, so not stored in metadata.
    """
    return GateMetadata(gate="edge", alpha=float(edge_alpha))


def _build_sibling_metadata(
    *,
    sibling_alpha: float,
) -> GateMetadata:
    """Build metadata for sibling-gate output."""
    return GateMetadata(gate="sibling", alpha=float(sibling_alpha))


def resolve_effective_sibling_alpha(
    sibling_alpha: float,
    sibling_gate_alpha_penalty: float,
) -> float:
    alpha = float(sibling_alpha)
    penalty = float(sibling_gate_alpha_penalty)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"sibling_alpha must lie in (0, 1); got {alpha!r}.")
    if penalty <= 0.0 or not math.isfinite(penalty):
        raise ValueError(
            f"sibling_gate_alpha_penalty must be finite and positive; got {penalty!r}."
        )
    effective = alpha / penalty
    if not 0.0 < effective < 1.0:
        raise ValueError(
            "Effective sibling alpha must lie in (0, 1); "
            f"got sibling_alpha={alpha!r}, penalty={penalty!r}, "
            f"effective={effective!r}."
        )
    return float(effective)


def _support_thresholds_signature(
    thresholds: CalibrationSupportThresholds,
) -> tuple[tuple[str, float | int], ...]:
    return tuple((field, getattr(thresholds, field)) for field in thresholds.__dataclass_fields__)


def build_gate_annotation_config_metadata(
    *,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    adaptive_projection_dimension_energy_fraction: float | None = None,
    spectral_include_internal_barycenters: bool = False,
    spectral_internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    ),
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    sibling_gate_profile_id: str | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_stability_tree_distance_metric: str = "hamming",
    root_stability_tree_linkage_method: str = "average",
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    root_selective_permutation_guard_tree_distance_metric: str = "hamming",
    root_selective_permutation_guard_tree_linkage_method: str = "average",
    enforce_internal_support_thresholds: bool = False,
    internal_support_thresholds: CalibrationSupportThresholds = (
        DEFAULT_INTERNAL_SUPPORT_THRESHOLDS
    ),
    external_selected_tail_model: ExternalSelectedTailCalibrationModel | None = None,
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
) -> GateAnnotationConfigMetadata:
    """Capture config values that affect gate annotation outputs."""
    return GateAnnotationConfigMetadata(
        spectral_minimum_dimension=int(spectral_minimum_dimension),
        adaptive_projection_dimension_energy_fraction=(
            None
            if adaptive_projection_dimension_energy_fraction is None
            else float(adaptive_projection_dimension_energy_fraction)
        ),
        spectral_include_internal_barycenters=bool(spectral_include_internal_barycenters),
        spectral_internal_distribution_mode=str(spectral_internal_distribution_mode),
        continuous_covariance_policy=validate_continuous_covariance_policy(
            continuous_covariance_policy
        ),
        continuous_covariance_min_child_leaf_count=(
            validate_continuous_covariance_min_child_leaf_count(
                continuous_covariance_min_child_leaf_count
            )
        ),
        edge_branch_length_variance_policy=(
            validate_edge_branch_length_variance_policy(edge_branch_length_variance_policy)
        ),
        sibling_gate_profile_id=(
            None if sibling_gate_profile_id is None else str(sibling_gate_profile_id)
        ),
        sibling_gate_method=str(sibling_gate_method),
        sibling_gate_alpha_penalty=float(sibling_gate_alpha_penalty),
        root_stability_guard_threshold=(
            None
            if root_stability_guard_threshold is None
            else float(root_stability_guard_threshold)
        ),
        root_stability_subsample_replicates=int(root_stability_subsample_replicates),
        root_stability_feature_fraction=float(root_stability_feature_fraction),
        root_stability_seed=int(root_stability_seed),
        root_stability_tree_distance_metric=str(root_stability_tree_distance_metric),
        root_stability_tree_linkage_method=str(root_stability_tree_linkage_method),
        root_selective_permutation_guard_replicates=int(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=int(root_selective_permutation_guard_seed),
        root_selective_permutation_guard_alpha=(
            None
            if root_selective_permutation_guard_alpha is None
            else float(root_selective_permutation_guard_alpha)
        ),
        root_selective_permutation_guard_scope=str(root_selective_permutation_guard_scope),
        root_selective_permutation_guard_tree_distance_metric=str(
            root_selective_permutation_guard_tree_distance_metric
        ),
        root_selective_permutation_guard_tree_linkage_method=str(
            root_selective_permutation_guard_tree_linkage_method
        ),
        enforce_internal_support_thresholds=bool(enforce_internal_support_thresholds),
        internal_support_thresholds_signature=_support_thresholds_signature(
            internal_support_thresholds
        ),
        external_selected_tail_calibration_enabled=(external_selected_tail_model is not None),
        external_selected_tail_rule_count=(
            0
            if external_selected_tail_model is None
            else int(len(external_selected_tail_model.rules))
        ),
        spectral_transport_passthrough_guard=bool(spectral_transport_passthrough_guard),
        spectral_transport_max_cost=float(spectral_transport_max_cost),
        spectral_transport_require_mp_blocks=bool(spectral_transport_require_mp_blocks),
        spectral_transport_block_log_tolerance=float(spectral_transport_block_log_tolerance),
        spectral_transport_unmatched_mode_penalty=float(spectral_transport_unmatched_mode_penalty),
    )


def _resolve_sibling_gate_inputs(
    tree,
    edge_gate_result: EdgeGateResult,
) -> _SiblingGateInputs:
    """Resolve sibling-gate inputs from edge-gate context."""
    resolved_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=edge_gate_result.spectral_context,
        )
    )
    (
        resolved_parent_principal_component_projections,
        resolved_parent_principal_component_eigenvalues,
    ) = collect_parent_principal_component_inputs_for_sibling_tests(
        resolved_projection_dimensions_from_edge_comparisons,
        spectral_context=edge_gate_result.spectral_context,
    )
    expected_parent_keys = set(resolved_projection_dimensions_from_edge_comparisons)
    projection_keys = set(resolved_parent_principal_component_projections)
    eigenvalue_keys = set(resolved_parent_principal_component_eigenvalues)
    if projection_keys != expected_parent_keys or eigenvalue_keys != expected_parent_keys:
        raise ValueError(
            "Sibling-gate parent PCA inputs must be keyed exactly by sibling projection parents. "
            f"expected={sorted(expected_parent_keys)!r}, "
            f"projection_keys={sorted(projection_keys)!r}, "
            f"eigenvalue_keys={sorted(eigenvalue_keys)!r}."
        )

    return _SiblingGateInputs(
        projection_dimensions_from_edge_comparisons=(
            resolved_projection_dimensions_from_edge_comparisons
        ),
        parent_principal_component_projections=(resolved_parent_principal_component_projections),
        parent_principal_component_eigenvalues=(resolved_parent_principal_component_eigenvalues),
    )


def _resolve_fixed_sibling_gate_feature_space(
    *,
    feature_space: FeatureSpace | None,
    leaf_data: pd.DataFrame | None,
) -> FeatureSpace:
    if feature_space is not None:
        return feature_space
    if leaf_data is None:
        raise ValueError(
            "Fixed-subspace sibling gates require feature_space metadata or leaf_data "
            "columns for feature-space inference."
        )
    return infer_feature_space_from_columns(tuple(leaf_data.columns))


def _resolve_edge_spectral_minimum_dimension(
    *,
    spectral_minimum_dimension: int,
) -> int:
    """Validate the edge projection floor.

    The edge gate owns its projection floor. Fixed-subspace sibling evidence may
    still be computed as diagnostic channels, but it must not silently force the
    edge gate into a full-rank test.
    """
    requested_minimum = int(spectral_minimum_dimension)
    if requested_minimum < 0:
        raise ValueError(
            f"spectral_minimum_dimension must be non-negative; got {spectral_minimum_dimension!r}."
        )
    return requested_minimum


def run_gate_annotation_pipeline(
    tree,
    annotations_df: pd.DataFrame,
    *,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    feature_space: FeatureSpace | None = None,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    adaptive_projection_dimension_energy_fraction: float | None = None,
    spectral_include_internal_barycenters: bool = False,
    spectral_internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    ),
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
    sibling_gate_profile: str | SiblingGateProfile | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_stability_tree_distance_metric: str = "hamming",
    root_stability_tree_linkage_method: str = "average",
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    root_selective_permutation_guard_tree_distance_metric: str = "hamming",
    root_selective_permutation_guard_tree_linkage_method: str = "average",
    enforce_internal_support_thresholds: bool = False,
    internal_support_thresholds: CalibrationSupportThresholds = (
        DEFAULT_INTERNAL_SUPPORT_THRESHOLDS
    ),
    external_selected_tail_model: ExternalSelectedTailCalibrationModel | None = None,
    external_selected_tail_context_by_parent: Mapping[object, Mapping[str, object]] | None = None,
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
) -> GateAnnotationBundle:
    """Run the edge-gate and sibling-gate annotation pipeline.

    The edge-divergence gate uses Tree-BH (Tree-structured Benjamini-Hochberg) for FDR
    correction. This is the only supported edge multiple-testing method.
    """
    stage_timings = {
        "edge_gate_contrast_covariance_sec": 0.0,
        "edge_gate_projection_sec": 0.0,
        "edge_gate_wald_statistic_sec": 0.0,
        "edge_gate_tree_bh_sec": 0.0,
        "sibling_gate_pair_record_collection_sec": 0.0,
        "sibling_gate_inflation_fit_sec": 0.0,
        "sibling_gate_adjusted_tests_sec": 0.0,
        "sibling_gate_fdr_sec": 0.0,
        "root_stability_guard_sec": 0.0,
        "root_selective_permutation_guard_sec": 0.0,
        "spectral_transport_passthrough_guard_sec": 0.0,
    }
    (
        sibling_gate_profile_id,
        sibling_gate_method,
        sibling_gate_alpha_penalty,
        root_stability_guard_threshold,
        root_stability_subsample_replicates,
        root_stability_feature_fraction,
        root_stability_seed,
        root_selective_permutation_guard_replicates,
        root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha,
        root_selective_permutation_guard_scope,
        spectral_transport_passthrough_guard,
        spectral_transport_max_cost,
        spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile=sibling_gate_profile,
        sibling_gate_method=sibling_gate_method,
        sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
        root_stability_guard_threshold=root_stability_guard_threshold,
        root_stability_subsample_replicates=root_stability_subsample_replicates,
        root_stability_feature_fraction=root_stability_feature_fraction,
        root_stability_seed=root_stability_seed,
        root_selective_permutation_guard_replicates=(root_selective_permutation_guard_replicates),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=(root_selective_permutation_guard_alpha),
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(spectral_transport_unmatched_mode_penalty),
    )
    continuous_covariance_policy = validate_continuous_covariance_policy(
        continuous_covariance_policy
    )
    continuous_covariance_min_child_leaf_count = (
        validate_continuous_covariance_min_child_leaf_count(
            continuous_covariance_min_child_leaf_count
        )
    )
    edge_branch_length_variance_policy = validate_edge_branch_length_variance_policy(
        edge_branch_length_variance_policy
    )
    if sibling_gate_method not in {
        "projected_wald_inflation",
        *FIXED_SUBSPACE_SIBLING_GATE_METHODS,
    }:
        raise ValueError(
            "Unknown sibling_gate_method "
            f"{sibling_gate_method!r}; allowed="
            f"{('projected_wald_inflation', *FIXED_SUBSPACE_SIBLING_GATE_METHODS)!r}."
        )
    effective_sibling_alpha = resolve_effective_sibling_alpha(
        sibling_alpha,
        sibling_gate_alpha_penalty,
    )
    validate_root_stability_guard_config(
        threshold=root_stability_guard_threshold,
        subsample_replicates=root_stability_subsample_replicates,
        feature_fraction=root_stability_feature_fraction,
    )
    effective_root_selective_alpha = (
        float(sibling_alpha)
        if root_selective_permutation_guard_alpha is None
        else float(root_selective_permutation_guard_alpha)
    )
    validate_root_selective_permutation_guard_config(
        replicates=root_selective_permutation_guard_replicates,
        alpha=effective_root_selective_alpha,
        scope=root_selective_permutation_guard_scope,
    )
    if (
        int(root_selective_permutation_guard_replicates) > 0
        and sibling_gate_method == "projected_wald_inflation"
    ):
        raise ValueError(
            "Selected-root permutation guard requires a fixed-subspace sibling "
            "gate; projected_wald_inflation would reintroduce adaptive "
            "projection into the guarded root statistic."
        )
    effective_spectral_minimum_dimension = _resolve_edge_spectral_minimum_dimension(
        spectral_minimum_dimension=spectral_minimum_dimension,
    )
    adaptive_projection_fraction = (
        None
        if adaptive_projection_dimension_energy_fraction is None
        else float(adaptive_projection_dimension_energy_fraction)
    )
    if adaptive_projection_fraction is not None and not 0.0 < adaptive_projection_fraction <= 1.0:
        raise ValueError(
            "adaptive_projection_dimension_energy_fraction must lie in (0, 1] "
            f"when set; got {adaptive_projection_dimension_energy_fraction!r}."
        )
    adaptive_projection_basis_dimension = None
    if adaptive_projection_fraction is not None:
        adaptive_feature_space = _resolve_fixed_sibling_gate_feature_space(
            feature_space=feature_space,
            leaf_data=leaf_data,
        )
        adaptive_projection_basis_dimension = int(adaptive_feature_space.contrast_dimension)

    # Run edge-divergence gate: child-parent edge tests
    edge_gate_start_sec = perf_counter()
    edge_annotated_df, spectral_context = annotate_child_parent_divergence(
        tree,
        annotations_df,
        significance_level_alpha=edge_alpha,
        leaf_data=leaf_data,
        feature_space=feature_space,
        spectral_minimum_dimension=effective_spectral_minimum_dimension,
        spectral_projection_basis_dimension=adaptive_projection_basis_dimension,
        spectral_include_internal_barycenters=(spectral_include_internal_barycenters),
        spectral_internal_distribution_mode=str(spectral_internal_distribution_mode),
        continuous_covariance_policy=continuous_covariance_policy,
        continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
        edge_branch_length_variance_policy=edge_branch_length_variance_policy,
        adaptive_projection_dimension_energy_fraction=adaptive_projection_fraction,
        stage_timings=stage_timings,
    )
    edge_gate_sec = float(perf_counter() - edge_gate_start_sec)
    stage_timings.update(spectral_context.stage_timings)
    validate_edge_gate_columns(edge_annotated_df)
    edge_metadata = _build_edge_metadata(
        edge_alpha=edge_alpha,
    )
    edge_gate_result = EdgeGateResult(
        annotated_df=edge_annotated_df,
        spectral_context=spectral_context,
        metadata=edge_metadata,
    )

    # Run sibling-divergence gate
    sibling_gate_start_sec = perf_counter()
    if sibling_gate_method == "projected_wald_inflation":
        sibling_inputs = _resolve_sibling_gate_inputs(
            tree,
            edge_gate_result,
        )
        annotated_df = annotate_sibling_divergence(
            tree,
            edge_annotated_df,
            significance_level_alpha=effective_sibling_alpha,
            sibling_projection_dimensions_from_edge_comparisons=(
                sibling_inputs.projection_dimensions_from_edge_comparisons
            ),
            parent_principal_component_projections=(
                sibling_inputs.parent_principal_component_projections
            ),
            parent_principal_component_eigenvalues=(
                sibling_inputs.parent_principal_component_eigenvalues
            ),
            feature_space=feature_space,
            continuous_covariance_policy=continuous_covariance_policy,
            continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
            adaptive_projection_dimension_energy_fraction=adaptive_projection_fraction,
            enforce_support_thresholds=enforce_internal_support_thresholds,
            support_thresholds=internal_support_thresholds,
            external_selected_tail_model=external_selected_tail_model,
            external_selected_tail_context_by_parent=(external_selected_tail_context_by_parent),
            stage_timings=stage_timings,
        )
    else:
        fixed_feature_space = _resolve_fixed_sibling_gate_feature_space(
            feature_space=feature_space,
            leaf_data=leaf_data,
        )
        annotated_df = annotate_fixed_subspace_sibling_divergence(
            tree,
            edge_annotated_df,
            significance_level_alpha=effective_sibling_alpha,
            feature_space=fixed_feature_space,
            method=sibling_gate_method,
            continuous_covariance_policy=continuous_covariance_policy,
            continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
        )
    sibling_gate_sec = float(perf_counter() - sibling_gate_start_sec)
    channel_feature_space = (
        feature_space
        if feature_space is not None
        else infer_feature_space_from_columns(tuple(leaf_data.columns))
        if leaf_data is not None
        else None
    )
    if channel_feature_space is not None:
        sparse_channel_method = (
            sibling_gate_method
            if sibling_gate_method in {"fixed_coordinate_bh", "fixed_block_bh"}
            else "fixed_coordinate_bh"
        )
        annotated_df = annotate_fixed_subspace_sibling_evidence_channels(
            tree,
            annotated_df,
            feature_space=channel_feature_space,
            sparse_method=sparse_channel_method,  # type: ignore[arg-type]
            dense_method="fixed_global_chi_square",
            continuous_covariance_policy=continuous_covariance_policy,
            continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
        )
    if root_stability_guard_threshold is not None:
        root_guard_start_sec = perf_counter()
        if leaf_data is None:
            raise ValueError("Root stability guard requires leaf_data.")
        guard_feature_space = (
            feature_space
            if feature_space is not None
            else infer_feature_space_from_columns(tuple(leaf_data.columns))
        )
        root_stability = compute_root_feature_subsample_stability(
            tree,
            leaf_data,
            guard_feature_space,
            subsample_replicates=root_stability_subsample_replicates,
            feature_fraction=root_stability_feature_fraction,
            seed=root_stability_seed,
            tree_distance_metric=root_stability_tree_distance_metric,
            tree_linkage_method=root_stability_tree_linkage_method,
        )
        annotated_df = apply_root_stability_guard(
            tree,
            annotated_df,
            root_stability,
            threshold=root_stability_guard_threshold,
        )
        stage_timings["root_stability_guard_sec"] = float(perf_counter() - root_guard_start_sec)
    if int(root_selective_permutation_guard_replicates) > 0:
        root_selective_start_sec = perf_counter()
        if leaf_data is None:
            raise ValueError("Selected-root permutation guard requires leaf_data.")
        guard_feature_space = (
            feature_space
            if feature_space is not None
            else infer_feature_space_from_columns(tuple(leaf_data.columns))
        )
        annotated_df = apply_root_selective_permutation_guard(
            tree,
            annotated_df,
            leaf_data,
            guard_feature_space,
            method=sibling_gate_method,
            bootstrap_replicates=int(root_selective_permutation_guard_replicates),
            seed=int(root_selective_permutation_guard_seed),
            alpha=effective_root_selective_alpha,
            scope=root_selective_permutation_guard_scope,
            tree_distance_metric=root_selective_permutation_guard_tree_distance_metric,
            tree_linkage_method=root_selective_permutation_guard_tree_linkage_method,
        )
        stage_timings["root_selective_permutation_guard_sec"] = float(
            perf_counter() - root_selective_start_sec
        )
    if bool(spectral_transport_passthrough_guard):
        spectral_transport_start_sec = perf_counter()
        annotated_df = annotate_spectral_transport_passthrough_support(
            tree,
            annotated_df,
            spectral_context,
            max_cost=float(spectral_transport_max_cost),
            require_mp_blocks=bool(spectral_transport_require_mp_blocks),
            eigenvalue_block_log_tolerance=float(spectral_transport_block_log_tolerance),
            unmatched_mode_penalty=float(spectral_transport_unmatched_mode_penalty),
        )
        stage_timings["spectral_transport_passthrough_guard_sec"] = float(
            perf_counter() - spectral_transport_start_sec
        )
    validate_edge_gate_columns(
        annotated_df,
        error_context="Sibling gate input/output edge columns differ from required contract",
    )
    validate_sibling_gate_columns(annotated_df)
    sibling_metadata = _build_sibling_metadata(
        sibling_alpha=sibling_alpha,
    )

    metadata = GateAnnotationMetadata(
        pipeline="gate_annotation",
        edge=edge_metadata,
        sibling=sibling_metadata,
        config=build_gate_annotation_config_metadata(
            spectral_minimum_dimension=effective_spectral_minimum_dimension,
            adaptive_projection_dimension_energy_fraction=(adaptive_projection_fraction),
            spectral_include_internal_barycenters=(spectral_include_internal_barycenters),
            spectral_internal_distribution_mode=str(spectral_internal_distribution_mode),
            continuous_covariance_policy=continuous_covariance_policy,
            continuous_covariance_min_child_leaf_count=(continuous_covariance_min_child_leaf_count),
            edge_branch_length_variance_policy=edge_branch_length_variance_policy,
            sibling_gate_profile_id=sibling_gate_profile_id,
            sibling_gate_method=sibling_gate_method,
            sibling_gate_alpha_penalty=sibling_gate_alpha_penalty,
            root_stability_guard_threshold=root_stability_guard_threshold,
            root_stability_subsample_replicates=root_stability_subsample_replicates,
            root_stability_feature_fraction=root_stability_feature_fraction,
            root_stability_seed=root_stability_seed,
            root_stability_tree_distance_metric=root_stability_tree_distance_metric,
            root_stability_tree_linkage_method=root_stability_tree_linkage_method,
            root_selective_permutation_guard_replicates=(
                root_selective_permutation_guard_replicates
            ),
            root_selective_permutation_guard_seed=(root_selective_permutation_guard_seed),
            root_selective_permutation_guard_alpha=(root_selective_permutation_guard_alpha),
            root_selective_permutation_guard_scope=(root_selective_permutation_guard_scope),
            root_selective_permutation_guard_tree_distance_metric=(
                root_selective_permutation_guard_tree_distance_metric
            ),
            root_selective_permutation_guard_tree_linkage_method=(
                root_selective_permutation_guard_tree_linkage_method
            ),
            enforce_internal_support_thresholds=enforce_internal_support_thresholds,
            internal_support_thresholds=internal_support_thresholds,
            external_selected_tail_model=external_selected_tail_model,
            spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
            spectral_transport_max_cost=spectral_transport_max_cost,
            spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
            spectral_transport_block_log_tolerance=(spectral_transport_block_log_tolerance),
            spectral_transport_unmatched_mode_penalty=(spectral_transport_unmatched_mode_penalty),
        ),
    )

    stage_timings["edge_gate_sec"] = edge_gate_sec
    stage_timings["sibling_gate_sec"] = sibling_gate_sec

    return GateAnnotationBundle(
        annotated_df=annotated_df,
        metadata=metadata,
        edge_gate_result=edge_gate_result,
        stage_timings=stage_timings,
    )


__all__ = [
    "SIBLING_GATE_PROFILES",
    "SiblingGateProfile",
    "apply_root_selective_permutation_guard",
    "apply_root_stability_guard",
    "build_gate_annotation_config_metadata",
    "compute_root_feature_subsample_stability",
    "resolve_effective_sibling_alpha",
    "resolve_sibling_gate_profile",
    "resolve_sibling_gate_profile_config",
    "run_gate_annotation_pipeline",
    "selected_global_sibling_min_permutation_p_value",
    "selected_root_permutation_p_value",
]
