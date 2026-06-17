"""Top-level gate annotation orchestration wrapper."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.core_utils.tree_utils import bottom_up_nodes
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    infer_feature_space_from_columns,
)

from ...statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from ...statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from ...statistics.contrast_covariance import (
    build_contrast_covariance,
    compute_whitened_wald_contrast,
)
from ...statistics.sibling_divergence.fixed_subspace_annotation import (
    FIXED_SUBSPACE_SIBLING_GATE_METHODS,
    annotate_fixed_subspace_sibling_divergence,
    fixed_coordinate_bh_p_value,
    fixed_subspace_sibling_p_value,
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
from ...statistics.sibling_divergence.pair_testing.collection.pair_observations import (
    identify_binary_sibling_children,
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
    GateAnnotationLeafDataMetadata,
    GateAnnotationMetadata,
    GateMetadata,
)
from .column_contracts import (
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
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


@dataclass(frozen=True)
class SiblingGateProfile:
    """Named sibling-gate configuration for auditable method candidates."""

    profile_id: str
    sibling_gate_method: str
    sibling_gate_alpha_penalty: float
    root_stability_guard_threshold: float | None
    root_stability_subsample_replicates: int
    root_stability_feature_fraction: float
    root_stability_seed: int
    status: str
    description: str
    root_selective_permutation_guard_replicates: int = 0
    root_selective_permutation_guard_seed: int = 0
    root_selective_permutation_guard_alpha: float | None = None
    root_selective_permutation_guard_scope: str = "root"
    spectral_transport_passthrough_guard: bool = False
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST
    spectral_transport_require_mp_blocks: bool = True
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    )
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    )


GLOBAL_PASSTHROUGH_REFINED_REPLICATES = 999


SIBLING_GATE_PROFILES: dict[str, SiblingGateProfile] = {
    "fixed_coordinate_conditional_topology_diagnostic_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_conditional_topology_diagnostic_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_only_not_production",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty and selected-root feature-subsample stability guard. "
            "Conditional topology traversal-law evidence is emitted by the "
            "diagnostic panel, not applied as a production calibration rule."
        ),
    ),
    "fixed_coordinate_guarded_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_guarded_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty and selected-root feature-subsample stability guard."
        ),
    ),
    "fixed_global_guarded_v1": SiblingGateProfile(
        profile_id="fixed_global_guarded_v1",
        sibling_gate_method="fixed_global_chi_square",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit full fixed-subspace chi-square sibling gate with "
            "selected-topology alpha penalty and selected-root feature-subsample "
            "stability guard."
        ),
    ),
    "fixed_coordinate_selective_root_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_selective_root_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty, selected-root feature-subsample stability guard, and "
            "selected-root permutation guard."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
    ),
    "fixed_coordinate_selective_traversal_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_selective_traversal_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty, selected-root feature-subsample stability guard, and "
            "selected-subtree permutation guard for traversal-open internal nodes."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
        root_selective_permutation_guard_scope="open_internal",
    ),
    "fixed_coordinate_selective_passthrough_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_selective_passthrough_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty, selected-root feature-subsample stability guard, and "
            "selected-subtree permutation guard only for splits reachable through "
            "a pass-through ancestor."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
        root_selective_permutation_guard_scope="passthrough_descendant",
    ),
    "fixed_coordinate_global_passthrough_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_global_passthrough_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty, selected-root feature-subsample stability guard, and "
            "a global selected-family sibling-min permutation guard for "
            "pass-through descendant splits."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
        root_selective_permutation_guard_scope=(
            "global_sibling_min_passthrough_descendant"
        ),
    ),
    "fixed_coordinate_global_passthrough_refined_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_global_passthrough_refined_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_candidate",
        description=(
            "Non-cross-fit fixed coordinate-BH sibling gate with selected-topology "
            "alpha penalty, selected-root feature-subsample stability guard, and "
            "a global selected-family sibling-min permutation guard that reruns "
            "Monte Carlo floor pass-through families at higher resolution."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
        root_selective_permutation_guard_scope=(
            "global_sibling_min_passthrough_descendant_refined"
        ),
    ),
    "fixed_coordinate_spectral_transport_passthrough_diagnostic_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_spectral_transport_passthrough_diagnostic_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="diagnostic_only_not_production",
        description=(
            "Refined fixed-coordinate selected-family pass-through guard plus a "
            "fail-closed MP mode-transport pass-through support guard. The "
            "spectral transport layer can only block pass-through; it cannot "
            "open sibling splits or create calibrated p-values."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
        root_selective_permutation_guard_scope=(
            "global_sibling_min_passthrough_descendant_refined"
        ),
        spectral_transport_passthrough_guard=True,
        spectral_transport_max_cost=DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
        spectral_transport_require_mp_blocks=True,
        spectral_transport_block_log_tolerance=(
            DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
        ),
        spectral_transport_unmatched_mode_penalty=(
            DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
        ),
    ),
    "fixed_coordinate_spectral_transport_passthrough_v1": SiblingGateProfile(
        profile_id="fixed_coordinate_spectral_transport_passthrough_v1",
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
        root_stability_guard_threshold=0.24,
        root_stability_subsample_replicates=12,
        root_stability_feature_fraction=0.8,
        root_stability_seed=0,
        status="opt_in_candidate_not_default",
        description=(
            "Refined fixed-coordinate selected-family pass-through guard plus a "
            "strict MP mode-transport pass-through support guard. The spectral "
            "transport layer can only block pass-through; it cannot open sibling "
            "splits or create calibrated p-values. The one-replicate targeted "
            "overlap gate passed, but the 50-replicate panel found signal "
            "regressions, so this profile remains opt-in and is not the default "
            "traversal rule."
        ),
        root_selective_permutation_guard_replicates=99,
        root_selective_permutation_guard_seed=0,
        root_selective_permutation_guard_alpha=0.01,
        root_selective_permutation_guard_scope=(
            "global_sibling_min_passthrough_descendant_refined"
        ),
        spectral_transport_passthrough_guard=True,
        spectral_transport_max_cost=DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
        spectral_transport_require_mp_blocks=True,
        spectral_transport_block_log_tolerance=(
            DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
        ),
        spectral_transport_unmatched_mode_penalty=(
            DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
        ),
    ),
}


def resolve_sibling_gate_profile(
    sibling_gate_profile: str | SiblingGateProfile | None,
) -> SiblingGateProfile | None:
    """Resolve a named or explicit sibling-gate profile."""
    if sibling_gate_profile is None:
        return None
    if isinstance(sibling_gate_profile, SiblingGateProfile):
        return sibling_gate_profile
    profile_id = str(sibling_gate_profile)
    try:
        return SIBLING_GATE_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sibling_gate_profile {profile_id!r}; "
            f"allowed={tuple(sorted(SIBLING_GATE_PROFILES))!r}."
        ) from exc


def _profile_value(
    *,
    field_name: str,
    current: object,
    default: object,
    profile_value: object,
) -> object:
    if current == default or current == profile_value:
        return profile_value
    raise ValueError(
        f"sibling_gate_profile conflicts with explicit {field_name}: "
        f"profile has {profile_value!r}, explicit value is {current!r}."
    )


def resolve_sibling_gate_profile_config(
    *,
    sibling_gate_profile: str | SiblingGateProfile | None = None,
    sibling_gate_method: str = "projected_wald_inflation",
    sibling_gate_alpha_penalty: float = 1.0,
    root_stability_guard_threshold: float | None = None,
    root_stability_subsample_replicates: int = 0,
    root_stability_feature_fraction: float = 0.8,
    root_stability_seed: int = 0,
    root_selective_permutation_guard_replicates: int = 0,
    root_selective_permutation_guard_seed: int = 0,
    root_selective_permutation_guard_alpha: float | None = None,
    root_selective_permutation_guard_scope: str = "root",
    spectral_transport_passthrough_guard: bool = False,
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    spectral_transport_require_mp_blocks: bool = True,
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    ),
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    ),
) -> tuple[
    str | None,
    str,
    float,
    float | None,
    int,
    float,
    int,
    int,
    int,
    float | None,
    str,
    bool,
    float,
    bool,
    float,
    float,
]:
    """Resolve profile and explicit sibling-gate settings to concrete values."""
    profile = resolve_sibling_gate_profile(sibling_gate_profile)
    if profile is None:
        return (
            None,
            str(sibling_gate_method),
            float(sibling_gate_alpha_penalty),
            (
                None
                if root_stability_guard_threshold is None
                else float(root_stability_guard_threshold)
            ),
            int(root_stability_subsample_replicates),
            float(root_stability_feature_fraction),
            int(root_stability_seed),
            int(root_selective_permutation_guard_replicates),
            int(root_selective_permutation_guard_seed),
            (
                None
                if root_selective_permutation_guard_alpha is None
                else float(root_selective_permutation_guard_alpha)
            ),
            str(root_selective_permutation_guard_scope),
            bool(spectral_transport_passthrough_guard),
            float(spectral_transport_max_cost),
            bool(spectral_transport_require_mp_blocks),
            float(spectral_transport_block_log_tolerance),
            float(spectral_transport_unmatched_mode_penalty),
        )

    method = _profile_value(
        field_name="sibling_gate_method",
        current=str(sibling_gate_method),
        default="projected_wald_inflation",
        profile_value=profile.sibling_gate_method,
    )
    penalty = _profile_value(
        field_name="sibling_gate_alpha_penalty",
        current=float(sibling_gate_alpha_penalty),
        default=1.0,
        profile_value=float(profile.sibling_gate_alpha_penalty),
    )
    threshold = _profile_value(
        field_name="root_stability_guard_threshold",
        current=(
            None
            if root_stability_guard_threshold is None
            else float(root_stability_guard_threshold)
        ),
        default=None,
        profile_value=profile.root_stability_guard_threshold,
    )
    replicates = _profile_value(
        field_name="root_stability_subsample_replicates",
        current=int(root_stability_subsample_replicates),
        default=0,
        profile_value=int(profile.root_stability_subsample_replicates),
    )
    fraction = _profile_value(
        field_name="root_stability_feature_fraction",
        current=float(root_stability_feature_fraction),
        default=0.8,
        profile_value=float(profile.root_stability_feature_fraction),
    )
    seed = _profile_value(
        field_name="root_stability_seed",
        current=int(root_stability_seed),
        default=0,
        profile_value=int(profile.root_stability_seed),
    )
    if int(profile.root_selective_permutation_guard_replicates) > 0:
        root_selective_replicates = _profile_value(
            field_name="root_selective_permutation_guard_replicates",
            current=int(root_selective_permutation_guard_replicates),
            default=0,
            profile_value=int(profile.root_selective_permutation_guard_replicates),
        )
        root_selective_seed = _profile_value(
            field_name="root_selective_permutation_guard_seed",
            current=int(root_selective_permutation_guard_seed),
            default=0,
            profile_value=int(profile.root_selective_permutation_guard_seed),
        )
        root_selective_alpha = _profile_value(
            field_name="root_selective_permutation_guard_alpha",
            current=(
                None
                if root_selective_permutation_guard_alpha is None
                else float(root_selective_permutation_guard_alpha)
            ),
            default=None,
            profile_value=profile.root_selective_permutation_guard_alpha,
        )
        root_selective_scope = _profile_value(
            field_name="root_selective_permutation_guard_scope",
            current=str(root_selective_permutation_guard_scope),
            default="root",
            profile_value=str(profile.root_selective_permutation_guard_scope),
        )
    else:
        root_selective_replicates = int(root_selective_permutation_guard_replicates)
        root_selective_seed = int(root_selective_permutation_guard_seed)
        root_selective_alpha = (
            None
            if root_selective_permutation_guard_alpha is None
            else float(root_selective_permutation_guard_alpha)
        )
        root_selective_scope = str(root_selective_permutation_guard_scope)
    spectral_guard = _profile_value(
        field_name="spectral_transport_passthrough_guard",
        current=bool(spectral_transport_passthrough_guard),
        default=False,
        profile_value=bool(profile.spectral_transport_passthrough_guard),
    )
    spectral_max_cost = _profile_value(
        field_name="spectral_transport_max_cost",
        current=float(spectral_transport_max_cost),
        default=float(DEFAULT_SPECTRAL_TRANSPORT_MAX_COST),
        profile_value=float(profile.spectral_transport_max_cost),
    )
    spectral_require_mp_blocks = _profile_value(
        field_name="spectral_transport_require_mp_blocks",
        current=bool(spectral_transport_require_mp_blocks),
        default=True,
        profile_value=bool(profile.spectral_transport_require_mp_blocks),
    )
    spectral_block_log_tolerance = _profile_value(
        field_name="spectral_transport_block_log_tolerance",
        current=float(spectral_transport_block_log_tolerance),
        default=float(DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE),
        profile_value=float(profile.spectral_transport_block_log_tolerance),
    )
    spectral_unmatched_mode_penalty = _profile_value(
        field_name="spectral_transport_unmatched_mode_penalty",
        current=float(spectral_transport_unmatched_mode_penalty),
        default=float(DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY),
        profile_value=float(profile.spectral_transport_unmatched_mode_penalty),
    )
    return (
        profile.profile_id,
        str(method),
        float(penalty),
        None if threshold is None else float(threshold),
        int(replicates),
        float(fraction),
        int(seed),
        int(root_selective_replicates),
        int(root_selective_seed),
        None if root_selective_alpha is None else float(root_selective_alpha),
        str(root_selective_scope),
        bool(spectral_guard),
        float(spectral_max_cost),
        bool(spectral_require_mp_blocks),
        float(spectral_block_log_tolerance),
        float(spectral_unmatched_mode_penalty),
    )


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
            "sibling_gate_alpha_penalty must be finite and positive; "
            f"got {penalty!r}."
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
    return tuple(
        (field, getattr(thresholds, field))
        for field in thresholds.__dataclass_fields__
    )


def build_gate_annotation_config_metadata(
    *,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    spectral_include_internal_barycenters: bool = False,
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
        spectral_include_internal_barycenters=bool(
            spectral_include_internal_barycenters
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
        root_stability_subsample_replicates=int(
            root_stability_subsample_replicates
        ),
        root_stability_feature_fraction=float(root_stability_feature_fraction),
        root_stability_seed=int(root_stability_seed),
        root_stability_tree_distance_metric=str(root_stability_tree_distance_metric),
        root_stability_tree_linkage_method=str(root_stability_tree_linkage_method),
        root_selective_permutation_guard_replicates=int(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=int(
            root_selective_permutation_guard_seed
        ),
        root_selective_permutation_guard_alpha=(
            None
            if root_selective_permutation_guard_alpha is None
            else float(root_selective_permutation_guard_alpha)
        ),
        root_selective_permutation_guard_scope=str(
            root_selective_permutation_guard_scope
        ),
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
        external_selected_tail_calibration_enabled=(
            external_selected_tail_model is not None
        ),
        external_selected_tail_rule_count=(
            0
            if external_selected_tail_model is None
            else int(len(external_selected_tail_model.rules))
        ),
        spectral_transport_passthrough_guard=bool(
            spectral_transport_passthrough_guard
        ),
        spectral_transport_max_cost=float(spectral_transport_max_cost),
        spectral_transport_require_mp_blocks=bool(spectral_transport_require_mp_blocks),
        spectral_transport_block_log_tolerance=float(
            spectral_transport_block_log_tolerance
        ),
        spectral_transport_unmatched_mode_penalty=float(
            spectral_transport_unmatched_mode_penalty
        ),
    )


def build_gate_annotation_leaf_data_metadata(
    leaf_data: pd.DataFrame | None,
    *,
    feature_space: FeatureSpace | None = None,
) -> GateAnnotationLeafDataMetadata:
    """Capture enough leaf-data identity to validate reusable annotations."""
    if leaf_data is None:
        return GateAnnotationLeafDataMetadata(present=False)

    content_hash = hashlib.sha256()
    content_hash.update(str(tuple(leaf_data.shape)).encode("utf-8"))
    content_hash.update(
        pd.util.hash_pandas_object(pd.Index(leaf_data.index), index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    content_hash.update(
        pd.util.hash_pandas_object(pd.Index(leaf_data.columns), index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    content_hash.update(
        pd.util.hash_pandas_object(leaf_data, index=True)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    return GateAnnotationLeafDataMetadata(
        present=True,
        shape=(int(leaf_data.shape[0]), int(leaf_data.shape[1])),
        content_hash=content_hash.hexdigest(),
        feature_space_signature=(
            None if feature_space is None else feature_space.signature
        ),
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
        parent_principal_component_projections=(
            resolved_parent_principal_component_projections
        ),
        parent_principal_component_eigenvalues=(
            resolved_parent_principal_component_eigenvalues
        ),
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


def _validate_root_stability_guard_config(
    *,
    threshold: float | None,
    subsample_replicates: int,
    feature_fraction: float,
) -> None:
    replicates = int(subsample_replicates)
    if replicates < 0:
        raise ValueError("root_stability_subsample_replicates must be nonnegative.")
    fraction = float(feature_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("root_stability_feature_fraction must lie in (0, 1].")
    if threshold is None:
        return
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not -1.0 <= threshold_value <= 1.0:
        raise ValueError(
            "root_stability_guard_threshold must be finite and lie in [-1, 1]; "
            f"got {threshold!r}."
        )
    if replicates <= 0:
        raise ValueError(
            "root_stability_subsample_replicates must be positive when "
            "root_stability_guard_threshold is configured."
        )


def _validate_root_selective_permutation_guard_config(
    *,
    replicates: int,
    alpha: float,
    scope: str = "root",
) -> None:
    count = int(replicates)
    if count < 0:
        raise ValueError(
            "root_selective_permutation_guard_replicates must be nonnegative."
        )
    alpha_value = float(alpha)
    if not 0.0 < alpha_value < 1.0:
        raise ValueError(
            "root_selective_permutation_guard_alpha must lie in (0, 1); "
            f"got {alpha!r}."
        )
    allowed_scopes = {
        "root",
        "open_internal",
        "passthrough_descendant",
        "global_sibling_min_passthrough_descendant",
        "global_sibling_min_passthrough_descendant_refined",
    }
    if str(scope) not in allowed_scopes:
        raise ValueError(
            "root_selective_permutation_guard_scope must be 'root', "
            "'open_internal', 'passthrough_descendant', or "
            "'global_sibling_min_passthrough_descendant'/'..._refined'; "
            f"got {scope!r}."
        )


def _tree_root_node(tree):
    if hasattr(tree, "root"):
        return tree.root()
    if "root" in tree.graph:
        return tree.graph["root"]
    roots = [node for node, degree in tree.in_degree() if degree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one root, got {roots!r}.")
    return roots[0]


def _descendant_leaf_label_sets(tree) -> dict[object, frozenset]:
    if hasattr(tree, "compute_descendant_sets"):
        return tree.compute_descendant_sets(use_labels=True)

    labels_by_node: dict[object, frozenset] = {}
    for node in tree.nodes:
        leaves: set[object] = set()
        for descendant in {node, *_nx_descendants(tree, node)}:
            if tree.out_degree(descendant) == 0:
                leaves.add(tree.nodes[descendant].get("label", descendant))
        labels_by_node[node] = frozenset(leaves)
    return labels_by_node


def _nx_descendants(tree, node) -> set[object]:
    frontier = list(tree.successors(node))
    seen: set[object] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(tree.successors(current))
    return seen


def _tree_root_split_labels(tree, data_index: pd.Index) -> np.ndarray | None:
    root = _tree_root_node(tree)
    children = list(tree.successors(root))
    if len(children) != 2:
        return None

    descendant_sets = _descendant_leaf_label_sets(tree)
    left = set(descendant_sets[children[0]])
    right = set(descendant_sets[children[1]])

    labels: list[int] = []
    missing: list[object] = []
    for label in data_index:
        if label in left:
            labels.append(0)
        elif label in right:
            labels.append(1)
        else:
            missing.append(label)
    if missing:
        raise ValueError(
            "leaf_data index must match leaf labels under the supplied tree root; "
            f"missing={missing[:5]!r}."
        )
    values = np.asarray(labels, dtype=int)
    if np.unique(values).size < 2:
        return None
    return values


def _build_selected_linkage_tree(
    data: pd.DataFrame,
    *,
    distance_metric: str,
    linkage_method: str,
):
    from kl_clustering_analysis.tree.poset_tree import PosetTree

    values = data.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("leaf_data must be a 2D feature matrix.")
    if values.shape[0] < 2:
        raise ValueError("Selected-root permutation requires at least two leaves.")
    if not np.isfinite(values).all():
        raise ValueError("leaf_data contains non-finite values.")
    distances = pdist(values, metric=str(distance_metric))
    return PosetTree.from_linkage(
        linkage(distances, method=str(linkage_method)),
        leaf_names=data.index.tolist(),
    )


def _fixed_root_sibling_p_value(
    tree,
    feature_space: FeatureSpace,
    *,
    method: str,
) -> float:
    root = _tree_root_node(tree)
    children = identify_binary_sibling_children(tree, root)
    if children is None:
        return 1.0
    left, right = children
    contrast = build_contrast_covariance(
        np.asarray(tree.nodes[left]["distribution"], dtype=float),
        np.asarray(tree.nodes[right]["distribution"], dtype=float),
        float(tree.nodes[left]["leaf_count"]),
        float(tree.nodes[right]["leaf_count"]),
        comparison="sibling",
        feature_space=feature_space,
    )
    return fixed_subspace_sibling_p_value(
        contrast.whitened_vector(),
        feature_space,
        method=method,  # type: ignore[arg-type]
    )


def _can_use_fast_bernoulli_coordinate_p_values(feature_space: FeatureSpace) -> bool:
    return all(
        block.family == "bernoulli"
        and len(tuple(block.column_indices)) == 1
        and int(block.contrast_dimension) == 1
        for block in feature_space.blocks
    )


def _can_use_fast_discrete_coordinate_p_values(feature_space: FeatureSpace) -> bool:
    return feature_space.family_label in {"bernoulli", "categorical"}


def _fixed_bernoulli_coordinate_sibling_p_value(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    *,
    ridge: float = 1e-12,
) -> float:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    variance_scale = 1.0 / float(first_sample_size) + 1.0 / float(second_sample_size)
    pooled = (
        float(first_sample_size) * first_values
        + float(second_sample_size) * second_values
    ) / (float(first_sample_size) + float(second_sample_size))
    variance = pooled * (1.0 - pooled) * variance_scale + float(ridge)
    z = (first_values - second_values) / np.sqrt(variance)
    return fixed_coordinate_bh_p_value(z)


def _fixed_discrete_coordinate_sibling_p_value(
    first: np.ndarray,
    second: np.ndarray,
    first_sample_size: float,
    second_sample_size: float,
    feature_space: FeatureSpace,
) -> float:
    z = compute_whitened_wald_contrast(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
        float(first_sample_size),
        float(second_sample_size),
        comparison="sibling",
        feature_space=feature_space,
    )
    return fixed_coordinate_bh_p_value(z)


def _fixed_binary_sibling_p_values(
    tree,
    feature_space: FeatureSpace,
    *,
    method: str,
) -> list[float]:
    """Return fixed-subspace sibling p-values for every binary parent."""
    p_values: list[float] = []
    use_fast_discrete_coordinate = (
        method == "fixed_coordinate_bh"
        and _can_use_fast_discrete_coordinate_p_values(feature_space)
    )
    for parent in tree.nodes:
        children = identify_binary_sibling_children(tree, parent)
        if children is None:
            continue
        left, right = children
        if use_fast_discrete_coordinate:
            p_values.append(
                _fixed_discrete_coordinate_sibling_p_value(
                    np.asarray(tree.nodes[left]["distribution"], dtype=float),
                    np.asarray(tree.nodes[right]["distribution"], dtype=float),
                    float(tree.nodes[left]["leaf_count"]),
                    float(tree.nodes[right]["leaf_count"]),
                    feature_space,
                )
            )
            continue
        contrast = build_contrast_covariance(
            np.asarray(tree.nodes[left]["distribution"], dtype=float),
            np.asarray(tree.nodes[right]["distribution"], dtype=float),
            float(tree.nodes[left]["leaf_count"]),
            float(tree.nodes[right]["leaf_count"]),
            comparison="sibling",
            feature_space=feature_space,
        )
        p_values.append(
            fixed_subspace_sibling_p_value(
                contrast.whitened_vector(),
                feature_space,
                method=method,  # type: ignore[arg-type]
            )
        )
    return p_values


def _selected_tree_fixed_sibling_min_p_value(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    tree_distance_metric: str,
    tree_linkage_method: str,
) -> float:
    tree = _build_selected_linkage_tree(
        data,
        distance_metric=tree_distance_metric,
        linkage_method=tree_linkage_method,
    )
    tree.populate_node_divergences(data, feature_space=feature_space)
    p_values = _fixed_binary_sibling_p_values(
        tree,
        feature_space,
        method=method,
    )
    return float(min(p_values)) if p_values else 1.0


def _selected_root_fixed_sibling_p_value(
    data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    tree_distance_metric: str,
    tree_linkage_method: str,
) -> float:
    tree = _build_selected_linkage_tree(
        data,
        distance_metric=tree_distance_metric,
        linkage_method=tree_linkage_method,
    )
    tree.populate_node_divergences(data, feature_space=feature_space)
    return _fixed_root_sibling_p_value(
        tree,
        feature_space,
        method=method,
    )


def _block_permutation_null_sample(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a feature/block permutation sample preserving block margins."""
    source = leaf_data.to_numpy(dtype=int)
    out = np.zeros_like(source)
    n_rows = int(source.shape[0])
    for block in feature_space.blocks:
        indices = np.asarray(block.column_indices, dtype=int)
        if block.family == "categorical":
            categories = np.argmax(source[:, indices], axis=1)
            permuted = rng.permutation(categories)
            out[np.arange(n_rows), indices[permuted]] = 1
            continue
        if block.family == "bernoulli":
            out[:, indices[0]] = rng.permutation(source[:, indices[0]])
            continue
        raise ValueError(
            "Root selective permutation supports Bernoulli and categorical "
            f"feature blocks only; got {block.family!r}."
        )
    return pd.DataFrame(
        out,
        index=leaf_data.index,
        columns=leaf_data.columns,
    )


def selected_root_permutation_p_value(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
    observed_p_value: float | None = None,
) -> dict[str, float]:
    """Estimate the selected-root p-value by feature/block permutation."""
    if method not in FIXED_SUBSPACE_SIBLING_GATE_METHODS:
        raise ValueError(
            "Selected-root permutation requires a fixed-subspace sibling gate; "
            f"got method={method!r}."
        )
    count = int(bootstrap_replicates)
    if count < 0:
        raise ValueError("bootstrap_replicates must be nonnegative.")
    observed = (
        float(observed_p_value)
        if observed_p_value is not None and np.isfinite(float(observed_p_value))
        else _selected_root_fixed_sibling_p_value(
            leaf_data,
            feature_space,
            method=method,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
        )
    )
    if count <= 0:
        return {
            "root_observed_p_value": float(observed),
            "root_selective_p_value": np.nan,
            "root_selective_null_min_p_value": np.nan,
            "root_selective_null_q05_p_value": np.nan,
        }

    rng = np.random.default_rng(int(seed))
    null_p_values: list[float] = []
    for _ in range(count):
        null_sample = _block_permutation_null_sample(leaf_data, feature_space, rng)
        null_p_values.append(
            _selected_root_fixed_sibling_p_value(
                null_sample,
                feature_space,
                method=method,
                tree_distance_metric=tree_distance_metric,
                tree_linkage_method=tree_linkage_method,
            )
        )
    null = np.asarray(null_p_values, dtype=float)
    selected = (1.0 + float(np.sum(null <= observed))) / (float(null.size) + 1.0)
    return {
        "root_observed_p_value": float(observed),
        "root_selective_p_value": float(selected),
        "root_selective_null_min_p_value": float(np.min(null)),
        "root_selective_null_q05_p_value": float(np.quantile(null, 0.05)),
    }


def selected_global_sibling_min_permutation_p_value(
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
    observed_p_value: float | None = None,
) -> dict[str, float]:
    """Estimate a whole selected-family minimum sibling p-value.

    This diagnostic null compares an observed selected sibling p-value against
    the minimum fixed-subspace sibling p-value over every binary parent in each
    reselected null tree. It is a conservative global-family correction for
    pass-through descendant searches.
    """
    if method not in FIXED_SUBSPACE_SIBLING_GATE_METHODS:
        raise ValueError(
            "Selected-family permutation requires a fixed-subspace sibling gate; "
            f"got method={method!r}."
        )
    count = int(bootstrap_replicates)
    if count < 0:
        raise ValueError("bootstrap_replicates must be nonnegative.")
    observed = (
        float(observed_p_value)
        if observed_p_value is not None and np.isfinite(float(observed_p_value))
        else _selected_tree_fixed_sibling_min_p_value(
            leaf_data,
            feature_space,
            method=method,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
        )
    )
    if count <= 0:
        return {
            "root_observed_p_value": float(observed),
            "root_selective_p_value": np.nan,
            "root_selective_null_min_p_value": np.nan,
            "root_selective_null_q05_p_value": np.nan,
        }

    rng = np.random.default_rng(int(seed))
    null_p_values: list[float] = []
    for _ in range(count):
        null_sample = _block_permutation_null_sample(leaf_data, feature_space, rng)
        null_p_values.append(
            _selected_tree_fixed_sibling_min_p_value(
                null_sample,
                feature_space,
                method=method,
                tree_distance_metric=tree_distance_metric,
                tree_linkage_method=tree_linkage_method,
            )
        )
    null = np.asarray(null_p_values, dtype=float)
    selected = (1.0 + float(np.sum(null <= observed))) / (float(null.size) + 1.0)
    return {
        "root_observed_p_value": float(observed),
        "root_selective_p_value": float(selected),
        "root_selective_null_min_p_value": float(np.min(null)),
        "root_selective_null_q05_p_value": float(np.quantile(null, 0.05)),
    }


def _selected_linkage_root_split_labels(
    data: pd.DataFrame,
    *,
    distance_metric: str,
    linkage_method: str,
) -> np.ndarray | None:
    values = data.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("leaf_data must be a 2D feature matrix.")
    n_rows = int(values.shape[0])
    if n_rows < 2:
        return None
    if not np.isfinite(values).all():
        raise ValueError("leaf_data contains non-finite values.")

    distances = pdist(values, metric=str(distance_metric))
    linkage_matrix = linkage(distances, method=str(linkage_method))
    clusters: dict[int, set[int]] = {index: {index} for index in range(n_rows)}
    for step_index, row in enumerate(linkage_matrix):
        left_id = int(row[0])
        right_id = int(row[1])
        clusters[n_rows + step_index] = clusters[left_id] | clusters[right_id]

    left_child = int(linkage_matrix[-1, 0])
    right_child = int(linkage_matrix[-1, 1])
    labels = np.zeros(n_rows, dtype=int)
    labels[list(clusters[right_child])] = 1
    if not clusters[left_child] or not clusters[right_child]:
        return None
    return labels


def compute_root_feature_subsample_stability(
    tree,
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    subsample_replicates: int,
    feature_fraction: float,
    seed: int,
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
) -> dict[str, float]:
    """Measure supplied-root stability under deterministic feature-block subsampling."""
    _validate_root_stability_guard_config(
        threshold=None,
        subsample_replicates=subsample_replicates,
        feature_fraction=feature_fraction,
    )
    if int(subsample_replicates) <= 0:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }

    original = _tree_root_split_labels(tree, leaf_data.index)
    if original is None:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }

    blocks = tuple(tuple(block.column_indices) for block in feature_space.blocks)
    if not blocks:
        raise ValueError("Feature-space blocks are required for root stability.")
    n_blocks = len(blocks)
    n_selected = max(1, int(round(float(feature_fraction) * n_blocks)))
    rng = np.random.default_rng(int(seed))

    scores: list[float] = []
    for _ in range(int(subsample_replicates)):
        selected_blocks = rng.choice(n_blocks, size=n_selected, replace=False)
        columns = [
            column
            for block_index in selected_blocks
            for column in blocks[int(block_index)]
        ]
        subsampled = leaf_data.iloc[:, columns]
        selected_labels = _selected_linkage_root_split_labels(
            subsampled,
            distance_metric=str(tree_distance_metric),
            linkage_method=str(tree_linkage_method),
        )
        if selected_labels is None:
            continue
        scores.append(float(adjusted_rand_score(original, selected_labels)))

    if not scores:
        return {
            "root_stability_subsample_mean_ari": np.nan,
            "root_stability_subsample_median_ari": np.nan,
            "root_stability_subsample_q10_ari": np.nan,
        }
    values = np.asarray(scores, dtype=float)
    return {
        "root_stability_subsample_mean_ari": float(np.mean(values)),
        "root_stability_subsample_median_ari": float(np.median(values)),
        "root_stability_subsample_q10_ari": float(np.quantile(values, 0.10)),
    }


def apply_root_stability_guard(
    tree,
    annotations_df: pd.DataFrame,
    root_stability: Mapping[str, float],
    *,
    threshold: float,
) -> pd.DataFrame:
    """Close an unstable root sibling gate without changing non-root decisions."""
    threshold_value = float(threshold)
    out = annotations_df.copy()
    for column in (
        "Root_Stability_Subsample_Mean_ARI",
        "Root_Stability_Subsample_Median_ARI",
        "Root_Stability_Subsample_Q10_ARI",
        "Root_Stability_Guard_Threshold",
    ):
        out[column] = np.nan
    out["Root_Stability_Guard_Blocked"] = False

    root = _tree_root_node(tree)
    if root not in out.index:
        return out

    mean_ari = float(root_stability.get("root_stability_subsample_mean_ari", np.nan))
    out.loc[root, "Root_Stability_Subsample_Mean_ARI"] = mean_ari
    out.loc[root, "Root_Stability_Subsample_Median_ARI"] = root_stability.get(
        "root_stability_subsample_median_ari",
        np.nan,
    )
    out.loc[root, "Root_Stability_Subsample_Q10_ARI"] = root_stability.get(
        "root_stability_subsample_q10_ari",
        np.nan,
    )
    out.loc[root, "Root_Stability_Guard_Threshold"] = threshold_value

    root_gate_open = bool(out.loc[root, "Sibling_BH_Different"])
    should_block = bool(np.isfinite(mean_ari) and mean_ari < threshold_value)
    if root_gate_open and should_block:
        out.loc[root, "Sibling_BH_Different"] = False
        out.loc[root, "Sibling_BH_Same"] = True
        out.loc[root, "Root_Stability_Guard_Blocked"] = True
    return out


def _annotation_bool(
    annotations_df: pd.DataFrame,
    node: object,
    column: str,
) -> bool:
    if node not in annotations_df.index or column not in annotations_df.columns:
        return False
    value = annotations_df.loc[node, column]
    return bool(pd.notna(value) and bool(value))


def _node_split_prerequisites(
    tree,
    annotations_df: pd.DataFrame,
    node: object,
) -> bool:
    children = list(tree.successors(node))
    if len(children) != 2:
        return False
    return any(
        _annotation_bool(
            annotations_df,
            child,
            "Child_Parent_Divergence_Significant",
        )
        for child in children
    )


def _node_sibling_gate_open(
    annotations_df: pd.DataFrame,
    node: object,
) -> bool:
    return _annotation_bool(
        annotations_df,
        node,
        "Sibling_BH_Different",
    ) and not _annotation_bool(
        annotations_df,
        node,
        "Sibling_Divergence_Skipped",
    )


def _root_stability_blocked(
    annotations_df: pd.DataFrame,
    node: object,
    root: object,
) -> bool:
    return (
        node == root
        and _annotation_bool(
            annotations_df,
            node,
            "Root_Stability_Guard_Blocked",
        )
    )


def _selective_guard_root_candidate(
    annotations_df: pd.DataFrame,
    node: object,
    root: object,
) -> bool:
    return _node_sibling_gate_open(annotations_df, node) or _root_stability_blocked(
        annotations_df,
        node,
        root,
    )


def _node_closed_by_explicit_guard(
    annotations_df: pd.DataFrame,
    node: object,
) -> bool:
    return any(
        _annotation_bool(annotations_df, node, column)
        for column in (
            "Root_Stability_Guard_Blocked",
            "Root_Selective_Permutation_Guard_Blocked",
            "Selective_Permutation_Guard_Blocked",
        )
    )


def _passthrough_descendant_guard_candidates(
    tree,
    annotations_df: pd.DataFrame,
    *,
    root: object,
) -> list[object]:
    """Return open split nodes reachable only through a closed sibling ancestor."""
    node_ids = tuple(tree.nodes)
    children = {node: list(tree.successors(node)) for node in node_ids}
    split_prerequisites = {
        node: _node_split_prerequisites(tree, annotations_df, node)
        for node in node_ids
    }
    sibling_open = {
        node: _node_sibling_gate_open(annotations_df, node) for node in node_ids
    }
    can_split = {
        node: bool(split_prerequisites[node] and sibling_open[node])
        for node in node_ids
    }
    has_descendant_split: dict[object, bool] = {}
    for node in bottom_up_nodes(tree):
        has_descendant_split[node] = any(
            can_split.get(child, False) or has_descendant_split.get(child, False)
            for child in children[node]
        )

    closed_passthrough_ancestors = {
        node
        for node in node_ids
        if split_prerequisites[node]
        and not can_split[node]
        and has_descendant_split.get(node, False)
        and not _node_closed_by_explicit_guard(annotations_df, node)
    }
    passthrough_reachable: dict[object, bool] = {}
    stack: list[tuple[object, bool]] = [(root, False)]
    while stack:
        node, ancestor_passthrough = stack.pop()
        passthrough_reachable[node] = bool(ancestor_passthrough)
        child_passthrough = bool(
            ancestor_passthrough or node in closed_passthrough_ancestors
        )
        for child in children[node]:
            stack.append((child, child_passthrough))

    return [
        node
        for node in annotations_df.index
        if node != root
        and can_split.get(node, False)
        and passthrough_reachable.get(node, False)
    ]


def apply_root_selective_permutation_guard(
    tree,
    annotations_df: pd.DataFrame,
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace,
    *,
    method: str,
    bootstrap_replicates: int,
    seed: int,
    alpha: float,
    scope: str = "root",
    tree_distance_metric: str = "hamming",
    tree_linkage_method: str = "average",
) -> pd.DataFrame:
    """Close selected-root or selected-subtree splits failing permutation evidence."""
    _validate_root_selective_permutation_guard_config(
        replicates=bootstrap_replicates,
        alpha=alpha,
        scope=scope,
    )
    out = annotations_df.copy()
    root_columns = (
        "Root_Selective_Permutation_Observed_P_Value",
        "Root_Selective_Permutation_P_Value",
        "Root_Selective_Permutation_Null_Min_P_Value",
        "Root_Selective_Permutation_Null_Q05_P_Value",
        "Root_Selective_Permutation_Guard_Alpha",
        "Root_Selective_Permutation_Guard_Replicates",
        "Root_Selective_Permutation_Guard_Seed",
    )
    generic_columns = (
        "Selective_Permutation_Observed_P_Value",
        "Selective_Permutation_P_Value",
        "Selective_Permutation_Null_Min_P_Value",
        "Selective_Permutation_Null_Q05_P_Value",
        "Selective_Permutation_Guard_Alpha",
        "Selective_Permutation_Guard_Replicates",
        "Selective_Permutation_Guard_Seed",
        "Selective_Permutation_Guard_Scope",
    )
    for column in (*root_columns, *generic_columns):
        out[column] = np.nan
    out["Selective_Permutation_Guard_Scope"] = ""
    out["Selective_Permutation_Base_P_Value"] = np.nan
    out["Selective_Permutation_Guard_Refined"] = False
    out["Root_Selective_Permutation_Guard_Would_Block"] = False
    out["Root_Selective_Permutation_Guard_Blocked"] = False
    out["Selective_Permutation_Guard_Would_Block"] = False
    out["Selective_Permutation_Guard_Blocked"] = False

    root = _tree_root_node(tree)
    if root not in out.index:
        return out

    descendant_sets = _descendant_leaf_label_sets(tree)
    scope_value = str(scope)
    evaluated_nodes: set[object] = set()
    evaluation_index = 0

    def evaluate_node(node: object, seed_offset: int) -> bool:
        if node not in descendant_sets or len(descendant_sets[node]) < 2:
            return False
        descendant_labels = descendant_sets[node]
        labels = [label for label in leaf_data.index if label in descendant_labels]
        subset = leaf_data.loc[labels]
        observed = (
            float(out.loc[node, "Sibling_Divergence_P_Value"])
            if "Sibling_Divergence_P_Value" in out.columns
            and pd.notna(out.loc[node, "Sibling_Divergence_P_Value"])
            else None
        )
        out.loc[node, "Selective_Permutation_Guard_Alpha"] = float(alpha)
        out.loc[node, "Selective_Permutation_Guard_Replicates"] = int(
            bootstrap_replicates
        )
        out.loc[node, "Selective_Permutation_Guard_Seed"] = int(seed) + seed_offset
        out.loc[node, "Selective_Permutation_Guard_Scope"] = scope_value
        result = selected_root_permutation_p_value(
            subset,
            feature_space,
            method=method,
            bootstrap_replicates=int(bootstrap_replicates),
            seed=int(seed) + seed_offset,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
            observed_p_value=observed,
        )
        out.loc[node, "Selective_Permutation_Observed_P_Value"] = result[
            "root_observed_p_value"
        ]
        out.loc[node, "Selective_Permutation_P_Value"] = result[
            "root_selective_p_value"
        ]
        out.loc[node, "Selective_Permutation_Null_Min_P_Value"] = result[
            "root_selective_null_min_p_value"
        ]
        out.loc[node, "Selective_Permutation_Null_Q05_P_Value"] = result[
            "root_selective_null_q05_p_value"
        ]
        selected_p = result["root_selective_p_value"]
        would_block = bool(np.isfinite(selected_p) and selected_p > float(alpha))
        out.loc[node, "Selective_Permutation_Guard_Would_Block"] = would_block
        if bool(out.loc[node, "Sibling_BH_Different"]) and would_block:
            out.loc[node, "Sibling_BH_Different"] = False
            out.loc[node, "Sibling_BH_Same"] = True
            out.loc[node, "Selective_Permutation_Guard_Blocked"] = True
        if node == root:
            out.loc[root, "Root_Selective_Permutation_Observed_P_Value"] = result[
                "root_observed_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_P_Value"] = result[
                "root_selective_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_Null_Min_P_Value"] = result[
                "root_selective_null_min_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_Null_Q05_P_Value"] = result[
                "root_selective_null_q05_p_value"
            ]
            out.loc[root, "Root_Selective_Permutation_Guard_Alpha"] = float(alpha)
            out.loc[root, "Root_Selective_Permutation_Guard_Replicates"] = int(
                bootstrap_replicates
            )
            out.loc[root, "Root_Selective_Permutation_Guard_Seed"] = int(
                seed
            ) + seed_offset
            out.loc[root, "Root_Selective_Permutation_Guard_Would_Block"] = (
                would_block
            )
            out.loc[root, "Root_Selective_Permutation_Guard_Blocked"] = bool(
                out.loc[root, "Selective_Permutation_Guard_Blocked"]
            )
        return True

    def evaluate_global_pass_through_nodes(
        nodes: list[object],
        seed_offset: int,
    ) -> bool:
        observed_values = [
            float(out.loc[node, "Sibling_Divergence_P_Value"])
            for node in nodes
            if node in out.index
            and "Sibling_Divergence_P_Value" in out.columns
            and pd.notna(out.loc[node, "Sibling_Divergence_P_Value"])
        ]
        if not observed_values:
            return False
        observed = float(min(observed_values))
        result = selected_global_sibling_min_permutation_p_value(
            leaf_data,
            feature_space,
            method=method,
            bootstrap_replicates=int(bootstrap_replicates),
            seed=int(seed) + seed_offset,
            tree_distance_metric=tree_distance_metric,
            tree_linkage_method=tree_linkage_method,
            observed_p_value=observed,
        )
        base_selected_p = result["root_selective_p_value"]
        used_replicates = int(bootstrap_replicates)
        refined = False
        base_floor = 1.0 / (float(bootstrap_replicates) + 1.0)
        if (
            scope_value == "global_sibling_min_passthrough_descendant_refined"
            and np.isfinite(base_selected_p)
            and float(base_selected_p) <= base_floor + 1e-12
            and int(bootstrap_replicates) < GLOBAL_PASSTHROUGH_REFINED_REPLICATES
        ):
            result = selected_global_sibling_min_permutation_p_value(
                leaf_data,
                feature_space,
                method=method,
                bootstrap_replicates=GLOBAL_PASSTHROUGH_REFINED_REPLICATES,
                seed=int(seed) + seed_offset,
                tree_distance_metric=tree_distance_metric,
                tree_linkage_method=tree_linkage_method,
                observed_p_value=observed,
            )
            used_replicates = GLOBAL_PASSTHROUGH_REFINED_REPLICATES
            refined = True
        selected_p = result["root_selective_p_value"]
        would_block = bool(np.isfinite(selected_p) and selected_p > float(alpha))
        for node in nodes:
            out.loc[node, "Selective_Permutation_Observed_P_Value"] = result[
                "root_observed_p_value"
            ]
            out.loc[node, "Selective_Permutation_Base_P_Value"] = base_selected_p
            out.loc[node, "Selective_Permutation_P_Value"] = selected_p
            out.loc[node, "Selective_Permutation_Null_Min_P_Value"] = result[
                "root_selective_null_min_p_value"
            ]
            out.loc[node, "Selective_Permutation_Null_Q05_P_Value"] = result[
                "root_selective_null_q05_p_value"
            ]
            out.loc[node, "Selective_Permutation_Guard_Alpha"] = float(alpha)
            out.loc[node, "Selective_Permutation_Guard_Replicates"] = int(
                used_replicates
            )
            out.loc[node, "Selective_Permutation_Guard_Seed"] = int(
                seed
            ) + seed_offset
            out.loc[node, "Selective_Permutation_Guard_Scope"] = scope_value
            out.loc[node, "Selective_Permutation_Guard_Refined"] = refined
            out.loc[node, "Selective_Permutation_Guard_Would_Block"] = would_block
            if bool(out.loc[node, "Sibling_BH_Different"]) and would_block:
                out.loc[node, "Sibling_BH_Different"] = False
                out.loc[node, "Sibling_BH_Same"] = True
                out.loc[node, "Selective_Permutation_Guard_Blocked"] = True
        return True

    if scope_value == "root":
        candidate_nodes = (
            [root] if _selective_guard_root_candidate(out, root, root) else []
        )
    elif scope_value == "open_internal":
        candidate_nodes = [
            node
            for node in out.index
            if node in descendant_sets
            and len(descendant_sets[node]) >= 2
            and (
                _node_sibling_gate_open(out, node)
                or _root_stability_blocked(out, node, root)
            )
        ]
    else:
        candidate_nodes = []
        if _selective_guard_root_candidate(out, root, root):
            candidate_nodes.append(root)

    for node in candidate_nodes:
        if evaluate_node(node, evaluation_index):
            evaluated_nodes.add(node)
            evaluation_index += 1

    if scope_value in {
        "passthrough_descendant",
        "global_sibling_min_passthrough_descendant",
        "global_sibling_min_passthrough_descendant_refined",
    }:
        pass_through_nodes = _passthrough_descendant_guard_candidates(
            tree,
            out,
            root=root,
        )
    else:
        pass_through_nodes = []

    if scope_value == "passthrough_descendant":
        for node in pass_through_nodes:
            if node in evaluated_nodes:
                continue
            if node not in descendant_sets or len(descendant_sets[node]) < 2:
                continue
            if evaluate_node(node, evaluation_index):
                evaluated_nodes.add(node)
                evaluation_index += 1
    elif scope_value in {
        "global_sibling_min_passthrough_descendant",
        "global_sibling_min_passthrough_descendant_refined",
    }:
        global_nodes = [
            node
            for node in pass_through_nodes
            if node not in evaluated_nodes
            and node in descendant_sets
            and len(descendant_sets[node]) >= 2
        ]
        if evaluate_global_pass_through_nodes(global_nodes, evaluation_index):
            evaluated_nodes.update(global_nodes)
    return out


def run_gate_annotation_pipeline(
    tree,
    annotations_df: pd.DataFrame,
    *,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    leaf_data: pd.DataFrame | None = None,
    feature_space: FeatureSpace | None = None,
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
    spectral_include_internal_barycenters: bool = False,
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
    external_selected_tail_context_by_parent: Mapping[
        object, Mapping[str, object]
    ] | None = None,
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
        root_selective_permutation_guard_replicates=(
            root_selective_permutation_guard_replicates
        ),
        root_selective_permutation_guard_seed=root_selective_permutation_guard_seed,
        root_selective_permutation_guard_alpha=(
            root_selective_permutation_guard_alpha
        ),
        root_selective_permutation_guard_scope=root_selective_permutation_guard_scope,
        spectral_transport_passthrough_guard=spectral_transport_passthrough_guard,
        spectral_transport_max_cost=spectral_transport_max_cost,
        spectral_transport_require_mp_blocks=spectral_transport_require_mp_blocks,
        spectral_transport_block_log_tolerance=spectral_transport_block_log_tolerance,
        spectral_transport_unmatched_mode_penalty=(
            spectral_transport_unmatched_mode_penalty
        ),
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
    _validate_root_stability_guard_config(
        threshold=root_stability_guard_threshold,
        subsample_replicates=root_stability_subsample_replicates,
        feature_fraction=root_stability_feature_fraction,
    )
    effective_root_selective_alpha = (
        float(sibling_alpha)
        if root_selective_permutation_guard_alpha is None
        else float(root_selective_permutation_guard_alpha)
    )
    _validate_root_selective_permutation_guard_config(
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

    # Run edge-divergence gate: child-parent edge tests
    edge_gate_start_sec = perf_counter()
    edge_annotated_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        annotations_df,
        significance_level_alpha=edge_alpha,
        leaf_data=leaf_data,
        feature_space=feature_space,
        spectral_minimum_dimension=spectral_minimum_dimension,
        spectral_include_internal_barycenters=(
            spectral_include_internal_barycenters
        ),
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
            enforce_support_thresholds=enforce_internal_support_thresholds,
            support_thresholds=internal_support_thresholds,
            external_selected_tail_model=external_selected_tail_model,
            external_selected_tail_context_by_parent=(
                external_selected_tail_context_by_parent
            ),
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
        )
    sibling_gate_sec = float(perf_counter() - sibling_gate_start_sec)
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
        stage_timings["root_stability_guard_sec"] = float(
            perf_counter() - root_guard_start_sec
        )
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
            eigenvalue_block_log_tolerance=float(
                spectral_transport_block_log_tolerance
            ),
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
            spectral_minimum_dimension=spectral_minimum_dimension,
            spectral_include_internal_barycenters=(
                spectral_include_internal_barycenters
            ),
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
            root_selective_permutation_guard_seed=(
                root_selective_permutation_guard_seed
            ),
            root_selective_permutation_guard_alpha=(
                root_selective_permutation_guard_alpha
            ),
            root_selective_permutation_guard_scope=(
                root_selective_permutation_guard_scope
            ),
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
            spectral_transport_block_log_tolerance=(
                spectral_transport_block_log_tolerance
            ),
            spectral_transport_unmatched_mode_penalty=(
                spectral_transport_unmatched_mode_penalty
            ),
        ),
        leaf_data=build_gate_annotation_leaf_data_metadata(
            leaf_data,
            feature_space=feature_space,
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
    "build_gate_annotation_leaf_data_metadata",
    "compute_root_feature_subsample_stability",
    "resolve_effective_sibling_alpha",
    "resolve_sibling_gate_profile",
    "resolve_sibling_gate_profile_config",
    "run_gate_annotation_pipeline",
    "selected_global_sibling_min_permutation_p_value",
    "selected_root_permutation_p_value",
]
