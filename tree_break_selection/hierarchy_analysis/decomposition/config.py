"""Configuration contracts for tree decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tree_break_selection import config as runtime_config
from tree_break_selection.hierarchy_analysis.decomposition.gates.profiles import (
    SiblingGateProfile,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION,
)
from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
    INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER,
    MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    DEFAULT_INTERNAL_SUPPORT_THRESHOLDS,
)
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.types.inflation_model import (
    CalibrationSupportThresholds,
)
from tree_break_selection.tree.distributions import (
    DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT,
    DEFAULT_CONTINUOUS_COVARIANCE_POLICY,
)

TRACE_LEVEL_COMPACT = "compact"
TRACE_LEVEL_FULL = "full"
TraceLevel = Literal["compact", "full"]


def validate_trace_level(trace_level: str) -> TraceLevel:
    """Validate the decomposition trace collection level."""
    if trace_level == TRACE_LEVEL_COMPACT:
        return TRACE_LEVEL_COMPACT
    if trace_level == TRACE_LEVEL_FULL:
        return TRACE_LEVEL_FULL
    raise ValueError(
        f"trace_level must be one of {(TRACE_LEVEL_COMPACT, TRACE_LEVEL_FULL)!r}; "
        f"got {trace_level!r}."
    )


@dataclass(frozen=True)
class DecompositionConfig:
    """Immutable knobs that affect decomposition traversal and gate annotation."""

    edge_alpha: float = DEFAULT_EDGE_ALPHA
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA
    spectral_minimum_dimension: int = EDGE_GATE_SPECTRAL_MINIMUM_PROJECTION_DIMENSION
    adaptive_projection_dimension_energy_fraction: float | None = None
    spectral_include_internal_barycenters: bool = False
    spectral_internal_distribution_mode: str = INTERNAL_DISTRIBUTION_EMPIRICAL_BARYCENTER
    spectral_mp_row_count_mode: str = MP_ROW_COUNT_LEAF_EFFECTIVE_ROWS
    continuous_covariance_policy: str = DEFAULT_CONTINUOUS_COVARIANCE_POLICY
    continuous_covariance_min_child_leaf_count: int = (
        DEFAULT_CONTINUOUS_COVARIANCE_MIN_CHILD_LEAF_COUNT
    )
    edge_branch_length_variance_policy: str = EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NONE
    sibling_gate_profile: str | SiblingGateProfile | None = None
    sibling_gate_method: str = "projected_wald_inflation"
    sibling_gate_alpha_penalty: float = 1.0
    root_stability_guard_threshold: float | None = None
    root_stability_subsample_replicates: int = 0
    root_stability_feature_fraction: float = 0.8
    root_stability_seed: int = 0
    root_stability_tree_distance_metric: str = "hamming"
    root_stability_tree_linkage_method: str = "average"
    root_selective_permutation_guard_replicates: int = 0
    root_selective_permutation_guard_seed: int = 0
    root_selective_permutation_guard_alpha: float | None = None
    root_selective_permutation_guard_scope: str = "root"
    root_selective_permutation_guard_tree_distance_metric: str = "hamming"
    root_selective_permutation_guard_tree_linkage_method: str = "average"
    enforce_internal_support_thresholds: bool = False
    internal_support_thresholds: CalibrationSupportThresholds = DEFAULT_INTERNAL_SUPPORT_THRESHOLDS
    spectral_transport_passthrough_guard: bool = False
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST
    spectral_transport_require_mp_blocks: bool = True
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    )
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    )
    passthrough: bool = runtime_config.PASSTHROUGH
    trace_level: TraceLevel = TRACE_LEVEL_COMPACT

    def __post_init__(self) -> None:
        validate_trace_level(str(self.trace_level))


__all__ = [
    "DecompositionConfig",
    "TRACE_LEVEL_COMPACT",
    "TRACE_LEVEL_FULL",
    "TraceLevel",
    "validate_trace_level",
]
