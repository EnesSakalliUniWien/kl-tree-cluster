"""Typed output from the gate annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.spectral_transport import (
    DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE,
    DEFAULT_SPECTRAL_TRANSPORT_MAX_COST,
    DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)


@dataclass(frozen=True)
class GateMetadata:
    """Metadata for one statistical gate stage."""

    gate: str
    alpha: float


@dataclass(frozen=True)
class GateAnnotationConfigMetadata:
    """Config values that affect gate annotation outputs."""

    spectral_minimum_dimension: int
    spectral_include_internal_barycenters: bool = False
    sibling_gate_profile_id: str | None = None
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
    internal_support_thresholds_signature: tuple[tuple[str, float | int], ...] = ()
    external_selected_tail_calibration_enabled: bool = False
    external_selected_tail_rule_count: int = 0
    spectral_transport_passthrough_guard: bool = False
    spectral_transport_max_cost: float = DEFAULT_SPECTRAL_TRANSPORT_MAX_COST
    spectral_transport_require_mp_blocks: bool = True
    spectral_transport_block_log_tolerance: float = (
        DEFAULT_SPECTRAL_TRANSPORT_BLOCK_LOG_TOLERANCE
    )
    spectral_transport_unmatched_mode_penalty: float = (
        DEFAULT_SPECTRAL_TRANSPORT_UNMATCHED_MODE_PENALTY
    )


@dataclass(frozen=True)
class GateAnnotationLeafDataMetadata:
    """Leaf-data identity used to validate reusable gate annotations."""

    present: bool
    shape: tuple[int, int] | None = None
    content_hash: str | None = None
    feature_space_signature: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class GateAnnotationMetadata:
    """Metadata for a complete gate annotation run."""

    pipeline: str
    edge: GateMetadata
    sibling: GateMetadata
    config: GateAnnotationConfigMetadata
    leaf_data: GateAnnotationLeafDataMetadata


@dataclass
class EdgeGateResult:
    """Edge-gate output passed into sibling-divergence annotation."""

    annotated_df: pd.DataFrame
    spectral_context: SpectralContext
    metadata: GateMetadata


@dataclass
class GateAnnotationBundle:
    """Gate annotation output plus metadata required for reuse."""

    annotated_df: pd.DataFrame
    metadata: GateAnnotationMetadata
    edge_gate_result: EdgeGateResult
    stage_timings: dict[str, float] = field(default_factory=dict)


__all__ = [
    "EdgeGateResult",
    "GateAnnotationBundle",
    "GateAnnotationConfigMetadata",
    "GateAnnotationLeafDataMetadata",
    "GateAnnotationMetadata",
    "GateMetadata",
]
