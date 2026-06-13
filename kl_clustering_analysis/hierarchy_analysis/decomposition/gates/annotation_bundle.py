"""Typed output from the gate annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

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
    enforce_internal_support_thresholds: bool = False
    internal_support_thresholds_signature: tuple[tuple[str, float | int], ...] = ()
    external_selected_tail_calibration_enabled: bool = False
    external_selected_tail_rule_count: int = 0


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
