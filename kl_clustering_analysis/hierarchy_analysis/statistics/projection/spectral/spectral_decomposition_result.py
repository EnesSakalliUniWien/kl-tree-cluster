"""Typed tree-level output for local spectral decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SpectralDecompositionResult:
    """Per-node spectral contracts consumed by projected-Wald gates."""

    test_projection_dimensions_by_node: dict[str, int]
    raw_mp_signal_counts_by_node: dict[str, int]
    effective_independent_rows_by_node: dict[str, int]
    mp_threshold_rows_by_node: dict[str, int]
    principal_component_projections_by_node: dict[str, np.ndarray]
    principal_component_eigenvalues_by_node: dict[str, np.ndarray]
    stage_timings: dict[str, float] = field(default_factory=dict)


__all__ = ["SpectralDecompositionResult"]
