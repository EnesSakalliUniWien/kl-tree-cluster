"""Typed output payload for one node's spectral decomposition work."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NodeSpectralResult:
    """Output payload for one node's spectral decomposition work."""

    node_id: str
    raw_mp_signal_count: int
    test_projection_dimension: int
    effective_independent_rows: int
    mp_threshold_rows: int
    projection_matrix: np.ndarray | None
    eigenvalues: np.ndarray | None
    descendant_leaf_row_count: int = 0
    internal_distribution_row_count: int = 0
    spectral_matrix_row_count: int = 0
    full_eigenvalues: np.ndarray | None = None
    active_feature_count: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)


__all__ = ["NodeSpectralResult"]
