"""Typed output payload for one node's spectral decomposition work."""

from __future__ import annotations

from dataclasses import dataclass

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


__all__ = ["NodeSpectralResult"]
