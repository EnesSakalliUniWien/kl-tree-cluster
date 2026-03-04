"""Typed containers for per-node spectral decomposition workers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NodeSpectralTask:
    """Input payload for one node's spectral decomposition work."""

    node_id: str
    row_indices: tuple[int, ...]
    internal_distributions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class NodeSpectralResult:
    """Output payload for one node's spectral decomposition work."""

    node_id: str
    projection_dimension: int
    projection_matrix: np.ndarray | None
    eigenvalues: np.ndarray | None


__all__ = [
    "NodeSpectralResult",
    "NodeSpectralTask",
]
