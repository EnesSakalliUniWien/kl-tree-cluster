"""Typed input payload for one node's spectral decomposition work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NodeSpectralTask:
    """Input payload for one node's spectral decomposition work."""

    node_id: str
    row_indices: tuple[int, ...]
    internal_distributions: tuple[np.ndarray, ...]


__all__ = ["NodeSpectralTask"]
