"""Typed input payload for one node's spectral decomposition work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tree_break_selection.tree.feature_space import FeatureSpace


@dataclass(frozen=True)
class NodeSpectralTask:
    """Input payload for one node's spectral decomposition work."""

    node_id: str
    row_indices: tuple[int, ...]
    internal_distributions: tuple[np.ndarray, ...]
    null_distribution: np.ndarray
    feature_space: FeatureSpace
    continuous_covariance_by_block: dict[str, np.ndarray] | None
    mp_row_count_mode: str


__all__ = ["NodeSpectralTask"]
