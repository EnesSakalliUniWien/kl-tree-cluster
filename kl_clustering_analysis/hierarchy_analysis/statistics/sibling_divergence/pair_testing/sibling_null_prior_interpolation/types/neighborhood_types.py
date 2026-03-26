"""Data types for tree-neighborhood sibling null prior interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeighborhoodReferenceSet:
    """Stable and signal edge neighborhoods used for sibling null prior interpolation."""

    stable_nodes: list[str]
    stable_p_values: np.ndarray
    stable_log_ks: np.ndarray
    signal_nodes: list[str]
    signal_p_values: np.ndarray


@dataclass(frozen=True)
class ChildSiblingNullPriorEstimate:
    """Per-child interpolated sibling null prior from tree-neighborhood estimation."""

    sibling_null_prior: float
    neighborhood_estimate: float
    ancestor_support: float
    neighborhood_interpolation_weight: float


__all__ = ["NeighborhoodReferenceSet", "ChildSiblingNullPriorEstimate"]
