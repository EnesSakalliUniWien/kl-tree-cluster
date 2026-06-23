"""Neighborhood reference-set type for sibling null-prior interpolation."""

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


__all__ = ["NeighborhoodReferenceSet"]
