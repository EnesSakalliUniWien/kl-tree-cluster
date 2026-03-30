"""Edge-level metadata type for sibling null-prior interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeLevelMetadata:
    """Edge-level Gate 2 metadata used for tree-neighborhood smoothing."""

    edge_child_ids: list[str]
    child_parent_edge_tested: np.ndarray
    child_parent_edge_significant: np.ndarray
    child_parent_edge_bh_p_values: np.ndarray
    edge_projection_dimensions: dict[str, int] | None


__all__ = ["EdgeLevelMetadata"]
