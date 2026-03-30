"""Per-child null-prior estimate type for sibling null-prior interpolation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChildSiblingNullPriorEstimate:
    """Per-child interpolated sibling null prior from tree-neighborhood estimation."""

    sibling_null_prior: float
    neighborhood_estimate: float
    ancestor_support: float
    neighborhood_interpolation_weight: float


__all__ = ["ChildSiblingNullPriorEstimate"]
