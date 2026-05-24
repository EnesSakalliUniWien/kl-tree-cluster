"""Sibling pair record dataclass for the pair_testing package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SiblingPairRecord:
    """Raw per-parent sibling-test record used by calibration pipelines."""

    parent: object
    left: object
    right: object
    stat: float
    reference_scale: float
    """Scale in the projected quadratic reference law; unit for orthonormal PCA."""
    degrees_of_freedom: float
    p_value: float
    branch_length_sum: float
    n_parent: int
    is_null_like: bool
    is_edge_blocked: bool = False
    sibling_null_weight: float = 0.0
    """Weight that this sibling pair represents empirical-null structure."""
    sibling_projection_dimension: float = 0.0
    """Projection dimension used by the sibling test and inflation context."""


__all__ = ["SiblingPairRecord"]
