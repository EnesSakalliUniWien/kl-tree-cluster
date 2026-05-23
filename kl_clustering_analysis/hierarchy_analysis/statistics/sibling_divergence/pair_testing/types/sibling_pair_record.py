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
    """Satterthwaite scale in the projected quadratic reference law."""
    degrees_of_freedom: float
    p_value: float
    branch_length_sum: float
    n_parent: int
    is_null_like: bool
    is_gate2_blocked: bool = False
    sibling_null_weight: float = 0.0
    """Weight that this sibling pair represents empirical-null structure."""
    sibling_calibration_scale: float = 0.0
    """Positive scale coordinate used for local post-selection scale calibration."""
    projection_dimension_source: str = ""
    """Which path supplied the sibling test projection dimension."""
    resolved_projection_dimension: float = float("nan")
    """Resolved projection dimension used by the sibling test."""
    used_parent_principal_component_basis: bool = False
    """Whether the sibling test projected into the parent principal-component basis."""


__all__ = ["SiblingPairRecord"]
