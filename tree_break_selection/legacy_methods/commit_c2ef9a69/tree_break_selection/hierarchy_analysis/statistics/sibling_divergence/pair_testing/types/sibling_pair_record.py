"""Sibling pair record dataclass for the pair_testing package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SiblingPairRecord:
    """Raw per-parent sibling-test record used by calibration pipelines."""

    parent: str
    left: str
    right: str
    stat: float
    degrees_of_freedom: float
    p_value: float
    branch_length_sum: float
    n_parent: int
    is_null_like: bool
    is_gate2_blocked: bool = False
    sibling_null_prior_from_edge_pvalue: float = 0.0
    """min(p_edge_left, p_edge_right) — prior probability that this sibling
    pair is null, derived from the Gate 2 child-parent edge p-values.
    High value → both children look null-like (no signal detected by Gate 2).
    Used as weight when estimating the post-selection inflation factor ĉ."""
    sibling_test_calibration_scale: float = 0.0
    """Rough sibling split scale used to match nearby calibration examples.

    Prefer the projection dimension derived from edge comparisons when
    available; otherwise fall back to the sibling test degrees of freedom.
    """
    projection_dimension_source: str = ""
    """Which path supplied the sibling test projection dimension."""
    resolved_projection_dimension: float = float("nan")
    """Resolved projection dimension used by the sibling test."""
    used_parent_principal_component_basis: bool = False
    """Whether the sibling test projected into the parent principal-component basis."""
    smoothed_sibling_null_prior: float | None = None
    """Tree-neighborhood-interpolated sibling null prior for nodes whose
    Gate 2 edge test was blocked (ancestor-blocked or untested)."""
    ancestor_support: float | None = None
    neighborhood_reliance: float | None = None


__all__ = ["SiblingPairRecord"]
