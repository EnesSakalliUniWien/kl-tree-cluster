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
    structural_dimension: float = 0.0  # sibling k derived from child edge dimensions
    smoothed_sibling_null_prior: float | None = None
    """Tree-neighborhood-interpolated sibling null prior for nodes whose
    Gate 2 edge test was blocked (ancestor-blocked or untested)."""
    ancestor_support: float | None = None
    blend_lambda: float | None = None


__all__ = ["SiblingPairRecord"]
