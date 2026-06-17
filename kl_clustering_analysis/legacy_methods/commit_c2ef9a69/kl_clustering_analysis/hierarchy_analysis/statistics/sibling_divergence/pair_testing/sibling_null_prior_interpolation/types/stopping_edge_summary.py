"""Stopping-edge summary type for sibling null-prior interpolation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoppingEdgeSummary:
    """Stopping-edge support attached to a child edge."""

    stopping_edge_p_value: float
    distance_to_stopping_edge: float


__all__ = ["StoppingEdgeSummary"]
