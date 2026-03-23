"""Type definitions for the pair_testing package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar


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
    edge_weight: float = 0.0  # min(p_edge_left, p_edge_right) for continuous calibration


class DeflatableSiblingRecord(Protocol):
    """Structural type for per-node records that can be focal-deflated."""

    parent: str
    stat: float
    degrees_of_freedom: float
    is_null_like: bool
    is_gate2_blocked: bool


_R = TypeVar("_R", bound=DeflatableSiblingRecord)


__all__ = ["SiblingPairRecord", "DeflatableSiblingRecord", "_R"]
