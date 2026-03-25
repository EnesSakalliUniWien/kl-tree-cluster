from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoppingEdgeInfo:
    """Recovered information for the ancestor edge that stopped descent."""

    stopping_child_node: str
    stopping_edge_p_value: float
    distance_to_stopping_edge: float
    generations_above: int


@dataclass(frozen=True)
class SignalNeighborInfo:
    """Nearest tested significant edge for signal-pressure computation."""

    sig_node: str | None
    sig_p_value: float
    distance_to_sig: float