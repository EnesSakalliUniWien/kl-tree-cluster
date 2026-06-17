from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoppingEdgeInfo:
    """Recovered information for the ancestor edge that stopped descent."""

    stopping_child_node: str
    stopping_edge_p_value: float
    distance_to_stopping_edge: float
    """Unweighted hop count on the undirected tree (not branch-length distance)."""
    generations_above: int
