from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalNeighborInfo:
    """Nearest tested significant edge for signal-pressure computation."""

    signal_node_id: str | None
    signal_p_value: float
    tree_distance_to_signal_node: float
    """Unweighted hop count on the undirected tree (not branch-length distance)."""
