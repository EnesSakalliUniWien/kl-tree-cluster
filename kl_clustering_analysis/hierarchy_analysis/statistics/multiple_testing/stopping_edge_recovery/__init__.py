"""Recover stopping-edge ancestry and signal neighborhood for Tree-BH-untested edges.

When Tree-BH stops exploring below a sibling group because a parent edge was
not rejected, all descendant edges remain untested. This package recovers:

1. The nearest tested non-significant ancestor edge that stopped descent.
2. The nearest tested significant edge for signal-pressure discounting.
"""

from .models import SignalNeighborInfo, StoppingEdgeInfo
from .signals import recover_signal_neighbors
from .stopping_edges import recover_stopping_edge_info

__all__ = [
    "SignalNeighborInfo",
    "StoppingEdgeInfo",
    "recover_signal_neighbors",
    "recover_stopping_edge_info",
]