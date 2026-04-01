"""Recover stopping-edge ancestry and signal neighborhood for Tree-BH-untested edges.

When Tree-BH stops exploring below a sibling group because a parent edge was
not rejected, all descendant edges remain untested. This package recovers:

1. The nearest tested non-significant ancestor edge that stopped descent.
2. The nearest tested significant edge for signal-pressure discounting.
"""

from ._tree import build_tree_distance_resolver
from .serialization import (
    STOPPING_EDGE_INFO_ATTR_KEY,
    build_stopping_edge_attrs,
    parse_stopping_edge_attrs,
)
from .signals import recover_signal_neighbors
from .stopping_edges import recover_stopping_edge_info
from .types import SignalNeighborInfo, StoppingEdgeAttrPayload, StoppingEdgeInfo

__all__ = [
    "SignalNeighborInfo",
    "StoppingEdgeAttrPayload",
    "STOPPING_EDGE_INFO_ATTR_KEY",
    "StoppingEdgeInfo",
    "build_stopping_edge_attrs",
    "build_tree_distance_resolver",
    "parse_stopping_edge_attrs",
    "recover_signal_neighbors",
    "recover_stopping_edge_info",
]
