"""Configuration constants for plot styling."""

from __future__ import annotations

# Centralized visual style constants for tree plotting.
UNASSIGNED_NODE_COLOR = "#CCCCCC"
INTERNAL_NODE_COLOR = "#D7D7D7"

LEAF_NODE_STYLE = {
    "alpha": 0.92,
    "linewidths": 0.6,
    "edgecolors": "#F4F4F4",
}

INTERNAL_NODE_STYLE = {
    "alpha": 0.78,
    "linewidths": 0.0,
    "node_color": INTERNAL_NODE_COLOR,
}

EDGE_DRAW_ORDER = ("missing", "not_different", "different")
EDGE_STYLES = {
    "missing": {
        "edge_color": "#8A8A8A",
        "width": 0.8,
        "style": "dashed",
        "alpha": 0.75,
        "legend_label": "Edge: sibling test missing",
    },
    "not_different": {
        "edge_color": "#BEBEBE",
        "width": 1.0,
        "style": "solid",
        "alpha": 0.8,
        "legend_label": "Edge: sibling not different",
    },
    "different": {
        "edge_color": "#1F1F1F",
        "width": 2.0,
        "style": "solid",
        "alpha": 0.98,
        "legend_label": "Edge: sibling different",
    },
}

HALO_SIZE_MULTIPLIER = 2.8
HALO_SIZE_OFFSET = 14
HALO_STYLES = {
    "significant": {
        "edgecolors": "#1C1C1C",
        "linewidths": 1.2,
        "linestyle": "solid",
        "legend_label": "Node: child-parent significant",
    },
    "tested_not_significant": {
        "edgecolors": "#6E6E6E",
        "linewidths": 1.2,
        "linestyle": (0, (1.0, 1.7)),
        "legend_label": "Node: tested, not significant",
    },
}

LEGEND_NODE_MARKER_SIZE = 7

__all__ = [
    "UNASSIGNED_NODE_COLOR",
    "INTERNAL_NODE_COLOR",
    "LEAF_NODE_STYLE",
    "INTERNAL_NODE_STYLE",
    "EDGE_DRAW_ORDER",
    "EDGE_STYLES",
    "HALO_SIZE_MULTIPLIER",
    "HALO_SIZE_OFFSET",
    "HALO_STYLES",
    "LEGEND_NODE_MARKER_SIZE",
]
