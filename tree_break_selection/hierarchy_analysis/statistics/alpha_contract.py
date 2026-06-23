"""Canonical statistical thresholds for the Tree-Break Selection gate pipeline."""

from __future__ import annotations

# Child-parent edge Tree-BH threshold. Conservative because same-data hierarchy
# construction strongly selects edge contrasts.
DEFAULT_EDGE_ALPHA: float = 0.001

# Sibling traversal/FDR threshold. Conservative to reduce false splitting near
# high-level tree boundaries.
DEFAULT_SIBLING_ALPHA: float = 0.01

__all__ = [
    "DEFAULT_EDGE_ALPHA",
    "DEFAULT_SIBLING_ALPHA",
]
