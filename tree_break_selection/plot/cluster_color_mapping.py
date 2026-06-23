"""
Utilities for consistent cluster color assignment across plots.

Key improvements over a plain ``cmap="tab10"`` approach:
- Uses *discrete* palettes (not interpolated continuous mapping)
- Uses a named large-N palette when many cluster colors are required
- Supports an explicit "unassigned" color for labels like -1 via ``under=``
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from math import gcd
from typing import Dict, Iterable, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


@dataclass(frozen=True)
class ClusterColorSpec:
    """Color configuration for integer cluster IDs 0..n-1 plus optional -1."""

    n_clusters: int
    colors: List[str]
    unassigned_color: str
    cmap: ListedColormap
    norm: BoundaryNorm
    id_to_color: Dict[int, str]


def _discrete_colors_from_matplotlib_cmap(name: str, n: int) -> List[str]:
    cmap = plt.get_cmap(name)
    if hasattr(cmap, "colors") and cmap.colors is not None:
        base = list(cmap.colors)
        if len(base) >= n:
            return [mcolors.to_hex(base[i]) for i in range(n)]
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _max_contrast_order(colors: List[str]) -> List[str]:
    """Reorder large palettes so adjacent integer IDs are less visually similar."""
    n = len(colors)
    if n <= 2:
        return colors

    step = max(1, n // 2 - 1)
    while gcd(step, n) != 1:
        step -= 1
    return [colors[(i * step) % n] for i in range(n)]


def _large_cluster_palette(n: int) -> List[str]:
    """Build a dependency-free large-N palette for diagnostic cluster plots."""
    colors: List[str] = []
    for i in range(n):
        hue = i / max(n, 1)
        saturation = 0.62 + 0.23 * ((i % 3) / 2)
        value = 0.74 + 0.16 * ((i % 2))
        colors.append(mcolors.to_hex(colorsys.hsv_to_rgb(hue, saturation, value)))
    return _max_contrast_order(colors)


def build_cluster_color_spec(
    n_clusters: int,
    *,
    base_cmap: str | None = None,
    unassigned_color: str = "#CCCCCC",
) -> ClusterColorSpec:
    """
    Build a discrete colormap + normalizer for cluster labels.

    - cluster labels are expected to be integers 0..n_clusters-1
    - unassigned labels like -1 will map to ``unassigned_color`` via ``under=``
    """
    n_clusters = int(n_clusters)
    if n_clusters < 0:
        raise ValueError("n_clusters must be >= 0")

    if n_clusters == 0:
        cmap = ListedColormap([unassigned_color])
        # Older Matplotlib versions don't accept bad/under kwargs in __init__.
        cmap.set_bad(unassigned_color)
        cmap.set_under(unassigned_color)
        boundaries = np.array([-0.5, 0.5], dtype=float)
        norm = BoundaryNorm(boundaries, ncolors=1, clip=False)
        return ClusterColorSpec(
            n_clusters=0,
            colors=[],
            unassigned_color=unassigned_color,
            cmap=cmap,
            norm=norm,
            id_to_color={-1: unassigned_color},
        )

    if base_cmap is not None:
        colors = _discrete_colors_from_matplotlib_cmap(base_cmap, n_clusters)
    else:
        if n_clusters <= 10:
            colors = _discrete_colors_from_matplotlib_cmap("tab10", n_clusters)
        elif n_clusters <= 20:
            colors = _discrete_colors_from_matplotlib_cmap("tab20", n_clusters)
        else:
            colors = _large_cluster_palette(n_clusters)

    cmap = ListedColormap(colors)
    # Older Matplotlib versions don't accept bad/under kwargs in __init__.
    cmap.set_bad(unassigned_color)
    cmap.set_under(unassigned_color)
    boundaries = np.arange(-0.5, n_clusters + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, ncolors=n_clusters, clip=False)

    id_to_color: Dict[int, str] = {i: colors[i] for i in range(n_clusters)}
    id_to_color[-1] = unassigned_color
    return ClusterColorSpec(
        n_clusters=n_clusters,
        colors=colors,
        unassigned_color=unassigned_color,
        cmap=cmap,
        norm=norm,
        id_to_color=id_to_color,
    )


def present_cluster_ids(labels: Iterable[int]) -> List[int]:
    """Sorted unique cluster IDs from label sequence (excludes -1)."""
    unique = {int(label) for label in labels}
    unique.discard(-1)
    return sorted(unique)
