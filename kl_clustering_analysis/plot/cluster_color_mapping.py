"""
Utilities for consistent cluster color assignment across plots.

Key improvements over a plain ``cmap="tab10"`` approach:
- Uses *discrete* palettes (not interpolated continuous mapping)
- Handles many clusters (>20) with a deterministic fallback palette
- Supports an explicit "unassigned" color for labels like -1 via ``under=``
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
from typing import Dict, Iterable, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def _golden_ratio_palette(n: int, *, s: float = 0.65, v: float = 0.95) -> List[str]:
    """Generate n visually distinct colors with deterministic hue spacing."""
    if n <= 0:
        return []
    phi = (1 + 5**0.5) / 2  # golden ratio
    hues = (np.arange(n) / phi) % 1.0
    colors = [colorsys.hsv_to_rgb(float(h), s, v) for h in hues]
    return [mcolors.to_hex(rgb) for rgb in colors]


def _discrete_colors_from_matplotlib_cmap(name: str, n: int) -> List[str]:
    cmap = plt.get_cmap(name)
    if hasattr(cmap, "colors") and cmap.colors is not None:
        base = list(cmap.colors)
        if len(base) >= n:
            return [mcolors.to_hex(base[i]) for i in range(n)]
    # Fall back to sampling (may interpolate for ListedColormap with small LUT)
    return [
        mcolors.to_hex(cmap(i / max(n - 1, 1)))
        for i in range(n)
    ]


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
            # Optional: use seaborn husl for large N if available.
            try:
                import seaborn as sns  # type: ignore

                colors = [mcolors.to_hex(c) for c in sns.husl_palette(n_clusters)]
            except Exception:
                colors = _golden_ratio_palette(n_clusters)

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
    unique = {int(x) for x in labels if x is not None}
    unique.discard(-1)
    return sorted(unique)
