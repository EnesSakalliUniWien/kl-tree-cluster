"""Reusable plotting engines for cluster colors, trees, and report panels."""

from .backend import DEFAULT_FILE_BACKEND, configure_matplotlib_backend
from .cluster_color_mapping import ClusterColorSpec, build_cluster_color_spec
from .cluster_tree_visualization import plot_tree_with_clusters
from .image_panel import draw_image_panel
from .multiscale_umap import load_overlay_data, render_multiscale_umap_overlay

__all__ = [
    "ClusterColorSpec",
    "DEFAULT_FILE_BACKEND",
    "build_cluster_color_spec",
    "configure_matplotlib_backend",
    "draw_image_panel",
    "load_overlay_data",
    "plot_tree_with_clusters",
    "render_multiscale_umap_overlay",
]
