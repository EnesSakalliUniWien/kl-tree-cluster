"""Reusable plotting engines for cluster colors, trees, and report panels."""

from importlib import import_module

from .backend import DEFAULT_FILE_BACKEND, configure_matplotlib_backend

configure_matplotlib_backend()

_PUBLIC_IMPORTS = {
    "ClusterColorSpec": ("cluster_color_mapping", "ClusterColorSpec"),
    "build_cluster_color_spec": ("cluster_color_mapping", "build_cluster_color_spec"),
    "draw_image_panel": ("image_panel", "draw_image_panel"),
    "load_overlay_data": ("multiscale_umap", "load_overlay_data"),
    "plot_tree_with_clusters": ("cluster_tree_visualization", "plot_tree_with_clusters"),
    "render_multiscale_umap_overlay": ("multiscale_umap", "render_multiscale_umap_overlay"),
}

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


def __getattr__(name: str):
    """Load a plotting engine only when its public symbol is requested."""
    try:
        module_name, attribute_name = _PUBLIC_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
