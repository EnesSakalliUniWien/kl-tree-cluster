"""
Plot helpers for benchmarking/validation outputs.

Functions are organized by purpose (summary, embedding comparison, manifold alignment,
and batch export helpers) but re-exported here for convenience.
"""

from .summary import create_validation_plot
from .embedding import (
    create_clustering_comparison_plot,
    create_clustering_comparison_plot_3d,
)
from .manifold import create_manifold_alignment_plot
from .export import (
    create_umap_plots_from_results,
    create_umap_3d_plots_from_results,
    create_manifold_plots_from_results,
    create_tree_plots_from_results,
)
from .runtime import (
    generate_benchmark_plots,
    log_detailed_results,
)

__all__ = [
    "create_validation_plot",
    "create_clustering_comparison_plot",
    "create_clustering_comparison_plot_3d",
    "create_manifold_alignment_plot",
    "create_umap_plots_from_results",
    "create_umap_3d_plots_from_results",
    "create_manifold_plots_from_results",
    "create_tree_plots_from_results",
    "generate_benchmark_plots",
    "log_detailed_results",
]
