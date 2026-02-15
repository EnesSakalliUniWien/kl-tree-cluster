"""Utility package for shared benchmark helpers."""

from .core import (
    _estimate_dbscan_eps,
    _knn_edge_weights,
    _normalize_labels,
    _resolve_n_neighbors,
)
from .decomposition import (
    _create_report_dataframe,
    _create_report_dataframe_from_labels,
    _labels_from_decomposition,
)
from .time import format_timestamp_utc
from .method_selection import resolve_methods_from_env
from .case_execution import (
    run_case_isolated,
    run_case_with_optional_isolation,
)
from .pdf_layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure
from .pdf_merge import merge_existing_pdfs
from .pdf_image_concat import (
    concat_manifold_plots,
    concat_plots_to_pdf,
    concat_tree_plots,
    concat_umap_3d_plots,
    concat_umap_plots,
)
from .pdf_figure_split import split_collected_figs_to_pdfs

__all__ = [
    "_normalize_labels",
    "_estimate_dbscan_eps",
    "_resolve_n_neighbors",
    "_knn_edge_weights",
    "_labels_from_decomposition",
    "_create_report_dataframe",
    "_create_report_dataframe_from_labels",
    "format_timestamp_utc",
    "resolve_methods_from_env",
    "run_case_isolated",
    "run_case_with_optional_isolation",
    "PDF_PAGE_SIZE_INCHES",
    "prepare_pdf_figure",
    "merge_existing_pdfs",
    "concat_plots_to_pdf",
    "concat_tree_plots",
    "concat_umap_plots",
    "concat_manifold_plots",
    "concat_umap_3d_plots",
    "split_collected_figs_to_pdfs",
]
