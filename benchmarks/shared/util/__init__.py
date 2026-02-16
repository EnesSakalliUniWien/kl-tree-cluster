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
from .case_inputs import prepare_case_inputs
from .time import format_timestamp_utc
from .method_selection import (
    resolve_methods_from_env,
    resolve_selected_methods_and_param_sets,
)
from .case_execution import (
    run_case_isolated,
    run_case_with_optional_isolation,
)
from .case_run import run_single_case
from .pdf.layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure
from .pdf.merge import merge_existing_pdfs
from .pdf.session import open_pdf_pages, resolve_pdf_output_path
from .pdf.image_concat import (
    concat_manifold_plots,
    concat_plots_to_pdf,
    concat_tree_plots,
    concat_umap_3d_plots,
    concat_umap_plots,
)
from .pdf.figure_split import split_collected_figs_to_pdfs

__all__ = [
    "_normalize_labels",
    "_estimate_dbscan_eps",
    "_resolve_n_neighbors",
    "_knn_edge_weights",
    "_labels_from_decomposition",
    "_create_report_dataframe",
    "_create_report_dataframe_from_labels",
    "prepare_case_inputs",
    "format_timestamp_utc",
    "resolve_methods_from_env",
    "resolve_selected_methods_and_param_sets",
    "run_case_isolated",
    "run_case_with_optional_isolation",
    "run_single_case",
    "PDF_PAGE_SIZE_INCHES",
    "prepare_pdf_figure",
    "merge_existing_pdfs",
    "resolve_pdf_output_path",
    "open_pdf_pages",
    "concat_plots_to_pdf",
    "concat_tree_plots",
    "concat_umap_plots",
    "concat_manifold_plots",
    "concat_umap_3d_plots",
    "split_collected_figs_to_pdfs",
]
