"""PDF-focused utility helpers for benchmark plotting and report assembly."""

from .figure_split import split_collected_figs_to_pdfs
from .image_concat import (
    concat_manifold_plots,
    concat_plots_to_pdf,
    concat_tree_plots,
    concat_umap_3d_plots,
    concat_umap_plots,
)
from .layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure
from .merge import merge_existing_pdfs
from .session import open_pdf_pages, resolve_pdf_output_path

__all__ = [
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
