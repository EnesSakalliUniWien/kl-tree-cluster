"""PDF page layout helpers for benchmark plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt

PDF_PAGE_SIZE_INCHES = (11.0, 8.5)  # Landscape Letter
PDF_WIDE_PAGE_SIZE_INCHES = (13.0, 8.5)
PDF_PAGE_SIZE_ATTR = "_tbs_pdf_page_size_inches"


def set_pdf_page_size(fig: plt.Figure, page_size_inches: tuple[float, float]) -> None:
    """Record the intended PDF page size for a figure."""
    setattr(fig, PDF_PAGE_SIZE_ATTR, tuple(page_size_inches))


def prepare_pdf_figure(fig: plt.Figure) -> None:
    """Normalize figure geometry before writing to PDF."""
    page_size = getattr(fig, PDF_PAGE_SIZE_ATTR, PDF_PAGE_SIZE_INCHES)
    fig.set_size_inches(*page_size, forward=True)


__all__ = [
    "PDF_PAGE_SIZE_INCHES",
    "PDF_WIDE_PAGE_SIZE_INCHES",
    "prepare_pdf_figure",
    "set_pdf_page_size",
]
