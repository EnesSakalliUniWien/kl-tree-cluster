"""PDF page layout helpers for benchmark plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt

PDF_PAGE_SIZE_INCHES = (11.0, 8.5)  # Landscape Letter


def prepare_pdf_figure(fig: plt.Figure) -> None:
    """Normalize figure geometry before writing to PDF."""
    fig.set_size_inches(*PDF_PAGE_SIZE_INCHES, forward=True)


__all__ = ["PDF_PAGE_SIZE_INCHES", "prepare_pdf_figure"]
