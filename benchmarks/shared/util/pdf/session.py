"""PDF session helpers for benchmark report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.shared.util.time import format_timestamp_utc

if TYPE_CHECKING:
    from matplotlib.backends.backend_pdf import PdfPages


def resolve_pdf_output_path(
    concat_output: str | None,
    *,
    plots_root: Path,
    started_at: datetime,
) -> Path:
    """Resolve the PDF output path, ensuring a timestamped default and `.pdf` suffix."""
    if concat_output:
        path = Path(concat_output)
        return path.with_suffix(".pdf") if path.suffix == "" else path

    stamp = format_timestamp_utc(started_at)
    return plots_root / f"benchmark_plots_{stamp}.pdf"


def open_pdf_pages(
    output_pdf: Path | None,
    *,
    plots_root: Path,
    started_at: datetime,
) -> tuple[Path, "PdfPages"]:
    """Create the output directory and open a `PdfPages` writer."""
    from matplotlib.backends.backend_pdf import PdfPages

    if output_pdf is None:
        output_pdf = resolve_pdf_output_path(
            None, plots_root=plots_root, started_at=started_at
        )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    return output_pdf, PdfPages(output_pdf)


__all__ = ["resolve_pdf_output_path", "open_pdf_pages"]
