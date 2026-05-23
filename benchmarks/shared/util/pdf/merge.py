"""PDF merge helper for existing case-level reports."""

from __future__ import annotations

import shutil
from pathlib import Path


def _load_pdf_writer():
    try:
        from pypdf import PdfWriter
    except ImportError as exc:
        raise RuntimeError(
            "PDF report merging requires pypdf. Install the visualization extra: "
            "`pip install -e .[viz]`."
        ) from exc
    return PdfWriter


def merge_existing_pdfs(
    pdf_paths: list[Path],
    output_pdf: Path,
    *,
    verbose: bool = True,
) -> bool:
    """Merge existing PDFs into one file using the declared Python PDF dependency."""
    existing = [p for p in pdf_paths if p.exists()]
    if not existing:
        if verbose:
            print("No case-level PDFs found to concatenate.")
        return False

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if len(existing) == 1:
        shutil.copyfile(existing[0], output_pdf)
        if verbose:
            print(f"Single PDF report copied to {output_pdf}")
        return True

    PdfWriter = _load_pdf_writer()
    writer = PdfWriter()
    for pdf_path in existing:
        writer.append(str(pdf_path))
    with output_pdf.open("wb") as handle:
        writer.write(handle)
    writer.close()
    if verbose:
        print(f"Concatenated {len(existing)} case PDFs to {output_pdf}")
    return True


__all__ = ["merge_existing_pdfs"]
