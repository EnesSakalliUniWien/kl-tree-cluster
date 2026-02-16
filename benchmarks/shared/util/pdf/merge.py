"""PDF merge helper for existing case-level reports."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def merge_existing_pdfs(
    pdf_paths: list[Path],
    output_pdf: Path,
    *,
    verbose: bool = True,
) -> bool:
    """Merge existing PDFs into one file using pdfunite/ghostscript when available."""
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

    pdfunite = shutil.which("pdfunite")
    if pdfunite:
        cmd = [pdfunite, *[str(p) for p in existing], str(output_pdf)]
        subprocess.run(cmd, check=True)
        if verbose:
            print(f"Concatenated {len(existing)} case PDFs to {output_pdf}")
        return True

    gs = shutil.which("gs")
    if gs:
        cmd = [
            gs,
            "-dBATCH",
            "-dNOPAUSE",
            "-q",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={output_pdf}",
            *[str(p) for p in existing],
        ]
        subprocess.run(cmd, check=True)
        if verbose:
            print(f"Concatenated {len(existing)} case PDFs to {output_pdf}")
        return True

    if verbose:
        print(
            "Could not concatenate PDFs automatically (missing 'pdfunite' and 'gs'). "
            f"Case-level PDFs remain in {existing[0].parent}"
        )
    return False


__all__ = ["merge_existing_pdfs"]
