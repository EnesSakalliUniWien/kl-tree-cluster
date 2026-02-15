"""Image-to-PDF concatenation helpers for benchmark plot files."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import re
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from .pdf_layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure

logger = logging.getLogger(__name__)


def _get_case_from_filename(path: Path) -> int | None:
    """Extract the case number from a plot filename."""
    match = re.search(r"case_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _group_files_by_case(plots_root: Path, pattern: str) -> Dict[int, List[Path]]:
    """Group matching plot files by extracted case number."""
    files_by_case = defaultdict(list)
    try:
        for path in plots_root.glob(pattern):
            case_num = _get_case_from_filename(path)
            if case_num is not None:
                files_by_case[case_num].append(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Error while scanning for plot files: %s", exc)
    return files_by_case


def concat_plots_to_pdf(
    plots_root: Path,
    pattern: str,
    output_pdf: Path,
    verbose: bool = False,
    dpi: int = 150,
):
    """Concatenate plot images into a single multi-page PDF grouped by test case."""
    files_by_case = _group_files_by_case(plots_root, pattern)

    if not files_by_case:
        if verbose:
            logger.info("No plots found for pattern '%s' in %s", pattern, plots_root)
        return None

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for case_num in sorted(files_by_case.keys()):
            title_fig = plt.figure(figsize=PDF_PAGE_SIZE_INCHES)
            title_fig.text(
                0.5,
                0.5,
                f"Test Case {case_num}",
                ha="center",
                va="center",
                fontsize=24,
                alpha=0.5,
            )
            prepare_pdf_figure(title_fig)
            pdf.savefig(title_fig)
            plt.close(title_fig)

            for path in sorted(files_by_case[case_num]):
                try:
                    with Image.open(path) as img:
                        fig = plt.figure(figsize=PDF_PAGE_SIZE_INCHES, dpi=dpi)
                        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94], frameon=False)
                        ax.imshow(img, aspect="equal", resample=True)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        prepare_pdf_figure(fig)
                        pdf.savefig(fig)
                        plt.close(fig)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("Failed to add image %s to PDF: %s", path, exc)

    if verbose:
        total = sum(len(v) for v in files_by_case.values())
        logger.info("Created PDF with %s plots: %s", total, output_pdf)
    return output_pdf


def concat_tree_plots(plots_root: Path, **kwargs):
    return concat_plots_to_pdf(
        plots_root, "tree_case_*.png", plots_root / "tree_plots.pdf", **kwargs
    )


def concat_umap_plots(plots_root: Path, **kwargs):
    return concat_plots_to_pdf(
        plots_root,
        "umap_comparison_case_*.png",
        plots_root / "umap_plots.pdf",
        **kwargs,
    )


def concat_manifold_plots(plots_root: Path, **kwargs):
    return concat_plots_to_pdf(
        plots_root, "manifold_case_*.png", plots_root / "manifold_plots.pdf", **kwargs
    )


def concat_umap_3d_plots(plots_root: Path, **kwargs):
    return concat_plots_to_pdf(
        plots_root, "umap3d_case_*.png", plots_root / "umap3d_plots.pdf", **kwargs
    )


__all__ = [
    "concat_plots_to_pdf",
    "concat_tree_plots",
    "concat_umap_plots",
    "concat_manifold_plots",
    "concat_umap_3d_plots",
]
