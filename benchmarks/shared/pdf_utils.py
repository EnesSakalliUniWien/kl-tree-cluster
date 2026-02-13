"""PDF utility helpers for the benchmarking pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict
import re
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

logger = logging.getLogger(__name__)

PDF_PAGE_SIZE_INCHES = (11.0, 8.5)  # Landscape Letter


def prepare_pdf_figure(fig: plt.Figure) -> None:
    """Normalize figure geometry before writing to PDF."""
    fig.set_size_inches(*PDF_PAGE_SIZE_INCHES, forward=True)


def _classify_figure(fig) -> str:
    """Best-effort classifier for a Matplotlib figure.

    Used when caller provides a mixed list of figures without explicit
    categories.
    """
    parts: list[str] = []

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        try:
            parts.append(str(suptitle.get_text()))
        except Exception:
            pass

    try:
        parts.extend([str(t.get_text()) for t in getattr(fig, "texts", [])])
    except Exception:
        pass

    try:
        for ax in fig.axes:
            parts.append(str(ax.get_title()))
            parts.append(str(ax.get_xlabel()))
            parts.append(str(ax.get_ylabel()))
    except Exception:
        pass

    text = " ".join(p for p in parts if p).lower()

    if "umap3d" in text or ("3d" in text and "umap" in text):
        return "umap3d"
    if "umap" in text:
        return "umap"
    if "manifold" in text:
        return "manifold"
    if "tree" in text or "hierarchical" in text:
        return "tree"
    if "validation" in text:
        return "validation"
    return "other"


def _get_case_from_filename(path: Path) -> int | None:
    """Extracts the case number from a plot filename."""
    match = re.search(r"case_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _group_files_by_case(plots_root: Path, pattern: str) -> Dict[int, List[Path]]:
    """Scans a directory for a glob pattern and groups the found files by test case number."""
    files_by_case = defaultdict(list)
    try:
        all_files = plots_root.glob(pattern)
        for path in all_files:
            case_num = _get_case_from_filename(path)
            if case_num is not None:
                files_by_case[case_num].append(path)
    except Exception as e:
        logger.error(f"Error while scanning for plot files: {e}")
    return files_by_case


def concat_plots_to_pdf(
    plots_root: Path,
    pattern: str,
    output_pdf: Path,
    verbose: bool = False,
    dpi: int = 150,
):
    """Concatenate plot images into a single multi-page PDF, grouped by test case."""
    files_by_case = _group_files_by_case(plots_root, pattern)

    if not files_by_case:
        if verbose:
            logger.info(f"No plots found for pattern '{pattern}' in {plots_root}")
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for case_num in sorted(files_by_case.keys()):
            # Add a separator page for each test case
            fig = plt.figure(figsize=PDF_PAGE_SIZE_INCHES)
            fig.text(
                0.5,
                0.5,
                f"Test Case {case_num}",
                ha="center",
                va="center",
                fontsize=24,
                alpha=0.5,
            )
            prepare_pdf_figure(fig)
            pdf.savefig(fig)
            plt.close(fig)

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
                except Exception as e:
                    logger.error(f"Failed to process and add image {path} to PDF: {e}")
    if verbose:
        logger.info(
            f"Created PDF with {sum(len(v) for v in files_by_case.values())} plots: {output_pdf}"
        )
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


def split_collected_figs_to_pdfs(
    figs_with_info: list, output_dir: Path, verbose: bool = False
):
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir.mkdir(parents=True, exist_ok=True)
    categorized_figs = defaultdict(lambda: defaultdict(list))

    for item in figs_with_info:
        # Support both raw Figure objects and metadata dicts
        if isinstance(item, dict):
            fig = item.get("figure")
            case_num = item.get("test_case_num", 1)
        else:
            fig = item
            case_num = 1

        if fig is None:
            continue

        cat = _classify_figure(fig)
        categorized_figs[cat][case_num].append(fig)

    results = {}
    for cat, cases in categorized_figs.items():
        if not cases:
            continue

        out_name = output_dir / f"{cat}_plots.pdf"
        with PdfPages(out_name) as pdf:
            for case_num in sorted(cases.keys()):
                title_fig = plt.figure(figsize=(11, 8.5))
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

                for fig in cases[case_num]:
                    prepare_pdf_figure(fig)
                    pdf.savefig(fig)
                    plt.close(fig)

        if verbose:
            total_figs = sum(len(fig_list) for fig_list in cases.values())
            logger.info(f"Saved {total_figs} '{cat}' figures to {out_name}")
        results[cat] = out_name

    plt.close("all")
    return results
