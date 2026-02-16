"""Figure-collection splitting helpers for categorized benchmark PDFs."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .layout import prepare_pdf_figure

logger = logging.getLogger(__name__)


def _classify_figure(fig) -> str:
    """Best-effort classifier for a Matplotlib figure."""
    parts: list[str] = []

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        try:
            parts.append(str(suptitle.get_text()))
        except Exception:
            pass

    try:
        parts.extend(str(t.get_text()) for t in getattr(fig, "texts", []))
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


def split_collected_figs_to_pdfs(
    figs_with_info: list, output_dir: Path, verbose: bool = False
):
    """Split a mixed figure collection into per-category multi-page PDFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    categorized_figs = defaultdict(lambda: defaultdict(list))

    for item in figs_with_info:
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
            logger.info("Saved %s '%s' figures to %s", total_figs, cat, out_name)
        results[cat] = out_name

    plt.close("all")
    return results


__all__ = ["split_collected_figs_to_pdfs"]
