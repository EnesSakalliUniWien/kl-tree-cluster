"""
Runtime helpers to render and save benchmark plots."""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from benchmarks.shared.util.pdf_layout import PDF_PAGE_SIZE_INCHES, prepare_pdf_figure
from .summary import create_validation_plot
from .export import (
    create_tree_plots_from_results,
    create_tree_then_umap_plots_from_results,
    create_manifold_plots_from_results,
    create_umap_3d_plots_from_results,
)

logger = logging.getLogger(__name__)


def log_detailed_results(df_results: pd.DataFrame) -> None:
    """Log the detailed results table row by row to avoid truncation."""
    logger.info("Detailed Results:")
    columns = [
        "Test",
        "Case_Name",
        "Method",
        "Params",
        "True",
        "Found",
        "Samples",
        "Features",
        "Noise",
        "ARI",
        "NMI",
        "Purity",
        "Status",
    ]
    available = [col for col in columns if col in df_results.columns]
    results_str = df_results[available].to_string(index=False)
    for line in results_str.split("\n"):
        logger.info(line)


def generate_benchmark_plots(
    df_results: pd.DataFrame,
    computed_results: list,
    plots_root: Path,
    verbose: bool,
    plot_umap: bool,
    plot_manifold: bool,
    plot_umap_3d: bool = False,
    save_png: bool = True,
    collect_figs: bool = False,
    *,
    pdf: PdfPages | None = None,
):
    """Generate benchmark plots in PDF-only mode."""
    # Prepare per-category collections (for interface compatibility).
    collected_by_category = {
        "validation": [],
        "trees": [],
        "umap": [],
        "manifold": [],
        "umap3d": [],
    }
    if df_results.empty:
        return None, collected_by_category
    if pdf is None:
        logger.debug("Skipping plot generation because no PdfPages handle was provided.")
        return None, collected_by_category

    _ = save_png
    _ = collect_figs

    fig = create_validation_plot(df_results)

    # Add a concise cover page for readability in long benchmark PDFs.
    cover = plt.figure(figsize=PDF_PAGE_SIZE_INCHES)
    cover.text(
        0.5,
        0.62,
        "Benchmark Report",
        ha="center",
        va="center",
        fontsize=26,
        weight="bold",
    )
    cover.text(
        0.5,
        0.50,
        f"Cases: {df_results['Test'].nunique() if 'Test' in df_results.columns else len(df_results)}",
        ha="center",
        va="center",
        fontsize=14,
    )
    cover.text(
        0.5,
        0.43,
        f"Methods: {df_results['Method'].nunique() if 'Method' in df_results.columns else 'n/a'}",
        ha="center",
        va="center",
        fontsize=14,
    )
    cover.text(
        0.5,
        0.33,
        f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%SZ')}",
        ha="center",
        va="center",
        fontsize=11,
        alpha=0.8,
    )
    pdf.savefig(cover)
    plt.close(cover)

    prepare_pdf_figure(fig)
    pdf.savefig(fig)
    plt.close(fig)
    fig = None

    if verbose:
        logger.info("Generating tree plots...")
    if plot_umap:
        if verbose:
            logger.info("Generating Tree→UMAP comparison pages...")
        create_tree_then_umap_plots_from_results(
            computed_results,
            plots_root,
            timestamp=None,
            verbose=verbose,
            save=False,
            collect=False,
            collected=collected_by_category["umap"],
            pdf=pdf,
        )
    else:
        create_tree_plots_from_results(
            test_results=computed_results,
            output_dir=plots_root,
            timestamp=None,
            verbose=False,
            save=False,
            collect=False,
            collected=collected_by_category["trees"],
            pdf=pdf,
        )

    if plot_manifold:
        if verbose:
            logger.info("Generating manifold diagnostics...")
        create_manifold_plots_from_results(
            computed_results,
            plots_root,
            timestamp=None,
            verbose=verbose,
            save=False,
            collect=False,
            collected=collected_by_category["manifold"],
            pdf=pdf,
        )

    if plot_umap_3d:
        if verbose:
            logger.info("Generating 3D UMAP visualizations...")
        create_umap_3d_plots_from_results(
            computed_results,
            plots_root,
            timestamp=None,
            verbose=verbose,
            save=False,
            collect=False,
            collected=collected_by_category["umap3d"],
            pdf=pdf,
        )

    return fig, collected_by_category
