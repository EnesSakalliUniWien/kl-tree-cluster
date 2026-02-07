"""
Runtime helpers to render and save benchmark plots."""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .summary import create_validation_plot
from .export import (
    create_tree_plots_from_results,
    create_umap_plots_from_results,
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
    """Generate validation plots; optionally avoid PNGs and collect figs for PDFs."""
    if df_results.empty:
        return None, {}

    fig = create_validation_plot(df_results)

    # Prepare per-category collections
    collected_by_category = {
        "validation": [],
        "trees": [],
        "umap": [],
        "manifold": [],
        "umap3d": [],
    }

    if save_png and verbose:
        fig.savefig("validation_results.png", dpi=150, bbox_inches="tight")
        logger.info("Validation plot saved to 'validation_results.png'")
        log_detailed_results(df_results)

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        fig = None
    elif collect_figs:
        # Pass test case number with the figure for grouping later
        collected_by_category["validation"].append({"figure": fig, "test_case_num": -1})
    elif not save_png:
        plt.close(fig)

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
            save=save_png,
            collect=collect_figs,
            collected=collected_by_category["umap"],
            pdf=pdf,
        )
    else:
        create_tree_plots_from_results(
            test_results=computed_results,
            output_dir=plots_root,
            timestamp=None,
            verbose=False,
            save=save_png,
            collect=collect_figs,
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
            save=save_png,
            collect=collect_figs,
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
            save=save_png,
            collect=collect_figs,
            collected=collected_by_category["umap3d"],
            pdf=pdf,
        )

    return fig, collected_by_category
