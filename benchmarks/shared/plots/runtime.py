"""
Runtime helpers to render and save benchmark plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from benchmarks.shared.result_records import (
    RESULT_COLUMNS,
    ComputedResultRecord,
)
from benchmarks.shared.util.pdf.layout import prepare_pdf_figure

from .cover_page import GROUP_ORDER, generate_overview_page, generate_section_page
from .export import (
    create_case_report_pages_from_results,
    create_manifold_plots_from_results,
    create_tree_plots_from_results,
    create_umap_3d_plots_from_results,
)
from .summary import create_validation_plot

logger = logging.getLogger(__name__)

_DETAILED_LOG_EXCLUSIONS = frozenset(
    {
        "case_category",
        "source_family",
        "feature_representation",
        "run_id",
        "benchmark_class",
        "benchmark_grid",
        "benchmark_repeat",
        "tree_build_sec",
        "populate_divergences_sec",
        "edge_gate_sec",
        "edge_gate_contrast_covariance_sec",
        "edge_gate_projection_sec",
        "edge_gate_wald_statistic_sec",
        "edge_gate_tree_bh_sec",
        "spectral_context_sec",
        "tangent_whitening_sec",
        "eigensolve_sec",
        "pca_projection_sec",
        "sibling_gate_sec",
        "sibling_gate_pair_record_collection_sec",
        "sibling_gate_inflation_fit_sec",
        "sibling_gate_adjusted_tests_sec",
        "sibling_gate_fdr_sec",
        "traversal_sec",
        "skip_reason",
        "labels_length",
    }
)


def log_detailed_results(df_results: pd.DataFrame) -> None:
    """Log the detailed results table row by row to avoid truncation."""
    logger.info("Detailed Results:")
    columns = [column for column in RESULT_COLUMNS if column not in _DETAILED_LOG_EXCLUSIONS]
    available = [col for col in columns if col in df_results.columns]
    results_str = df_results[available].to_string(index=False)
    for line in results_str.split("\n"):
        logger.info(line)


def generate_benchmark_plots(
    df_results: pd.DataFrame,
    computed_results: list[ComputedResultRecord],
    plots_root: Path,
    verbose: bool,
    plot_umap: bool,
    plot_manifold: bool,
    plot_umap_3d: bool = False,
    save_png: bool = True,
    collect_figs: bool = False,
    include_cover_pages: bool = True,
    include_validation_page: bool = True,
    *,
    pdf: PdfPages | None = None,
):
    """Generate benchmark plots in PDF-only mode."""
    # Prepare per-category collections for downstream consumers.
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

    # --- Experiment-setup cover pages ---
    # Full benchmark reports add their own global overview/section pages during
    # top-level PDF assembly. Allow per-case runs to skip these to avoid
    # duplicated/misaligned explanation pages in concatenated reports.
    if include_cover_pages:
        n_cases = df_results["test_case"].nunique()
        cover_figs = [generate_overview_page(n_cases=n_cases)]
        for group in GROUP_ORDER:
            section_fig = generate_section_page(group)
            if section_fig is not None:
                cover_figs.append(section_fig)
        for cfig in cover_figs:
            pdf.savefig(cfig)
            plt.close(cfig)
        logger.info("Wrote %d cover/setup pages to PDF.", len(cover_figs))

    fig = None
    if include_validation_page:
        fig = create_validation_plot(df_results)
        prepare_pdf_figure(fig)
        pdf.savefig(fig)
        plt.close(fig)
        fig = None

    if verbose:
        logger.info("Generating tree plots...")
    if plot_umap:
        if verbose:
            logger.info("Generating Tree→UMAP comparison pages...")
        create_case_report_pages_from_results(
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
