"""
Benchmarking/validation pipeline for the clustering algorithm.

Runs configurable test cases, computes metrics, and optionally produces plots.
The implementation was previously housed in ``tests/validation_utils`` and is
kept here so it can be reused outside the test suite.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.config import DEFAULT_METHODS

# Logging helpers extracted to a small utility module to keep the pipeline
# focused and testable.
from benchmarks.shared.logging import log_validation_completion as _log_validation_completion
from benchmarks.shared.logging import log_validation_start as _log_validation_start

# Metrics helpers extracted to their own module for reuse and easier testing.
from benchmarks.shared.plots import generate_benchmark_plots
from benchmarks.shared.results import (
    BenchmarkResultRow,
    ComputedResultRecord,
    benchmark_rows_to_dataframe,
)
from benchmarks.shared.runners.method_registry import METHOD_SPECS

# Small utilities (avoid circular imports by keeping them lightweight)
from benchmarks.shared.util.case_run import run_single_case
from benchmarks.shared.util.method_selection import resolve_selected_methods_and_param_sets
from benchmarks.shared.util.pdf.session import open_pdf_pages, resolve_pdf_output_path
from kl_clustering_analysis import config

# Configure logger (library-friendly: leave handlers/levels to callers)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# PDF layout/merge helpers live under `benchmarks.shared.util.pdf.*`.
def benchmark_cluster_algorithm(
    test_cases=None,
    significance_level=config.SIBLING_ALPHA,
    verbose=True,
    plot_umap=False,
    plot_manifold=False,
    methods: list[str] | None = None,
    method_params: dict[str, list[dict[str, object]]] | None = None,
    concat_plots_pdf: bool = False,
    concat_output: str | None = None,
    matrix_audit: bool = False,
):
    """
    Benchmark the cluster decomposition algorithm across multiple test cases.

    Parameters
    ----------
    test_cases : list of dict, optional
        List of test case configurations. Each dict should contain:
        - n_samples: number of samples
        - n_features: number of features
        - n_clusters: true number of clusters
        - cluster_std: noise level (standard deviation)
        - seed: random seed
        If None, uses default test cases.
    significance_level : float, default=config.SIBLING_ALPHA
        Significance level used for sibling-independence gating in decomposition
    verbose : bool, default=True
        If True, prints progress and displays validation results
    plot_umap : bool, default=False
        If True, generates t-SNE plots comparing KL clustering with K-means and spectral clustering
    plot_manifold : bool, default=False
        If True, saves UMAP-vs-Isomap manifold diagnostics for each test case

    methods : list of str, optional
        Clustering methods to run (defaults to the full registry).
    method_params : dict, optional
        Optional per-method parameter grids. Values are lists of param dicts.
    concat_plots_pdf : bool, default=False
        When True, stream generated benchmark plots directly into a PDF.
    concat_output : str, optional
        Output PDF path. Defaults to ``plots_root / f'benchmark_plots_{timestamp}.pdf'`` when omitted.
    matrix_audit : bool, default=False
        If True, export TensorBoard summaries for key matrices per test case.

    Returns
    -------
    df_results : pd.DataFrame
        Results dataframe with columns: Test, True, Found, Samples, Features,
        Noise, ARI, NMI, Purity
    fig : None
        Plotting has been removed; always returns None.
    """

    # Default test cases
    run_started_at = datetime.now(timezone.utc)
    project_root = Path(__file__).resolve().parents[2]
    plots_root = project_root / "benchmarks" / "results" / "plots"
    if test_cases is None:
        test_cases = get_default_test_cases()

    if verbose:
        _log_validation_start(len(test_cases))

    result_rows: list[BenchmarkResultRow] = []

    # Store computed results to avoid recalculation
    computed_results: list[ComputedResultRecord] = []

    output_pdf = (
        resolve_pdf_output_path(
            concat_output,
            plots_root=plots_root,
            started_at=run_started_at,
        )
        if concat_plots_pdf
        else None
    )

    if (plot_umap or plot_manifold) and not concat_plots_pdf:
        logger.warning(
            "plot_umap/plot_manifold requested without concat_plots_pdf; skipping plot generation."
        )

    # PDF-only plotting mode.
    stream_pdf = bool(concat_plots_pdf)

    # Run test cases
    selected_methods, param_sets = resolve_selected_methods_and_param_sets(
        methods=methods,
        method_params=method_params,
        default_methods=DEFAULT_METHODS,
        method_specs=METHOD_SPECS,
    )

    total_cases = len(test_cases)
    total_runs = total_cases * sum(len(params) for params in param_sets.values())

    pdf_pages = None

    try:

        if stream_pdf:
            output_pdf, pdf_pages = open_pdf_pages(
                output_pdf,
                plots_root=plots_root,
                started_at=run_started_at,
            )

        for i, tc in enumerate(test_cases, 1):
            case_result_rows, case_computed_results = run_single_case(
                tc=tc,
                case_position=i,
                total_cases=total_cases,
                selected_methods=selected_methods,
                param_sets=param_sets,
                significance_level=significance_level,
                output_pdf=output_pdf,
                plots_root=plots_root,
                matrix_audit=matrix_audit,
                verbose=verbose,
            )
            result_rows.extend(case_result_rows)
            computed_results.extend(case_computed_results)

        if verbose:
            _log_validation_completion(total_runs, len(test_cases))

        df_results = benchmark_rows_to_dataframe(result_rows)

        should_generate_plots = bool(concat_plots_pdf)
        if not should_generate_plots:
            return df_results, None

        plot_kwargs = {
            "save_png": False,
            "collect_figs": False,
        }
        if pdf_pages is not None:
            plot_kwargs["pdf"] = pdf_pages

        fig, _new_figs_by_category = generate_benchmark_plots(
            df_results,
            computed_results,
            plots_root,
            verbose,
            plot_umap,
            plot_manifold,
            **plot_kwargs,
        )

        return df_results, fig

    finally:
        if pdf_pages is not None:
            pdf_pages.close()


__all__ = [
    "benchmark_cluster_algorithm",
]
