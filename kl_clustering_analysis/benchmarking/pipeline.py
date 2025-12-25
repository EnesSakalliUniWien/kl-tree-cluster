"""
Benchmarking/validation pipeline for the clustering algorithm.

Runs configurable test cases, computes metrics, and optionally produces plots.
The implementation was previously housed in ``tests/validation_utils`` and is
kept here so it can be reused outside the test suite.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Metrics helpers extracted to their own module for reuse and easier testing.
from kl_clustering_analysis.benchmarking.metrics import (
    _calculate_ari_nmi_purity_metrics,
)


# Logging helpers extracted to a small utility module to keep the pipeline
# focused and testable. The names are preserved for backward compatibility.
from kl_clustering_analysis.benchmarking.logging import (
    log_validation_start as _log_validation_start,
    log_test_case_start as _log_test_case_start,
    log_validation_completion as _log_validation_completion,
)


try:  # Prefer test config when available (repo usage)
    from tests.test_cases_config import get_default_test_cases  # type: ignore
except ImportError:
    # No test config available; use an empty default
    def get_default_test_cases():
        return []


from kl_clustering_analysis.benchmarking.method_registry import METHOD_SPECS
from kl_clustering_analysis import config
from kl_clustering_analysis.benchmarking.plots import (
    generate_benchmark_plots,
)

from kl_clustering_analysis.benchmarking.generators import generate_case_data


# PDF concatenation helper extracted to its own module.
from kl_clustering_analysis.benchmarking.pdf_utils import (
    concat_plots_to_pdf as _concat_plots_to_pdf,
)

# Small utilities (avoid circular imports by keeping them lightweight)
from kl_clustering_analysis.benchmarking.utils_decomp import (
    _create_report_dataframe_from_labels,
)


# Configure logger (library-friendly: leave handlers/levels to callers)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


DEFAULT_METHODS = ["kl", "leiden", "louvain", "dbscan", "optics", "hdbscan"]


def _format_params(params: dict[str, object]) -> str:
    """Format parameters for result rows."""
    if not params:
        return ""
    parts = [f"{key}={value}" for key, value in sorted(params.items())]
    return ", ".join(parts)


def _format_timestamp_utc(dt: datetime) -> str:
    """Return a filesystem-safe UTC timestamp like 20250101_235959Z."""
    if dt.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _resolve_pdf_output_path(
    concat_output: str | None,
    *,
    plots_root: Path,
    started_at: datetime,
) -> Path:
    """Resolve the PDF output path, ensuring a timestamped default and .pdf suffix."""
    if concat_output:
        path = Path(concat_output)
        return path.with_suffix(".pdf") if path.suffix == "" else path

    stamp = _format_timestamp_utc(started_at)
    return plots_root / f"benchmark_plots_{stamp}.pdf"


# PDF concatenation helper moved to `kl_clustering_analysis.benchmarking.pdf_utils`.
# See that module for the implementation.
def benchmark_cluster_algorithm(
    test_cases=None,
    significance_level=config.SIBLING_ALPHA,
    verbose=True,
    plot_umap=False,
    plot_manifold=False,
    methods: list[str] | None = None,
    method_params: dict[str, list[dict[str, object]]] | None = None,
    concat_plots_pdf: bool = False,
    concat_pattern: str = "tree_case_*.png",
    concat_output: str | None = None,
    save_individual_plots: bool = False,
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
        When True, concatenate generated plots matching ``concat_pattern`` into a PDF.
    concat_pattern : str, default="tree_case_*.png"
        Glob used to collect plots for concatenation (relative to plots_root).
    concat_output : str, optional
        Output PDF path. Defaults to ``plots_root / f'benchmark_plots_{timestamp}.pdf'`` when omitted.
    save_individual_plots : bool, default=False
        When False, skip writing per-plot PNGs; collect figures for optional PDF output instead.
        The pipeline now defaults to not producing PNG artifacts; pass ``save_individual_plots=True``
        explicitly to enable PNG output (not recommended).

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
    plots_root = project_root / "results" / "plots"
    if test_cases is None:
        test_cases = get_default_test_cases()

    if verbose:
        _log_validation_start(len(test_cases))

    results_data = []

    # Store computed results to avoid recalculation
    computed_results = []

    output_pdf = (
        _resolve_pdf_output_path(
            concat_output,
            plots_root=plots_root,
            started_at=run_started_at,
        )
        if concat_plots_pdf
        else None
    )

    # When concatenating directly to a PDF without intermediate PNGs, stream to
    # PdfPages immediately and close each figure right away (prevents figure leaks).
    stream_pdf = bool(concat_plots_pdf and not save_individual_plots)

    # Run test cases
    selected_methods = methods or DEFAULT_METHODS
    method_params = method_params or {}
    for method_id in selected_methods:
        if method_id not in METHOD_SPECS:
            raise ValueError(f"Unknown method: {method_id}")

    param_sets = {
        method_id: (method_params.get(method_id) or METHOD_SPECS[method_id].param_grid)
        for method_id in selected_methods
    }
    total_cases = len(test_cases)
    total_runs = total_cases * sum(len(params) for params in param_sets.values())

    pdf_pages = None
    try:
        if stream_pdf:
            from matplotlib.backends.backend_pdf import PdfPages

            if output_pdf is None:
                output_pdf = _resolve_pdf_output_path(
                    None, plots_root=plots_root, started_at=run_started_at
                )
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf_pages = PdfPages(output_pdf)

        for i, tc in enumerate(test_cases, 1):
            case_name = tc.get("name", f"Case {i}")
            if verbose:
                _log_test_case_start(i, total_cases, case_name)

            # Generate and process data
            data_t, y_t, X_original, meta = generate_case_data(tc)

            needs_distance_matrix = any(
                method_id in {"leiden", "louvain", "dbscan", "optics", "hdbscan"}
                for method_id in selected_methods
            )

            needs_distance_condensed = "kl" in selected_methods or needs_distance_matrix

            distance_condensed = None
            distance_matrix = None

            # Special-case: if the generator supplied an adjacency matrix (SBM), use it
            if meta.get("generator") == "sbm" and meta.get("adjacency") is not None:
                adj = np.asarray(meta.get("adjacency"), dtype=float)
                # Treat adjacency as similarity; convert to a distance matrix where smaller means closer
                distance_matrix = 1.0 - adj
                # Ensure diagonal is zero for a valid distance matrix
                np.fill_diagonal(distance_matrix, 0.0)
                # Condensed distance form for methods that need it (e.g., KL)
                if needs_distance_condensed:
                    distance_condensed = squareform(distance_matrix)
            else:
                if needs_distance_condensed:
                    distance_condensed = pdist(
                        data_t.values, metric=config.TREE_DISTANCE_METRIC
                    )
                if needs_distance_matrix:
                    if distance_condensed is None:
                        distance_condensed = pdist(
                            data_t.values, metric=config.TREE_DISTANCE_METRIC
                        )
                    distance_matrix = squareform(distance_condensed)

            for method_id in selected_methods:
                spec = METHOD_SPECS[method_id]
                params_list = param_sets[method_id]
                for params in params_list:
                    meta_run = meta.copy()
                    if method_id == "kl":
                        # KL runner supports per-run linkage/method configuration.
                        # Compute per-run distance_condensed if the metric differs.
                        metric = params.get(
                            "tree_distance_metric", config.TREE_DISTANCE_METRIC
                        )
                        linkage_method = params.get(
                            "tree_linkage_method", config.TREE_LINKAGE_METHOD
                        )
                        distance_condensed_kl = pdist(data_t.values, metric=metric)
                        result = spec.runner(
                            data_t,
                            distance_condensed_kl,
                            significance_level,
                            tree_linkage_method=str(linkage_method),
                        )
                    elif method_id in {"leiden", "louvain"}:
                        result = spec.runner(
                            distance_matrix,
                            params,
                            tc.get("seed"),
                        )
                    else:
                        result = spec.runner(distance_matrix, params)

                    if result.status == "ok" and result.labels is not None:
                        labels = result.labels
                        report_df = _create_report_dataframe_from_labels(
                            labels, data_t.index
                        )
                        found_clusters = result.found_clusters
                        labels_len = len(labels)
                        ari, nmi, purity = _calculate_ari_nmi_purity_metrics(
                            found_clusters, report_df, data_t.index, y_t
                        )
                    else:
                        labels_len = 0
                        found_clusters = 0
                        ari, nmi, purity = np.nan, np.nan, np.nan

                    results_data.append(
                        {
                            "Test": i,
                            "Case_Name": case_name,
                            "Method": spec.name,
                            "Params": _format_params(params),
                            "True": meta["n_clusters"],
                            "Found": found_clusters,
                            "Samples": meta["n_samples"],
                            "Features": meta["n_features"],
                            "Noise": meta["noise"],
                            "ARI": ari,
                            "NMI": nmi,
                            "Purity": purity,
                            "Status": result.status,
                            "Skip_Reason": result.skip_reason or "",
                            "Labels_Length": labels_len,
                        }
                    )

                    if result.status == "ok" and result.labels is not None:
                        meta_run["found_clusters"] = found_clusters
                        computed_results.append(
                            {
                                "test_case_num": i,
                                "method_name": spec.name,
                                "params": params,
                                "labels": result.labels,
                                "data": data_t,
                                "meta": meta_run,
                                "X_original": X_original,
                                "y_true": y_t,
                                "tree": result.extra.get("tree")
                                if result.extra
                                else None,
                                "decomposition": result.extra.get("decomposition")
                                if result.extra
                                else None,
                                "stats": result.extra.get("stats")
                                if result.extra
                                else None,
                            }
                        )

        if verbose:
            _log_validation_completion(total_runs, len(test_cases))

        df_results = pd.DataFrame(results_data)

        if not verbose and not plot_umap and not plot_manifold and not concat_plots_pdf:
            return df_results, None

        plot_kwargs = {
            "save_png": save_individual_plots,
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

        if concat_plots_pdf and save_individual_plots:
            # Legacy: allow arbitrary pattern -> single PDF of saved PNGs
            if output_pdf is None:
                output_pdf = _resolve_pdf_output_path(
                    concat_output,
                    plots_root=plots_root,
                    started_at=run_started_at,
                )
            _concat_plots_to_pdf(
                plots_root=plots_root,
                pattern=concat_pattern,
                output_pdf=output_pdf,
                verbose=bool(verbose),
            )

        return df_results, fig
    finally:
        if pdf_pages is not None:
            pdf_pages.close()


__all__ = [
    "benchmark_cluster_algorithm",
]
