"""
Benchmarking/validation pipeline for the clustering algorithm.

Runs configurable test cases, computes metrics, and optionally produces plots.
The implementation was previously housed in ``tests/validation_utils`` and is
kept here so it can be reused outside the test suite.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import SparseEfficiencyWarning

# Metrics helpers extracted to their own module for reuse and easier testing.
from benchmarks.shared.metrics import (
    _calculate_ari_nmi_purity_metrics,
)


# Logging helpers extracted to a small utility module to keep the pipeline
# focused and testable. The names are preserved for backward compatibility.
from benchmarks.shared.logging import (
    log_validation_start as _log_validation_start,
    log_test_case_start as _log_test_case_start,
    log_validation_completion as _log_validation_completion,
)


from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.cases.overlapping import get_overlapping_cases


from benchmarks.shared.runners.method_registry import METHOD_SPECS
from kl_clustering_analysis import config
from benchmarks.shared.plots import (
    generate_benchmark_plots,
)

from benchmarks.shared.generators import generate_case_data

from benchmarks.shared.audit_utils import (
    export_decomposition_audit,
    export_matrix_audit,
)

# Small utilities (avoid circular imports by keeping them lightweight)
from benchmarks.shared.util.decomposition import (
    _create_report_dataframe_from_labels,
)
from benchmarks.shared.util.time import format_timestamp_utc


# Configure logger (library-friendly: leave handlers/levels to callers)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


DEFAULT_METHODS = [
    "kl",
    "leiden",
    "louvain",
    "kmeans",
    "spectral",
    "dbscan",
    "optics",
    "hdbscan",
]


def _format_params(params: dict[str, object]) -> str:
    """Format parameters for result rows."""
    if not params:
        return ""
    parts = [f"{key}={value}" for key, value in sorted(params.items())]
    return ", ".join(parts)


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

    stamp = format_timestamp_utc(started_at)
    return plots_root / f"benchmark_plots_{stamp}.pdf"


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _maybe_add_matrix(target: dict[str, object], name: str, value: object) -> None:
    if value is None:
        return
    arr = np.asarray(value)
    if arr.size == 0:
        return
    if arr.dtype == object:
        return
    if not np.issubdtype(arr.dtype, np.number):
        return
    target[name] = arr


# PDF layout/merge helpers live under `benchmarks.shared.util.pdf_*`.
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
        When True, concatenate generated plots matching ``concat_pattern`` into a PDF.
    concat_pattern : str, default="tree_case_*.png"
        Glob used to collect plots for concatenation (relative to plots_root).
    concat_output : str, optional
        Output PDF path. Defaults to ``plots_root / f'benchmark_plots_{timestamp}.pdf'`` when omitted.
    save_individual_plots : bool, default=False
        Deprecated. PNG output is disabled; benchmark plots are emitted to PDF only.
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

    if save_individual_plots:
        logger.warning(
            "save_individual_plots=True is ignored: benchmark plotting is PDF-only."
        )
        save_individual_plots = False
    if (plot_umap or plot_manifold) and not concat_plots_pdf:
        logger.warning(
            "plot_umap/plot_manifold requested without concat_plots_pdf; skipping plot generation."
        )

    # PDF-only plotting mode.
    stream_pdf = bool(concat_plots_pdf)

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
        with warnings.catch_warnings():
            sparse_warn_mode = os.getenv("KL_TE_SPARSE_WARNING", "").strip().lower()
            if sparse_warn_mode == "error":
                warnings.filterwarnings("error", category=SparseEfficiencyWarning)
            elif sparse_warn_mode in {"default", "always", "once", "module"}:
                warnings.filterwarnings(sparse_warn_mode, category=SparseEfficiencyWarning)
            else:
                warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

            if stream_pdf:
                from matplotlib.backends.backend_pdf import PdfPages

                if output_pdf is None:
                    output_pdf = _resolve_pdf_output_path(
                        None, plots_root=plots_root, started_at=run_started_at
                    )
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                pdf_pages = PdfPages(output_pdf)

            for i, tc in enumerate(test_cases, 1):
                case_idx = tc.get("test_case_num", i)
                case_name = tc.get("name", f"Case {case_idx}")
                if verbose:
                    _log_test_case_start(case_idx, total_cases, case_name)

                if output_pdf:
                    audit_root_inc = output_pdf.parent
                    if audit_root_inc.name == "plots":
                        audit_root_inc = audit_root_inc.parent
                else:
                    audit_root_inc = (
                        plots_root.parent if plots_root.name == "plots" else plots_root
                    )
                if matrix_audit:
                    os.environ["KL_TE_MATRIX_AUDIT_ROOT"] = str(audit_root_inc)

                # Generate and process data
                data_t, y_t, X_original, meta = generate_case_data(tc)

                needs_distance_matrix = any(
                    method_id in {"leiden", "louvain", "dbscan", "optics", "hdbscan"}
                    for method_id in selected_methods
                )

                needs_distance_condensed = (
                    "kl" in selected_methods or needs_distance_matrix
                )

                distance_condensed = None
                distance_matrix = None

                # Special-case: if the generator supplied an adjacency matrix (SBM), use it
                modularity_matrix = None
                modularity_expected = None
                modularity_shifted = None
                modularity_norm = None
                if meta.get("generator") == "sbm" and meta.get("adjacency") is not None:
                    adj = np.asarray(meta.get("adjacency"), dtype=float)
                    # Use modularity matrix transformation for graph data
                    # Modularity: B = A - k_i*k_j / 2m captures community structure
                    # because B[i,j] > 0 when nodes share more edges than expected
                    degrees = adj.sum(axis=1)
                    m = adj.sum() / 2  # Number of edges
                    if m > 0:
                        expected = np.outer(degrees, degrees) / (2 * m)
                        B = adj - expected
                        # Shift to make non-negative similarity
                        B_shifted = B - B.min()
                        B_norm = B_shifted / (B_shifted.max() + 1e-10)
                        distance_matrix = 1.0 - B_norm
                        if matrix_audit:
                            modularity_expected = expected
                            modularity_matrix = B
                            modularity_shifted = B_shifted
                            modularity_norm = B_norm
                    else:
                        # Fallback for empty graphs
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

                method_audits: list[tuple[str, dict[str, object]]] = []
                for method_id in selected_methods:
                    spec = METHOD_SPECS[method_id]
                    params_list = param_sets[method_id]
                    for params in params_list:
                        run_params = dict(params)
                        if method_id in {"kmeans", "spectral"}:
                            raw_k = run_params.get("n_clusters")
                            if raw_k is None or str(raw_k).strip().lower() in {
                                "true",
                                "expected",
                                "auto",
                            }:
                                run_params["n_clusters"] = int(meta["n_clusters"])

                        meta_run = meta.copy()
                        if method_id.startswith("kl"):
                            # KL runner supports per-run linkage/method configuration.
                            # Compute per-run distance_condensed if the metric differs.
                            metric = run_params.get(
                                "tree_distance_metric", config.TREE_DISTANCE_METRIC
                            )
                            linkage_method = run_params.get(
                                "tree_linkage_method", config.TREE_LINKAGE_METHOD
                            )
                            # For SBM cases, the pre-computed modularity distance
                            # MUST be used. Raw hamming/rogerstanimoto on adjacency
                            # rows does not capture community structure. No fallback.
                            if meta.get("generator") == "sbm":
                                if distance_condensed is None:
                                    raise ValueError(
                                        f"SBM case '{meta.get('name', '?')}' requires "
                                        "pre-computed modularity distance but "
                                        "distance_condensed is None."
                                    )
                                distance_condensed_kl = distance_condensed
                            else:
                                distance_condensed_kl = pdist(
                                    data_t.values, metric=metric
                                )
                            result = spec.runner(
                                data_t,
                                distance_condensed_kl,
                                significance_level,
                                tree_linkage_method=str(linkage_method),
                            )
                        elif method_id in {"leiden", "louvain"}:
                            result = spec.runner(
                                distance_matrix,
                                run_params,
                                tc.get("seed"),
                            )
                        elif method_id in {"kmeans", "spectral"}:
                            result = spec.runner(
                                data_t.values,
                                run_params,
                                tc.get("seed"),
                            )
                        else:
                            result = spec.runner(distance_matrix, run_params)

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
                                "Test": case_idx,
                                "Case_Name": case_name,
                                "Case_Category": meta.get("category", "unknown"),
                                "Method": spec.name,
                                "Params": _format_params(run_params),
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
                                    "test_case_num": case_idx,
                                    "method_name": spec.name,
                                    "params": run_params,
                                    "ari": float(ari) if np.isfinite(ari) else np.nan,
                                    "nmi": float(nmi) if np.isfinite(nmi) else np.nan,
                                    "purity": float(purity)
                                    if np.isfinite(purity)
                                    else np.nan,
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
                                    "posthoc_merge_audit": result.extra.get(
                                        "posthoc_merge_audit"
                                    )
                                    if result.extra
                                    else None,
                                }
                            )

                        if matrix_audit:
                            method_name = _slugify(spec.name)
                            params_slug = _slugify(_format_params(run_params))
                            method_tag = (
                                method_name
                                if not params_slug
                                else f"{method_name}__{params_slug}"
                            )
                            matrices: dict[str, object] = {}
                            if method_id.startswith("kl"):
                                matrices["distance_condensed"] = distance_condensed_kl
                            if (
                                result.extra
                                and result.extra.get("linkage_matrix") is not None
                            ):
                                matrices["linkage_matrix"] = result.extra.get(
                                    "linkage_matrix"
                                )
                            if matrices:
                                method_audits.append((method_tag, matrices))

                # Export detailed audit logs for decomposition methods incrementally
                # We do this INSIDE the loop to ensure logs are saved even if later cases crash
                if matrix_audit:
                    case_slug = _slugify(case_name)
                    case_tag = f"case_{case_idx}_{case_slug}"
                    case_matrices = {
                        "data_matrix": data_t.values.astype(float),
                        "y_true": y_t,
                    }
                    _maybe_add_matrix(case_matrices, "X_original", X_original)
                    _maybe_add_matrix(case_matrices, "distance_matrix", distance_matrix)
                    _maybe_add_matrix(
                        case_matrices, "distance_condensed", distance_condensed
                    )
                    if meta.get("generator") == "sbm" and meta.get("adjacency") is not None:
                        _maybe_add_matrix(
                            case_matrices,
                            "adjacency",
                            np.asarray(meta.get("adjacency"), dtype=float),
                        )
                    _maybe_add_matrix(case_matrices, "sbm_expected", modularity_expected)
                    _maybe_add_matrix(case_matrices, "sbm_modularity", modularity_matrix)
                    _maybe_add_matrix(
                        case_matrices, "sbm_modularity_shifted", modularity_shifted
                    )
                    _maybe_add_matrix(case_matrices, "sbm_modularity_norm", modularity_norm)

                    _maybe_add_matrix(
                        case_matrices, "generator_distributions", meta.get("distributions")
                    )
                    _maybe_add_matrix(
                        case_matrices,
                        "generator_divergence_matrix",
                        meta.get("divergence_matrix"),
                    )
                    _maybe_add_matrix(
                        case_matrices,
                        "generator_divergence_from_ancestor",
                        meta.get("divergence_from_ancestor"),
                    )
                    _maybe_add_matrix(
                        case_matrices,
                        "generator_distributions_over_time",
                        meta.get("distributions_over_time"),
                    )

                    leaf_distributions = meta.get("leaf_distributions")
                    if isinstance(leaf_distributions, dict) and leaf_distributions:
                        keys = sorted(leaf_distributions.keys(), key=str)
                        values = [np.asarray(leaf_distributions[k]) for k in keys]
                        if all(
                            v.size > 0
                            and v.dtype != object
                            and np.issubdtype(v.dtype, np.number)
                            for v in values
                        ):
                            try:
                                stacked = np.stack(values, axis=0)
                            except ValueError:
                                stacked = None
                            _maybe_add_matrix(case_matrices, "leaf_distributions", stacked)

                    export_matrix_audit(
                        matrices=case_matrices,
                        output_root=audit_root_inc,
                        tag_prefix=case_tag,
                        step=case_idx,
                        include_products=True,
                        verbose=verbose,
                    )

                    for method_tag, matrices in method_audits:
                        export_matrix_audit(
                            matrices=matrices,
                            output_root=audit_root_inc,
                            tag_prefix=f"{case_tag}/method_{method_tag}",
                            step=case_idx,
                            include_products=True,
                            verbose=verbose,
                        )

                # Detect how many runs were added for this specific test case
                current_case_results = [
                    r for r in computed_results if r.get("test_case_num") == case_idx
                ]

                export_decomposition_audit(
                    computed_results=current_case_results,
                    output_root=audit_root_inc,
                    verbose=verbose,
                )

            if verbose:
                _log_validation_completion(total_runs, len(test_cases))

            df_results = pd.DataFrame(results_data)

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
