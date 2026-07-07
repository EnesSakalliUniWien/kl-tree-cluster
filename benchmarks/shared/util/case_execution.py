"""Case execution helpers for benchmark runner scripts."""

from __future__ import annotations

import multiprocessing as mp
import os

import pandas as pd

from benchmarks.shared.plots.backend import configure_matplotlib_backend


def _get_benchmark_fn():
    """Lazy import to break the circular dependency with benchmarks.shared.plots."""
    from benchmarks.shared.pipeline import benchmark_cluster_algorithm

    return benchmark_cluster_algorithm


def _run_case_worker(
    queue: "mp.Queue",
    case: dict,
    methods_to_test: list[str],
    case_plot_umap: bool,
    case_plot_manifold: bool,
    enable_plots: bool,
    pdf_path: str | None,
    include_validation_page: bool,
    method_params: dict[str, list[dict[str, object]]] | None = None,
    tree_consensus_label_dir: str | None = None,
) -> None:
    """Execute one case in a fresh process and return rows via a queue."""
    # Reduce native runtime contention in spawned workers. This materially
    # lowers intermittent SIGSEGV/SIGBUS failures in heavy numeric workloads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    configure_matplotlib_backend()

    try:
        df_res, _ = _get_benchmark_fn()(
            test_cases=[case],
            methods=methods_to_test,
            method_params=method_params,
            verbose=False,
            plot_umap=case_plot_umap,
            plot_manifold=case_plot_manifold,
            concat_plots_pdf=enable_plots,
            concat_output=pdf_path,
            matrix_audit=False,
            include_cover_pages=False,
            include_validation_page=include_validation_page,
            tree_consensus_label_dir=tree_consensus_label_dir,
        )
        queue.put({"ok": True, "rows": df_res.to_dict(orient="records")})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def run_case_isolated(
    *,
    case: dict,
    methods_to_test: list[str],
    method_params: dict[str, list[dict[str, object]]] | None,
    case_plot_umap: bool,
    case_plot_manifold: bool,
    enable_plots: bool,
    pdf_path: str | None,
    timeout_sec: int,
    include_validation_page: bool,
    tree_consensus_label_dir: str | None = None,
) -> pd.DataFrame:
    """Run a single benchmark case in an isolated subprocess."""
    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_run_case_worker,
        args=(
            queue,
            case,
            methods_to_test,
            case_plot_umap,
            case_plot_manifold,
            enable_plots,
            pdf_path,
            include_validation_page,
            method_params,
            tree_consensus_label_dir,
        ),
    )
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise RuntimeError(
            f"Case subprocess timed out after {timeout_sec}s (likely plotting memory pressure)."
        )

    payload = None
    if not queue.empty():
        payload = queue.get()

    if proc.exitcode != 0 and payload is None:
        raise RuntimeError(f"Case subprocess exited with code {proc.exitcode}.")
    if payload is None:
        raise RuntimeError("Case subprocess completed without returning a result payload.")
    if not payload["ok"]:
        raise RuntimeError(payload["error"])

    rows = payload["rows"]
    return pd.DataFrame(rows)


def run_case_with_optional_isolation(
    *,
    case: dict,
    methods_to_test: list[str],
    case_plot_umap: bool,
    case_plot_manifold: bool,
    enable_plots: bool,
    pdf_path: str | None,
    isolate_umap_cases: bool,
    timeout_sec: int,
    method_params: dict[str, list[dict[str, object]]] | None = None,
    include_validation_page: bool = True,
    tree_consensus_label_dir: str | None = None,
) -> pd.DataFrame:
    """Run a benchmark case, optionally in a subprocess."""
    if case_plot_umap and isolate_umap_cases:
        return run_case_isolated(
            case=case,
            methods_to_test=methods_to_test,
            method_params=method_params,
            case_plot_umap=case_plot_umap,
            case_plot_manifold=case_plot_manifold,
            enable_plots=enable_plots,
            pdf_path=pdf_path,
            timeout_sec=timeout_sec,
            include_validation_page=include_validation_page,
            tree_consensus_label_dir=tree_consensus_label_dir,
        )

    df_res, _ = _get_benchmark_fn()(
        test_cases=[case],
        methods=methods_to_test,
        method_params=method_params,
        verbose=False,
        plot_umap=case_plot_umap,
        plot_manifold=case_plot_manifold,
        concat_plots_pdf=enable_plots,
        concat_output=pdf_path,
        matrix_audit=False,  # Disable heavy TensorBoard exports to prevent memory crashes
        include_cover_pages=False,
        include_validation_page=include_validation_page,
        tree_consensus_label_dir=tree_consensus_label_dir,
    )
    return df_res


__all__ = [
    "run_case_isolated",
    "run_case_with_optional_isolation",
]
