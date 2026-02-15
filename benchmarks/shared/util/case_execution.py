"""Case execution helpers for benchmark runner scripts."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os

import pandas as pd

from benchmarks.shared.pipeline import benchmark_cluster_algorithm


def _run_case_worker(
    queue: "mp.Queue",
    case: dict,
    methods_to_test: list[str],
    case_plot_umap: bool,
    case_plot_manifold: bool,
    enable_plots: bool,
    pdf_path: str | None,
) -> None:
    """Execute one case in a fresh process and return rows via a queue."""
    # Reduce native runtime contention in spawned workers. This materially
    # lowers intermittent SIGSEGV/SIGBUS failures in heavy numeric workloads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        df_res, _ = benchmark_cluster_algorithm(
            test_cases=[case],
            methods=methods_to_test,
            verbose=False,
            plot_umap=case_plot_umap,
            plot_manifold=case_plot_manifold,
            concat_plots_pdf=enable_plots,
            concat_output=pdf_path,
            save_individual_plots=False,
            matrix_audit=False,
        )
        queue.put({"ok": True, "rows": df_res.to_dict(orient="records")})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def run_case_isolated(
    *,
    case: dict,
    methods_to_test: list[str],
    case_plot_umap: bool,
    case_plot_manifold: bool,
    enable_plots: bool,
    pdf_path: str | None,
    timeout_sec: int,
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
        return pd.DataFrame()
    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "Unknown case subprocess failure."))

    rows = payload.get("rows", [])
    return pd.DataFrame(rows)


def run_case_with_optional_isolation(
    *,
    case: dict,
    case_id: str,
    methods_to_test: list[str],
    case_plot_umap: bool,
    case_plot_manifold: bool,
    enable_plots: bool,
    pdf_path: str | None,
    isolate_umap_cases: bool,
    timeout_sec: int,
    retry_count: int,
) -> pd.DataFrame:
    """Run a benchmark case with optional subprocess isolation and retry fallback."""
    if case_plot_umap and isolate_umap_cases:
        df_res = None
        last_err: Exception | None = None
        for attempt in range(1, retry_count + 2):
            try:
                df_res = run_case_isolated(
                    case=case,
                    methods_to_test=methods_to_test,
                    case_plot_umap=case_plot_umap,
                    case_plot_manifold=case_plot_manifold,
                    enable_plots=enable_plots,
                    pdf_path=pdf_path,
                    timeout_sec=timeout_sec,
                )
                break
            except RuntimeError as exc:
                last_err = exc
                msg = str(exc)
                if "exited with code -11" in msg:
                    if attempt <= retry_count:
                        print(
                            f"  WARN: isolated subprocess SIGSEGV on '{case_id}' "
                            f"(attempt {attempt}/{retry_count + 1}); retrying...",
                            flush=True,
                        )
                        gc.collect()
                        continue
                    print(
                        f"  WARN: isolated subprocess kept SIGSEGV-ing on '{case_id}' "
                        f"after {retry_count + 1} attempts; falling back "
                        "to in-process execution for this case.",
                        flush=True,
                    )
                    break
                raise

        if df_res is None and isinstance(last_err, RuntimeError):
            if "exited with code -11" not in str(last_err):
                raise last_err
    else:
        df_res = None

    if df_res is not None:
        return df_res

    df_res, _ = benchmark_cluster_algorithm(
        test_cases=[case],
        methods=methods_to_test,
        verbose=False,
        plot_umap=case_plot_umap,
        plot_manifold=case_plot_manifold,
        concat_plots_pdf=enable_plots,
        concat_output=pdf_path,
        save_individual_plots=False,
        matrix_audit=False,  # Disable heavy TensorBoard exports to prevent memory crashes
    )
    return df_res


__all__ = [
    "run_case_isolated",
    "run_case_with_optional_isolation",
]
