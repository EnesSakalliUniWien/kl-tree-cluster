import sys
import os
import gc
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path

# Add the project root to the python path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.env import get_env_bool, get_env_int
from benchmarks.shared.time_utils import format_timestamp_utc
import pandas as pd
import numpy as np

try:
    from benchmarks.shared.debug_trace import diagnose_benchmark_failures
except ImportError:
    diagnose_benchmark_failures = None

def _resolve_methods_to_test() -> list[str]:
    """Resolve method list from KL_TE_METHODS, defaulting to all registry methods."""
    raw = (os.getenv("KL_TE_METHODS") or "all").strip()
    if raw.lower() == "all":
        return list(METHOD_SPECS.keys())

    selected = [m.strip() for m in raw.split(",") if m.strip()]
    if not selected:
        raise ValueError("KL_TE_METHODS is empty. Provide comma-separated method ids or 'all'.")

    unknown = [m for m in selected if m not in METHOD_SPECS]
    if unknown:
        available = ", ".join(sorted(METHOD_SPECS.keys()))
        raise ValueError(
            f"Unknown methods in KL_TE_METHODS: {unknown}. Available: {available}"
        )
    return selected


def _concat_case_pdfs(case_pdfs: list[Path], output_pdf: Path) -> bool:
    """Concatenate case-level PDFs into a single report PDF."""
    existing = [p for p in case_pdfs if p.exists()]
    if not existing:
        print("No case-level PDFs found to concatenate.")
        return False

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    if len(existing) == 1:
        shutil.copyfile(existing[0], output_pdf)
        print(f"Single PDF report copied to {output_pdf}")
        return True

    pdfunite = shutil.which("pdfunite")
    if pdfunite:
        cmd = [pdfunite, *[str(p) for p in existing], str(output_pdf)]
        subprocess.run(cmd, check=True)
        print(f"Concatenated {len(existing)} case PDFs to {output_pdf}")
        return True

    gs = shutil.which("gs")
    if gs:
        cmd = [
            gs,
            "-dBATCH",
            "-dNOPAUSE",
            "-q",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={output_pdf}",
            *[str(p) for p in existing],
        ]
        subprocess.run(cmd, check=True)
        print(f"Concatenated {len(existing)} case PDFs to {output_pdf}")
        return True

    print(
        "Could not concatenate PDFs automatically (missing 'pdfunite' and 'gs'). "
        f"Case-level PDFs remain in {existing[0].parent}"
    )
    return False


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


def _run_case_isolated(
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


def run_benchmarks():
    print("Fetching default test cases...")
    test_cases = get_default_test_cases()
    print(f"Found {len(test_cases)} test cases.")

    methods_to_test = _resolve_methods_to_test()
    required_tree_methods = ["kl", "kl_rogerstanimoto"]
    added_tree_methods = [m for m in required_tree_methods if m not in methods_to_test]
    if added_tree_methods:
        methods_to_test.extend(added_tree_methods)
        print(f"Added required tree methods for plotting: {added_tree_methods}")
    print(f"Methods: {methods_to_test}")

    # Keep plots enabled by default; UMAP comparison pages are always generated
    # when plotting is on.
    enable_plots = get_env_bool("KL_TE_ENABLE_PLOTS", default=True)
    enable_umap = enable_plots
    enable_manifold = (
        get_env_bool("KL_TE_ENABLE_MANIFOLD", default=False) and enable_plots
    )
    isolate_umap_cases = get_env_bool("KL_TE_UMAP_ISOLATE_CASES", default=enable_umap)
    case_timeout_sec = get_env_int("KL_TE_CASE_TIMEOUT_SEC", 1800)
    case_retry_count = max(0, get_env_int("KL_TE_CASE_RETRY_COUNT", 4))
    if enable_umap and "KL_TE_EMBEDDING_BACKEND" not in os.environ:
        os.environ["KL_TE_EMBEDDING_BACKEND"] = "umap"
    if enable_umap and "KL_TE_EMBEDDING_BACKEND_3D" not in os.environ:
        os.environ["KL_TE_EMBEDDING_BACKEND_3D"] = "umap"
    if enable_umap and "KL_TE_FORCE_UMAP_FOR_LARGE" not in os.environ:
        os.environ["KL_TE_FORCE_UMAP_FOR_LARGE"] = "1"
    if enable_umap:
        try:
            import umap  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "UMAP is required but not available. "
                "Install a compatible stack (e.g., NumPy <= 2.3 with numba/umap) "
                "or set KL_TE_ENABLE_PLOTS=0."
            ) from exc
    print(
        f"Plot settings: enable_plots={enable_plots}, "
        f"enable_umap={enable_umap}, enable_manifold={enable_manifold}, "
        f"isolate_umap_cases={isolate_umap_cases}, "
        f"case_retry_count={case_retry_count}"
    )

    # Single benchmark results root
    timestamp = format_timestamp_utc()
    base_output_dir = repo_root / "benchmarks" / "results"
    run_dir = base_output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "full_benchmark_comparison.csv"
    pdf_dir = run_dir / "plots"
    if enable_plots:
        pdf_dir.mkdir(exist_ok=True)

    # Load existing results if any to resume
    if output_path.exists():
        try:
            all_results = pd.read_csv(output_path)
            existing_case_keys: set[int] = set()
            if "test_case" in all_results.columns:
                existing_case_keys = set(
                    pd.to_numeric(all_results["test_case"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            elif "Test" in all_results.columns:
                existing_case_keys = set(
                    pd.to_numeric(all_results["Test"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            print(
                f"Resuming... Found {len(existing_case_keys)} already completed test indices."
            )
        except Exception as e:
            print(f"Error reading existing results, starting fresh: {e}")
            all_results = pd.DataFrame()
            existing_case_keys = set()
    else:
        all_results = pd.DataFrame()
        existing_case_keys = set()

    for i, case in enumerate(test_cases):
        case_key = i + 1
        case_id = case.get("name", case.get("id", f"case_{i}"))
        case_type = case.get("type", "unknown")

        if case_key in existing_case_keys:
            print(
                f"[{i + 1}/{len(test_cases)}] Skipping case: {case_id} "
                f"(test index {case_key} already done)"
            )
            continue

        print(
            f"[{i + 1}/{len(test_cases)}] Running case: {case_id} (Type: {case_type})",
            flush=True,
        )

        case["test_case_num"] = case_key

        try:
            pdf_path = (
                str((pdf_dir / f"{case_id}.pdf").absolute()) if enable_plots else None
            )

            # Use both *_features/samples and *_cols/rows keys to detect large cases.
            n_features = int(case.get("n_features", case.get("n_cols", 0)) or 0)
            n_samples = int(case.get("n_samples", case.get("n_rows", 0)) or 0)
            is_large = n_features > 400 or n_samples > 1000
            case_plot_umap = enable_umap
            case_plot_manifold = enable_manifold and not is_large

            if case_plot_umap and isolate_umap_cases:
                df_res = None
                last_err: Exception | None = None
                for attempt in range(1, case_retry_count + 2):
                    try:
                        df_res = _run_case_isolated(
                            case=case,
                            methods_to_test=methods_to_test,
                            case_plot_umap=case_plot_umap,
                            case_plot_manifold=case_plot_manifold,
                            enable_plots=enable_plots,
                            pdf_path=pdf_path,
                            timeout_sec=case_timeout_sec,
                        )
                        break
                    except RuntimeError as e:
                        last_err = e
                        msg = str(e)
                        if "exited with code -11" in msg:
                            if attempt <= case_retry_count:
                                print(
                                    f"  WARN: isolated subprocess SIGSEGV on '{case_id}' "
                                    f"(attempt {attempt}/{case_retry_count + 1}); retrying...",
                                    flush=True,
                                )
                                gc.collect()
                                continue
                            print(
                                f"  WARN: isolated subprocess kept SIGSEGV-ing on '{case_id}' "
                                f"after {case_retry_count + 1} attempts; falling back "
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

            if df_res is None:
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

            if not df_res.empty:
                df_res = df_res.rename(
                    columns={
                        "Method": "method",
                        "ARI": "ari",
                        "Case_Name": "case_id",
                        "Test": "test_case",
                        "True": "true_clusters",
                        "Found": "found_clusters",
                    }
                )

                name_map = {spec.name: method_id for method_id, spec in METHOD_SPECS.items()}
                df_res["method"] = df_res["method"].map(
                    lambda x: name_map.get(str(x), str(x))
                )

                all_results = pd.concat([all_results, df_res], ignore_index=True)

                write_header = not output_path.exists()
                df_res.to_csv(output_path, mode="a", header=write_header, index=False)

                print("  Results:", flush=True)
                for method in methods_to_test:
                    row = df_res[df_res["method"] == method]
                    if not row.empty:
                        ari_val = row["ari"].values[0]
                        print(f"    {method}: ARI={ari_val:.4f}", flush=True)
                    else:
                        print(f"    {method}: No result", flush=True)
            else:
                print("  No results returned.", flush=True)

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
        finally:
            gc.collect()

    print("-" * 50)
    print("Benchmark Complete.")

    if enable_plots:
        report_pdf = run_dir / "full_benchmark_report.pdf"
        ordered_case_pdfs = []
        for case in test_cases:
            case_id = case.get("name", case.get("id"))
            if not case_id:
                continue
            ordered_case_pdfs.append(pdf_dir / f"{case_id}.pdf")
        _concat_case_pdfs(ordered_case_pdfs, report_pdf)

    if all_results.empty:
        print("No results collected.")
        return

    df = all_results[all_results["method"].isin(methods_to_test)].copy()

    summary = df.groupby(["method"])["ari"].mean()
    print("\nMean ARI by Method:")
    print(summary)

    if "test_case" in df.columns:
        pivot = df.pivot(index="test_case", columns="method", values="ari")
        if "kl" in pivot.columns and "kl_rogerstanimoto" in pivot.columns:
            pivot["diff"] = pivot["kl"] - pivot["kl_rogerstanimoto"]

            diff_cases = pivot[pivot["diff"].abs() > 1e-6]
            if len(diff_cases) > 0:
                print(f"\nFound {len(diff_cases)} cases with different ARI scores:")
                print(diff_cases[["kl", "kl_rogerstanimoto", "diff"]])
            else:
                print(
                    "\nNo significant differences found between 'kl' (hamming) and 'kl_rogerstanimoto'."
                )

        print(f"\nDetailed results are saved to {output_path}")

        # Run failure diagnosis when available.
        if diagnose_benchmark_failures is not None:
            print("\nRunning failure diagnosis...")
            actual_audit_dir = run_dir / "audit"
            diagnose_benchmark_failures(
                str(output_path), str(actual_audit_dir), str(run_dir / "failure_report.md")
            )
        else:
            print("\nSkipping failure diagnosis: benchmarks.shared.debug_trace not found.")
    else:
        print("Could not pivot results: 'case_id' column missing.")
        print(df.head())


if __name__ == "__main__":
    run_benchmarks()
