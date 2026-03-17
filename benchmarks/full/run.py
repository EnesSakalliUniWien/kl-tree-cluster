import gc
import os
import sys
from pathlib import Path

# Load shared path bootstrap helper from benchmarks root.
_script_path = Path(__file__).resolve()
_benchmarks_root = (
    _script_path.parent if _script_path.parent.name == "benchmarks" else _script_path.parents[1]
)
if str(_benchmarks_root) not in sys.path:
    sys.path.insert(0, str(_benchmarks_root))
from _bootstrap import ensure_repo_root_on_path

# Ensure repo root is importable before any benchmarks/kl_clustering imports.
repo_root = ensure_repo_root_on_path(__file__)

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.calibration.run import run_calibration_suite
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.env import get_env_bool, get_env_int
from benchmarks.shared.plots.cover_page import (
    GROUP_ORDER,
    category_group,
    generate_overview_page,
    write_section_page_to_pdf,
)
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_execution import run_case_with_optional_isolation
from benchmarks.shared.util.method_selection import resolve_methods_from_env
from benchmarks.shared.util.pdf.merge import merge_existing_pdfs
from benchmarks.shared.util.time import format_timestamp_utc
from kl_clustering_analysis import config

try:
    from benchmarks.shared.debug_trace import diagnose_benchmark_failures
except ImportError:
    diagnose_benchmark_failures = None


def _compute_resume_coverage(
    existing_results: pd.DataFrame,
    expected_methods: list[str],
) -> tuple[set[int], dict[int, list[str]], int]:
    """Compute fully-complete and partial case coverage from an existing CSV."""
    if existing_results.empty:
        return set(), {}, 0

    case_col = "test_case" if "test_case" in existing_results.columns else "Test"
    if case_col not in existing_results.columns:
        return set(), {}, 0

    method_col = "method" if "method" in existing_results.columns else "Method"
    if method_col not in existing_results.columns:
        return set(), {}, 0

    expected_set = set(expected_methods)
    name_to_id = {spec.name: method_id for method_id, spec in METHOD_SPECS.items()}

    case_keys = pd.to_numeric(existing_results[case_col], errors="coerce")
    methods_raw = existing_results[method_col].astype(str)

    def _normalize_method(value: str) -> str | None:
        if value in expected_set:
            return value
        mapped = name_to_id.get(value)
        if mapped in expected_set:
            return mapped
        return None

    methods_norm = methods_raw.map(_normalize_method)
    progress = pd.DataFrame({"case_key": case_keys, "method_id": methods_norm})
    progress = progress.dropna(subset=["case_key", "method_id"])
    if progress.empty:
        return set(), {}, 0

    progress["case_key"] = progress["case_key"].astype(int)

    methods_by_case = (
        progress.groupby("case_key")["method_id"].agg(lambda s: set(s.tolist())).to_dict()
    )

    completed_cases: set[int] = set()
    missing_methods_by_case: dict[int, list[str]] = {}
    for case_key, seen_methods in methods_by_case.items():
        missing = [method_id for method_id in expected_methods if method_id not in seen_methods]
        if not missing:
            completed_cases.add(int(case_key))
        else:
            missing_methods_by_case[int(case_key)] = missing

    return completed_cases, missing_methods_by_case, len(methods_by_case)


def run_benchmarks():
    # Default to single-threaded spectral decomposition workers to avoid
    # thread oversubscription (outer benchmark parallelism + BLAS threads).
    # Users can still override by setting KL_TE_N_JOBS explicitly.
    spectral_jobs = os.environ.setdefault("KL_TE_N_JOBS", "1")

    print("Fetching default test cases...")
    test_cases = get_default_test_cases()
    print(f"Found {len(test_cases)} test cases.")

    methods_to_test = resolve_methods_from_env(METHOD_SPECS)
    # Ensure the primary KL (Hamming + average) method is always included for
    # tree plot generation.
    if "kl" not in methods_to_test:
        methods_to_test.insert(0, "kl")
        print("Added required tree method: kl")
    print(f"Methods: {methods_to_test}")
    print(f"Spectral settings: KL_TE_N_JOBS={spectral_jobs}")

    # Keep plots enabled by default; UMAP comparison pages are always generated
    # when plotting is on.
    enable_plots = get_env_bool("KL_TE_ENABLE_PLOTS", default=True)
    enable_umap = enable_plots
    enable_manifold = get_env_bool("KL_TE_ENABLE_MANIFOLD", default=False) and enable_plots
    isolate_umap_cases = get_env_bool("KL_TE_UMAP_ISOLATE_CASES", default=enable_umap)
    case_timeout_sec = get_env_int("KL_TE_CASE_TIMEOUT_SEC", 1800)
    case_retry_count = max(0, get_env_int("KL_TE_CASE_RETRY_COUNT", 4))
    if enable_umap and "KL_TE_EMBEDDING_BACKEND" not in os.environ:
        os.environ["KL_TE_EMBEDDING_BACKEND"] = "umap"
    if enable_umap and "KL_TE_EMBEDDING_BACKEND_3D" not in os.environ:
        os.environ["KL_TE_EMBEDDING_BACKEND_3D"] = "umap"
    if enable_umap and "KL_TE_FORCE_UMAP_FOR_LARGE" not in os.environ:
        os.environ["KL_TE_FORCE_UMAP_FOR_LARGE"] = "1"
    run_calibration = get_env_bool("KL_TE_RUN_CALIBRATION", default=False)
    calibration_null_reps = get_env_int("KL_TE_CAL_NULL_REPS", 30)
    calibration_treebh_reps = get_env_int("KL_TE_CAL_TREEBH_REPS", 200)
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
    if run_calibration:
        print(
            "Calibration settings: "
            f"null_reps={calibration_null_reps}, treebh_reps={calibration_treebh_reps}"
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
            completed_case_keys, missing_methods_by_case, tracked_case_count = (
                _compute_resume_coverage(
                    all_results,
                    methods_to_test,
                )
            )
            partial_case_count = max(0, tracked_case_count - len(completed_case_keys))
            print(
                "Resuming... "
                f"Found {len(completed_case_keys)} fully completed test indices "
                f"and {partial_case_count} partial cases."
            )
        except Exception as e:
            print(f"Error reading existing results, starting fresh: {e}")
            all_results = pd.DataFrame()
            completed_case_keys = set()
            missing_methods_by_case = {}
    else:
        all_results = pd.DataFrame()
        completed_case_keys = set()
        missing_methods_by_case = {}

    for i, case in enumerate(test_cases):
        case_key = i + 1
        case_id = case.get("name", case.get("id", f"case_{i}"))
        case_type = case.get("type", "unknown")

        if case_key in completed_case_keys:
            print(
                f"[{i + 1}/{len(test_cases)}] Skipping case: {case_id} "
                f"(test index {case_key} already done)"
            )
            continue

        methods_for_case = missing_methods_by_case.get(case_key, methods_to_test)
        if case_key in missing_methods_by_case:
            print(
                f"[{i + 1}/{len(test_cases)}] Resuming partial case: {case_id} "
                f"(missing methods: {methods_for_case})",
                flush=True,
            )

        print(
            f"[{i + 1}/{len(test_cases)}] Running case: {case_id} (Type: {case_type})",
            flush=True,
        )

        case["test_case_num"] = case_key

        try:
            pdf_path = str((pdf_dir / f"{case_id}.pdf").absolute()) if enable_plots else None

            # Use both *_features/samples and *_cols/rows keys to detect large cases.
            n_features = int(case.get("n_features", case.get("n_cols", 0)) or 0)
            n_samples = int(case.get("n_samples", case.get("n_rows", 0)) or 0)
            is_large = n_features > 400 or n_samples > 1000
            case_plot_umap = enable_umap
            case_plot_manifold = enable_manifold and not is_large

            df_res = run_case_with_optional_isolation(
                case=case,
                case_id=str(case_id),
                methods_to_test=methods_for_case,
                case_plot_umap=case_plot_umap,
                case_plot_manifold=case_plot_manifold,
                enable_plots=enable_plots,
                pdf_path=pdf_path,
                isolate_umap_cases=isolate_umap_cases,
                timeout_sec=case_timeout_sec,
                retry_count=case_retry_count,
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
                df_res["method"] = df_res["method"].map(lambda x: name_map.get(str(x), str(x)))

                all_results = pd.concat([all_results, df_res], ignore_index=True)

                write_header = not output_path.exists()
                df_res.to_csv(output_path, mode="a", header=write_header, index=False)

                print("  Results:", flush=True)
                for method in methods_for_case:
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

    calibration_pdf: Path | None = None
    if run_calibration:
        print("\nRunning calibration suite (null/Type-I + TreeBH)...")
        try:
            cal_outputs = run_calibration_suite(
                run_dir=run_dir,
                alpha=float(config.SIBLING_ALPHA),
                n_reps_null=max(1, int(calibration_null_reps)),
                n_reps_treebh=max(1, int(calibration_treebh_reps)),
            )
            calibration_pdf = Path(cal_outputs["plots_pdf"])
            print(f"Calibration report written: {cal_outputs['report_md']}")
            print(f"Calibration plots written: {cal_outputs['plots_pdf']}")
        except Exception as e:
            print(f"Calibration suite failed: {e}")

    if enable_plots:
        report_pdf = run_dir / "full_benchmark_report.pdf"

        # --- Overview cover page ---
        cover_pdf = run_dir / "_cover_overview.pdf"
        from matplotlib.backends.backend_pdf import PdfPages as _PP

        overview_fig = generate_overview_page(n_cases=len(test_cases), timestamp=timestamp)
        with _PP(str(cover_pdf)) as _pp:
            _pp.savefig(overview_fig)
        plt.close(overview_fig)
        print(f"Generated overview page: {cover_pdf}")

        # --- Build section page PDFs ---
        section_pdfs: dict[str, Path] = {}
        for group in GROUP_ORDER:
            sec_path = run_dir / f"_section_{group}.pdf"
            result = write_section_page_to_pdf(group, str(sec_path))
            if result is not None:
                section_pdfs[group] = sec_path

        # --- Assemble merge order: overview, then (section + cases) per group ---
        ordered_case_pdfs: list[Path] = [cover_pdf]

        # Determine each case's group from its category
        seen_groups: set[str] = set()
        for case in test_cases:
            case_id = case.get("name", case.get("id"))
            if not case_id:
                continue
            cat = case.get("category", "")
            group = category_group(cat)

            # Insert the section page before the first case of each group
            if group not in seen_groups and group in section_pdfs:
                ordered_case_pdfs.append(section_pdfs[group])
                seen_groups.add(group)

            ordered_case_pdfs.append(pdf_dir / f"{case_id}.pdf")

        # Append section pages for any groups that had no cases (shouldn't happen)
        for group in GROUP_ORDER:
            if group not in seen_groups and group in section_pdfs:
                ordered_case_pdfs.append(section_pdfs[group])

        if calibration_pdf is not None and calibration_pdf.exists():
            ordered_case_pdfs.append(calibration_pdf)
            print(f"Appending calibration plots to full report: {calibration_pdf}")
        merge_existing_pdfs(ordered_case_pdfs, report_pdf, verbose=True)

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
