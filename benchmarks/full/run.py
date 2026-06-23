import gc
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]

from benchmarks.shared.plots.backend import configure_matplotlib_backend

configure_matplotlib_backend()

import matplotlib.pyplot as plt
import pandas as pd

from benchmarks.diagnostics.failure.debug_trace import diagnose_benchmark_failures
from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.cases.geometry import case_recipe_geometry
from benchmarks.shared.config import DEFAULT_METHODS
from benchmarks.shared.env import get_env_bool, get_env_int
from benchmarks.shared.plots.cover_page import (
    GROUP_ORDER,
    category_group,
    generate_overview_page,
    write_case_manifest_pages_to_pdf,
    write_section_page_to_pdf,
)
from benchmarks.shared.relationship_analysis import analyze_benchmark_relationships
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_execution import run_case_with_optional_isolation
from benchmarks.shared.util.method_selection import resolve_methods_from_env
from benchmarks.shared.util.pdf.merge import merge_existing_pdfs
from benchmarks.shared.util.time import format_timestamp_utc


def _compute_resume_coverage(
    existing_results: pd.DataFrame,
    expected_methods: list[str],
) -> tuple[set[int], dict[int, list[str]], int]:
    """Compute fully-complete and partial case coverage from an existing CSV."""
    if existing_results.empty:
        return set(), {}, 0

    if "test_case" not in existing_results.columns:
        return set(), {}, 0

    if "method" not in existing_results.columns:
        return set(), {}, 0

    expected_set = set(expected_methods)

    case_keys = pd.to_numeric(existing_results["test_case"], errors="coerce")
    methods_raw = existing_results["method"].astype(str)
    methods_norm = methods_raw.where(methods_raw.isin(expected_set))
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


def _stamp_full_run_case_identity(
    results: pd.DataFrame,
    *,
    case_key: int,
    case_id: str,
) -> pd.DataFrame:
    """Map one-case pipeline output back to the full benchmark case identity."""
    if results.empty:
        return results
    if "case_id" not in results.columns:
        raise ValueError("Benchmark result rows must include case_id.")
    observed_case_ids = set(results["case_id"].dropna().astype(str))
    if observed_case_ids != {case_id}:
        raise ValueError(
            f"Benchmark returned case_id values {sorted(observed_case_ids)!r}; "
            f"expected only {case_id!r}."
        )
    results = results.copy()
    results["test_case"] = int(case_key)
    return results


def run_benchmarks():
    # Default to single-threaded spectral decomposition workers to avoid
    # thread oversubscription (outer benchmark parallelism + BLAS threads).
    # Users can still override by setting TBS_N_JOBS explicitly.
    spectral_jobs = os.environ.setdefault("TBS_N_JOBS", "1")

    case_suite = os.environ.get("TBS_CASE_SUITE", "full").strip().lower()
    print(f"Fetching benchmark case suite: {case_suite}")
    test_cases = get_test_cases_by_suite(case_suite)
    print(f"Found {len(test_cases)} test cases.")

    methods_to_test = resolve_methods_from_env(
        METHOD_SPECS,
        default_methods=DEFAULT_METHODS,
    )
    # Ensure the primary TBS (Hamming + average) method is always included for
    # tree plot generation.
    if "tbs" not in methods_to_test:
        methods_to_test.insert(0, "tbs")
        print("Added required tree method: tbs")
    print(f"Methods: {methods_to_test}")
    print(f"Spectral settings: TBS_N_JOBS={spectral_jobs}")

    # Keep plots enabled by default; UMAP comparison pages are always generated
    # when plotting is on.
    enable_plots = get_env_bool("TBS_ENABLE_PLOTS", default=True)
    enable_umap = enable_plots
    enable_manifold = get_env_bool("TBS_ENABLE_MANIFOLD", default=False) and enable_plots
    isolate_umap_cases = get_env_bool("TBS_UMAP_ISOLATE_CASES", default=enable_umap)
    case_timeout_sec = get_env_int("TBS_CASE_TIMEOUT_SEC", 1800)
    if enable_umap and "TBS_EMBEDDING_BACKEND" not in os.environ:
        os.environ["TBS_EMBEDDING_BACKEND"] = "umap"
    if enable_umap and "TBS_EMBEDDING_BACKEND_3D" not in os.environ:
        os.environ["TBS_EMBEDDING_BACKEND_3D"] = "umap"
    if enable_umap and "TBS_FORCE_UMAP_FOR_LARGE" not in os.environ:
        os.environ["TBS_FORCE_UMAP_FOR_LARGE"] = "1"
    run_relationship_analysis = get_env_bool("TBS_RUN_RELATIONSHIP_ANALYSIS", default=True)
    enable_relationship_plots = get_env_bool(
        "TBS_ENABLE_RELATIONSHIP_PLOTS",
        default=enable_plots,
    )
    if enable_umap:
        import umap  # noqa: F401
    print(
        f"Plot settings: enable_plots={enable_plots}, "
        f"enable_umap={enable_umap}, enable_manifold={enable_manifold}, "
        f"isolate_umap_cases={isolate_umap_cases}"
    )
    print(
        "Relationship analysis settings: "
        f"enabled={run_relationship_analysis}, plots={enable_relationship_plots}"
    )

    # Single benchmark results root
    timestamp = format_timestamp_utc()
    base_output_dir = repo_root / "benchmarks" / "results"
    configured_run_dir = os.environ.get("TBS_RUN_DIR")
    run_dir = (
        Path(configured_run_dir).expanduser().resolve()
        if configured_run_dir
        else base_output_dir / f"run_{timestamp}_{case_suite}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if configured_run_dir:
        print(f"Using configured run directory: {run_dir}")

    output_path = run_dir / f"{case_suite}_benchmark_comparison.csv"
    pdf_dir = run_dir / "plots"
    if enable_plots:
        pdf_dir.mkdir(exist_ok=True)

    # Load existing results if any to resume
    if output_path.exists():
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
    else:
        all_results = pd.DataFrame()
        completed_case_keys = set()
        missing_methods_by_case = {}

    for i, case in enumerate(test_cases):
        case_key = i + 1
        case_id = case["name"]
        case_type = case["category"]

        if case_key in completed_case_keys:
            print(
                f"[{i + 1}/{len(test_cases)}] Skipping case: {case_id} "
                f"(test index {case_key} already done)"
            )
            continue

        if case_key in missing_methods_by_case:
            methods_for_case = missing_methods_by_case[case_key]
        else:
            methods_for_case = methods_to_test
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

        pdf_path = str((pdf_dir / f"{case_id}.pdf").absolute()) if enable_plots else None

        n_samples, n_features = case_recipe_geometry(case)
        is_large = n_features > 400 or n_samples > 1000
        case_plot_umap = enable_umap
        case_plot_manifold = enable_manifold and not is_large

        df_res = run_case_with_optional_isolation(
            case=case,
            methods_to_test=methods_for_case,
            case_plot_umap=case_plot_umap,
            case_plot_manifold=case_plot_manifold,
            enable_plots=enable_plots,
            pdf_path=pdf_path,
            isolate_umap_cases=isolate_umap_cases,
            timeout_sec=case_timeout_sec,
            include_validation_page=False,
        )
        df_res = _stamp_full_run_case_identity(
            df_res,
            case_key=case_key,
            case_id=str(case_id),
        )

        if not df_res.empty:
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

        gc.collect()

    print("-" * 50)
    print("Benchmark Complete.")

    relationship_outputs = None
    relationship_pdf: Path | None = None
    if all_results.empty:
        print("No results collected.")
    else:
        df = all_results[all_results["method"].isin(methods_to_test)].copy()

        if run_relationship_analysis:
            print("\nRunning benchmark relationship analysis...")
            relationship_outputs = analyze_benchmark_relationships(
                df,
                run_dir,
                source_path=output_path,
                include_plots=enable_relationship_plots,
            )
            relationship_pdf = (
                Path(relationship_outputs.plots_pdf)
                if relationship_outputs.plots_pdf is not None
                else None
            )
            print(f"Relationship report written: {relationship_outputs.report_md}")
            if relationship_pdf is not None:
                print(f"Relationship plots written: {relationship_pdf}")

        summary = df.groupby(["method"])["ari"].mean()
        print("\nMean ARI by Method (ok rows only; skipped rows have NaN ARI):")
        print(summary)
        status_summary = df.groupby(["method", "status"]).size().unstack(fill_value=0)
        print("\nRun status counts by Method:")
        print(status_summary)

        if "test_case" in df.columns:
            print(f"\nDetailed results are saved to {output_path}")

            print("\nRunning failure diagnosis...")
            actual_audit_dir = run_dir / "audit"
            diagnose_benchmark_failures(
                str(output_path),
                str(actual_audit_dir),
                str(run_dir / "failure_report.md"),
            )
        else:
            print("Could not pivot results: 'case_id' column missing.")
            print(df.head())

    if enable_plots:
        report_pdf = run_dir / f"{case_suite}_benchmark_report.pdf"

        # --- Overview cover page ---
        cover_pdf = run_dir / "_cover_overview.pdf"
        from matplotlib.backends.backend_pdf import PdfPages as _PP

        overview_fig = generate_overview_page(n_cases=len(test_cases), timestamp=timestamp)
        with _PP(str(cover_pdf)) as _pp:
            _pp.savefig(overview_fig)
        plt.close(overview_fig)
        print(f"Generated overview page: {cover_pdf}")

        manifest_pdf = run_dir / "_case_generation_manifest.pdf"
        write_case_manifest_pages_to_pdf(test_cases, manifest_pdf)
        print(f"Generated case manifest: {manifest_pdf}")

        # --- Build section page PDFs for groups that actually occur in this run ---
        present_groups = {
            category_group(case["category"])
            for case in test_cases
        }
        section_pdfs: dict[str, Path] = {}
        for group in GROUP_ORDER:
            if group not in present_groups:
                continue
            sec_path = run_dir / f"_section_{group}.pdf"
            result = write_section_page_to_pdf(group, str(sec_path))
            if result is not None:
                section_pdfs[group] = sec_path

        # --- Assemble merge order: overview, then (section + cases) per group ---
        ordered_case_pdfs: list[Path] = [cover_pdf, manifest_pdf]

        # Determine each case's group from its category
        seen_groups: set[str] = set()
        for case in test_cases:
            case_id = case["name"]
            cat = case["category"]
            group = category_group(cat)

            # Insert the section page before the first case of each group
            if group not in seen_groups and group in section_pdfs:
                ordered_case_pdfs.append(section_pdfs[group])
                seen_groups.add(group)

            ordered_case_pdfs.append(pdf_dir / f"{case_id}.pdf")

        if relationship_pdf is not None and relationship_pdf.exists():
            ordered_case_pdfs.append(relationship_pdf)
            print(f"Appending relationship plots to full report: {relationship_pdf}")
        merge_existing_pdfs(ordered_case_pdfs, report_pdf, verbose=True)


if __name__ == "__main__":
    run_benchmarks()
