import sys
from pathlib import Path

# Add the project root to the python path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.debug_trace import diagnose_benchmark_failures
from datetime import datetime, timezone
import pandas as pd
import numpy as np


def _format_timestamp_utc() -> str:
    """Return a filesystem-safe UTC timestamp like 20250101_235959Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def run_benchmarks():
    print("Fetching default test cases...")
    test_cases = get_default_test_cases()
    print(f"Found {len(test_cases)} test cases.")

    # We want to compare the default 'kl' (now hamming) against 'kl_rogerstanimoto'
    methods_to_test = ["kl", "kl_rogerstanimoto"]

    # Results directory relative to script
    timestamp = _format_timestamp_utc()
    base_output_dir = Path(__file__).parent / "results"
    run_dir = base_output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "full_benchmark_comparison.csv"
    pdf_dir = run_dir / "plots"
    pdf_dir.mkdir(exist_ok=True)

    # Load existing results if any to resume
    if output_path.exists():
        try:
            all_results = pd.read_csv(output_path)
            existing_cases = set(all_results["case_id"].unique())
            print(f"Resuming... Found {len(existing_cases)} already completed cases.")
        except Exception as e:
            print(f"Error reading existing results, starting fresh: {e}")
            all_results = pd.DataFrame()
            existing_cases = set()
    else:
        all_results = pd.DataFrame()
        existing_cases = set()

    for i, case in enumerate(test_cases):
        case_id = case.get("name", case.get("id", f"case_{i}"))
        case_type = case.get("type", "unknown")

        if case_id in existing_cases:
            print(
                f"[{i + 1}/{len(test_cases)}] Skipping case: {case_id} (Already done)"
            )
            continue

        print(
            f"[{i + 1}/{len(test_cases)}] Running case: {case_id} (Type: {case_type})",
            flush=True,
        )

        case["test_case_num"] = i + 1

        try:
            pdf_path = str((pdf_dir / f"{case_id}.pdf").absolute())

            # Disable heavy plotting for large/high-dimensional cases to avoid segfaults (Isomap/UMAP)
            # Case 92 (1000x600) caused a segfault. Case 91 (800x350) passed.
            n_cols = case.get("n_cols", 0)
            n_rows = case.get("n_rows", 0)
            is_large = n_cols > 400 or n_rows > 1000

            df_res, _ = benchmark_cluster_algorithm(
                test_cases=[case],
                methods=methods_to_test,
                verbose=False,
                plot_umap=not is_large,
                plot_manifold=not is_large,
                concat_plots_pdf=True,
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

                name_map = {
                    "KL Divergence": "kl",
                    "KL (Rogers-Tanimoto)": "kl_rogerstanimoto",
                }
                df_res["method"] = df_res["method"].map(lambda x: name_map.get(x, x))

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

    print("-" * 50)
    print("Benchmark Complete.")

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

        # Run failure diagnosis
        print("\nRunning failure diagnosis...")
        # project_root is repo_root
        # results_dir is run_dir
        actual_audit_dir = run_dir / "audit"

        diagnose_benchmark_failures(
            str(output_path), str(actual_audit_dir), str(run_dir / "failure_report.md")
        )
    else:
        print("Could not pivot results: 'case_id' column missing.")
        print(df.head())


if __name__ == "__main__":
    run_benchmarks()
