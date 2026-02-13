#!/usr/bin/env python
"""
Special benchmark run: Overlapping to Gaussian cases.
Generates a single unified PDF report.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmarks.shared.cases import get_test_cases_by_category, list_categories
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.time_utils import format_timestamp_utc
import pandas as pd


def main():
    print("Starting specialized benchmark: Overlapping to Gaussian")

    categories = [
        "overlapping_binary_heavy",
        "overlapping_binary_moderate",
        "overlapping_binary_partial",
        "overlapping_binary_highd",
        "overlapping_binary_unbalanced",
        "overlapping_gaussian",
        "gaussian_clear",
        "gaussian_mixed",
        "gaussian_extreme_noise",
        "improved_gaussian",
    ]

    available = list_categories()
    categories = [c for c in categories if c in available]

    test_cases = []
    for cat in categories:
        cases = get_test_cases_by_category(cat)
        for c in cases:
            c["name"] = f"{cat}_{c.get('name', 'case')}"
        test_cases.extend(cases)

    print(f"Found {len(test_cases)} test cases across {len(categories)} categories.")

    timestamp = format_timestamp_utc()
    # Single benchmark results root
    output_root = Path(__file__).parent / "results"
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.csv"
    pdf_path = run_dir / "report.pdf"

    print(f"Running benchmarks (saving to {run_dir})...")

    try:
        results_df, _ = benchmark_cluster_algorithm(
            test_cases=test_cases,
            methods=["kl", "kl_rogerstanimoto"],
            verbose=True,
            plot_umap=True,
            concat_plots_pdf=True,
            concat_output=str(pdf_path.absolute()),
            save_individual_plots=False,
        )

        if not results_df.empty:
            results_df.to_csv(results_path, index=False)
            print(f"\nSuccess! Results saved to {results_path}")
            print(f"Final report generated at {pdf_path}")

            summary = results_df.groupby("Method")["ARI"].mean().round(4)
            print("\nMean ARI Summary:")
            print(summary)
        else:
            print("\nNo results were produced.")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
