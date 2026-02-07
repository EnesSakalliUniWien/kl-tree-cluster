#!/usr/bin/env python
"""
Phylogenetic benchmark - tests clustering on data with hierarchical tree structure.

This benchmark uses a random binary tree to simulate evolutionary divergence,
testing how well the method recovers the tree structure from sampled traits.

Usage:
    python benchmarks/phylogenetic/run.py
"""

import sys
from pathlib import Path

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from benchmarks.phylogenetic.generator import generate_phylogenetic_data
from benchmarks.phylogenetic.cases import PHYLOGENETIC_CASES
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.runners.method_registry import METHOD_SPECS


def main():
    print("=" * 70)
    print("PHYLOGENETIC BENCHMARK")
    print("=" * 70)
    print()

    # Create output directory relative to script
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Convert dictionary cases to the format expected by the pipeline if necessary
    # Or just use them directly if the pipeline supports it
    test_cases = []
    for group_name, cases in PHYLOGENETIC_CASES.items():
        for case in cases:
            case_copy = case.copy()
            if "name" not in case_copy:
                case_copy["name"] = group_name
            test_cases.append(case_copy)

    print(f"Running {len(test_cases)} phylogenetic test cases...")

    # Run benchmark pipeline
    # Note: the pipeline might need updates to handle our new structure
    # For now, we'll try running it with 'kl' method
    methods = ["kl", "kl_rogerstanimoto"]

    output_path = output_dir / "phylogenetic_results.csv"
    pdf_path = output_dir / "phylogenetic_plots.pdf"

    try:
        results_df, _ = benchmark_cluster_algorithm(
            test_cases=test_cases,
            methods=methods,
            verbose=True,
            plot_umap=True,
            concat_plots_pdf=True,
            concat_output=str(pdf_path.absolute()),
        )

        if not results_df.empty:
            results_df.to_csv(output_path, index=False)
            print(f"\nSaved results to {output_path}")
            print(f"Saved plots to {pdf_path}")

            # Print summary
            summary = results_df.groupby("Method")["ARI"].mean().round(4)
            print("\nMean ARI by Method:")
            print(summary)
        else:
            print("\nNo results returned from pipeline.")

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
