#!/usr/bin/env python
"""Quick benchmark to test SBM cases with the updated pipeline."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm


def main():
    print("Testing SBM cases with updated pipeline...")

    # Get only SBM cases
    all_cases = get_default_test_cases()
    sbm_cases = [c for c in all_cases if c.get("generator") == "sbm"]

    print(f"Found {len(sbm_cases)} SBM test cases")

    # Run benchmark
    df, _ = benchmark_cluster_algorithm(
        test_cases=sbm_cases,
        methods=["kl"],
        verbose=True,
        plot_umap=False,
        plot_manifold=False,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df[["Case_Name", "Method", "ARI", "True", "Found"]].to_string())

    print(f"\nMean ARI: {df['ARI'].mean():.4f}")


if __name__ == "__main__":
    main()
