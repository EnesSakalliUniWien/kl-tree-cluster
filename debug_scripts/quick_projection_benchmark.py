"""Quick benchmark comparing sparse vs orthonormal projection methods.

Runs the full benchmark pipeline on a small subset of test cases
to compare both projection methods.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

import kl_clustering_analysis.config as cfg
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    _PROJECTION_CACHE,
    _PROJECTOR_CACHE,
)
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from benchmarks.shared.cases import SMALL_TEST_CASES
from benchmarks.shared.cases.overlapping import (
    OVERLAPPING_BINARY_MODERATE_CASES,
    OVERLAPPING_GAUSSIAN_CASES,
)


def run_with_method(method: str, test_cases: list) -> tuple[pd.DataFrame, float]:
    """Run benchmark with specified projection method."""
    # Set method
    original = cfg.PROJECTION_METHOD
    cfg.PROJECTION_METHOD = method

    # Clear caches
    _PROJECTION_CACHE.clear()
    _PROJECTOR_CACHE.clear()

    print(f"\n{'=' * 60}")
    print(f"Running with PROJECTION_METHOD = {method}")
    print(f"{'=' * 60}")

    start = time.time()
    try:
        df, _ = benchmark_cluster_algorithm(
            test_cases=test_cases,
            significance_level=0.05,
            verbose=True,
            plot_umap=False,
            plot_manifold=False,
            concat_plots_pdf=False,
            save_individual_plots=False,
        )
        elapsed = time.time() - start
        return df, elapsed
    finally:
        cfg.PROJECTION_METHOD = original


def main():
    print("\n" + "=" * 70)
    print("QUICK BENCHMARK: Sparse vs Orthonormal Projection")
    print("=" * 70)
    print(f"\nCurrent default: PROJECTION_METHOD = {cfg.PROJECTION_METHOD}")

    # Use small test cases for speed
    # Use overlapping test cases (a subset for speed)
    test_cases = OVERLAPPING_BINARY_MODERATE_CASES[:2] + OVERLAPPING_GAUSSIAN_CASES[:2]
    print(f"\nTest cases: {[tc['name'] for tc in test_cases]}")

    # Run with orthonormal
    df_orth, time_orth = run_with_method("orthonormal", test_cases)

    # Run with sparse
    df_sparse, time_sparse = run_with_method("sparse", test_cases)

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Time (s)':<12} {'Mean ARI':<12} {'Mean NMI':<12}")
    print("-" * 55)

    # Filter to KL Divergence method only for comparison
    df_orth_kl = df_orth[df_orth["Method"] == "KL Divergence"]
    df_sparse_kl = df_sparse[df_sparse["Method"] == "KL Divergence"]

    orth_ari = df_orth_kl["ARI"].mean() if "ARI" in df_orth_kl.columns else np.nan
    orth_nmi = df_orth_kl["NMI"].mean() if "NMI" in df_orth_kl.columns else np.nan
    sparse_ari = df_sparse_kl["ARI"].mean() if "ARI" in df_sparse_kl.columns else np.nan
    sparse_nmi = df_sparse_kl["NMI"].mean() if "NMI" in df_sparse_kl.columns else np.nan

    print(f"{'Orthonormal':<15} {time_orth:<12.2f} {orth_ari:<12.4f} {orth_nmi:<12.4f}")
    print(
        f"{'Sparse':<15} {time_sparse:<12.2f} {sparse_ari:<12.4f} {sparse_nmi:<12.4f}"
    )
    print("-" * 55)

    ari_diff = orth_ari - sparse_ari
    nmi_diff = orth_nmi - sparse_nmi
    time_ratio = time_sparse / time_orth if time_orth > 0 else 1.0

    print(f"{'Difference':<15} {'':<12} {ari_diff:+<12.4f} {nmi_diff:+<12.4f}")
    print(
        f"\nSpeed: Sparse is {time_ratio:.2f}x {'faster' if time_ratio < 1 else 'slower'} than Orthonormal"
    )

    # Per-case comparison
    print("\n" + "-" * 70)
    print("PER-CASE COMPARISON (KL Divergence only)")
    print("-" * 70)
    print(f"{'Case':<20} {'Orth ARI':<12} {'Sparse ARI':<12} {'Diff':<10}")
    print("-" * 55)

    for tc in test_cases:
        name = tc["name"]
        orth_row = df_orth_kl[df_orth_kl["Case_Name"] == name]
        sparse_row = df_sparse_kl[df_sparse_kl["Case_Name"] == name]

        # Average across different tree configs
        orth_val = orth_row["ARI"].mean() if len(orth_row) > 0 else np.nan
        sparse_val = sparse_row["ARI"].mean() if len(sparse_row) > 0 else np.nan
        diff = (
            orth_val - sparse_val
            if not (np.isnan(orth_val) or np.isnan(sparse_val))
            else np.nan
        )

        diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
        print(f"{name:<20} {orth_val:<12.4f} {sparse_val:<12.4f} {diff_str:<10}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if abs(ari_diff) < 0.01:
        print("→ Both methods produce SIMILAR clustering results")
    elif ari_diff > 0:
        print(f"→ Orthonormal is BETTER by {ari_diff:.4f} ARI")
    else:
        print(f"→ Sparse is BETTER by {-ari_diff:.4f} ARI")

    if time_ratio < 0.9:
        print(f"→ Sparse is {1 / time_ratio:.1f}x FASTER")
    elif time_ratio > 1.1:
        print(f"→ Orthonormal is {time_ratio:.1f}x FASTER")
    else:
        print("→ Both methods have SIMILAR speed")


if __name__ == "__main__":
    main()
