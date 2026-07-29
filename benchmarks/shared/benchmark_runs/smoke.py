"""Smoke benchmark suite selection and reporting helpers."""

from __future__ import annotations

import pandas as pd

from benchmarks.shared.cases.geometry import case_recipe_cluster_count

SMOKE_SUBSET_NAMES = {
    # Gaussian
    "gauss_clear_small",
    "gauss_clear_large",
    "gauss_moderate_3c",
    # Binary
    "binary_perfect_4c",
    "binary_low_noise_4c",
    "binary_moderate_6c",
    "sparse_features_72x72",
    # Categorical
    "cat_clear_3cat_4c",
    # SBM
    "sbm_clear_small",
    # Overlapping
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
    # Gaussian overlap
    "gauss_overlap_3c_small",
    # Edge cases
    "binary_2clusters",
    "binary_many_features",
}


def select_smoke_cases(all_cases: list[dict]) -> list[dict]:
    """Return the canonical fast smoke subset in source case order."""
    return [case for case in all_cases if case["name"] in SMOKE_SUBSET_NAMES]


def print_smoke_case_manifest(subset: list[dict], *, total_cases: int, spectral_jobs: str) -> None:
    """Print the selected smoke cases and their true cluster counts."""
    print(f"Selected {len(subset)}/{total_cases} cases:")
    print(f"Spectral settings: TBS_N_JOBS={spectral_jobs}")
    for case in subset:
        print(f"  {case['name']:<35s}  K={case_recipe_cluster_count(case)}")
    print()


def print_tbs_smoke_summary(df_results: pd.DataFrame) -> None:
    """Print a compact TBS-only smoke benchmark summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    tbs = df_results[df_results["method"] == "tbs"].copy()
    tbs = tbs.sort_values("case_id")

    print(f"\n{'Case':<36s} {'True':>4s} {'Found':>5s} {'ARI':>7s} {'NMI':>7s} {'Status'}")
    print("-" * 70)
    for _, row in tbs.iterrows():
        ari_val = row["ari"]
        nmi_val = row["nmi"]
        ari_str = f"{ari_val:.3f}" if not (ari_val != ari_val) else "  N/A"
        nmi_str = f"{nmi_val:.3f}" if not (nmi_val != nmi_val) else "  N/A"
        true_str = f"{row['true_clusters']:>4.0f}" if row["true_clusters"] > 0 else " N/A"
        marker = (
            " ✓"
            if row["true_clusters"] > 0 and row["found_clusters"] == row["true_clusters"]
            else ""
        )
        print(
            f"{row['case_id']:<36s} {true_str} {row['found_clusters']:>5.0f} "
            f"{ari_str:>7s} {nmi_str:>7s} {row['status']}{marker}"
        )

    tbs_with_truth = tbs[tbs["true_clusters"] > 0]
    exact_k = (tbs_with_truth["found_clusters"] == tbs_with_truth["true_clusters"]).sum()
    print(f"\nExact K: {exact_k}/{len(tbs_with_truth)}")
    ari_valid = tbs["ari"].dropna()
    print(f"Mean ARI: {ari_valid.mean():.3f}" if len(ari_valid) > 0 else "Mean ARI: N/A")
    print(f"Median ARI: {tbs['ari'].median():.3f}")


__all__ = [
    "SMOKE_SUBSET_NAMES",
    "print_smoke_case_manifest",
    "print_tbs_smoke_summary",
    "select_smoke_cases",
]
