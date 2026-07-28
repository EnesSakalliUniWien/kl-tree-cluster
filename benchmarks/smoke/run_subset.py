#!/usr/bin/env python3
"""
Quick subset benchmark — picks ~15 representative cases across categories
and runs them with plots.
"""

import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.cases.geometry import case_recipe_cluster_count
from benchmarks.shared.pipeline import benchmark_cluster_algorithm

# Default to single-threaded spectral decomposition workers to avoid
# thread oversubscription. Users can override by setting TBS_N_JOBS.
spectral_jobs = os.environ.setdefault("TBS_N_JOBS", "1")

# Pick a representative subset: mix of easy/hard, different K, different types
SUBSET_NAMES = {
    # Gaussian
    "gauss_clear_small",  # K=3, easy
    "gauss_clear_large",  # K=5, easy
    "gauss_moderate_3c",  # K=3, moderate
    # Binary
    "binary_perfect_4c",  # K=4, perfect separation
    "binary_low_noise_4c",  # K=4, low noise
    "binary_moderate_6c",  # K=6, moderate
    "sparse_features_72x72",  # K=4, sparse
    # Categorical
    "cat_clear_3cat_4c",  # K=4, clear
    # SBM
    "sbm_clear_small",  # SBM
    # Overlapping
    "overlap_mod_4c_small",  # K=4, moderate overlap
    "overlap_heavy_4c_small_feat",  # K=4, heavy overlap
    # Gaussian overlap
    "gauss_overlap_3c_small",  # K=3
    # Edge cases
    "binary_2clusters",  # K=2
    "binary_many_features",  # K=4, high-d
}

all_cases = get_default_test_cases()
subset = [c for c in all_cases if c["name"] in SUBSET_NAMES]
print(f"Selected {len(subset)}/{len(all_cases)} cases:")
print(f"Spectral settings: TBS_N_JOBS={spectral_jobs}")
for c in subset:
    print(f"  {c['name']:<35s}  K={case_recipe_cluster_count(c)}")
print()

df_results, fig = benchmark_cluster_algorithm(
    test_cases=subset,
    verbose=True,
    plot_umap=True,
    concat_plots_pdf=True,
    methods=["tbs"],
)

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
        " ✓" if row["true_clusters"] > 0 and row["found_clusters"] == row["true_clusters"] else ""
    )
    print(
        f"{row['case_id']:<36s} {true_str} {row['found_clusters']:>5.0f} {ari_str:>7s} {nmi_str:>7s} {row['status']}{marker}"
    )

tbs_with_truth = tbs[tbs["true_clusters"] > 0]
exact_k = (tbs_with_truth["found_clusters"] == tbs_with_truth["true_clusters"]).sum()
print(f"\nExact K: {exact_k}/{len(tbs_with_truth)}")
ari_valid = tbs["ari"].dropna()
print(f"Mean ARI: {ari_valid.mean():.3f}" if len(ari_valid) > 0 else "Mean ARI: N/A")
print(f"Median ARI: {tbs['ari'].median():.3f}")
