#!/usr/bin/env python3
"""
Quick subset benchmark — picks ~15 representative cases across categories
and runs them with plots.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm

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
    # Real data
    "feature_matrix_go_terms",  # real GO-term binary matrix
}

all_cases = get_default_test_cases()
subset = [c for c in all_cases if c["name"] in SUBSET_NAMES]
print(f"Selected {len(subset)}/{len(all_cases)} cases:")
for c in subset:
    print(f"  {c['name']:<35s}  K={c.get('n_clusters', '?')}")
print()

df_results, fig = benchmark_cluster_algorithm(
    test_cases=subset,
    verbose=True,
    plot_umap=True,
    concat_plots_pdf=True,
    methods=["kl"],
)

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
kl = df_results[df_results["Method"] == "KL Divergence"].copy()
kl = kl.sort_values("Case_Name")

print(f"\n{'Case':<36s} {'True':>4s} {'Found':>5s} {'ARI':>7s} {'NMI':>7s} {'Status'}")
print("-" * 70)
for _, row in kl.iterrows():
    ari_val = row['ARI']
    nmi_val = row['NMI']
    ari_str = f"{ari_val:.3f}" if not (ari_val != ari_val) else "  N/A"
    nmi_str = f"{nmi_val:.3f}" if not (nmi_val != nmi_val) else "  N/A"
    true_str = f"{row['True']:>4.0f}" if row['True'] > 0 else " N/A"
    marker = " ✓" if row['True'] > 0 and row["Found"] == row["True"] else ""
    print(
        f"{row['Case_Name']:<36s} {true_str} {row['Found']:>5.0f} {ari_str:>7s} {nmi_str:>7s} {row['Status']}{marker}"
    )

kl_with_truth = kl[kl["True"] > 0]
exact_k = (kl_with_truth["Found"] == kl_with_truth["True"]).sum()
print(f"\nExact K: {exact_k}/{len(kl_with_truth)}")
ari_valid = kl["ARI"].dropna()
print(f"Mean ARI: {ari_valid.mean():.3f}" if len(ari_valid) > 0 else "Mean ARI: N/A")
print(f"Median ARI: {kl['ARI'].median():.3f}")
