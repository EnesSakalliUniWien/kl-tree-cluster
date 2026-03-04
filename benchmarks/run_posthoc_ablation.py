#!/usr/bin/env python3
"""
Ablation: compare POSTHOC_MERGE = True vs False on the subset benchmark.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("KL_TE_N_JOBS", "1")

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from kl_clustering_analysis import config

SUBSET_NAMES = {
    "gauss_clear_small",
    "gauss_clear_large",
    "gauss_moderate_3c",
    "binary_perfect_4c",
    "binary_low_noise_4c",
    "binary_moderate_6c",
    "sparse_features_72x72",
    "cat_clear_3cat_4c",
    "sbm_clear_small",
    "overlap_mod_4c_small",
    "overlap_heavy_4c_small_feat",
    "gauss_overlap_3c_small",
    "binary_2clusters",
    "binary_many_features",
    "feature_matrix_go_terms",
}

all_cases = get_default_test_cases()
subset = [c for c in all_cases if c["name"] in SUBSET_NAMES]
print(f"Selected {len(subset)} cases\n")

# --- Run 1: POSTHOC_MERGE = True ---
print("=" * 70)
print("RUN 1: POSTHOC_MERGE = True")
print("=" * 70)
config.POSTHOC_MERGE = True
df_on, _ = benchmark_cluster_algorithm(
    test_cases=subset,
    verbose=True,
    plot_umap=False,
    methods=["kl"],
)

# --- Run 2: POSTHOC_MERGE = False ---
print("\n" + "=" * 70)
print("RUN 2: POSTHOC_MERGE = False")
print("=" * 70)
config.POSTHOC_MERGE = False
df_off, _ = benchmark_cluster_algorithm(
    test_cases=subset,
    verbose=True,
    plot_umap=False,
    methods=["kl"],
)

# --- Compare ---
print("\n" + "=" * 70)
print("COMPARISON: POSTHOC_MERGE ON vs OFF")
print("=" * 70)

kl_on = df_on[df_on["Method"] == "KL Divergence"].set_index("Case_Name").sort_index()
kl_off = df_off[df_off["Method"] == "KL Divergence"].set_index("Case_Name").sort_index()

header = f"{'Case':<36s} {'True':>4s} {'K_ON':>5s} {'K_OFF':>5s} {'ARI_ON':>7s} {'ARI_OFF':>7s} {'Diff':>6s}"
print(f"\n{header}")
print("-" * 75)

diffs = 0
for name in kl_on.index:
    if name not in kl_off.index:
        continue
    r_on, r_off = kl_on.loc[name], kl_off.loc[name]
    ari_on = r_on["ARI"] if r_on["ARI"] == r_on["ARI"] else float("nan")
    ari_off = r_off["ARI"] if r_off["ARI"] == r_off["ARI"] else float("nan")
    k_on, k_off = int(r_on["Found"]), int(r_off["Found"])
    true_k = int(r_on["True"]) if r_on["True"] > 0 else 0
    diff = ari_off - ari_on if ari_on == ari_on and ari_off == ari_off else float("nan")
    marker = " <--" if k_on != k_off else ""
    if k_on != k_off:
        diffs += 1
    true_str = f"{true_k:>4d}" if true_k > 0 else " N/A"
    ari_on_s = f"{ari_on:.3f}" if ari_on == ari_on else "  N/A"
    ari_off_s = f"{ari_off:.3f}" if ari_off == ari_off else "  N/A"
    diff_s = f"{diff:+.3f}" if diff == diff else "  N/A"
    print(
        f"{name:<36s} {true_str} {k_on:>5d} {k_off:>5d} {ari_on_s:>7s} {ari_off_s:>7s} {diff_s:>6s}{marker}"
    )

ari_on_valid = kl_on["ARI"].dropna()
ari_off_valid = kl_off["ARI"].dropna()
print(f"\nMean ARI  ON: {ari_on_valid.mean():.3f}   OFF: {ari_off_valid.mean():.3f}")
print(f"Cases with different K: {diffs}")
print(f"Conclusion: {'IDENTICAL' if diffs == 0 else 'DIFFERENT'} results")

# Restore default
config.POSTHOC_MERGE = True
