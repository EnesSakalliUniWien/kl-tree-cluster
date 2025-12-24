#!/usr/bin/env python3
"""
Run two benchmark comparisons with global weighting OFF and ON.
This script properly sets the config before importing benchmark functions.
"""

import sys
from datetime import datetime

# FIRST RUN: Global weighting OFF
print("=" * 80)
print("RUN 1: GLOBAL WEIGHTING DISABLED")
print("=" * 80)
print()

# Import and modify config BEFORE importing benchmark
import kl_clustering_analysis.config as config

config.USE_GLOBAL_DIVERGENCE_WEIGHTING = False

print(
    f"Configuration set: USE_GLOBAL_DIVERGENCE_WEIGHTING = {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}"
)
print()

# Now import benchmark (it will see the modified config)
from kl_clustering_analysis.benchmarking import benchmark_cluster_algorithm

print("Running benchmark with global weighting DISABLED...")
print()

df_results_off, fig_off = benchmark_cluster_algorithm(
    significance_level=0.01,
    verbose=True,
    plot_umap=True,
    plot_manifold=True,
)

# Save results
timestamp_off = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file_off = f"results/benchmark_global_OFF_{timestamp_off}.csv"
df_results_off.to_csv(results_file_off, index=False)

print()
print("=" * 80)
print(f"✅ Run 1 complete: {results_file_off}")
print("=" * 80)
print()
print()

# SECOND RUN: Global weighting ON
print("=" * 80)
print("RUN 2: GLOBAL WEIGHTING ENABLED")
print("=" * 80)
print()

# Modify config for second run
config.USE_GLOBAL_DIVERGENCE_WEIGHTING = True
print(
    f"Configuration set: USE_GLOBAL_DIVERGENCE_WEIGHTING = {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}"
)
print()

print("Running benchmark with global weighting ENABLED...")
print()

df_results_on, fig_on = benchmark_cluster_algorithm(
    significance_level=0.01,
    verbose=True,
    plot_umap=True,
    plot_manifold=True,
)

# Save results
timestamp_on = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file_on = f"results/benchmark_global_ON_{timestamp_on}.csv"
df_results_on.to_csv(results_file_on, index=False)

print()
print("=" * 80)
print(f"✅ Run 2 complete: {results_file_on}")
print("=" * 80)
print()
print()

# COMPARISON
print("=" * 80)
print("QUICK COMPARISON")
print("=" * 80)
print()

# Filter to KL Divergence only
df_off_kl = df_results_off[df_results_off["Method"] == "KL Divergence"].copy()
df_on_kl = df_results_on[df_results_on["Method"] == "KL Divergence"].copy()

# Merge for comparison
df_compare = df_off_kl.merge(df_on_kl, on="Test", suffixes=("_off", "_on"))

# Calculate differences
df_compare["cluster_diff"] = df_compare["Found_on"] - df_compare["Found_off"]
df_compare["ari_diff"] = df_compare["ARI_on"] - df_compare["ARI_off"]
df_compare["nmi_diff"] = df_compare["NMI_on"] - df_compare["NMI_off"]

print(f"Total KL Divergence test cases: {len(df_compare)}")
print()

print("Cluster Count Changes (ON - OFF):")
print(f"  Mean: {df_compare['cluster_diff'].mean():.2f}")
print(f"  Median: {df_compare['cluster_diff'].median():.2f}")
print(f"  Min: {df_compare['cluster_diff'].min()}")
print(f"  Max: {df_compare['cluster_diff'].max()}")
print(f"  Fewer clusters (ON): {(df_compare['cluster_diff'] < 0).sum()}")
print(f"  Same clusters: {(df_compare['cluster_diff'] == 0).sum()}")
print(f"  More clusters (ON): {(df_compare['cluster_diff'] > 0).sum()}")
print()

print("ARI Changes (ON - OFF):")
print(f"  Mean: {df_compare['ari_diff'].mean():.4f}")
print(f"  Median: {df_compare['ari_diff'].median():.4f}")
print(f"  Improved (>0.01): {(df_compare['ari_diff'] > 0.01).sum()}")
print(f"  Unchanged (±0.01): {(df_compare['ari_diff'].abs() <= 0.01).sum()}")
print(f"  Degraded (<-0.01): {(df_compare['ari_diff'] < -0.01).sum()}")
print()

print("NMI Changes (ON - OFF):")
print(f"  Mean: {df_compare['nmi_diff'].mean():.4f}")
print(f"  Median: {df_compare['nmi_diff'].median():.4f}")
print()

# Show top cases with largest changes
print("=" * 80)
print("TOP 5 CASES WITH LARGEST CLUSTER REDUCTION (ON < OFF)")
print("=" * 80)
top_reduced = df_compare.nsmallest(5, "cluster_diff")[
    [
        "Case_Name_off",
        "True_off",
        "Found_off",
        "Found_on",
        "cluster_diff",
        "ARI_off",
        "ARI_on",
        "ari_diff",
    ]
]
top_reduced.columns = [
    "case",
    "true",
    "off",
    "on",
    "diff",
    "ari_off",
    "ari_on",
    "ari_diff",
]
print(top_reduced.to_string(index=False))
print()

print("=" * 80)
print("TOP 5 CASES WITH LARGEST ARI IMPROVEMENT (ON > OFF)")
print("=" * 80)
top_improved = df_compare.nlargest(5, "ari_diff")[
    [
        "Case_Name_off",
        "True_off",
        "Found_off",
        "Found_on",
        "ARI_off",
        "ARI_on",
        "ari_diff",
    ]
]
top_improved.columns = ["case", "true", "off", "on", "ari_off", "ari_on", "ari_diff"]
print(top_improved.to_string(index=False))
print()

# Save detailed comparison
comparison_file = f"results/comparison_OFF_vs_ON_{timestamp_on}.csv"
df_compare[
    [
        "Case_Name_off",
        "True_off",
        "Found_off",
        "Found_on",
        "cluster_diff",
        "ARI_off",
        "ARI_on",
        "ari_diff",
        "NMI_off",
        "NMI_on",
        "nmi_diff",
        "Purity_off",
        "Purity_on",
    ]
].to_csv(comparison_file, index=False)

print("=" * 80)
print("FILES SAVED")
print("=" * 80)
print(f"  OFF: {results_file_off}")
print(f"  ON:  {results_file_on}")
print(f"  Comparison: {comparison_file}")
print("=" * 80)
