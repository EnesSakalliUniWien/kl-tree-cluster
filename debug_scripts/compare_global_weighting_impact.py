#!/usr/bin/env python3
"""
Compare the impact of global divergence weighting on clustering results.
Analyzes differences between baseline (no weighting) and global weighting runs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load both result sets
baseline_file = "results/validation_results_baseline_2025-12-20_19-44-33.csv"
global_weighted_file = (
    "results/validation_results_global_weighting_2025-12-20_19-38-18.csv"
)

print("=" * 80)
print("GLOBAL DIVERGENCE WEIGHTING IMPACT ANALYSIS")
print("=" * 80)
print()

# Load data
df_baseline = pd.read_csv(baseline_file)
df_weighted = pd.read_csv(global_weighted_file)

print(f"Baseline results: {len(df_baseline)} rows")
print(f"Weighted results: {len(df_weighted)} rows")
print()

# Filter to only KL Divergence algorithm for meaningful comparison
df_baseline_kl = df_baseline[df_baseline["Method"] == "KL Divergence"].copy()
df_weighted_kl = df_weighted[df_weighted["Method"] == "KL Divergence"].copy()

print(f"KL Divergence test cases - Baseline: {len(df_baseline_kl)}")
print(f"KL Divergence test cases - Weighted: {len(df_weighted_kl)}")
print()

# Merge on Test number to compare
df_comparison = df_baseline_kl.merge(
    df_weighted_kl, on="Test", suffixes=("_baseline", "_weighted")
)

print("=" * 80)
print("CLUSTER COUNT COMPARISON")
print("=" * 80)

# Calculate differences
df_comparison["cluster_diff"] = (
    df_comparison["Found_weighted"] - df_comparison["Found_baseline"]
)
df_comparison["ari_diff"] = (
    df_comparison["ARI_weighted"] - df_comparison["ARI_baseline"]
)
df_comparison["nmi_diff"] = (
    df_comparison["NMI_weighted"] - df_comparison["NMI_baseline"]
)
df_comparison["purity_diff"] = (
    df_comparison["Purity_weighted"] - df_comparison["Purity_baseline"]
)

print(f"\nCluster count changes:")
print(f"  Mean difference: {df_comparison['cluster_diff'].mean():.2f}")
print(f"  Median difference: {df_comparison['cluster_diff'].median():.2f}")
print(f"  Std difference: {df_comparison['cluster_diff'].std():.2f}")
print(
    f"  Cases with fewer clusters (weighted): {(df_comparison['cluster_diff'] < 0).sum()}"
)
print(f"  Cases with same clusters: {(df_comparison['cluster_diff'] == 0).sum()}")
print(
    f"  Cases with more clusters (weighted): {(df_comparison['cluster_diff'] > 0).sum()}"
)

print("\n" + "=" * 80)
print("PERFORMANCE METRICS COMPARISON")
print("=" * 80)

print(f"\nARI (Adjusted Rand Index) changes:")
print(f"  Mean difference: {df_comparison['ari_diff'].mean():.4f}")
print(f"  Median difference: {df_comparison['ari_diff'].median():.4f}")
print(f"  Cases improved: {(df_comparison['ari_diff'] > 0.01).sum()}")
print(f"  Cases degraded: {(df_comparison['ari_diff'] < -0.01).sum()}")
print(f"  Cases unchanged (±0.01): {((df_comparison['ari_diff'].abs()) <= 0.01).sum()}")

print(f"\nNMI (Normalized Mutual Information) changes:")
print(f"  Mean difference: {df_comparison['nmi_diff'].mean():.4f}")
print(f"  Median difference: {df_comparison['nmi_diff'].median():.4f}")
print(f"  Cases improved: {(df_comparison['nmi_diff'] > 0.01).sum()}")
print(f"  Cases degraded: {(df_comparison['nmi_diff'] < -0.01).sum()}")
print(f"  Cases unchanged (±0.01): {((df_comparison['nmi_diff'].abs()) <= 0.01).sum()}")

print(f"\nPurity changes:")
print(f"  Mean difference: {df_comparison['purity_diff'].mean():.4f}")
print(f"  Median difference: {df_comparison['purity_diff'].median():.4f}")
print(f"  Cases improved: {(df_comparison['purity_diff'] > 0.01).sum()}")
print(f"  Cases degraded: {(df_comparison['purity_diff'] < -0.01).sum()}")
print(
    f"  Cases unchanged (±0.01): {((df_comparison['purity_diff'].abs()) <= 0.01).sum()}"
)

print("\n" + "=" * 80)
print("TOP 10 CASES WITH LARGEST CLUSTER REDUCTION")
print("=" * 80)
df_reduced = df_comparison.nsmallest(10, "cluster_diff", keep="all")[
    [
        "Case_Name_baseline",
        "True_baseline",
        "Found_baseline",
        "Found_weighted",
        "cluster_diff",
        "ARI_baseline",
        "ARI_weighted",
        "ari_diff",
    ]
]
df_reduced.columns = [
    "case",
    "true",
    "baseline",
    "weighted",
    "diff",
    "ari_base",
    "ari_weight",
    "ari_diff",
]
print(df_reduced.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 10 CASES WITH LARGEST CLUSTER INCREASE")
print("=" * 80)
df_increased = df_comparison.nlargest(10, "cluster_diff", keep="all")[
    [
        "Case_Name_baseline",
        "True_baseline",
        "Found_baseline",
        "Found_weighted",
        "cluster_diff",
        "ARI_baseline",
        "ARI_weighted",
        "ari_diff",
    ]
]
df_increased.columns = [
    "case",
    "true",
    "baseline",
    "weighted",
    "diff",
    "ari_base",
    "ari_weight",
    "ari_diff",
]
print(df_increased.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 10 CASES WITH LARGEST ARI IMPROVEMENT")
print("=" * 80)
df_ari_improved = df_comparison.nlargest(10, "ari_diff")[
    [
        "Case_Name_baseline",
        "True_baseline",
        "ARI_baseline",
        "ARI_weighted",
        "ari_diff",
        "Found_baseline",
        "Found_weighted",
    ]
]
df_ari_improved.columns = [
    "case",
    "true",
    "ari_base",
    "ari_weight",
    "ari_diff",
    "clust_base",
    "clust_weight",
]
print(df_ari_improved.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 10 CASES WITH LARGEST ARI DEGRADATION")
print("=" * 80)
df_ari_degraded = df_comparison.nsmallest(10, "ari_diff")[
    [
        "Case_Name_baseline",
        "True_baseline",
        "ARI_baseline",
        "ARI_weighted",
        "ari_diff",
        "Found_baseline",
        "Found_weighted",
    ]
]
df_ari_degraded.columns = [
    "case",
    "true",
    "ari_base",
    "ari_weight",
    "ari_diff",
    "clust_base",
    "clust_weight",
]
print(df_ari_degraded.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY BY DATASET TYPE")
print("=" * 80)

# Group by case name pattern (e.g., "binary_perfect", "gauss_clear", etc.)
df_comparison["case_type"] = df_comparison["Case_Name_baseline"].str.extract(
    r"([a-z_]+)_\d+[a-z]?$"
)[0]
df_comparison["case_type"] = df_comparison["case_type"].fillna(
    df_comparison["Case_Name_baseline"].str.extract(r"([a-z_]+)")[0]
)

summary_by_type = (
    df_comparison.groupby("case_type")
    .agg(
        {
            "cluster_diff": ["mean", "median", "count"],
            "ari_diff": ["mean", "median"],
            "nmi_diff": ["mean", "median"],
        }
    )
    .round(4)
)

print(summary_by_type.to_string())

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

# Overall statistics
print(f"\nTotal test cases analyzed: {len(df_comparison)}")
print(f"\nGlobal weighting effect on clustering:")
print(
    f"  Reduces over-segmentation: {(df_comparison['cluster_diff'] < -1).sum()} cases"
)
print(
    f"  Minimal change (±1 cluster): {(df_comparison['cluster_diff'].abs() <= 1).sum()} cases"
)
print(f"  Increases segmentation: {(df_comparison['cluster_diff'] > 1).sum()} cases")

print(f"\nGlobal weighting effect on accuracy:")
print(f"  Improves ARI (>0.01): {(df_comparison['ari_diff'] > 0.01).sum()} cases")
print(f"  Neutral ARI (±0.01): {(df_comparison['ari_diff'].abs() <= 0.01).sum()} cases")
print(f"  Degrades ARI (<-0.01): {(df_comparison['ari_diff'] < -0.01).sum()} cases")

# Correlation analysis
print(f"\nCorrelation between cluster reduction and ARI improvement:")
print(
    f"  Pearson r: {df_comparison['cluster_diff'].corr(df_comparison['ari_diff']):.4f}"
)

print("\n" + "=" * 80)
print("VISUALIZATION")
print("=" * 80)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cluster count comparison
ax = axes[0, 0]
ax.scatter(
    df_comparison["Found_baseline"], df_comparison["Found_weighted"], alpha=0.5, s=30
)
max_val = max(
    df_comparison["Found_baseline"].max(), df_comparison["Found_weighted"].max()
)
ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")
ax.set_xlabel("Baseline Clusters")
ax.set_ylabel("Weighted Clusters")
ax.set_title("Cluster Count: Baseline vs Weighted")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: ARI comparison
ax = axes[0, 1]
ax.scatter(
    df_comparison["ARI_baseline"], df_comparison["ARI_weighted"], alpha=0.5, s=30
)
ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
ax.set_xlabel("Baseline ARI")
ax.set_ylabel("Weighted ARI")
ax.set_title("ARI: Baseline vs Weighted")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Distribution of cluster differences
ax = axes[1, 0]
ax.hist(df_comparison["cluster_diff"], bins=30, edgecolor="black", alpha=0.7)
ax.axvline(0, color="r", linestyle="--", linewidth=2, label="No change")
ax.axvline(
    df_comparison["cluster_diff"].mean(),
    color="g",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {df_comparison['cluster_diff'].mean():.2f}",
)
ax.set_xlabel("Cluster Count Difference (Weighted - Baseline)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Cluster Count Changes")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Distribution of ARI differences
ax = axes[1, 1]
ax.hist(df_comparison["ari_diff"], bins=30, edgecolor="black", alpha=0.7)
ax.axvline(0, color="r", linestyle="--", linewidth=2, label="No change")
ax.axvline(
    df_comparison["ari_diff"].mean(),
    color="g",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {df_comparison['ari_diff'].mean():.4f}",
)
ax.set_xlabel("ARI Difference (Weighted - Baseline)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of ARI Changes")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/global_weighting_comparison.png", dpi=150, bbox_inches="tight")
print("\n✅ Visualization saved to: results/global_weighting_comparison.png")

# Create detailed CSV with comparison
df_comparison_export = df_comparison[
    [
        "Case_Name_baseline",
        "True_baseline",
        "Found_baseline",
        "Found_weighted",
        "cluster_diff",
        "ARI_baseline",
        "ARI_weighted",
        "ari_diff",
        "NMI_baseline",
        "NMI_weighted",
        "nmi_diff",
        "Purity_baseline",
        "Purity_weighted",
        "purity_diff",
    ]
].copy()

df_comparison_export.columns = [
    "case",
    "true_clusters",
    "baseline_clusters",
    "weighted_clusters",
    "cluster_diff",
    "baseline_ari",
    "weighted_ari",
    "ari_diff",
    "baseline_nmi",
    "weighted_nmi",
    "nmi_diff",
    "baseline_purity",
    "weighted_purity",
    "purity_diff",
]

comparison_file = "results/global_weighting_detailed_comparison.csv"
df_comparison_export.to_csv(comparison_file, index=False)
print(f"✅ Detailed comparison saved to: {comparison_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
