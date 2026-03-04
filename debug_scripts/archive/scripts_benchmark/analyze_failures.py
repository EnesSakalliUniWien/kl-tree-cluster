"""Deep analysis of benchmark failure modes."""

import pandas as pd
import numpy as np

df = pd.read_csv("benchmarks/results/full_benchmark_alpha_0p001_now.csv")

# Focus on kl method
kl = df[df["method"] == "kl"].copy()
kl["k_ratio"] = kl["found_clusters"] / kl["true_clusters"]


def categorize(row):
    if row["found_clusters"] == 1:
        return "UNDER_K1"
    if row["k_ratio"] > 3:
        return "OVER_SPLIT_SEVERE"
    if row["k_ratio"] > 1.5:
        return "OVER_SPLIT_MILD"
    if row["k_ratio"] < 0.5:
        return "UNDER_SPLIT"
    if row["ari"] >= 0.9:
        return "GOOD"
    if row["ari"] >= 0.5:
        return "MODERATE"
    return "POOR"


kl["category"] = kl.apply(categorize, axis=1)

print("=== FAILURE CATEGORY DISTRIBUTION (kl method) ===")
print(kl["category"].value_counts().to_string())
print()

for cat in [
    "UNDER_K1",
    "OVER_SPLIT_SEVERE",
    "OVER_SPLIT_MILD",
    "UNDER_SPLIT",
    "POOR",
    "MODERATE",
    "GOOD",
]:
    sub = kl[kl["category"] == cat]
    if len(sub) == 0:
        continue
    print(f"\n=== {cat} ({len(sub)} cases) ===")
    for _, r in sub.iterrows():
        print(
            f"  {r['test_case']:40s} ARI={r['ari']:7.4f}  K={r['found_clusters']:3d}/{r['true_clusters']:3d}  n={r['Samples']:4d} p={r['Features']:5d} noise={r['Noise']}"
        )

print("\n\n=== CROSS-METHOD COMPARISON (where they differ) ===")
kl_pivot = df.pivot_table(
    index="test_case",
    columns="method",
    values=["ari", "found_clusters", "true_clusters"],
    aggfunc="first",
)
kl_pivot["ari_diff"] = kl_pivot[("ari", "kl")] - kl_pivot[("ari", "kl_rogerstanimoto")]
kl_pivot["abs_diff"] = kl_pivot["ari_diff"].abs()
big_diff = kl_pivot[kl_pivot["abs_diff"] > 0.1].sort_values("abs_diff", ascending=False)
print(f"\nCases with ARI diff > 0.1 ({len(big_diff)} cases):")
for tc in big_diff.index:
    row = kl_pivot.loc[tc]
    print(
        f"  {tc:40s}  kl={row[('ari', 'kl')]:7.4f}  rt={row[('ari', 'kl_rogerstanimoto')]:7.4f}  diff={row['ari_diff']:+7.4f}"
    )

print("\n\n=== SAMPLE SIZE vs PERFORMANCE ===")
for bucket in [(0, 50), (50, 100), (100, 300), (300, 1000)]:
    sub = kl[(kl["Samples"] >= bucket[0]) & (kl["Samples"] < bucket[1])]
    if len(sub) > 0:
        print(
            f"  n={bucket[0]:4d}-{bucket[1]:4d}: {len(sub):2d} cases, mean ARI={sub['ari'].mean():.3f}, median={sub['ari'].median():.3f}"
        )

print("\n\n=== FEATURE DIM vs PERFORMANCE ===")
for bucket in [(0, 50), (50, 200), (200, 1000), (1000, 10000)]:
    sub = kl[(kl["Features"] >= bucket[0]) & (kl["Features"] < bucket[1])]
    if len(sub) > 0:
        print(
            f"  p={bucket[0]:5d}-{bucket[1]:5d}: {len(sub):2d} cases, mean ARI={sub['ari'].mean():.3f}, median={sub['ari'].median():.3f}"
        )

print("\n\n=== NOISE LEVEL vs PERFORMANCE ===")
for noise_val in sorted(kl["Noise"].unique()):
    sub = kl[kl["Noise"] == noise_val]
    if len(sub) > 0:
        print(
            f"  noise={noise_val}: {len(sub):2d} cases, mean ARI={sub['ari'].mean():.3f}, median={sub['ari'].median():.3f}, K=1: {(sub['found_clusters'] == 1).sum()}"
        )

print("\n\n=== SAMPLES PER CLUSTER (n/K) vs PERFORMANCE ===")
kl["samples_per_k"] = kl["Samples"] / kl["true_clusters"]
for bucket in [(0, 10), (10, 20), (20, 50), (50, 200)]:
    sub = kl[(kl["samples_per_k"] >= bucket[0]) & (kl["samples_per_k"] < bucket[1])]
    if len(sub) > 0:
        print(
            f"  n/K={bucket[0]:3d}-{bucket[1]:3d}: {len(sub):2d} cases, mean ARI={sub['ari'].mean():.3f}, K=1: {(sub['found_clusters'] == 1).sum()}, over-split(>3x): {(sub['k_ratio'] > 3).sum()}"
        )
