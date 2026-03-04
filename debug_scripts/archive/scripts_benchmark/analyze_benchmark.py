#!/usr/bin/env python
"""Analyze the benchmark results."""

import pandas as pd

df = pd.read_csv("benchmarks/results/full_benchmark_alpha_0p001_now.csv")

# Remove duplicates (keep last)
df = df.drop_duplicates(["case_id", "method"], keep="last")

print("=" * 60)
print("BENCHMARK ANALYSIS")
print("=" * 60)
print()

# Summary by method
print("=== Mean ARI by Method ===")
summary = df.groupby("method")["ari"].agg(["mean", "std", "count"])
print(summary)
print()

# Pivot to compare
pivot = df.pivot(index="case_id", columns="method", values="ari")
pivot["diff"] = pivot["kl"] - pivot["kl_rogerstanimoto"]

print("=== Detailed Comparison ===")
print()
print("Cases where KL (Hamming) WINS (diff > 0.05):")
wins = pivot[pivot["diff"] > 0.05].sort_values("diff", ascending=False)
print(f"  Count: {len(wins)}")
print(wins[["kl", "kl_rogerstanimoto", "diff"]].head(15).to_string())
print()

print("Cases where Rogers-Tanimoto WINS (diff < -0.05):")
losses = pivot[pivot["diff"] < -0.05].sort_values("diff")
print(f"  Count: {len(losses)}")
print(losses[["kl", "kl_rogerstanimoto", "diff"]].head(15).to_string())
print()

print("Cases roughly equal (|diff| <= 0.05):")
equal = pivot[pivot["diff"].abs() <= 0.05]
print(f"  Count: {len(equal)}")
print()

# Perfect cases (ARI >= 0.99)
print("=== Cases with perfect clustering (ARI >= 0.99) ===")
perfect = df[df["ari"] >= 0.99].groupby("method").size()
print(perfect)
print()

# Poor cases (ARI < 0.5)
print("=== Cases with poor clustering (ARI < 0.5) ===")
poor = df[df["ari"] < 0.5].groupby("method").size()
print(poor)
print()

# Breakdown by case type
print("=== Mean ARI by Case Type ===")


# Extract case type from case_id
def get_case_type(case_id):
    if case_id.startswith("case_"):
        return "legacy"
    elif "gauss" in case_id:
        return "gaussian"
    elif "binary" in case_id:
        return "binary"
    elif "sparse" in case_id:
        return "sparse"
    elif "sbm" in case_id:
        return "sbm"
    elif "cat" in case_id:
        return "categorical"
    elif "phylo" in case_id:
        return "phylogenetic"
    elif "temporal" in case_id:
        return "temporal"
    elif "overlap" in case_id:
        return "overlap"
    else:
        return "other"


df["case_type"] = df["case_id"].apply(get_case_type)
type_summary = df.pivot_table(
    index="case_type", columns="method", values="ari", aggfunc="mean"
)
type_summary["diff"] = type_summary["kl"] - type_summary["kl_rogerstanimoto"]
print(type_summary.round(3).to_string())
