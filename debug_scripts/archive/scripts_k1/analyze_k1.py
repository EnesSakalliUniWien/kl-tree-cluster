"""Analyze benchmark results with focus on K=1 cases."""
import pandas as pd
import sys

CSV = "benchmarks/results/run_20260216_133601Z/full_benchmark_comparison.csv"
df = pd.read_csv(CSV)

print("=" * 80)
print("BENCHMARK SUMMARY (cousin_weighted_wald)")
print("=" * 80)

# Overall metrics per method
for method in ["kl", "kl_rogerstanimoto"]:
    m = df[df["method"] == method]
    n = len(m)
    mean_ari = m["ari"].mean()
    median_ari = m["ari"].median()
    exact_k = (m["found_clusters"] == m["true_clusters"]).sum()
    k1 = (m["found_clusters"] == 1).sum()
    over_split = (m["found_clusters"] > m["true_clusters"]).sum()
    under_split = (m["found_clusters"] < m["true_clusters"]).sum()
    print(f"\n{method} ({n} cases):")
    print(f"  Mean ARI:     {mean_ari:.3f}")
    print(f"  Median ARI:   {median_ari:.3f}")
    print(f"  Exact K:      {exact_k}/{n}")
    print(f"  K=1 (merge):  {k1}")
    print(f"  Over-split:   {over_split}")
    print(f"  Under-split:  {under_split}")

# Compare with previous run (cousin_adjusted_wald from copilot-instructions)
print("\n" + "=" * 80)
print("COMPARISON: Previous (cousin_adjusted_wald) vs Current (cousin_weighted_wald)")
print("=" * 80)
print("                    | Previous (adj_wald) | Current (w_wald)")
print("-" * 60)
kl = df[df["method"] == "kl"]
kl_k1 = (kl["found_clusters"] == 1).sum()
kl_exact = (kl["found_clusters"] == kl["true_clusters"]).sum()
print(f"  KL Mean ARI       |       0.757         |    {kl['ari'].mean():.3f}")
print(f"  KL Median ARI     |       1.000         |    {kl['ari'].median():.3f}")
print(f"  KL Exact K        |       59/95         |    {kl_exact}/{len(kl)}")
print(f"  KL K=1            |       10            |    {kl_k1}")

# K=1 cases detail
print("\n" + "=" * 80)
print("K=1 CASES (KL method) — All cases where found_clusters=1")
print("=" * 80)
kl_k1_cases = kl[kl["found_clusters"] == 1].copy()
kl_k1_cases = kl_k1_cases.sort_values("test_case")
cols = ["test_case", "case_id", "Case_Category", "true_clusters", "found_clusters", "Samples", "Features", "Noise", "ari"]
print(kl_k1_cases[cols].to_string(index=False))

# For each K=1 case, check what other methods found
print("\n" + "=" * 80)
print("K=1 CASES — Cross-method comparison")
print("=" * 80)
k1_case_ids = kl_k1_cases["test_case"].tolist()
for case_id in k1_case_ids:
    case = df[df["test_case"] == case_id]
    case_name = case["case_id"].iloc[0]
    true_k = case["true_clusters"].iloc[0]
    print(f"\nCase {case_id}: {case_name} (true K={true_k}, n={case['Samples'].iloc[0]}, p={case['Features'].iloc[0]}, noise={case['Noise'].iloc[0]})")
    for _, row in case.iterrows():
        marker = " ***" if row["found_clusters"] == 1 and row["method"].startswith("kl") else ""
        print(f"  {row['method']:25s}  K={row['found_clusters']:3.0f}  ARI={row['ari']:.4f}{marker}")

# Category breakdown
print("\n" + "=" * 80)
print("KL ARI BY CATEGORY")
print("=" * 80)
cat_stats = kl.groupby("Case_Category").agg(
    n=("ari", "count"),
    mean_ari=("ari", "mean"),
    exact_k=("found_clusters", lambda x: (x == kl.loc[x.index, "true_clusters"]).sum()),
    k1=("found_clusters", lambda x: (x == 1).sum()),
).sort_values("mean_ari")
print(cat_stats.to_string())

# ARI=0 cases (complete failures)
print("\n" + "=" * 80)
print("COMPLETE FAILURES (ARI ≤ 0.01) — KL method")
print("=" * 80)
failures = kl[kl["ari"] <= 0.01].sort_values("ari")
print(failures[cols].to_string(index=False))
print(f"\nTotal failures: {len(failures)}/{len(kl)}")

# Over-splitting cases
print("\n" + "=" * 80)
print("OVER-SPLITTING CASES (found_k > 2*true_k) — KL method")
print("=" * 80)
over = kl[kl["found_clusters"] > 2 * kl["true_clusters"]].sort_values("found_clusters", ascending=False)
if len(over) > 0:
    print(over[cols].to_string(index=False))
else:
    print("None")
