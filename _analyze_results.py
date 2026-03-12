import pandas as pd

df = pd.read_csv("benchmarks/results/run_20260304_180235Z/full_benchmark_comparison.csv")

kl = df[df["method"] == "kl"].copy()
kl_k = (kl["found_clusters"] == kl["true_clusters"]).sum()
print(f"KL: Exact K match = {kl_k}/{len(kl)} ({100*kl_k/len(kl):.1f}%)")
print(f'KL: K=1 cases = {(kl["found_clusters"]==1).sum()}')
print(f'KL: Mean ARI = {kl["ari"].mean():.4f}')
print(f'KL: Median ARI = {kl["ari"].median():.4f}')
print()

low = kl[kl["ari"] < 0.5][["test_case", "true_clusters", "found_clusters", "ari"]].sort_values(
    "ari"
)
print("=== KL Cases with ARI < 0.5 ===")
print(low.to_string(index=False))
print()

merged = df.pivot_table(index="test_case", columns="method", values="ari")
kl_wins = (merged["kl"] > merged["kmeans"]).sum()
km_wins = (merged["kl"] < merged["kmeans"]).sum()
ties = (merged["kl"] == merged["kmeans"]).sum()
print(f"KL vs K-Means: KL wins {kl_wins}, K-Means wins {km_wins}, ties {ties}")

kl_vs_leiden = (merged["kl"] > merged["leiden"]).sum()
lei_wins = (merged["kl"] < merged["leiden"]).sum()
print(f"KL vs Leiden: KL wins {kl_vs_leiden}, Leiden wins {lei_wins}")
print()

# By category
print("=== KL Mean ARI by Category ===")
cat_ari = (
    kl.groupby("Case_Category")["ari"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)
print(cat_ari.to_string())
print()

# Cases where KL beats everyone
print("=== Cases where KL has highest ARI ===")
best_method = merged.idxmax(axis=1)
kl_best = (best_method == "kl").sum()
print(f"KL is best method in {kl_best}/{len(merged)} cases")
