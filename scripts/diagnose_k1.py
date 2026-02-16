"""Diagnose K=1 cases: trace calibration model and root-level decisions."""
import pandas as pd
import numpy as np
import logging
import sys

# Enable logging to see calibration details
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

CSV = "benchmarks/results/run_20260216_133601Z/full_benchmark_comparison.csv"
df = pd.read_csv(CSV)

# K=1 cases for KL method
kl = df[df["method"] == "kl"]
k1_cases = kl[kl["found_clusters"] == 1]

print("=" * 80)
print("K=1 CASE CLASSIFICATION")
print("=" * 80)

# Classify by failure pattern from failure report
# P=nan means sibling test couldn't compute (edge test failed/skipped)
# P>α means sibling test decided "same" (over-calibrated?)
# Root split OK but K=1 means post-hoc merge collapsed

# Group by category
categories = k1_cases.groupby("Case_Category").size().sort_values(ascending=False)
print("\nK=1 by category:")
for cat, count in categories.items():
    cases = k1_cases[k1_cases["Case_Category"] == cat]
    print(f"  {cat}: {count} cases")
    for _, row in cases.iterrows():
        print(f"    - {row['case_id']}: true_k={row['true_clusters']}, n={row['Samples']}, p={row['Features']}, noise={row['Noise']}")

# Separate into: small sample, overlap, phylogenetic, other
print("\n" + "=" * 80)
print("ROOT CAUSE CLASSIFICATION")
print("=" * 80)

small_sample = k1_cases[k1_cases["Samples"] <= 60]
phylo = k1_cases[k1_cases["Case_Category"].str.contains("phylo")]
overlap = k1_cases[k1_cases["Case_Category"].str.contains("overlap")]
cat_cases = k1_cases[k1_cases["Case_Category"].str.contains("categorical")]
other = k1_cases[
    ~k1_cases.index.isin(small_sample.index) &
    ~k1_cases.index.isin(phylo.index) &
    ~k1_cases.index.isin(overlap.index) &
    ~k1_cases.index.isin(cat_cases.index)
]

print(f"\n1. Small sample (n ≤ 60): {len(small_sample)} cases")
for _, r in small_sample.iterrows():
    n_per_k = r["Samples"] / r["true_clusters"]
    print(f"   {r['case_id']}: n={r['Samples']}, K={r['true_clusters']}, n/K={n_per_k:.0f}, p={r['Features']}")

print(f"\n2. Phylogenetic data (type mismatch): {len(phylo)} cases")
for _, r in phylo.iterrows():
    print(f"   {r['case_id']}: n={r['Samples']}, K={r['true_clusters']}, p={r['Features']}, noise={r['Noise']}")

print(f"\n3. Heavy overlap (noise ≥ 0.35): {len(overlap)} cases")
for _, r in overlap.iterrows():
    print(f"   {r['case_id']}: n={r['Samples']}, K={r['true_clusters']}, p={r['Features']}, noise={r['Noise']}")

print(f"\n4. Categorical (high cardinality): {len(cat_cases)} cases")
for _, r in cat_cases.iterrows():
    print(f"   {r['case_id']}: n={r['Samples']}, K={r['true_clusters']}, p={r['Features']}, noise={r['Noise']}")

print(f"\n5. Other: {len(other)} cases")
for _, r in other.iterrows():
    print(f"   {r['case_id']}: n={r['Samples']}, K={r['true_clusters']}, p={r['Features']}, noise={r['Noise']}")

# Now compare: which K=1 cases are NEW (not K=1 with cousin_adjusted_wald)?
# From copilot-instructions.md, the previous run had 10 K=1 cases
# We now have 18. Let's identify the 8 new ones.
print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS RESULTS (cousin_adjusted_wald had K=1 in 10 cases)")
print("=" * 80)

# The previous known K=1 cases were mostly SBM and phylogenetic.
# New K=1 cases likely include gaussian/binary cases that previously worked.
# Let's check which gaussian/binary cases now fail that shouldn't:
gaussian_k1 = k1_cases[k1_cases["Case_Category"].str.contains("gauss")]
binary_k1 = k1_cases[k1_cases["Case_Category"].str.contains("binary")]
print(f"\nGaussian K=1: {len(gaussian_k1)} cases (likely all NEW failures)")
for _, r in gaussian_k1.iterrows():
    print(f"   {r['case_id']}: true_k={r['true_clusters']}, n={r['Samples']}, p={r['Features']}")

print(f"\nBinary K=1: {len(binary_k1)} cases")
for _, r in binary_k1.iterrows():
    print(f"   {r['case_id']}: true_k={r['true_clusters']}, n={r['Samples']}, p={r['Features']}")

# Now let's actually run one case through the pipeline to trace calibration
print("\n" + "=" * 80)
print("DIAGNOSTIC: Running gauss_clear_small through pipeline with logging")
print("=" * 80)

try:
    from sklearn.datasets import make_blobs
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    from kl_clustering_analysis.tree.poset_tree import PosetTree
    from kl_clustering_analysis import config

    # Reproduce gauss_clear_small: n=30, p=20, K=3, cluster_std=0.5, seed=100
    np.random.seed(100)
    data_raw, labels = make_blobs(
        n_samples=30, n_features=20, centers=3, cluster_std=0.5, random_state=100
    )

    # Binarize
    X_binary = (data_raw > np.median(data_raw, axis=0)).astype(int)
    bin_df = pd.DataFrame(X_binary, index=[f"S{i}" for i in range(len(X_binary))], columns=[f"F{j}" for j in range(X_binary.shape[1])])

    # Build tree
    Z = linkage(pdist(bin_df.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=bin_df.index.tolist())

    print(f"  Tree nodes: {tree.number_of_nodes()}")
    print(f"  Tree edges: {tree.number_of_edges()}")
    print(f"  Config: SIBLING_TEST_METHOD={config.SIBLING_TEST_METHOD}")

    # Run decompose
    results = tree.decompose(
        leaf_data=bin_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )

    print(f"\n  Found clusters: {len(results)}")
    print(f"  Cluster sizes: {[len(v) for v in results.values()]}")

    # Check the stats_df for calibration audit
    stats_df = tree.stats_df
    if stats_df is not None and hasattr(stats_df, 'attrs'):
        audit = stats_df.attrs.get("sibling_divergence_audit", {})
        print(f"\n  Calibration audit:")
        for k, v in audit.items():
            if k != "diagnostics":
                print(f"    {k}: {v}")
            else:
                print(f"    diagnostics:")
                for dk, dv in v.items():
                    print(f"      {dk}: {dv}")

    # Show root node stats
    root = [n for n in tree.nodes if tree.in_degree(n) == 0][0]
    print(f"\n  Root node: {root}")
    if stats_df is not None and root in stats_df.index:
        root_row = stats_df.loc[root]
        for col in stats_df.columns:
            val = root_row[col]
            if pd.notna(val):
                print(f"    {col}: {val}")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
