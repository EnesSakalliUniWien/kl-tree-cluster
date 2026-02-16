"""
Purpose: Analyze sparse-feature behavior for benchmark case 24.
Inputs: Benchmark case 24 data and in-script analysis settings.
Outputs: Console diagnostics for case 24.
Expected runtime: ~10-60 seconds.
How to run: python debug_scripts/case_studies/q_case24_sparse_feature_analysis__diagnostic__case24.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_cases_config import DEFAULT_TEST_CASES_CONFIG
from benchmarks.shared.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)

# Get Case 24 config
sparse_cases = DEFAULT_TEST_CASES_CONFIG.get("binary_sparse_features", [])
case_config = None
for c in sparse_cases:
    if c.get("name") == "sparse_features_moderate":
        case_config = c
        break

if case_config is None:
    print("Case not found!")
    exit(1)

print("=" * 70)
print("CASE 24: sparse_features_moderate")
print("=" * 70)
print(f"Config: {case_config}")

# Generate the data
leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
    n_rows=case_config["n_rows"],
    n_cols=case_config["n_cols"],
    n_clusters=case_config["n_clusters"],
    entropy_param=case_config["entropy_param"],
    feature_sparsity=case_config.get("feature_sparsity"),
    balanced_clusters=case_config.get("balanced_clusters", True),
    random_seed=case_config["seed"],
)

# Convert to numpy arrays
sample_names = sorted(leaf_matrix_dict.keys(), key=lambda x: int(x[1:]))
X = np.array([leaf_matrix_dict[name] for name in sample_names])
y = np.array([cluster_assignments[name] for name in sample_names])

print(f"\nGenerated data shape: {X.shape}")
print(f"Cluster distribution: {np.bincount(y)}")

# Analyze the data
print("\n" + "=" * 70)
print("DATA ANALYSIS")
print("=" * 70)

# 1. Feature sparsity analysis
feature_means = X.mean(axis=0)
print(f"\nFeature means (θ values):")
print(f"  Min: {feature_means.min():.4f}")
print(f"  Max: {feature_means.max():.4f}")
print(f"  Mean: {feature_means.mean():.4f}")
print(f"  Median: {np.median(feature_means):.4f}")

# Count sparse features (θ near 0 or 1)
sparse_low = np.sum(feature_means < 0.1)
sparse_high = np.sum(feature_means > 0.9)
moderate = np.sum((feature_means >= 0.1) & (feature_means <= 0.9))
print(f"\nFeature sparsity breakdown:")
print(f"  θ < 0.1: {sparse_low} features")
print(f"  θ > 0.9: {sparse_high} features")
print(f"  0.1 ≤ θ ≤ 0.9: {moderate} features")

# 2. Variance analysis
feature_variances = X.var(axis=0)
bernoulli_weights = 4 * feature_means * (1 - feature_means)
effective_df = np.sum(bernoulli_weights)
print(
    f"\nVariance-weighted effective df: {effective_df:.2f} (out of {X.shape[1]} features)"
)

# 3. Per-cluster analysis
print("\n" + "=" * 70)
print("PER-CLUSTER ANALYSIS")
print("=" * 70)

for cluster_id in range(case_config["n_clusters"]):
    mask = y == cluster_id
    X_cluster = X[mask]
    cluster_means = X_cluster.mean(axis=0)
    print(f"\nCluster {cluster_id}: n={mask.sum()}")
    print(
        f"  Feature means - min: {cluster_means.min():.4f}, max: {cluster_means.max():.4f}"
    )

    # How different from global mean?
    diff_from_global = np.abs(cluster_means - feature_means)
    print(f"  Max diff from global: {diff_from_global.max():.4f}")
    print(f"  Mean diff from global: {diff_from_global.mean():.4f}")

    # Features with large differences
    sig_diff = np.sum(diff_from_global > 0.1)
    print(f"  Features with |Δθ| > 0.1: {sig_diff}")

# 4. Pairwise cluster comparison
print("\n" + "=" * 70)
print("PAIRWISE CLUSTER DIVERGENCE")
print("=" * 70)

from scipy.stats import chi2

for i in range(case_config["n_clusters"]):
    for j in range(i + 1, case_config["n_clusters"]):
        X_i = X[y == i]
        X_j = X[y == j]

        theta_i = X_i.mean(axis=0)
        theta_j = X_j.mean(axis=0)

        # Compute chi-square test statistic
        n_i, n_j = len(X_i), len(X_j)
        n_pooled = n_i * n_j / (n_i + n_j)

        # Pooled theta for variance
        theta_pooled = (X_i.sum(axis=0) + X_j.sum(axis=0)) / (n_i + n_j)

        # Avoid division by zero
        var_pooled = theta_pooled * (1 - theta_pooled)
        var_pooled = np.maximum(var_pooled, 1e-10)

        # Chi-square statistic
        diff = theta_i - theta_j
        chi_sq = n_pooled * np.sum(diff**2 / var_pooled)

        # Effective df (variance-weighted)
        weights = 4 * theta_pooled * (1 - theta_pooled)
        df_eff = np.sum(weights)

        # P-value
        p_val = chi2.sf(chi_sq, df=df_eff)
        ratio = chi_sq / df_eff

        print(f"\nCluster {i} vs {j}:")
        print(f"  Chi-sq: {chi_sq:.2f}, df_eff: {df_eff:.2f}, ratio: {ratio:.3f}")
        print(f"  P-value: {p_val:.6f}")
        print(f"  Significant at 0.05? {'YES' if p_val < 0.05 else 'NO'}")

# 5. Test with projection
print("\n" + "=" * 70)
print("RANDOM PROJECTION TEST")
print("=" * 70)

from scipy.linalg import qr

k = max(10, int(4 * np.log(X.shape[0])))
print(f"Projection dimension k = {k}")

for i in range(case_config["n_clusters"]):
    for j in range(i + 1, case_config["n_clusters"]):
        X_i = X[y == i]
        X_j = X[y == j]

        # Standardize
        X_all = np.vstack([X_i, X_j])
        mean = X_all.mean(axis=0)
        std = X_all.std(axis=0)
        std[std < 1e-10] = 1.0

        X_i_std = (X_i - mean) / std
        X_j_std = (X_j - mean) / std

        # Random projection with orthonormal rows
        np.random.seed(42)
        G = np.random.standard_normal((k, X.shape[1]))
        Q, _ = qr(G.T, mode="economic")
        R = Q.T  # k x d orthonormal rows

        # Project centroids
        z_i = R @ X_i_std.mean(axis=0)
        z_j = R @ X_j_std.mean(axis=0)

        # Test statistic (squared distance scaled by n)
        n_i, n_j = len(X_i), len(X_j)
        n_eff = n_i * n_j / (n_i + n_j)
        diff = z_i - z_j
        T = n_eff * np.sum(diff**2)

        p_val = chi2.sf(T, df=k)
        ratio = T / k

        print(f"\nCluster {i} vs {j} (projected):")
        print(f"  T = {T:.2f}, df = {k}, ratio = {ratio:.3f}")
        print(f"  P-value: {p_val:.6f}")
        print(f"  Significant at 0.05? {'YES' if p_val < 0.05 else 'NO'}")

# 6. Feature selection analysis
print("\n" + "=" * 70)
print("FEATURE SELECTION IMPACT")
print("=" * 70)

# Sort features by variance
var_order = np.argsort(feature_variances)[::-1]  # Descending

for keep_frac in [1.0, 0.5, 0.25, 0.1]:
    n_keep = max(1, int(keep_frac * X.shape[1]))
    selected = var_order[:n_keep]
    X_sel = X[:, selected]

    # Test first pair
    X_0 = X_sel[y == 0]
    X_1 = X_sel[y == 1]

    # Standardize
    X_all = np.vstack([X_0, X_1])
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0)
    std[std < 1e-10] = 1.0

    X_0_std = (X_0 - mean) / std
    X_1_std = (X_1 - mean) / std

    # Projection
    k_sel = min(k, n_keep)
    np.random.seed(42)
    G = np.random.standard_normal((k_sel, n_keep))
    Q, _ = qr(G.T, mode="economic")
    R = Q.T

    z_0 = R @ X_0_std.mean(axis=0)
    z_1 = R @ X_1_std.mean(axis=0)

    n_eff = len(X_0) * len(X_1) / (len(X_0) + len(X_1))
    diff = z_0 - z_1
    T = n_eff * np.sum(diff**2)

    p_val = chi2.sf(T, df=k_sel)
    ratio = T / k_sel

    print(f"\nTop {keep_frac * 100:.0f}% features ({n_keep}):")
    print(f"  Cluster 0 vs 1: T={T:.2f}, ratio={ratio:.3f}, p={p_val:.6f}")
