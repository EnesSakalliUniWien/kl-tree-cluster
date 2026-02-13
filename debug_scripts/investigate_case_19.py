"""Investigate why test case 19 (d=1200, n=72, k=4) is under-splitting.

This script analyzes the signal strength and chi-square test behavior
to understand why the sibling divergence test fails to detect cluster structure.
"""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.spatial.distance import pdist, cdist
from scipy.stats import chi2
from sklearn.metrics import silhouette_score

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis import config

print("=" * 70)
print("INVESTIGATION: Test Case 19 Under-Splitting")
print("Parameters: n=72, d=1200, k=4, entropy=0.25")
print("=" * 70)

# Generate the data
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=72,
    n_cols=1200,
    n_clusters=4,
    entropy_param=0.25,
    balanced_clusters=True,
    random_seed=314,
)

sample_names = list(data_dict.keys())
X = np.array([data_dict[name] for name in sample_names])
true_labels = np.array([cluster_dict[name] for name in sample_names])

print(f"\nData shape: {X.shape}")
print(f"Cluster sizes: {np.bincount(true_labels)}")

# ============================================================
# 1. DATA SEPARABILITY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("1. DATA SEPARABILITY ANALYSIS")
print("=" * 70)

sil_score = silhouette_score(X, true_labels, metric="hamming")
print(f"Silhouette score (hamming): {sil_score:.4f}")

print("\nBetween-cluster hamming distances:")
for i in range(4):
    for j in range(i + 1, 4):
        mask_i = true_labels == i
        mask_j = true_labels == j
        d = cdist(X[mask_i], X[mask_j], metric="hamming").mean()
        print(f"  Cluster {i} vs {j}: {d:.4f}")

print("\nWithin-cluster hamming distances:")
for i in range(4):
    mask = true_labels == i
    d = pdist(X[mask], metric="hamming").mean()
    print(f"  Cluster {i}: {d:.4f}")

# Feature difference analysis
print("\nFeature difference analysis (cluster means):")
for i in range(4):
    for j in range(i + 1, 4):
        mask_i = true_labels == i
        mask_j = true_labels == j
        mean_i = X[mask_i].mean(axis=0)
        mean_j = X[mask_j].mean(axis=0)
        diff = np.abs(mean_i - mean_j)
        n_diff_02 = np.sum(diff > 0.2)
        n_diff_01 = np.sum(diff > 0.1)
        max_diff = diff.max()
        print(
            f"  Cluster {i} vs {j}: max_diff={max_diff:.3f}, "
            f"|diff|>0.2: {n_diff_02}, |diff|>0.1: {n_diff_01}"
        )

# ============================================================
# 2. CHI-SQUARE TEST POWER ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("2. CHI-SQUARE TEST POWER ANALYSIS")
print("=" * 70)

# Simulate what happens at root level (children: 30 and 42 samples)
n_left, n_right = 30, 42
n_effective = (2.0 * n_left * n_right) / (n_left + n_right)
d = 1200

print(f"\nRoot split scenario: n_left={n_left}, n_right={n_right}")
print(f"Effective sample size: {n_effective:.1f}")
print(f"Number of features: {d}")

# Current projection dimension
k = int(np.ceil(config.PROJECTION_K_MULTIPLIER * np.log(n_effective)))
k = max(k, config.PROJECTION_MIN_K)
print(f"Projection dimension k: {k}")

# Critical value for significance
alpha = config.SIBLING_ALPHA
critical_value = chi2.ppf(1 - alpha, df=k)
print(f"\nFor alpha={alpha}, df={k}:")
print(f"  Critical value: {critical_value:.2f}")
print(f"  Observed test statistic: 27.0 (from debug)")
print(f"  Significant: {27.0 > critical_value}")

# ============================================================
# 3. SIGNAL DILUTION PROBLEM
# ============================================================
print("\n" + "=" * 70)
print("3. SIGNAL DILUTION PROBLEM")
print("=" * 70)

# The issue: chi-square formula divides by d, diluting signal
# χ² = k * ||R·Δθ̂||² / (d * avg_var * (1/n_L + 1/n_R))

# If only m < d features differ, the signal is diluted by d/m factor
# Let's estimate how many features actually differ

# Split data roughly like the tree would
# (This is approximate - actual tree split depends on linkage)
half = X.shape[0] // 2
left_mean = X[:half].mean(axis=0)
right_mean = X[half:].mean(axis=0)
diff = np.abs(left_mean - right_mean)

# Count informative features at different thresholds
for thresh in [0.05, 0.1, 0.15, 0.2]:
    n_informative = np.sum(diff > thresh)
    dilution_factor = d / max(n_informative, 1)
    print(
        f"Features with |diff| > {thresh}: {n_informative} "
        f"(dilution factor: {dilution_factor:.1f}x)"
    )

# ============================================================
# 4. POTENTIAL SOLUTIONS
# ============================================================
print("\n" + "=" * 70)
print("4. POTENTIAL SOLUTIONS")
print("=" * 70)

print("""
The chi-square test statistic is computed as:
  χ² = k * ||R·Δθ̂||² / (d * avg_var * (1/n_L + 1/n_R))

Problem: Dividing by d=1200 dilutes the signal when only a subset
of features are truly different between clusters.

SOLUTION OPTIONS:

A) Use variance-weighted effective dimension instead of raw d
   - df_eff = Σ 4·θ(1-θ) weights features by information content
   - Already implemented for local KL, but not for sibling test
   - Would reduce denominator from 1200 to ~800-1000

B) Use adaptive significance level for high d/n ratio
   - Already have USE_ADAPTIVE_ALPHA in config (disabled)
   - Scale α upward when n << d to compensate for reduced power

C) Increase projection dimension k
   - Current: k = 8*log(n) ≈ 29
   - Higher k preserves more signal but increases df
   - Trade-off: need k high enough to capture signal, low enough for power

D) Use permutation-based p-value instead of chi-square
   - More accurate null distribution
   - Computationally expensive but avoids chi-square approximation issues

E) Pre-filter to informative features before projection
   - Remove features with low variance or low MI
   - Reduces d in denominator to actual informative features
   - Caution: selection bias (already tested, causes problems)

F) Scale test statistic by estimated signal sparsity
   - If signal is sparse (only m << d features differ), adjust accordingly
   - Requires estimating sparsity, which is tricky
""")

# ============================================================
# 5. QUICK TEST: What if we use variance-weighted d?
# ============================================================
print("\n" + "=" * 70)
print("5. SIMULATION: Variance-Weighted Denominator")
print("=" * 70)

# Compute variance-weighted effective dimension
parent_mean = X.mean(axis=0)  # Approximate parent distribution
var_weights = 4 * parent_mean * (1 - parent_mean)
d_effective = np.sum(var_weights)
print(f"Raw dimension d: {d}")
print(f"Variance-weighted d_eff: {d_effective:.1f}")
print(f"Reduction factor: {d / d_effective:.2f}x")

# If we used d_effective in the denominator:
# New test stat = old_stat * (d / d_eff) = 27 * (1200 / d_eff)
old_stat = 27.0
new_stat = old_stat * (d / d_effective)
new_p_value = chi2.sf(new_stat, df=k)
print(f"\nWith variance-weighted denominator:")
print(f"  Adjusted test statistic: {new_stat:.2f}")
print(f"  New p-value: {new_p_value:.4f}")
print(f"  Significant at α={alpha}? {new_p_value < alpha}")

# ============================================================
# 6. QUICK TEST: What if we increase k?
# ============================================================
print("\n" + "=" * 70)
print("6. SIMULATION: Higher Projection Dimension")
print("=" * 70)

for k_mult in [8, 12, 16, 20]:
    k_new = int(np.ceil(k_mult * np.log(n_effective)))
    k_new = max(k_new, config.PROJECTION_MIN_K)

    # Higher k means test statistic scales proportionally (more signal preserved)
    # but critical value also increases
    # Expected test stat scales as k_new/k (roughly)
    expected_stat = 27.0 * (k_new / k)
    crit = chi2.ppf(1 - alpha, df=k_new)

    print(
        f"k_multiplier={k_mult}: k={k_new}, expected_stat≈{expected_stat:.1f}, "
        f"critical={crit:.1f}, significant={expected_stat > crit}"
    )

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The under-splitting is caused by signal dilution in high dimensions.
The chi-square denominator uses raw d=1200, but only a subset of
features actually differ between clusters.

RECOMMENDED FIX:
Use variance-weighted effective dimension (d_eff) in the denominator
instead of raw d. This naturally down-weights uninformative features
and increases test power for detecting true cluster differences.
""")
