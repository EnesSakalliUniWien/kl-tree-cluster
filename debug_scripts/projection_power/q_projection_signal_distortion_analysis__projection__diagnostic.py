"""
Purpose: Analyze how projection affects signal for realistic cluster differences.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/projection_power/q_projection_signal_distortion_analysis__projection__diagnostic.py
"""

import numpy as np
from scipy.linalg import qr
from scipy.stats import chi2

np.random.seed(42)

d = 3000
n_left, n_right = 30, 30
n_eff = 2 * n_left * n_right / (n_left + n_right)

# Simulate TRUE clusters: distinct theta patterns
# In binary data, clusters differ in a BLOCK of features, not randomly scattered
theta_cluster_A = np.concatenate([np.full(1500, 0.8), np.full(1500, 0.2)])
theta_cluster_B = np.concatenate([np.full(1500, 0.2), np.full(1500, 0.8)])

pooled = 0.5 * (theta_cluster_A + theta_cluster_B)  # = 0.5 everywhere
inv_n = 1 / n_left + 1 / n_right
var_diff = pooled * (1 - pooled) * inv_n  # = 0.25 * inv_n = uniform

diff = theta_cluster_A - theta_cluster_B  # [0.6, 0.6, ..., -0.6, -0.6]
z = diff / np.sqrt(var_diff)

print("=" * 70)
print("BLOCK-DIFFERENCE SCENARIO (realistic cluster separation)")
print("=" * 70)
print(f"d={d}, n_left={n_left}, n_right={n_right}")
print(f"Cluster A: first 1500 features have θ=0.8, rest θ=0.2")
print(f"Cluster B: first 1500 features have θ=0.2, rest θ=0.8")
print(f"||z||² = {np.sum(z**2):.1f}")
print()

# Full chi-square
stat_full = np.sum(z**2)
p_full = chi2.sf(stat_full, df=d)
print(f"Full chi-square: T={stat_full:.1f}, df={d}, p={p_full:.2e}")

# Projected chi-square with different k
for k in [14, 50, 100, 500, 1000]:
    rng = np.random.default_rng(42)
    G = rng.standard_normal((k, d))
    Q, _ = qr(G.T, mode="economic")
    R = Q.T
    z_proj = R @ z
    stat = np.sum(z_proj**2)
    p = chi2.sf(stat, df=k)

    # Expected value of ||Rz||² = ||z||² * k/d (JL property)
    expected = np.sum(z**2) * k / d
    print(f"k={k:4d}: T={stat:8.1f} (expected ~{expected:.1f}), df={k}, p={p:.2e}")

print()
print("=" * 70)
print("KEY INSIGHT: Projection PRESERVES distances but at REDUCED scale!")
print("=" * 70)
print(f"||z||² = {np.sum(z**2):.1f}")
print(f"With k=14: ||Rz||² ≈ {np.sum(z**2) * 14 / d:.1f} (signal preserved but scaled)")
print(f"Expected χ²(14) mean = 14, so test stat >> expected → still significant!")
print()
print("The issue is NOT projection destroying signal.")
print("The issue is that our ACTUAL data does not have this strong signal.")
