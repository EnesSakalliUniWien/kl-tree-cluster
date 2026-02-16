"""
Purpose: Quantify effective signal strength in benchmark case 55.
Inputs: Benchmark case 55 data and analysis parameters.
Outputs: Console summary of measured signal components.
Expected runtime: ~10-60 seconds.
How to run: python debug_scripts/case_studies/q_case55_signal_strength__diagnostic__case55.py
"""

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis import config


def analyze_case_55_signal():
    """Analyze the actual signal strength between clusters in Case 55."""
    print("=" * 70)
    print("CASE 55: Actual signal strength analysis")
    print("=" * 70)

    # Generate Case 55 data
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=360,
        n_cols=3000,
        n_clusters=12,
        entropy_param=0.20,
        balanced_clusters=False,
        random_seed=2024,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    print(f"\nData: {data_df.shape[0]} samples × {data_df.shape[1]} features")
    print(f"True clusters: 12")
    print(f"Entropy (noise): 0.20")

    # Get unique clusters
    unique_clusters = np.unique(true_labels)
    print(f"\nCluster sizes:")
    for c in unique_clusters:
        size = np.sum(true_labels == c)
        print(f"  Cluster {c}: {size} samples")

    # Compute mean distributions per cluster
    cluster_means = {}
    for c in unique_clusters:
        mask = true_labels == c
        cluster_means[c] = data_df.values[mask].mean(axis=0)

    # Compute pairwise signal strength
    print(f"\n{'=' * 70}")
    print("PAIRWISE SIGNAL STRENGTH BETWEEN TRUE CLUSTERS")
    print("=" * 70)
    print(
        f"{'Pair':<12} {'||z||²':<12} {'E[χ²] under H₀':<15} {'Ratio':<10} {'p-value'}"
    )
    print("-" * 70)

    d = data_df.shape[1]
    n_per_cluster = 30  # approximate for standardization

    signals = []
    for i, c1 in enumerate(unique_clusters):
        for c2 in unique_clusters[i + 1 :]:
            n1 = np.sum(true_labels == c1)
            n2 = np.sum(true_labels == c2)

            theta1 = cluster_means[c1]
            theta2 = cluster_means[c2]

            # Standardize difference
            pooled = 0.5 * (theta1 + theta2)
            pooled = np.clip(pooled, 1e-10, 1 - 1e-10)
            inv_n = 1 / n1 + 1 / n2
            var_diff = pooled * (1 - pooled) * inv_n
            var_diff = np.maximum(var_diff, 1e-10)

            diff = theta1 - theta2
            z = diff / np.sqrt(var_diff)
            z_sq = np.sum(z**2)

            p_full = chi2.sf(z_sq, df=d)
            ratio = z_sq / d

            signals.append((c1, c2, z_sq, ratio))
            print(f"{c1}-{c2:<8} {z_sq:<12.1f} {d:<15} {ratio:<10.2f} {p_full:.2e}")

    print()
    avg_signal = np.mean([s[2] for s in signals])
    avg_ratio = np.mean([s[3] for s in signals])
    print(f"Average ||z||²: {avg_signal:.1f}")
    print(f"Average ratio (||z||²/d): {avg_ratio:.2f}")
    print(f"For significant detection, need ratio >> 1.0")

    # Now check what happens with projection
    print(f"\n{'=' * 70}")
    print("PROJECTED SIGNAL (k=14, current setting)")
    print("=" * 70)

    k = 14
    rng = np.random.default_rng(42)
    G = rng.standard_normal((k, d))
    Q, _ = qr(G.T, mode="economic")
    R = Q.T

    print(
        f"{'Pair':<12} {'||Rz||²':<12} {'E[χ²(k)] under H₀':<18} {'Ratio':<10} {'p-value'}"
    )
    print("-" * 70)

    for c1, c2, z_sq_full, _ in signals[:10]:  # Show first 10
        n1 = np.sum(true_labels == c1)
        n2 = np.sum(true_labels == c2)

        theta1 = cluster_means[c1]
        theta2 = cluster_means[c2]

        pooled = 0.5 * (theta1 + theta2)
        pooled = np.clip(pooled, 1e-10, 1 - 1e-10)
        inv_n = 1 / n1 + 1 / n2
        var_diff = pooled * (1 - pooled) * inv_n
        var_diff = np.maximum(var_diff, 1e-10)

        diff = theta1 - theta2
        z = diff / np.sqrt(var_diff)

        z_proj = R @ z
        z_sq_proj = np.sum(z_proj**2)

        p_proj = chi2.sf(z_sq_proj, df=k)
        ratio = z_sq_proj / k

        print(f"{c1}-{c2:<8} {z_sq_proj:<12.1f} {k:<18} {ratio:<10.2f} {p_proj:.2e}")

    print()
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if avg_ratio < 2.0:
        print("⚠️  Weak signal: True clusters have low separation in feature space")
        print("   This is expected with entropy=0.20 (high noise)")
        print("   Features are noisy Bernoulli with θ close to 0.5")
        print()
        print("OPTIONS:")
        print("1. Accept that high-noise data has limited detectability")
        print("2. Use covariance-aware categorical testing (no feature prefiltering)")
        print("3. Increase sample size requirements")
        print("4. Lower significance threshold (more aggressive)")
    else:
        print("✓ Strong signal detected between true clusters")
        print("  Projection should preserve this signal")


if __name__ == "__main__":
    analyze_case_55_signal()
