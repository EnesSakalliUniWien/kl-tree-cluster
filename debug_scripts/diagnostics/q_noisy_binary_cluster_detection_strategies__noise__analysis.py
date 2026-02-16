"""
Purpose: Analyze approaches to detect clusters in noisy binary data.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/diagnostics/q_noisy_binary_cluster_detection_strategies__noise__analysis.py
"""

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2, fisher_exact, combine_pvalues
from scipy.spatial.distance import pdist, squareform

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)


def generate_noisy_data():
    """Generate Case 17 data."""
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=72,
        n_cols=160,
        n_clusters=4,
        entropy_param=0.25,
        balanced_clusters=True,
        random_seed=314,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])
    return data_df, true_labels


def approach_1_more_samples():
    """Approach 1: More samples reduce estimation variance."""
    print("=" * 70)
    print("APPROACH 1: Increase sample size")
    print("=" * 70)

    for n_samples in [72, 144, 288, 576, 1000]:
        data_dict, cluster_dict = generate_random_feature_matrix(
            n_rows=n_samples,
            n_cols=160,
            n_clusters=4,
            entropy_param=0.25,
            balanced_clusters=True,
            random_seed=314,
        )
        data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
        true_labels = np.array([cluster_dict[name] for name in data_df.index])

        # Compute cluster means
        cluster_means = {}
        for c in np.unique(true_labels):
            mask = true_labels == c
            cluster_means[c] = data_df.values[mask].mean(axis=0)

        # Compute average signal strength
        signals = []
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
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
                z_sq = np.sum(z**2)
                signals.append(z_sq)

        avg_signal = np.mean(signals)
        d = 160
        ratio = avg_signal / d

        print(f"  n={n_samples:4d}: avg ||z||²={avg_signal:6.1f}, ratio={ratio:.2f}")


def approach_2_more_features():
    """Approach 2: More features increase cumulative signal."""
    print("\n" + "=" * 70)
    print("APPROACH 2: Increase number of features")
    print("=" * 70)

    for n_features in [160, 500, 1000, 3000, 10000]:
        data_dict, cluster_dict = generate_random_feature_matrix(
            n_rows=72,
            n_cols=n_features,
            n_clusters=4,
            entropy_param=0.25,
            balanced_clusters=True,
            random_seed=314,
        )
        data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
        true_labels = np.array([cluster_dict[name] for name in data_df.index])

        cluster_means = {}
        for c in np.unique(true_labels):
            mask = true_labels == c
            cluster_means[c] = data_df.values[mask].mean(axis=0)

        signals = []
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
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
                z_sq = np.sum(z**2)
                p = chi2.sf(z_sq, df=n_features)
                signals.append((z_sq, p))

        avg_signal = np.mean([s[0] for s in signals])
        avg_p = np.mean([s[1] for s in signals])
        ratio = avg_signal / n_features

        print(
            f"  d={n_features:5d}: avg ||z||²={avg_signal:8.1f}, ratio={ratio:.2f}, avg_p={avg_p:.3f}"
        )


def approach_3_feature_selection():
    """Approach 3: Select informative features before testing."""
    print("\n" + "=" * 70)
    print("APPROACH 3: Feature selection (keep most variable features)")
    print("=" * 70)

    data_df, true_labels = generate_noisy_data()
    d = data_df.shape[1]

    cluster_means = {}
    for c in np.unique(true_labels):
        mask = true_labels == c
        cluster_means[c] = data_df.values[mask].mean(axis=0)

    # Compute per-feature variance across cluster means
    all_means = np.array([cluster_means[c] for c in range(4)])
    feature_variance = np.var(all_means, axis=0)

    print(
        f"  Feature variance range: [{feature_variance.min():.4f}, {feature_variance.max():.4f}]"
    )
    print(f"  Features with variance > 0.01: {np.sum(feature_variance > 0.01)}")

    for top_k_pct in [100, 50, 25, 10, 5]:
        top_k = int(d * top_k_pct / 100)
        top_indices = np.argsort(feature_variance)[-top_k:]

        signals = []
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
                n1 = np.sum(true_labels == c1)
                n2 = np.sum(true_labels == c2)

                theta1 = cluster_means[c1][top_indices]
                theta2 = cluster_means[c2][top_indices]

                pooled = 0.5 * (theta1 + theta2)
                pooled = np.clip(pooled, 1e-10, 1 - 1e-10)
                inv_n = 1 / n1 + 1 / n2
                var_diff = pooled * (1 - pooled) * inv_n
                var_diff = np.maximum(var_diff, 1e-10)

                diff = theta1 - theta2
                z = diff / np.sqrt(var_diff)
                z_sq = np.sum(z**2)
                p = chi2.sf(z_sq, df=top_k)
                signals.append((z_sq, p, top_k))

        avg_signal = np.mean([s[0] for s in signals])
        avg_p = np.mean([s[1] for s in signals])
        ratio = avg_signal / top_k
        sig_pairs = sum(1 for s in signals if s[1] < 0.05)

        print(
            f"  Top {top_k_pct:3d}% ({top_k:3d} features): ratio={ratio:.2f}, avg_p={avg_p:.3f}, sig_pairs={sig_pairs}/6"
        )


def approach_4_different_distance():
    """Approach 4: Use distance-based methods instead of chi-square."""
    print("\n" + "=" * 70)
    print("APPROACH 4: Distance-based clustering quality")
    print("=" * 70)

    data_df, true_labels = generate_noisy_data()

    # Compute pairwise distances
    for metric in ["hamming", "rogerstanimoto", "jaccard", "euclidean"]:
        try:
            D = squareform(pdist(data_df.values, metric=metric))
        except:
            continue

        # Compute within-cluster vs between-cluster distance ratio
        within_dists = []
        between_dists = []

        for i in range(len(true_labels)):
            for j in range(i + 1, len(true_labels)):
                if true_labels[i] == true_labels[j]:
                    within_dists.append(D[i, j])
                else:
                    between_dists.append(D[i, j])

        avg_within = np.mean(within_dists)
        avg_between = np.mean(between_dists)
        ratio = avg_between / avg_within if avg_within > 0 else 0

        print(
            f"  {metric:15s}: within={avg_within:.4f}, between={avg_between:.4f}, ratio={ratio:.3f}"
        )


def approach_5_cumulative_evidence():
    """Approach 5: Combine evidence across multiple features."""
    print("\n" + "=" * 70)
    print("APPROACH 5: Combine p-values (Fisher's method)")
    print("=" * 70)

    data_df, true_labels = generate_noisy_data()
    d = data_df.shape[1]

    cluster_means = {}
    for c in np.unique(true_labels):
        mask = true_labels == c
        cluster_means[c] = data_df.values[mask].mean(axis=0)

    # For each cluster pair, compute per-feature p-values and combine
    for c1 in range(4):
        for c2 in range(c1 + 1, 4):
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

            # Per-feature z-scores and p-values
            feature_pvals = 2 * (1 - chi2.cdf(z**2, df=1))  # Two-sided
            feature_pvals = np.clip(feature_pvals, 1e-100, 1)  # Avoid log(0)

            # Fisher's combined p-value
            combined_stat, combined_p = combine_pvalues(feature_pvals, method="fisher")

            # Standard chi-square
            chi2_stat = np.sum(z**2)
            chi2_p = chi2.sf(chi2_stat, df=d)

            print(f"  Pair {c1}-{c2}: χ² p={chi2_p:.3e}, Fisher p={combined_p:.3e}")


def approach_6_lower_entropy():
    """Approach 6: Lower entropy (less noise) makes clusters detectable."""
    print("\n" + "=" * 70)
    print("APPROACH 6: Reduce noise (lower entropy parameter)")
    print("=" * 70)

    for entropy in [0.25, 0.20, 0.15, 0.10, 0.05]:
        data_dict, cluster_dict = generate_random_feature_matrix(
            n_rows=72,
            n_cols=160,
            n_clusters=4,
            entropy_param=entropy,
            balanced_clusters=True,
            random_seed=314,
        )
        data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(float)
        true_labels = np.array([cluster_dict[name] for name in data_df.index])

        cluster_means = {}
        for c in np.unique(true_labels):
            mask = true_labels == c
            cluster_means[c] = data_df.values[mask].mean(axis=0)

        signals = []
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
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
                z_sq = np.sum(z**2)
                p = chi2.sf(z_sq, df=160)
                signals.append((z_sq, p))

        avg_signal = np.mean([s[0] for s in signals])
        sig_pairs = sum(1 for s in signals if s[1] < 0.05)
        ratio = avg_signal / 160

        print(
            f"  entropy={entropy:.2f}: avg ||z||²={avg_signal:6.1f}, ratio={ratio:.2f}, sig_pairs={sig_pairs}/6"
        )


def summary():
    """Summary of findings."""
    print("\n" + "=" * 70)
    print("SUMMARY: How to detect clusters in noise")
    print("=" * 70)
    print("""
The fundamental issue is SIGNAL-TO-NOISE RATIO (SNR):
- With entropy=0.25, each feature has 25% noise
- SNR = (cluster_separation)² / variance_of_estimate

WHAT HELPS:
1. ✓ MORE SAMPLES: Reduces variance → increases SNR
   - Need ~500+ samples for entropy=0.25 to be detectable

2. ✓ MORE FEATURES: Signal accumulates across features
   - But variance also accumulates, so ratio stays ~1.0
   - Need structured (correlated) features for this to help

3. ✓ FEATURE SELECTION: Remove noise features
   - If we can identify informative features, ratio improves
   - BUT: with uniform noise, no features are special

4. ✗ DISTANCE METRICS: Limited help
   - All metrics suffer from the same noise
   - Ratio between/within is close to 1.0

5. ✓ FISHER'S METHOD: Combines per-feature evidence
   - Can detect aggregate signal even when each feature is weak
   - More sensitive than single chi-square test

6. ✓ LOWER NOISE: Best solution when possible
   - entropy=0.15: marginally detectable
   - entropy=0.10: clearly detectable
   - entropy=0.05: perfectly detectable

KEY INSIGHT:
The algorithm is CORRECTLY saying "I cannot reliably distinguish these clusters"
because the data genuinely doesn't support it. This is NOT a bug - it's
honest statistical inference!

RECOMMENDATIONS:
1. Accept that high-noise data may not have detectable structure
2. Consider using Fisher's combined test for more power
3. Add minimum sample-size requirements (e.g., 50 per cluster)
4. Document detection limits: "With entropy > 0.20, detection requires n > 200"
""")


if __name__ == "__main__":
    approach_1_more_samples()
    approach_2_more_features()
    approach_3_feature_selection()
    approach_4_different_distance()
    approach_5_cumulative_evidence()
    approach_6_lower_entropy()
    summary()
