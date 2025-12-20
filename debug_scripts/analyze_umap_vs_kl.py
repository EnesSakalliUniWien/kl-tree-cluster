"""
Analyze why UMAP separates clusters but KL-based algorithm finds only 1.

Test case 101: overlap_hd_6c_2k
- 600 samples × 2000 features
- 6 ground-truth clusters
- entropy_param = 0.33
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_cases_config import get_default_test_cases
from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)


def analyze_test_case(case_idx=100):
    """Analyze why UMAP and KL-test disagree."""

    cases = get_default_test_cases()
    tc = cases[case_idx]

    print("=" * 80)
    print(f"Analyzing Test Case {case_idx + 1}: {tc['name']}")
    print("=" * 80)
    print(
        f"  n_rows={tc['n_rows']}, n_cols={tc['n_cols']}, n_clusters={tc['n_clusters']}"
    )
    print(f"  entropy_param={tc['entropy_param']}")
    print()

    # Generate the data
    np.random.seed(tc["seed"])
    leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=tc["n_rows"],
        n_cols=tc["n_cols"],
        n_clusters=tc["n_clusters"],
        entropy_param=tc["entropy_param"],
        balanced_clusters=tc["balanced_clusters"],
        random_seed=tc["seed"],
    )

    # Convert to numpy arrays
    sample_names = sorted(leaf_matrix_dict.keys(), key=lambda x: int(x[1:]))
    X = np.array([leaf_matrix_dict[name] for name in sample_names])
    y = np.array([cluster_assignments[name] for name in sample_names])

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # =========================================================================
    # 1. Feature-level analysis (what KL-test sees)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. FEATURE-LEVEL ANALYSIS (KL-test perspective)")
    print("=" * 80)

    # Compute cluster means per feature
    n_clusters = tc["n_clusters"]
    cluster_means = np.array([X[y == c].mean(axis=0) for c in range(n_clusters)])

    print("\nCluster feature means:")
    for c in range(n_clusters):
        print(
            f"  Cluster {c}: mean={cluster_means[c].mean():.4f} ± {cluster_means[c].std():.4f}"
        )

    # How different are cluster means from each other?
    mean_diffs = []
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            diff = np.abs(cluster_means[c1] - cluster_means[c2]).mean()
            mean_diffs.append(diff)
    print(
        f"\nMean absolute difference between cluster means: {np.mean(mean_diffs):.6f}"
    )

    # With entropy=0.33, expected difference is small
    # Each feature has ~33% noise, so cluster-specific signal is diluted

    # =========================================================================
    # 2. Distance-level analysis (what UMAP sees)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. DISTANCE-LEVEL ANALYSIS (UMAP perspective)")
    print("=" * 80)

    # Compute pairwise distances
    all_dists = pairwise_distances(X)

    # Within-cluster distances
    within_dists = []
    for c in range(n_clusters):
        mask = y == c
        cluster_dists = all_dists[np.ix_(mask, mask)]
        within_dists.extend(cluster_dists[np.triu_indices_from(cluster_dists, k=1)])
    within_dists = np.array(within_dists)

    # Between-cluster distances
    between_dists = []
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            mask1 = y == c1
            mask2 = y == c2
            between_dists.extend(all_dists[np.ix_(mask1, mask2)].flatten())
    between_dists = np.array(between_dists)

    print(
        f"Within-cluster distances:  mean={within_dists.mean():.2f} ± {within_dists.std():.2f}"
    )
    print(
        f"Between-cluster distances: mean={between_dists.mean():.2f} ± {between_dists.std():.2f}"
    )
    print(
        f"Distance ratio (between/within): {between_dists.mean() / within_dists.mean():.4f}"
    )

    # Check overlap in distance distributions
    within_max = np.percentile(within_dists, 95)
    between_min = np.percentile(between_dists, 5)
    print(f"\nWithin 95th percentile: {within_max:.2f}")
    print(f"Between 5th percentile: {between_min:.2f}")
    if between_min > within_max:
        print("→ Clusters are SEPARABLE by distance (UMAP can find them)")
    else:
        print(
            "→ Distance distributions OVERLAP (but UMAP may still find local structure)"
        )

    # =========================================================================
    # 3. Why the discrepancy?
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. WHY THE DISCREPANCY?")
    print("=" * 80)

    print("""
KEY INSIGHT: UMAP and KL-test measure different things!

UMAP:
- Uses LOCAL neighborhood structure
- Works in DISTANCE space (Euclidean or other metric)
- With 2000 features, even small per-feature differences SUM UP
- Distance = sqrt(sum of squared differences) accumulates signal
- UMAP is sensitive to LOCAL density variations

KL-test (chi-square):
- Tests FEATURE-BY-FEATURE distributional differences
- With entropy=0.33, each feature has only ~67% of true cluster signal
- Tests if P(feature|cluster1) ≠ P(feature|cluster2)
- High entropy = each feature looks similar across clusters
- Statistical test is CONSERVATIVE (requires strong evidence)

The mathematical reason:
- Euclidean distance: d = sqrt(Σ (x_i - y_i)²)
  With 2000 features, many small differences accumulate!
  
- KL divergence per feature: KL_i = θ_i * log(θ_i/θ_j)
  With entropy=0.33, θ_i ≈ θ_j, so KL_i ≈ 0 for each feature
  
CONCLUSION:
- UMAP finds structure because distances ACCUMULATE across 2000 features
- KL-test finds nothing because each individual feature is too noisy
- This is a VALID negative result for KL-test (clusters aren't distinguishable per-feature)
""")

    # =========================================================================
    # 4. Quantify the effect
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. QUANTIFYING THE EFFECT")
    print("=" * 80)

    # Compute expected distance difference
    # In high-D binary data, distance ≈ sqrt(p * 2 * θ * (1-θ)) for noise
    # But cluster signal adds extra distance between clusters

    p = tc["n_cols"]  # number of features
    entropy = tc["entropy_param"]

    # Expected per-feature variance contribution
    # With entropy=0.33, features are ~67% cluster-specific, 33% noise
    signal_fraction = 1 - 2 * entropy  # fraction of signal preserved
    print(f"Signal fraction per feature: {signal_fraction:.2%}")

    # But distances accumulate!
    print(
        f"With {p} features, distance signal accumulates: sqrt({p}) ≈ {np.sqrt(p):.1f}x amplification"
    )

    # Compute actual signal-to-noise ratio for distances
    mean_within = within_dists.mean()
    mean_between = between_dists.mean()
    std_within = within_dists.std()

    distance_snr = (mean_between - mean_within) / std_within
    print(f"Distance SNR: {distance_snr:.2f}")
    print(f"  (UMAP can separate if SNR > ~1-2)")

    # Compute signal for KL test (per-feature)
    # This is what the chi-square test sees
    overall_theta = X.mean(axis=0)  # global mean per feature

    # Per-cluster theta
    cluster_theta = cluster_means

    # KL divergence from global to each cluster
    eps = 1e-10
    kl_per_cluster = []
    for c in range(n_clusters):
        kl = np.sum(
            overall_theta * np.log((overall_theta + eps) / (cluster_theta[c] + eps))
            + (1 - overall_theta)
            * np.log((1 - overall_theta + eps) / (1 - cluster_theta[c] + eps))
        )
        kl_per_cluster.append(kl)

    print(f"\nKL divergence (global → cluster):")
    for c in range(n_clusters):
        print(f"  Cluster {c}: KL = {kl_per_cluster[c]:.4f}")

    print(f"\nMean KL across clusters: {np.mean(kl_per_cluster):.4f}")

    # For chi-square test: test statistic = 2 * n * KL
    n_per_cluster = tc["n_rows"] // n_clusters
    test_stat = 2 * n_per_cluster * np.mean(kl_per_cluster)

    # df = effective features (variance-weighted)
    var_weights = 4 * overall_theta * (1 - overall_theta)
    df_eff = np.sum(var_weights)

    print(f"\nChi-square analysis:")
    print(
        f"  Test statistic: 2 * n * KL = 2 * {n_per_cluster} * {np.mean(kl_per_cluster):.4f} = {test_stat:.2f}"
    )
    print(f"  Effective df: {df_eff:.1f}")
    print(f"  Ratio (stat/df): {test_stat / df_eff:.4f}")
    print(f"  (Need ratio >> 1 for significance)")

    from scipy.stats import chi2

    p_value = chi2.sf(test_stat, df=df_eff)
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.01:
        print("  → SIGNIFICANT (would split)")
    else:
        print("  → NOT SIGNIFICANT (would not split)")

    return X, y, tc


if __name__ == "__main__":
    X, y, tc = analyze_test_case(100)  # Test case 101 (0-indexed)
