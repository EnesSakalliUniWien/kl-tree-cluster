#!/usr/bin/env python3
"""
Analyze the mathematical relationship between branch length and divergence.

Key findings from exploration:
1. Branch length = parent_height - node_height (from scipy linkage)
2. Height comes from distance metric (hamming/jaccard) during clustering
3. The root's children have the LARGEST branch lengths when clusters are well-separated

This script explores:
- How branch length relates to actual distributional divergence
- Whether branch length can predict test significance
- The mathematical model connecting them
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy import stats
import matplotlib.pyplot as plt

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree


def generate_data(n_per_cluster, n_features, divergence, seed):
    """Generate two-cluster data with known divergence."""
    rng = np.random.RandomState(seed)
    k = 4  # categories

    ancestor = rng.randint(0, k, size=n_features)

    def evolve(seq, branch_len):
        P = np.full((k, k), (1.0 / k) * (1 - np.exp(-k * branch_len / (k - 1))))
        np.fill_diagonal(
            P, (1.0 / k) + ((k - 1.0) / k) * np.exp(-k * branch_len / (k - 1))
        )
        return np.array([rng.choice(k, p=P[s]) for s in seq])

    cluster_a = evolve(ancestor, divergence)
    cluster_b = evolve(ancestor, divergence)

    samples = []
    labels = []
    terminal = 0.05

    for _ in range(n_per_cluster):
        samples.append(evolve(cluster_a, terminal))
        labels.append(0)
    for _ in range(n_per_cluster):
        samples.append(evolve(cluster_b, terminal))
        labels.append(1)

    return np.array(samples), np.array(labels)


def compute_empirical_divergence(X, y, n_features):
    """Compute Jensen-Shannon divergence between the two clusters."""
    X_a = X[y == 0]
    X_b = X[y == 1]

    # Empirical distributions (proportion of each category per feature)
    k = 4
    p_a = np.zeros((n_features, k))
    p_b = np.zeros((n_features, k))

    for j in range(n_features):
        for c in range(k):
            p_a[j, c] = (X_a[:, j] == c).mean()
            p_b[j, c] = (X_b[:, j] == c).mean()

    # Add smoothing
    eps = 1e-10
    p_a = np.clip(p_a, eps, 1 - eps)
    p_b = np.clip(p_b, eps, 1 - eps)
    p_a = p_a / p_a.sum(axis=1, keepdims=True)
    p_b = p_b / p_b.sum(axis=1, keepdims=True)

    # JS divergence per feature
    m = 0.5 * (p_a + p_b)
    kl_am = np.sum(p_a * np.log(p_a / m), axis=1)
    kl_bm = np.sum(p_b * np.log(p_b / m), axis=1)
    js_per_feature = 0.5 * (kl_am + kl_bm)

    return js_per_feature.mean(), js_per_feature


def expected_hamming_distance(divergence, n_categories=4):
    """Expected Hamming distance under Jukes-Cantor model.

    After branch length b, probability that a site differs:
    P(different) = (K-1)/K * (1 - exp(-Kb/(K-1)))

    For two independent branches of length b from a common ancestor:
    P(site differs between them) ≈ 2 * P(different) * (1 - P(different)/2)
    """
    k = n_categories
    p_mut = ((k - 1) / k) * (1 - np.exp(-k * divergence / (k - 1)))

    # Two independent branches
    # P(both same) = (1-p)^2 + p^2/k + ... ≈ (1-p)^2 for simplicity
    # P(different) ≈ 2*p*(1-p) for small p
    p_diff = 2 * p_mut * (1 - p_mut) + (p_mut**2) * (k - 1) / k

    return p_diff


def main():
    print("=" * 70)
    print("BRANCH LENGTH ↔ DIVERGENCE MATHEMATICAL ANALYSIS")
    print("=" * 70)

    n_per_cluster = 50
    n_features = 100
    seed = 42

    # Test range of divergences
    divergences = np.linspace(0.05, 2.0, 20)

    results = []

    for div in divergences:
        X, y = generate_data(n_per_cluster, n_features, div, seed)
        sample_names = [f"S{i}" for i in range(len(X))]

        # Build tree
        Z = linkage(pdist(X, metric="hamming"), method="weighted")
        tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

        # Get root info
        root = tree.root()
        root_height = tree.nodes[root].get("height", 0.0)

        # Get root's children branch lengths
        children = list(tree.successors(root))
        child_heights = [tree.nodes[c].get("height", 0.0) for c in children]
        branch_lengths = [root_height - h for h in child_heights]
        avg_branch_length = np.mean(branch_lengths)

        # Compute empirical divergence
        js_mean, js_per_feature = compute_empirical_divergence(X, y, n_features)

        # Expected Hamming
        expected_hamming = expected_hamming_distance(div)

        # Actual Hamming between cluster centroids (mode per feature)
        X_a = X[y == 0]
        X_b = X[y == 1]
        mode_a = np.array([np.bincount(X_a[:, j]).argmax() for j in range(n_features)])
        mode_b = np.array([np.bincount(X_b[:, j]).argmax() for j in range(n_features)])
        actual_hamming = (mode_a != mode_b).mean()

        results.append(
            {
                "true_divergence": div,
                "root_height": root_height,
                "avg_branch_length": avg_branch_length,
                "js_divergence": js_mean,
                "expected_hamming": expected_hamming,
                "actual_hamming": actual_hamming,
            }
        )

    df = pd.DataFrame(results)

    print("\n--- Results Table ---")
    print(df.to_string(index=False))

    # Correlations
    print("\n\n--- Correlations ---")
    print(
        f"Branch Length vs True Divergence:   r = {df['avg_branch_length'].corr(df['true_divergence']):.4f}"
    )
    print(
        f"Branch Length vs JS Divergence:     r = {df['avg_branch_length'].corr(df['js_divergence']):.4f}"
    )
    print(
        f"Root Height vs True Divergence:     r = {df['root_height'].corr(df['true_divergence']):.4f}"
    )
    print(
        f"Root Height vs JS Divergence:       r = {df['root_height'].corr(df['js_divergence']):.4f}"
    )
    print(
        f"Expected Hamming vs Actual Hamming: r = {df['expected_hamming'].corr(df['actual_hamming']):.4f}"
    )

    # Fit linear model: branch_length = a * divergence + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["true_divergence"], df["avg_branch_length"]
    )
    print(f"\n--- Linear Fit: branch_length = a * divergence + b ---")
    print(f"  a (slope):     {slope:.4f}")
    print(f"  b (intercept): {intercept:.4f}")
    print(f"  R²:            {r_value**2:.4f}")
    print(f"  p-value:       {p_value:.2e}")

    # Relationship to expected values
    print("\n\n--- Expected vs Observed ---")
    print("Under Jukes-Cantor, expected Hamming distance saturates at ~0.75 for K=4")
    print(f"Max observed root height: {df['root_height'].max():.4f}")
    print(f"Max observed branch len:  {df['avg_branch_length'].max():.4f}")
    print(f"Saturation point (JC):    ~0.75")

    # Mathematical interpretation
    print("\n\n" + "=" * 70)
    print("MATHEMATICAL INTERPRETATION")
    print("=" * 70)
    print("""
For Hamming distance with K categories (Jukes-Cantor model):

  d_Hamming(A,B) = P(site differs between A and B)
  
After branch length b from common ancestor:
  P(mutation) = (K-1)/K * (1 - exp(-Kb/(K-1)))

For two clusters evolved with branch length b each:
  Expected Hamming ≈ 2 * P(mutation) * (1 - P(mutation)/2)

The scipy linkage 'height' is the average pairwise Hamming distance
at which two clusters merge. Thus:

  root_height ≈ mean pairwise Hamming between all samples
  branch_length = root_height - child_height
                ≈ additional divergence from parent to child cluster

KEY INSIGHT:
  Branch length at root's children directly reflects the evolutionary
  divergence between clusters. This can be used to:
  
  1. SET THRESHOLD: Require minimum branch length to split
  2. CALIBRATE TESTS: Weight p-values by expected divergence for branch length
  3. VALIDATE SPLITS: Compare observed divergence to expected for branch length
""")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(df["true_divergence"], df["avg_branch_length"], "b-o", linewidth=2)
    ax.set_xlabel("True Divergence (JC branch length)")
    ax.set_ylabel("Observed Branch Length (from tree)")
    ax.set_title("Branch Length vs True Divergence")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["true_divergence"], df["root_height"], "r-o", linewidth=2)
    ax.set_xlabel("True Divergence")
    ax.set_ylabel("Root Height")
    ax.set_title("Root Height vs True Divergence")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["expected_hamming"], df["actual_hamming"], "g-o", linewidth=2)
    ax.plot([0, 0.8], [0, 0.8], "k--", alpha=0.5)
    ax.set_xlabel("Expected Hamming (JC model)")
    ax.set_ylabel("Actual Hamming (cluster centroids)")
    ax.set_title("Expected vs Actual Hamming")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["avg_branch_length"], df["js_divergence"], "m-o", linewidth=2)
    ax.set_xlabel("Branch Length")
    ax.set_ylabel("JS Divergence")
    ax.set_title("Branch Length vs JS Divergence")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(repo_root / "results" / "branch_length_analysis.png", dpi=150)
    print(f"\nPlot saved to: results/branch_length_analysis.png")


if __name__ == "__main__":
    main()
