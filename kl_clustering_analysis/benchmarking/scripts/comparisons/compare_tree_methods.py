"""Compare Tree Inference Methods for Binary Data.

Evaluates different hierarchical clustering approaches to find
the best tree construction method for the KL-TE algorithm.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances
import sys
import os
from pathlib import Path
from itertools import product

sys.path.insert(0, ".")

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "comparisons"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DISTANCE METRICS FOR BINARY DATA
# ============================================================


def jaccard_distance(X):
    """Jaccard distance: 1 - |A∩B| / |A∪B|. Good for sparse binary."""
    return pdist(X, metric="jaccard")


def dice_distance(X):
    """Dice/Sørensen: 1 - 2|A∩B| / (|A| + |B|). Emphasizes overlap."""
    return pdist(X, metric="dice")


def rogerstanimoto_distance(X):
    """Rogers-Tanimoto: considers both matches and mismatches."""
    return pdist(X, metric="rogerstanimoto")


def hamming_distance(X):
    """Hamming: proportion of differing bits."""
    return pdist(X, metric="hamming")


def sokalsneath_distance(X):
    """Sokal-Sneath: weights double presence more."""
    return pdist(X, metric="sokalsneath")


def mutual_info_distance(X):
    """Distance based on normalized mutual information."""
    n = X.shape[0]
    mi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Treat each row as a discrete distribution
            mi = mutual_info_score(X[i], X[j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    # Convert to distance (higher MI = lower distance)
    max_mi = mi_matrix.max() if mi_matrix.max() > 0 else 1
    dist_matrix = 1 - (mi_matrix / max_mi)
    return squareform(dist_matrix)


def kl_distance(X):
    """Symmetric KL divergence between row distributions."""
    n = X.shape[0]
    # Add small epsilon for stability
    eps = 1e-10
    X_prob = (X + eps) / (X + eps).sum(axis=1, keepdims=True)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Symmetric KL
            kl_ij = np.sum(X_prob[i] * np.log(X_prob[i] / X_prob[j]))
            kl_ji = np.sum(X_prob[j] * np.log(X_prob[j] / X_prob[i]))
            dist_matrix[i, j] = dist_matrix[j, i] = (kl_ij + kl_ji) / 2
    return squareform(dist_matrix)


# ============================================================
# TREE INFERENCE CONFIGURATIONS
# ============================================================

DISTANCE_METRICS = {
    "hamming": hamming_distance,
    "jaccard": jaccard_distance,
    "dice": dice_distance,
    "rogers_tanimoto": rogerstanimoto_distance,
    # 'mutual_info': mutual_info_distance,  # Slow for large n
}

LINKAGE_METHODS = ["complete", "average", "ward", "single"]


def run_clustering_with_config(data_df, distance_func, linkage_method):
    """Run clustering with specified distance metric and linkage."""
    try:
        X = data_df.values.astype(float)

        # Compute distance
        if linkage_method == "ward":
            # Ward requires Euclidean distance
            dist = pdist(X, metric="euclidean")
        else:
            dist = distance_func(X)

        # Handle NaN/Inf
        dist = np.nan_to_num(dist, nan=0.5, posinf=1.0, neginf=0.0)

        # Build tree
        Z = linkage(dist, method=linkage_method)
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
        decomposition = tree.decompose(leaf_data=data_df)

        # Extract clusters
        cluster_assignments = decomposition.get("cluster_assignments", {})

        label_map = {}
        for cl_id, info in cluster_assignments.items():
            for leaf in info["leaves"]:
                label_map[leaf] = cl_id

        labels = np.array([label_map.get(name, 0) for name in data_df.index])
        n_found = len(cluster_assignments) if cluster_assignments else 1

        return labels, n_found

    except Exception as e:
        print(f"      Error: {e}")
        return np.zeros(len(data_df), dtype=int), 1


def evaluate_configuration(
    distance_name, linkage_method, n_samples, n_features, n_clusters, noise, seed
):
    """Evaluate a single configuration."""
    data_dict, cluster_dict = generate_random_feature_matrix(
        n_rows=n_samples,
        n_cols=n_features,
        n_clusters=n_clusters,
        entropy_param=noise,
        balanced_clusters=True,
        random_seed=seed,
    )

    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
    true_labels = np.array([cluster_dict[name] for name in data_df.index])

    distance_func = DISTANCE_METRICS[distance_name]
    pred_labels, n_found = run_clustering_with_config(
        data_df, distance_func, linkage_method
    )

    ari = adjusted_rand_score(true_labels, pred_labels)

    return {
        "distance": distance_name,
        "linkage": linkage_method,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": n_clusters,
        "noise": noise,
        "seed": seed,
        "ari": ari,
        "n_found": n_found,
        "k_error": abs(n_found - n_clusters),
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TREE INFERENCE METHOD COMPARISON")
    print("=" * 70)

    # Test configurations
    N_SAMPLES = [50, 100, 150]
    N_FEATURES = [30, 50, 100]
    N_CLUSTERS = [2, 3, 4]
    NOISE_LEVELS = [0.05, 0.10, 0.15]
    N_SEEDS = 3

    configs = list(
        product(
            DISTANCE_METRICS.keys(),
            LINKAGE_METHODS,
            N_SAMPLES,
            N_FEATURES,
            N_CLUSTERS,
            NOISE_LEVELS,
            range(N_SEEDS),
        )
    )

    print(f"\nConfigurations to test: {len(configs)}")
    print(f"  Distance metrics: {list(DISTANCE_METRICS.keys())}")
    print(f"  Linkage methods: {LINKAGE_METHODS}")

    # Run evaluation
    results = []
    for i, (dist, link, n_s, n_f, n_c, noise, seed) in enumerate(configs):
        result = evaluate_configuration(dist, link, n_s, n_f, n_c, noise, seed)
        results.append(result)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(configs)} done...")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "tree_method_comparison.csv", index=False)

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS BY METHOD")
    print("=" * 70)

    # Summary by distance + linkage
    summary = (
        df.groupby(["distance", "linkage"])
        .agg(
            {
                "ari": ["mean", "std"],
                "k_error": "mean",
            }
        )
        .round(3)
    )
    summary.columns = ["Mean ARI", "Std", "Mean K Error"]
    summary = summary.sort_values("Mean ARI", ascending=False)
    print("\n" + summary.to_string())

    # Best method
    best_idx = summary["Mean ARI"].idxmax()
    print(
        f"\n✅ BEST METHOD: {best_idx[0]} + {best_idx[1]} (ARI = {summary.loc[best_idx, 'Mean ARI']:.3f})"
    )

    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Heatmap of distance × linkage
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df.groupby(["distance", "linkage"])["ari"].mean().unstack()
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xlabel("Linkage Method", fontsize=12)
    ax.set_ylabel("Distance Metric", fontsize=12)
    ax.set_title("Mean ARI by Tree Inference Method", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="ARI")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tree_method_heatmap.png", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'tree_method_heatmap.png'}")
    plt.close()

    # 2. Performance by noise level
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best methods by noise
    ax = axes[0]
    top_methods = summary.nlargest(4, "Mean ARI").index.tolist()
    for dist, link in top_methods:
        subset = df[(df["distance"] == dist) & (df["linkage"] == link)]
        means = subset.groupby("noise")["ari"].mean()
        ax.plot(
            means.index * 100,
            means.values,
            "o-",
            label=f"{dist}+{link}",
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("Mean ARI", fontsize=12)
    ax.set_title("Top 4 Methods: Noise Tolerance", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # K accuracy
    ax = axes[1]
    for dist, link in top_methods:
        subset = df[(df["distance"] == dist) & (df["linkage"] == link)]
        means = subset.groupby("noise")["k_error"].mean()
        ax.plot(
            means.index * 100,
            means.values,
            "o-",
            label=f"{dist}+{link}",
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("Mean |k_found - k_true|", fontsize=12)
    ax.set_title("Cluster Count Accuracy", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tree_method_comparison.png", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'tree_method_comparison.png'}")
    plt.close()

    # 3. Bar chart of all methods
    fig, ax = plt.subplots(figsize=(12, 6))

    method_ari = (
        df.groupby(["distance", "linkage"])["ari"].mean().sort_values(ascending=True)
    )
    colors = plt.cm.RdYlGn(method_ari.values)

    y_pos = range(len(method_ari))
    bars = ax.barh(y_pos, method_ari.values, color=colors, edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{d} + {l}" for d, l in method_ari.index])
    ax.set_xlabel("Mean ARI", fontsize=12)
    ax.set_title(
        "Tree Inference Methods Ranked by Performance", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, method_ari.values):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "tree_method_ranking.png", dpi=150)
    print(f"  Saved: {RESULTS_DIR / 'tree_method_ranking.png'}")
    plt.close()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
Based on the comparison:

1. BEST: {best_idx[0]} distance + {best_idx[1]} linkage
   - Mean ARI: {summary.loc[best_idx, "Mean ARI"]:.3f}
   
2. Consider switching from current 'hamming + complete' if another method ranks higher.

3. For binary data:
   - Jaccard: Better for sparse data (ignores shared 0s)
   - Dice: Emphasizes shared 1s more than Jaccard
   - Ward: Good compact clusters but assumes Euclidean

4. Linkage effects:
   - Complete: Tends to create compact, equally-sized clusters
   - Average: More balanced, commonly used in phylogenetics  
   - Ward: Minimizes variance, very compact clusters
   - Single: Creates elongated chains (usually avoid)
""")
