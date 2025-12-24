"""Quick power analysis for the fixed df implementation."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def run_clustering(data_df):
    Z = linkage(
        pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomposition = tree.decompose(leaf_data=data_df)
    cluster_assignments = decomposition.get("cluster_assignments", {})

    label_map = {}
    cluster_id = 0
    for cl_key, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id
        cluster_id += 1

    labels = [label_map.get(name, 0) for name in data_df.index]
    n_found = len(cluster_assignments) if cluster_assignments else 1
    return np.array(labels), n_found


def main():
    print("=" * 70)
    print("QUICK POWER ANALYSIS (with fixed df)")
    print("=" * 70)

    # Smaller grids for quick test
    N_SAMPLES = [50, 100, 200]
    N_FEATURES = [30, 50, 100]
    N_CLUSTERS = [3, 4, 5]
    NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
    N_SEEDS = 2

    results = []
    total = (
        len(N_SAMPLES) * len(N_FEATURES) * len(N_CLUSTERS) * len(NOISE_LEVELS) * N_SEEDS
    )

    print(f"\nRunning {total} configurations...")

    for n_samples in N_SAMPLES:
        for n_features in N_FEATURES:
            for n_clusters in N_CLUSTERS:
                for noise in NOISE_LEVELS:
                    for seed in range(N_SEEDS):
                        data_dict, cluster_dict = generate_random_feature_matrix(
                            n_rows=n_samples,
                            n_cols=n_features,
                            n_clusters=n_clusters,
                            entropy_param=noise,
                            balanced_clusters=True,
                            random_seed=42 + seed,
                        )

                        data_df = pd.DataFrame.from_dict(
                            data_dict, orient="index"
                        ).astype(int)
                        true_labels = np.array(
                            [cluster_dict[name] for name in data_df.index]
                        )

                        pred_labels, n_found = run_clustering(data_df)
                        ari = adjusted_rand_score(true_labels, pred_labels)

                        results.append(
                            {
                                "n_samples": n_samples,
                                "n_features": n_features,
                                "n_clusters": n_clusters,
                                "noise": noise,
                                "seed": seed,
                                "ari": ari,
                                "n_found": n_found,
                                "perfect": 1 if ari > 0.99 else 0,
                            }
                        )

    df = pd.DataFrame(results)

    print(f"\nTotal configurations tested: {len(df)}")
    print(f"\nOverall Mean ARI: {df['ari'].mean():.3f}")
    print(f"Perfect Clustering Rate: {df['perfect'].mean() * 100:.1f}%")

    print("\n" + "-" * 70)
    print("BY NOISE LEVEL:")
    print("-" * 70)
    noise_summary = (
        df.groupby("noise")
        .agg(
            {
                "ari": ["mean", "std", "min", "max"],
                "perfect": "mean",
                "n_found": "mean",
            }
        )
        .round(3)
    )
    noise_summary.columns = ["Mean ARI", "Std", "Min", "Max", "Perfect %", "Mean k"]
    noise_summary["Perfect %"] = (noise_summary["Perfect %"] * 100).round(1)
    print(noise_summary.to_string())

    print("\n" + "-" * 70)
    print("BY SAMPLE SIZE:")
    print("-" * 70)
    sample_summary = (
        df.groupby("n_samples").agg({"ari": "mean", "perfect": "mean"}).round(3)
    )
    sample_summary.columns = ["Mean ARI", "Perfect %"]
    sample_summary["Perfect %"] = (sample_summary["Perfect %"] * 100).round(1)
    print(sample_summary.to_string())

    print("\n" + "-" * 70)
    print("BY FEATURE COUNT:")
    print("-" * 70)
    feature_summary = (
        df.groupby("n_features").agg({"ari": "mean", "perfect": "mean"}).round(3)
    )
    feature_summary.columns = ["Mean ARI", "Perfect %"]
    feature_summary["Perfect %"] = (feature_summary["Perfect %"] * 100).round(1)
    print(feature_summary.to_string())

    print("\n" + "-" * 70)
    print("BY CLUSTER COUNT:")
    print("-" * 70)
    cluster_summary = (
        df.groupby("n_clusters").agg({"ari": "mean", "perfect": "mean"}).round(3)
    )
    cluster_summary.columns = ["Mean ARI", "Perfect %"]
    cluster_summary["Perfect %"] = (cluster_summary["Perfect %"] * 100).round(1)
    print(cluster_summary.to_string())

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
