#!/usr/bin/env python3
"""Quick diagnostic: see what K and ARI each SMALL_TEST_CASE produces."""
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from benchmarks.shared.pipeline import benchmark_cluster_algorithm

SMALL_TEST_CASES = [
    {
        "name": "clear",
        "n_samples": 24,
        "n_features": 12,
        "n_clusters": 3,
        "cluster_std": 0.4,
        "seed": 0,
    },
    {
        "name": "moderate",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.0,
        "seed": 1,
    },
    {
        "name": "noisy",
        "n_samples": 30,
        "n_features": 16,
        "n_clusters": 3,
        "cluster_std": 1.6,
        "seed": 2,
    },
]

df, _ = benchmark_cluster_algorithm(
    test_cases=[c.copy() for c in SMALL_TEST_CASES],
    verbose=False,
    plot_umap=False,
    methods=["kl"],
)

kl = df[df["Method"] == "KL Divergence"]
print(kl[["Case_Name", "True", "Found", "ARI", "NMI", "Purity", "Status"]].to_string())
print(kl[["Case_Name", "True", "Found", "ARI", "NMI", "Purity", "Status"]].to_string())
