"""Isolate and run the gaussian_extreme_noise case #2 (n=300, p=2000, K=30)."""

import sys
import time

from benchmarks.shared.pipeline import benchmark_cluster_algorithm

case = {
    "name": "gaussian_extreme_noise_2",
    "n_samples": 300,
    "n_features": 2000,
    "n_clusters": 30,
    "cluster_std": 2,
    "seed": 44,
}

print(f"Running isolated case: {case['name']}")
print(f"  n={case['n_samples']}, p={case['n_features']}, K={case['n_clusters']}, std={case['cluster_std']}")
t0 = time.perf_counter()

df, fig = benchmark_cluster_algorithm(
    test_cases=[case],
    methods=["kl"],
    verbose=True,
    plot_umap=False,
)

elapsed = time.perf_counter() - t0
print(f"\nCompleted in {elapsed:.1f}s")
print(df.to_string())
