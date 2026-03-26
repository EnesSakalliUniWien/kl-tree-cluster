#!/usr/bin/env python3
"""Check whether posthoc merge changes affect the 'clear' case under the production sibling test."""
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")


# Reproduce exact 'clear' test case data generation
# Need to check how benchmark_cluster_algorithm generates data
from benchmarks.shared.pipeline import benchmark_cluster_algorithm
from kl_clustering_analysis import config

CLEAR_CASE = {
    "name": "clear",
    "n_samples": 24,
    "n_features": 12,
    "n_clusters": 3,
    "cluster_std": 0.4,
    "seed": 0,
}

orig_sibling_method = config.SIBLING_TEST_METHOD
orig_posthoc_merge = config.POSTHOC_MERGE

config.SIBLING_TEST_METHOD = "cousin_adjusted_wald"

for merge in [True, False]:
    config.POSTHOC_MERGE = merge
    df, _ = benchmark_cluster_algorithm(
        test_cases=[CLEAR_CASE.copy()],
        verbose=False,
        plot_umap=False,
        methods=["kl"],
    )
    kl = df[df["Method"] == "KL Divergence"]
    row = kl.iloc[0]
    print(
        f"method={config.SIBLING_TEST_METHOD:<25s} merge={merge!s:<6s}  K={row['Found']}/{row['True']}  ARI={row['ARI']:.3f}"
    )

# Restore defaults
config.SIBLING_TEST_METHOD = orig_sibling_method
config.POSTHOC_MERGE = orig_posthoc_merge
