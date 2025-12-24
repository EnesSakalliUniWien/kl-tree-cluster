#!/usr/bin/env python3
"""
Simple diagnostic: Run one test case and check if global weights are computed.
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from kl_clustering_analysis import config

print(
    f"Config: USE_GLOBAL_DIVERGENCE_WEIGHTING = {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}"
)
print(f"Config: GLOBAL_WEIGHT_METHOD = {config.GLOBAL_WEIGHT_METHOD}")
print()

# Run a single simple test case
from kl_clustering_analysis.benchmarking.clustering_test_suite import generate_2d_spiral
from kl_clustering_analysis.benchmarking.run_clustering_benchmark import (
    test_clustering_algorithm,
)

# Generate simple test data
X, labels = generate_2d_spiral(n_samples=200, n_turns=1.5, noise=0.05)

print("Running clustering test...")
result = test_clustering_algorithm(
    X=X,
    true_labels=labels,
    method="KL Divergence",
    test_name="Diagnostic Test",
    case_name="Simple Spiral",
)

print(f"\nResult:")
print(f"  True clusters: {result['true_clusters']}")
print(f"  Found clusters: {result['found_clusters']}")
print(f"  ARI: {result['ari']:.4f}")
print(f"  Status: {result['status']}")

# Now check if we can access the tree to see global weights
print("\nChecking for global weight computation...")
print("(This would require modifying the benchmark to return the tree)")
