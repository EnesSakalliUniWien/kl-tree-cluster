"""
Analyze Test Case Quality

This script evaluates the current test cases to identify:
1. Redundancy (similar cases that don't add value)
2. Gaps (missing important scenarios)
3. Quality issues (cases where ground truth may be ambiguous)
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_cases_config import DEFAULT_TEST_CASES_CONFIG, get_default_test_cases
from kl_clustering_analysis.benchmarking.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
from scipy.stats import chi2
from scipy.linalg import qr

print("=" * 80)
print("TEST CASE QUALITY ANALYSIS")
print("=" * 80)

# Get all test cases
all_cases = get_default_test_cases()
print(f"\nTotal test cases: {len(all_cases)}")

# Categorize by generator
gaussian_cases = [c for c in all_cases if c.get("generator", "blobs") == "blobs"]
binary_cases = [c for c in all_cases if c.get("generator") == "binary"]

print(f"  Gaussian (blobs): {len(gaussian_cases)}")
print(f"  Binary: {len(binary_cases)}")

# Summarize by category
print("\n" + "=" * 80)
print("CATEGORY BREAKDOWN")
print("=" * 80)

for category, cases in DEFAULT_TEST_CASES_CONFIG.items():
    print(f"\n{category}: {len(cases)} cases")

    # Show parameter ranges
    if cases[0].get("generator") == "binary":
        n_rows = [c.get("n_rows", c.get("n_samples")) for c in cases]
        n_cols = [c.get("n_cols", c.get("n_features")) for c in cases]
        entropy = [c.get("entropy_param") for c in cases]
        n_clusters = [c.get("n_clusters") for c in cases]

        print(f"  n_rows: {min(n_rows)} - {max(n_rows)}")
        print(f"  n_cols: {min(n_cols)} - {max(n_cols)}")
        print(f"  entropy: {min(entropy):.2f} - {max(entropy):.2f}")
        print(f"  n_clusters: {min(n_clusters)} - {max(n_clusters)}")
    else:
        n_samples = [c.get("n_samples") for c in cases]
        n_features = [c.get("n_features") for c in cases]
        cluster_std = [c.get("cluster_std") for c in cases]
        n_clusters = [c.get("n_clusters") for c in cases]

        print(f"  n_samples: {min(n_samples)} - {max(n_samples)}")
        print(f"  n_features: {min(n_features)} - {max(n_features)}")
        print(f"  cluster_std: {min(cluster_std):.1f} - {max(cluster_std):.1f}")
        print(f"  n_clusters: {min(n_clusters)} - {max(n_clusters)}")


print("\n" + "=" * 80)
print("ISSUE 1: REDUNDANT CASES")
print("=" * 80)


# Find cases with identical parameters (except seed)
def case_signature(c):
    """Create a signature for a case (excluding seed)."""
    if c.get("generator") == "binary":
        return (
            "binary",
            c.get("n_rows", c.get("n_samples")),
            c.get("n_cols", c.get("n_features")),
            c.get("n_clusters"),
            c.get("entropy_param"),
            c.get("balanced_clusters", True),
            c.get("feature_sparsity"),
        )
    else:
        return (
            "blobs",
            c.get("n_samples"),
            c.get("n_features"),
            c.get("n_clusters"),
            c.get("cluster_std"),
        )


# Group by signature
from collections import defaultdict

signature_groups = defaultdict(list)
for i, c in enumerate(all_cases, 1):
    sig = case_signature(c)
    signature_groups[sig].append((i, c))

# Find duplicates
duplicates = {sig: cases for sig, cases in signature_groups.items() if len(cases) > 1}
print(f"\nFound {len(duplicates)} duplicate signatures:")
for sig, cases in duplicates.items():
    case_nums = [c[0] for c in cases]
    print(f"  Cases {case_nums}: {sig[:4]}...")  # Show first 4 elements


print("\n" + "=" * 80)
print("ISSUE 2: SIGNAL QUALITY IN BINARY CASES")
print("=" * 80)


def measure_signal_quality(case):
    """
    Measure actual signal strength in a binary test case.
    Returns ratio of test statistic to df (>1 means detectable signal).
    """
    if case.get("generator") != "binary":
        return None

    n_rows = case.get("n_rows", case.get("n_samples"))
    n_cols = case.get("n_cols", case.get("n_features"))
    n_clusters = case.get("n_clusters")
    entropy = case.get("entropy_param")
    seed = case.get("seed")

    # Generate data
    leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        n_clusters=n_clusters,
        entropy_param=entropy,
        feature_sparsity=case.get("feature_sparsity"),
        balanced_clusters=case.get("balanced_clusters", True),
        random_seed=seed,
    )

    # Convert to arrays
    sample_names = list(leaf_matrix_dict.keys())
    X = np.array([leaf_matrix_dict[name] for name in sample_names])
    y = np.array([cluster_assignments[name] for name in sample_names])

    # Test pairwise cluster separability
    unique_clusters = np.unique(y)
    if len(unique_clusters) < 2:
        return 0.0

    # Test first two clusters
    c0, c1 = unique_clusters[:2]
    X0 = X[y == c0]
    X1 = X[y == c1]

    # Standardize
    X_all = np.vstack([X0, X1])
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0)
    std[std < 1e-10] = 1.0

    X0_std = (X0 - mean) / std
    X1_std = (X1 - mean) / std

    # Random projection
    k = min(max(10, int(4 * np.log(n_rows))), n_cols)
    np.random.seed(42)
    G = np.random.standard_normal((k, n_cols))
    Q, _ = qr(G.T, mode="economic")
    R = Q.T

    # Project centroids
    z0 = R @ X0_std.mean(axis=0)
    z1 = R @ X1_std.mean(axis=0)

    # Test statistic
    n0, n1 = len(X0), len(X1)
    n_eff = n0 * n1 / (n0 + n1)
    diff = z0 - z1
    T = n_eff * np.sum(diff**2)

    ratio = T / k
    return ratio


print("\nMeasuring signal quality for binary cases...")
print("(ratio > 1 indicates detectable signal)\n")

weak_signal_cases = []
for i, case in enumerate(all_cases, 1):
    if case.get("generator") != "binary":
        continue

    ratio = measure_signal_quality(case)
    if ratio is not None:
        n_rows = case.get("n_rows")
        n_cols = case.get("n_cols")
        entropy = case.get("entropy_param")
        n_clusters = case.get("n_clusters")

        status = "✓" if ratio > 1.5 else ("~" if ratio > 1.0 else "✗")

        if ratio < 1.5:
            weak_signal_cases.append((i, case.get("name"), ratio, entropy))

        # Only show problematic cases
        if ratio < 1.5:
            print(
                f"Case {i:2d}: {case.get('name', 'unnamed'):30s} "
                f"({n_rows}x{n_cols}, {n_clusters}c, e={entropy:.2f}) "
                f"ratio={ratio:.2f} {status}"
            )

print(f"\n{len(weak_signal_cases)} cases with weak signal (ratio < 1.5)")


print("\n" + "=" * 80)
print("ISSUE 3: MISSING SCENARIOS")
print("=" * 80)

print("\nCurrent coverage:")
print("  ✓ Gaussian blobs with various noise levels")
print("  ✓ Binary balanced clusters")
print("  ✓ Binary unbalanced clusters")
print("  ✓ Sparse features (low-variance)")
print("  ✓ Various sizes (small to large)")

print("\nPotential gaps:")
print("  ? Hierarchical structure (clusters within clusters)")
print("  ? Non-spherical clusters (elongated, curved)")
print("  ? Very small samples per cluster (<10)")
print("  ? Two-cluster cases (important baseline)")
print("  ? Different cluster sizes within same dataset")


print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
1. REMOVE REDUNDANT CASES:
   - Many cases with identical parameters except seed
   - Keep one representative case per unique configuration

2. FIX WEAK-SIGNAL CASES:
   - Cases with entropy >= 0.25 often have no detectable signal
   - Consider lowering entropy to 0.15 or enabling feature selection

3. ADD MISSING SCENARIOS:
   - Add 2-cluster binary cases (important baseline)
   - Add very small cluster cases (n=5-10 per cluster)
   - Add hierarchical structure cases
   
4. IMPROVE PARAMETER COVERAGE:
   - More systematic grid over (samples, features, clusters, noise)
   - Current cases are somewhat arbitrary
""")

# Count actual unique configurations
unique_configs = len(signature_groups)
print(f"\nSUMMARY:")
print(f"  Total cases: {len(all_cases)}")
print(f"  Unique configurations: {unique_configs}")
print(
    f"  Potential reduction: {len(all_cases) - unique_configs} cases could be removed"
)
