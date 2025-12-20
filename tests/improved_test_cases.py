"""
PROPOSED: Improved Test Case Configurations

This module contains well-designed test cases that:
1. Have verifiable cluster structure (signal > noise)
2. Cover diverse scenarios systematically
3. Avoid redundancy
"""

# =============================================================================
# GAUSSIAN BLOB CASES - For continuous-to-binary testing
# =============================================================================

GAUSSIAN_TEST_CASES = [
    # Clear separation (should get 100% correct)
    {
        "name": "gauss_clear_small",
        "n_samples": 30,
        "n_features": 20,
        "n_clusters": 3,
        "cluster_std": 0.5,
        "seed": 100,
    },
    {
        "name": "gauss_clear_medium",
        "n_samples": 60,
        "n_features": 40,
        "n_clusters": 4,
        "cluster_std": 0.6,
        "seed": 101,
    },
    {
        "name": "gauss_clear_large",
        "n_samples": 100,
        "n_features": 80,
        "n_clusters": 5,
        "cluster_std": 0.7,
        "seed": 102,
    },
    # Moderate overlap (80-95% ARI expected)
    {
        "name": "gauss_moderate_3c",
        "n_samples": 45,
        "n_features": 30,
        "n_clusters": 3,
        "cluster_std": 1.2,
        "seed": 200,
    },
    {
        "name": "gauss_moderate_5c",
        "n_samples": 75,
        "n_features": 50,
        "n_clusters": 5,
        "cluster_std": 1.5,
        "seed": 201,
    },
    # Challenging (50-80% ARI expected)
    {
        "name": "gauss_noisy_3c",
        "n_samples": 60,
        "n_features": 40,
        "n_clusters": 3,
        "cluster_std": 2.5,
        "seed": 300,
    },
    {
        "name": "gauss_noisy_many",
        "n_samples": 120,
        "n_features": 80,
        "n_clusters": 8,
        "cluster_std": 2.0,
        "seed": 301,
    },
]


# =============================================================================
# BINARY CASES - Core test suite
# =============================================================================

BINARY_TEST_CASES = [
    # -------------------------------------------------------------------------
    # TIER 1: Perfect separation (entropy=0, should get 100% ARI)
    # -------------------------------------------------------------------------
    {
        "name": "binary_perfect_2c",
        "generator": "binary",
        "n_rows": 40,
        "n_cols": 50,
        "n_clusters": 2,
        "entropy_param": 0.00,
        "balanced_clusters": True,
        "seed": 1000,
    },
    {
        "name": "binary_perfect_4c",
        "generator": "binary",
        "n_rows": 80,
        "n_cols": 100,
        "n_clusters": 4,
        "entropy_param": 0.00,
        "balanced_clusters": True,
        "seed": 1001,
    },
    {
        "name": "binary_perfect_8c",
        "generator": "binary",
        "n_rows": 160,
        "n_cols": 200,
        "n_clusters": 8,
        "entropy_param": 0.00,
        "balanced_clusters": True,
        "seed": 1002,
    },
    # -------------------------------------------------------------------------
    # TIER 2: Low noise (entropy=0.05-0.10, should get 95%+ ARI)
    # -------------------------------------------------------------------------
    {
        "name": "binary_low_noise_2c",
        "generator": "binary",
        "n_rows": 40,
        "n_cols": 50,
        "n_clusters": 2,
        "entropy_param": 0.05,
        "balanced_clusters": True,
        "seed": 2000,
    },
    {
        "name": "binary_low_noise_4c",
        "generator": "binary",
        "n_rows": 80,
        "n_cols": 100,
        "n_clusters": 4,
        "entropy_param": 0.05,
        "balanced_clusters": True,
        "seed": 2001,
    },
    {
        "name": "binary_low_noise_8c",
        "generator": "binary",
        "n_rows": 160,
        "n_cols": 200,
        "n_clusters": 8,
        "entropy_param": 0.10,
        "balanced_clusters": True,
        "seed": 2002,
    },
    {
        "name": "binary_low_noise_12c",
        "generator": "binary",
        "n_rows": 240,
        "n_cols": 300,
        "n_clusters": 12,
        "entropy_param": 0.10,
        "balanced_clusters": True,
        "seed": 2003,
    },
    # -------------------------------------------------------------------------
    # TIER 3: Moderate noise (entropy=0.12-0.18, should get 80-95% ARI)
    # -------------------------------------------------------------------------
    {
        "name": "binary_moderate_4c",
        "generator": "binary",
        "n_rows": 80,
        "n_cols": 100,
        "n_clusters": 4,
        "entropy_param": 0.12,
        "balanced_clusters": True,
        "seed": 3000,
    },
    {
        "name": "binary_moderate_6c",
        "generator": "binary",
        "n_rows": 120,
        "n_cols": 150,
        "n_clusters": 6,
        "entropy_param": 0.15,
        "balanced_clusters": True,
        "seed": 3001,
    },
    {
        "name": "binary_moderate_8c",
        "generator": "binary",
        "n_rows": 200,
        "n_cols": 250,
        "n_clusters": 8,
        "entropy_param": 0.12,
        "balanced_clusters": True,
        "seed": 3002,
    },
    {
        "name": "binary_moderate_many_feat",
        "generator": "binary",
        "n_rows": 100,
        "n_cols": 1000,
        "n_clusters": 4,
        "entropy_param": 0.15,
        "balanced_clusters": True,
        "seed": 3003,
    },
    # -------------------------------------------------------------------------
    # TIER 4: Challenging (entropy=0.20, may need feature selection)
    # -------------------------------------------------------------------------
    {
        "name": "binary_hard_4c",
        "generator": "binary",
        "n_rows": 100,
        "n_cols": 200,
        "n_clusters": 4,
        "entropy_param": 0.20,
        "balanced_clusters": True,
        "seed": 4000,
    },
    {
        "name": "binary_hard_8c",
        "generator": "binary",
        "n_rows": 280,
        "n_cols": 400,
        "n_clusters": 8,
        "entropy_param": 0.15,
        "balanced_clusters": True,
        "seed": 4001,
    },
    # -------------------------------------------------------------------------
    # UNBALANCED CLUSTERS
    # -------------------------------------------------------------------------
    {
        "name": "binary_unbalanced_low",
        "generator": "binary",
        "n_rows": 100,
        "n_cols": 150,
        "n_clusters": 4,
        "entropy_param": 0.10,
        "balanced_clusters": False,
        "seed": 5000,
    },
    {
        "name": "binary_unbalanced_med",
        "generator": "binary",
        "n_rows": 150,
        "n_cols": 200,
        "n_clusters": 6,
        "entropy_param": 0.12,
        "balanced_clusters": False,
        "seed": 5001,
    },
    {
        "name": "binary_unbalanced_hard",
        "generator": "binary",
        "n_rows": 200,
        "n_cols": 300,
        "n_clusters": 8,
        "entropy_param": 0.15,
        "balanced_clusters": False,
        "seed": 5002,
    },
    # -------------------------------------------------------------------------
    # SPARSE FEATURES (low variance, tests df weighting)
    # -------------------------------------------------------------------------
    {
        "name": "binary_sparse_low",
        "generator": "binary",
        "n_rows": 80,
        "n_cols": 100,
        "n_clusters": 4,
        "entropy_param": 0.10,
        "feature_sparsity": 0.05,
        "balanced_clusters": True,
        "seed": 6000,
    },
    {
        "name": "binary_sparse_med",
        "generator": "binary",
        "n_rows": 100,
        "n_cols": 200,
        "n_clusters": 4,
        "entropy_param": 0.12,
        "feature_sparsity": 0.10,
        "balanced_clusters": True,
        "seed": 6001,
    },
    {
        "name": "binary_sparse_high",
        "generator": "binary",
        "n_rows": 100,
        "n_cols": 500,
        "n_clusters": 4,
        "entropy_param": 0.10,
        "feature_sparsity": 0.05,
        "balanced_clusters": True,
        "seed": 6002,
    },
    # -------------------------------------------------------------------------
    # EDGE CASES
    # -------------------------------------------------------------------------
    {
        "name": "binary_2clusters",
        "generator": "binary",
        "n_rows": 50,
        "n_cols": 60,
        "n_clusters": 2,
        "entropy_param": 0.10,
        "balanced_clusters": True,
        "seed": 7000,
    },
    {
        "name": "binary_small_samples",
        "generator": "binary",
        "n_rows": 40,
        "n_cols": 80,
        "n_clusters": 4,
        "entropy_param": 0.08,
        "balanced_clusters": True,
        "seed": 7001,
    },  # 10 per cluster
    {
        "name": "binary_many_features",
        "generator": "binary",
        "n_rows": 100,
        "n_cols": 2000,
        "n_clusters": 4,
        "entropy_param": 0.10,
        "balanced_clusters": True,
        "seed": 7002,
    },
    {
        "name": "binary_many_clusters",
        "generator": "binary",
        "n_rows": 300,
        "n_cols": 400,
        "n_clusters": 15,
        "entropy_param": 0.10,
        "balanced_clusters": True,
        "seed": 7003,
    },
]


# =============================================================================
# COMBINED TEST SUITE
# =============================================================================

IMPROVED_TEST_CASES = GAUSSIAN_TEST_CASES + BINARY_TEST_CASES


def get_improved_test_cases():
    """Get the improved test case suite."""
    return [c.copy() for c in IMPROVED_TEST_CASES]


# Summary
if __name__ == "__main__":
    print("IMPROVED TEST CASES SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(IMPROVED_TEST_CASES)}")
    print(f"  Gaussian: {len(GAUSSIAN_TEST_CASES)}")
    print(f"  Binary: {len(BINARY_TEST_CASES)}")

    print("\nBinary cases by entropy:")
    for e in [0.00, 0.05, 0.10, 0.12, 0.15, 0.20]:
        count = sum(1 for c in BINARY_TEST_CASES if c.get("entropy_param") == e)
        if count > 0:
            print(f"  entropy={e:.2f}: {count} cases")

    print("\nBinary cases by cluster count:")
    for nc in sorted(set(c.get("n_clusters") for c in BINARY_TEST_CASES)):
        count = sum(1 for c in BINARY_TEST_CASES if c.get("n_clusters") == nc)
        print(f"  {nc} clusters: {count} cases")
