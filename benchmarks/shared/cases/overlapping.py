"""Overlapping cluster test cases for benchmarking.

Tests the algorithm's ability to handle clusters with significant overlap.
Includes both binary (entropy-based) and Gaussian (std-based) overlap cases.
"""

OVERLAPPING_BINARY_HEAVY_CASES = [
    # Heavy overlap (entropy 0.40-0.48) - nearly random, almost no signal
    {
        "name": "overlap_heavy_4c_small_feat",
        "generator": "binary",
        "n_rows": 500,
        "n_cols": 50,
        "n_clusters": 4,
        "entropy_param": 0.40,
        "balanced_clusters": True,
        "seed": 8000,
    },
    {
        "name": "overlap_heavy_4c_med_feat",
        "generator": "binary",
        "n_rows": 500,
        "n_cols": 200,
        "n_clusters": 4,
        "entropy_param": 0.40,
        "balanced_clusters": True,
        "seed": 8001,
    },
    {
        "name": "overlap_heavy_8c_large_feat",
        "generator": "binary",
        "n_rows": 800,
        "n_cols": 500,
        "n_clusters": 8,
        "entropy_param": 0.40,
        "balanced_clusters": True,
        "seed": 8002,
    },
    {
        "name": "overlap_extreme_4c",
        "generator": "binary",
        "n_rows": 600,
        "n_cols": 150,
        "n_clusters": 4,
        "entropy_param": 0.48,
        "balanced_clusters": True,
        "seed": 8003,
    },
    {
        "name": "overlap_extreme_6c_highd",
        "generator": "binary",
        "n_rows": 600,
        "n_cols": 1000,
        "n_clusters": 6,
        "entropy_param": 0.45,
        "balanced_clusters": True,
        "seed": 8004,
    },
]

OVERLAPPING_BINARY_MODERATE_CASES = [
    # Moderate overlap (entropy 0.28-0.35) - some structure, many merges expected
    {
        "name": "overlap_mod_4c_small",
        "generator": "binary",
        "n_rows": 400,
        "n_cols": 80,
        "n_clusters": 4,
        "entropy_param": 0.30,
        "balanced_clusters": True,
        "seed": 8100,
    },
    {
        "name": "overlap_mod_6c_med",
        "generator": "binary",
        "n_rows": 600,
        "n_cols": 250,
        "n_clusters": 6,
        "entropy_param": 0.30,
        "balanced_clusters": True,
        "seed": 8101,
    },
    {
        "name": "overlap_mod_8c_large",
        "generator": "binary",
        "n_rows": 800,
        "n_cols": 400,
        "n_clusters": 8,
        "entropy_param": 0.28,
        "balanced_clusters": True,
        "seed": 8102,
    },
    {
        "name": "overlap_mod_10c_highd",
        "generator": "binary",
        "n_rows": 1000,
        "n_cols": 800,
        "n_clusters": 10,
        "entropy_param": 0.32,
        "balanced_clusters": True,
        "seed": 8103,
    },
]

OVERLAPPING_BINARY_PARTIAL_CASES = [
    # Partial overlap (entropy 0.22-0.26) - some clusters merge, others separate
    {
        "name": "overlap_part_4c_small",
        "generator": "binary",
        "n_rows": 400,
        "n_cols": 60,
        "n_clusters": 4,
        "entropy_param": 0.24,
        "balanced_clusters": True,
        "seed": 8200,
    },
    {
        "name": "overlap_part_6c_med",
        "generator": "binary",
        "n_rows": 600,
        "n_cols": 200,
        "n_clusters": 6,
        "entropy_param": 0.25,
        "balanced_clusters": True,
        "seed": 8201,
    },
    {
        "name": "overlap_part_8c_large",
        "generator": "binary",
        "n_rows": 800,
        "n_cols": 350,
        "n_clusters": 8,
        "entropy_param": 0.22,
        "balanced_clusters": True,
        "seed": 8202,
    },
    {
        "name": "overlap_part_10c_highd",
        "generator": "binary",
        "n_rows": 1000,
        "n_cols": 600,
        "n_clusters": 10,
        "entropy_param": 0.23,
        "balanced_clusters": True,
        "seed": 8203,
    },
]

OVERLAPPING_BINARY_HIGHD_CASES = [
    # High-dimensional overlap - tests overlap in high feature space
    {
        "name": "overlap_hd_4c_1k",
        "generator": "binary",
        "n_rows": 500,
        "n_cols": 1000,
        "n_clusters": 4,
        "entropy_param": 0.35,
        "balanced_clusters": True,
        "seed": 8300,
    },
    {
        "name": "overlap_hd_6c_2k",
        "generator": "binary",
        "n_rows": 600,
        "n_cols": 2000,
        "n_clusters": 6,
        "entropy_param": 0.33,
        "balanced_clusters": True,
        "seed": 8301,
    },
    {
        "name": "overlap_hd_8c_3k",
        "generator": "binary",
        "n_rows": 800,
        "n_cols": 3000,
        "n_clusters": 8,
        "entropy_param": 0.30,
        "balanced_clusters": True,
        "seed": 8302,
    },
    {
        "name": "overlap_hd_10c_5k",
        "generator": "binary",
        "n_rows": 1000,
        "n_cols": 5000,
        "n_clusters": 10,
        "entropy_param": 0.28,
        "balanced_clusters": True,
        "seed": 8303,
    },
]

OVERLAPPING_BINARY_UNBALANCED_CASES = [
    # Unbalanced overlapping - small clusters get lost in overlap
    {
        "name": "overlap_unbal_4c_small",
        "generator": "binary",
        "n_rows": 400,
        "n_cols": 100,
        "n_clusters": 4,
        "entropy_param": 0.35,
        "balanced_clusters": False,
        "seed": 8400,
    },
    {
        "name": "overlap_unbal_6c_med",
        "generator": "binary",
        "n_rows": 600,
        "n_cols": 300,
        "n_clusters": 6,
        "entropy_param": 0.30,
        "balanced_clusters": False,
        "seed": 8401,
    },
    {
        "name": "overlap_unbal_8c_large",
        "generator": "binary",
        "n_rows": 800,
        "n_cols": 500,
        "n_clusters": 8,
        "entropy_param": 0.32,
        "balanced_clusters": False,
        "seed": 8402,
    },
    {
        "name": "overlap_unbal_10c_highd",
        "generator": "binary",
        "n_rows": 1000,
        "n_cols": 1000,
        "n_clusters": 10,
        "entropy_param": 0.28,
        "balanced_clusters": False,
        "seed": 8403,
    },
]

OVERLAPPING_GAUSSIAN_CASES = [
    # High cluster_std means Gaussian blobs overlap significantly
    {
        "name": "gauss_overlap_3c_small",
        "n_samples": 300,
        "n_features": 30,
        "n_clusters": 3,
        "cluster_std": 3.5,
        "seed": 9000,
    },
    {
        "name": "gauss_overlap_4c_med",
        "n_samples": 400,
        "n_features": 80,
        "n_clusters": 4,
        "cluster_std": 4.0,
        "seed": 9001,
    },
    {
        "name": "gauss_overlap_6c_large",
        "n_samples": 600,
        "n_features": 150,
        "n_clusters": 6,
        "cluster_std": 3.5,
        "seed": 9002,
    },
    {
        "name": "gauss_overlap_8c_highd",
        "n_samples": 800,
        "n_features": 300,
        "n_clusters": 8,
        "cluster_std": 3.5,
        "seed": 9003,
    },
    # Extreme overlap - essentially uniform distribution
    {
        "name": "gauss_overlap_extreme_4c",
        "n_samples": 500,
        "n_features": 100,
        "n_clusters": 4,
        "cluster_std": 6.0,
        "seed": 9100,
    },
    {
        "name": "gauss_overlap_extreme_6c",
        "n_samples": 600,
        "n_features": 200,
        "n_clusters": 6,
        "cluster_std": 5.5,
        "seed": 9101,
    },
    {
        "name": "gauss_overlap_many_10c",
        "n_samples": 1000,
        "n_features": 150,
        "n_clusters": 10,
        "cluster_std": 4.0,
        "seed": 9102,
    },
    {
        "name": "gauss_overlap_dense_8c",
        "n_samples": 800,
        "n_features": 50,
        "n_clusters": 8,
        "cluster_std": 3.0,
        "seed": 9103,
    },
]


def get_overlapping_cases():
    """Get all overlapping test cases as a flat list."""
    return (
        [c.copy() for c in OVERLAPPING_BINARY_HEAVY_CASES]
        + [c.copy() for c in OVERLAPPING_BINARY_MODERATE_CASES]
        + [c.copy() for c in OVERLAPPING_BINARY_PARTIAL_CASES]
        + [c.copy() for c in OVERLAPPING_BINARY_HIGHD_CASES]
        + [c.copy() for c in OVERLAPPING_BINARY_UNBALANCED_CASES]
        + [c.copy() for c in OVERLAPPING_GAUSSIAN_CASES]
    )


# Category mapping for grouped access
OVERLAPPING_CASES = {
    "overlapping_binary_heavy": OVERLAPPING_BINARY_HEAVY_CASES,
    "overlapping_binary_moderate": OVERLAPPING_BINARY_MODERATE_CASES,
    "overlapping_binary_partial": OVERLAPPING_BINARY_PARTIAL_CASES,
    "overlapping_binary_highd": OVERLAPPING_BINARY_HIGHD_CASES,
    "overlapping_binary_unbalanced": OVERLAPPING_BINARY_UNBALANCED_CASES,
    "overlapping_gaussian": OVERLAPPING_GAUSSIAN_CASES,
}
