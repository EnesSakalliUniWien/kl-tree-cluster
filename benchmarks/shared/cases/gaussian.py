"""Gaussian cluster test cases.

Gaussian blobs are generated via sklearn.make_blobs and then median-binarized
per feature: (X > median(X, axis=0)).astype(int).

The improved_gaussian series provides a well-structured progressive
difficulty strategy.  The extreme_noise cases are stress tests where signal
is deliberately buried in noise.
"""

from benchmarks.shared.cases.representations import representation_variants_by_name

GAUSSIAN_CASES = {
    "gaussian_extreme_noise": [
        {
            "name": "gauss_extreme_noise_3c",
            "generator": "blobs",
            "n_samples": 30,
            "n_features": 30,
            "n_clusters": 3,
            "cluster_std": 5.0,
            "seed": 42,
        },
        {
            "name": "gauss_extreme_noise_highd",
            "generator": "blobs",
            "n_samples": 40,
            "n_features": 20000,
            "n_clusters": 4,
            "cluster_std": 7.5,
            "seed": 43,
        },
        {
            "name": "gauss_extreme_noise_many",
            "generator": "blobs",
            "n_samples": 300,
            "n_features": 2000,
            "n_clusters": 30,
            "cluster_std": 2,
            "seed": 44,
        },
    ],
    "improved_gaussian": [
        # Clear separation (should get 100% correct)
        {
            "name": "gauss_clear_small",
            "generator": "blobs",
            "n_samples": 30,
            "n_features": 20,
            "n_clusters": 3,
            "cluster_std": 0.5,
            "seed": 100,
        },
        {
            "name": "gauss_clear_medium",
            "generator": "blobs",
            "n_samples": 60,
            "n_features": 40,
            "n_clusters": 4,
            "cluster_std": 0.6,
            "seed": 101,
        },
        {
            "name": "gauss_clear_large",
            "generator": "blobs",
            "n_samples": 100,
            "n_features": 80,
            "n_clusters": 5,
            "cluster_std": 0.7,
            "seed": 102,
        },
        # Moderate overlap (80-95% ARI expected)
        {
            "name": "gauss_moderate_3c",
            "generator": "blobs",
            "n_samples": 45,
            "n_features": 30,
            "n_clusters": 3,
            "cluster_std": 1.2,
            "seed": 200,
        },
        {
            "name": "gauss_moderate_5c",
            "generator": "blobs",
            "n_samples": 75,
            "n_features": 50,
            "n_clusters": 5,
            "cluster_std": 1.5,
            "seed": 201,
        },
        # Challenging (50-80% ARI expected)
        {
            "name": "gauss_noisy_3c",
            "generator": "blobs",
            "n_samples": 60,
            "n_features": 40,
            "n_clusters": 3,
            "cluster_std": 2.5,
            "seed": 300,
        },
        {
            "name": "gauss_noisy_many",
            "generator": "blobs",
            "n_samples": 120,
            "n_features": 80,
            "n_clusters": 8,
            "cluster_std": 2.0,
            "seed": 301,
        },
    ],
    # ---- Null case (K=1): no cluster structure ----
    "gaussian_null": [
        {
            "name": "gauss_null_small",
            "generator": "blobs",
            "n_samples": 60,
            "n_features": 30,
            "n_clusters": 1,
            "cluster_std": 1.0,
            "seed": 400,
        },
        {
            "name": "gauss_null_large",
            "generator": "blobs",
            "n_samples": 200,
            "n_features": 80,
            "n_clusters": 1,
            "cluster_std": 1.0,
            "seed": 401,
        },
    ],
}


GAUSSIAN_CASES["continuous_gaussian_examples"] = [
    *representation_variants_by_name(
        GAUSSIAN_CASES["improved_gaussian"],
        selected_names=("gauss_clear_medium", "gauss_moderate_3c"),
        generator="blobs_continuous",
        name_suffix="_continuous",
        representation_role="continuous_example",
    ),
    *representation_variants_by_name(
        GAUSSIAN_CASES["gaussian_null"],
        selected_names=("gauss_null_large",),
        generator="blobs_continuous",
        name_suffix="_continuous",
        representation_role="continuous_null_example",
    ),
    *representation_variants_by_name(
        GAUSSIAN_CASES["gaussian_extreme_noise"],
        selected_names=("gauss_extreme_noise_highd",),
        generator="blobs_continuous",
        name_suffix="_continuous",
        representation_role="continuous_stress_example",
    ),
]
