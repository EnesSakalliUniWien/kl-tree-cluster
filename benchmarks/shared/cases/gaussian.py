"""Gaussian cluster test cases."""

GAUSSIAN_CASES = {
    "gaussian_clear": [
        {
            "n_samples": 30,
            "n_features": 30,
            "n_clusters": 3,
            "cluster_std": 0.5,
            "seed": 100,
        },
        {
            "n_samples": 40,
            "n_features": 40,
            "n_clusters": 4,
            "cluster_std": 0.8,
            "seed": 101,
        },
        {
            "n_samples": 50,
            "n_features": 50,
            "n_clusters": 5,
            "cluster_std": 1.0,
            "seed": 102,
        },
    ],
    "gaussian_mixed": [
        {
            "n_samples": 35,
            "n_features": 35,
            "n_clusters": 3,
            "cluster_std": 1.2,
            "seed": 103,
        },
        {
            "n_samples": 45,
            "n_features": 45,
            "n_clusters": 4,
            "cluster_std": 1.5,
            "seed": 104,
        },
        {
            "n_samples": 55,
            "n_features": 55,
            "n_clusters": 5,
            "cluster_std": 1.8,
            "seed": 105,
        },
        {
            "n_samples": 25,
            "n_features": 25,
            "n_clusters": 2,
            "cluster_std": 0.7,
            "seed": 106,
        },
        {
            "n_samples": 65,
            "n_features": 65,
            "n_clusters": 6,
            "cluster_std": 2.0,
            "seed": 107,
        },
    ],
    "gaussian_extreme_noise": [
        {
            "n_samples": 30,
            "n_features": 30,
            "n_clusters": 3,
            "cluster_std": 5.0,
            "seed": 42,
        },
        {
            "n_samples": 40,
            "n_features": 20000,
            "n_clusters": 4,
            "cluster_std": 7.5,
            "seed": 43,
        },
        {
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
    ],
}
