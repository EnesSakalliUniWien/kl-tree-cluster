"""Outlier-focused Gaussian benchmark cases."""

OUTLIER_CASES = {
    "gaussian_outlier_singleton": [
        {
            "name": "gauss_single_outlier_4c",
            "generator": "gaussian_outliers",
            "n_samples": 161,
            "n_features": 40,
            "n_inlier_clusters": 4,
            "n_clusters": 5,
            "cluster_std": 0.7,
            "outlier_count": 1,
            "outlier_distance": 12.0,
            "outlier_std": 0.0,
            "outlier_spatial_mode": "clustered",
            "outlier_label_mode": "singleton",
            "balanced_clusters": True,
            "seed": 13200,
        },
    ],
    "gaussian_outlier_contamination": [
        {
            "name": "gauss_outlier_cluster_4c",
            "generator": "gaussian_outliers",
            "n_samples": 166,
            "n_features": 40,
            "n_inlier_clusters": 4,
            "n_clusters": 5,
            "cluster_std": 0.75,
            "outlier_count": 6,
            "outlier_distance": 10.0,
            "outlier_std": 0.25,
            "outlier_spatial_mode": "clustered",
            "outlier_label_mode": "grouped",
            "balanced_clusters": True,
            "seed": 13201,
        },
        {
            "name": "gauss_outlier_shell_4c",
            "generator": "gaussian_outliers",
            "n_samples": 172,
            "n_features": 40,
            "n_inlier_clusters": 4,
            "n_clusters": 5,
            "cluster_std": 0.75,
            "outlier_count": 12,
            "outlier_distance": 9.0,
            "outlier_std": 0.0,
            "outlier_spatial_mode": "uniform_shell",
            "outlier_label_mode": "grouped",
            "balanced_clusters": True,
            "seed": 13202,
        },
    ],
}


def _continuous_outlier_variant(case: dict) -> dict:
    """Return the continuous-coordinate A/B companion for a median-binary case."""
    variant = case.copy()
    variant["name"] = f"{case['name']}_continuous"
    variant["generator"] = "gaussian_outliers_continuous"
    variant["baseline_case_name"] = case["name"]
    return variant


OUTLIER_CASES["gaussian_outlier_singleton_continuous"] = [
    _continuous_outlier_variant(case)
    for case in OUTLIER_CASES["gaussian_outlier_singleton"]
]
OUTLIER_CASES["gaussian_outlier_contamination_continuous"] = [
    _continuous_outlier_variant(case)
    for case in OUTLIER_CASES["gaussian_outlier_contamination"]
]
