"""Dimensionality benchmark cases with fixed informative subspace and added noise dimensions."""

DIMENSIONALITY_CASES = {
    "gaussian_dimensionality_consolidated": [
        {
            "name": "dim_consolidated_4c_24f",
            "generator": "dimensional_gaussian",
            "n_samples": 160,
            "n_clusters": 4,
            "informative_dims": 12,
            "n_features": 24,
            "separation": 2.8,
            "informative_std": 1.0,
            "noise_std": 1.0,
            "informative_corr": 0.15,
            "noise_corr": 0.0,
            "signal_mode": "consolidated",
            "balanced_clusters": True,
            "seed": 13000,
        },
        {
            "name": "dim_consolidated_4c_72f",
            "generator": "dimensional_gaussian",
            "n_samples": 160,
            "n_clusters": 4,
            "informative_dims": 12,
            "n_features": 72,
            "separation": 2.8,
            "informative_std": 1.0,
            "noise_std": 1.0,
            "informative_corr": 0.15,
            "noise_corr": 0.1,
            "signal_mode": "consolidated",
            "balanced_clusters": True,
            "seed": 13001,
        },
        {
            "name": "dim_consolidated_4c_272f",
            "generator": "dimensional_gaussian",
            "n_samples": 160,
            "n_clusters": 4,
            "informative_dims": 12,
            "n_features": 272,
            "separation": 2.8,
            "informative_std": 1.0,
            "noise_std": 1.0,
            "informative_corr": 0.15,
            "noise_corr": 0.1,
            "signal_mode": "consolidated",
            "balanced_clusters": True,
            "seed": 13002,
        },
    ],
    "gaussian_dimensionality_diffuse": [
        {
            "name": "dim_diffuse_6c_36f",
            "generator": "dimensional_gaussian",
            "n_samples": 180,
            "n_clusters": 6,
            "informative_dims": 18,
            "n_features": 36,
            "separation": 2.2,
            "informative_std": 1.1,
            "noise_std": 1.0,
            "informative_corr": 0.25,
            "noise_corr": 0.0,
            "signal_mode": "diffuse",
            "balanced_clusters": True,
            "seed": 13100,
        },
        {
            "name": "dim_diffuse_6c_136f",
            "generator": "dimensional_gaussian",
            "n_samples": 180,
            "n_clusters": 6,
            "informative_dims": 18,
            "n_features": 136,
            "separation": 2.2,
            "informative_std": 1.1,
            "noise_std": 1.0,
            "informative_corr": 0.25,
            "noise_corr": 0.1,
            "signal_mode": "diffuse",
            "balanced_clusters": True,
            "seed": 13101,
        },
        {
            "name": "dim_diffuse_6c_536f",
            "generator": "dimensional_gaussian",
            "n_samples": 180,
            "n_clusters": 6,
            "informative_dims": 18,
            "n_features": 536,
            "separation": 2.2,
            "informative_std": 1.1,
            "noise_std": 1.0,
            "informative_corr": 0.25,
            "noise_corr": 0.1,
            "signal_mode": "diffuse",
            "balanced_clusters": True,
            "seed": 13102,
        },
    ],
}


def _continuous_dimensional_variant(case: dict) -> dict:
    """Return the continuous-coordinate A/B companion for a median-binary case."""
    variant = case.copy()
    variant["name"] = f"{case['name']}_continuous"
    variant["generator"] = "dimensional_gaussian_continuous"
    variant["baseline_case_name"] = case["name"]
    return variant


DIMENSIONALITY_CASES["gaussian_dimensionality_consolidated_continuous"] = [
    _continuous_dimensional_variant(case)
    for case in DIMENSIONALITY_CASES["gaussian_dimensionality_consolidated"]
]
DIMENSIONALITY_CASES["gaussian_dimensionality_diffuse_continuous"] = [
    _continuous_dimensional_variant(case)
    for case in DIMENSIONALITY_CASES["gaussian_dimensionality_diffuse"]
]
