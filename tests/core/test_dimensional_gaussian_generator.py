from __future__ import annotations

import numpy as np

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.generators.generate_dimensional_gaussian import (
    DimensionalGaussianConfig,
    generate_dimensional_gaussian,
)


def test_dimensional_gaussian_generator_concentrates_signal_in_informative_block() -> None:
    config = DimensionalGaussianConfig(
        n_samples=160,
        n_clusters=4,
        informative_dims=12,
        noise_dims=120,
        separation=3.0,
        informative_corr=0.15,
        noise_corr=0.1,
        signal_mode="consolidated",
        random_seed=42,
    )

    data, labels, metadata = generate_dimensional_gaussian(config)

    informative = data[:, : config.informative_dims]
    noise = data[:, config.informative_dims :]
    cluster_mean_spread_informative = informative.reshape(data.shape[0], -1)
    informative_by_cluster = np.vstack(
        [informative[labels == cluster_id].mean(axis=0) for cluster_id in range(config.n_clusters)]
    )
    noise_by_cluster = np.vstack(
        [noise[labels == cluster_id].mean(axis=0) for cluster_id in range(config.n_clusters)]
    )

    informative_spread = float(np.mean(np.var(informative_by_cluster, axis=0)))
    noise_spread = float(np.mean(np.var(noise_by_cluster, axis=0)))

    assert data.shape == (160, 132)
    assert labels.shape == (160,)
    assert metadata["informative_dims"] == 12
    assert metadata["noise_dims"] == 120
    assert informative_spread > noise_spread
    assert cluster_mean_spread_informative.shape[1] == 12


def test_dimensional_gaussian_case_generation_returns_binary_benchmark_matrix() -> None:
    case = {
        "name": "dimensional_test_case",
        "generator": "dimensional_gaussian",
        "n_samples": 90,
        "n_clusters": 3,
        "informative_dims": 10,
        "n_features": 70,
        "separation": 2.6,
        "informative_corr": 0.2,
        "noise_corr": 0.1,
        "signal_mode": "diffuse",
        "seed": 7,
    }

    data_df, labels, x_original, metadata = generate_case_data(case)

    assert data_df.shape == (90, 70)
    assert x_original.shape == (90, 70)
    assert set(np.unique(data_df.values)).issubset({0, 1})
    assert len(labels) == 90
    assert metadata["generator"] == "dimensional_gaussian"
    assert metadata["informative_dims"] == 10
    assert metadata["noise_dims"] == 60
    assert metadata["signal_mode"] == "diffuse"


def test_default_cases_include_dimensional_gaussian_family() -> None:
    all_cases = get_default_test_cases()
    family = [case for case in all_cases if case.get("generator") == "dimensional_gaussian"]

    assert len(family) == 6
    assert {case["category"] for case in family} == {
        "gaussian_dimensionality_consolidated",
        "gaussian_dimensionality_diffuse",
    }
