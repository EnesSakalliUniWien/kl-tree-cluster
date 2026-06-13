from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.diagnostics.path_b.phase1_path_b_foundation import (
    add_phase1_variant_columns,
    build_phase1_param_grid,
    cluster_count_distribution,
    q5_predictive_gain_summary,
    summarize_phase1_benchmark,
)


def test_phase1_param_grid_crosses_k_min_and_passthrough() -> None:
    grid = build_phase1_param_grid(k_min_values=(0, 2), passthrough_values=(True, False))

    assert len(grid) == 4
    assert {row["spectral_minimum_dimension"] for row in grid} == {0, 2}
    assert {row["passthrough"] for row in grid} == {True, False}
    assert all(row["tree_distance_metric"] == "hamming" for row in grid)


def test_phase1_summary_selects_penalized_ari_optimum() -> None:
    results = pd.DataFrame(
        {
            "test_case": [1, 2, 1, 2],
            "params": [
                "passthrough=True, spectral_minimum_dimension=0",
                "passthrough=True, spectral_minimum_dimension=0",
                "passthrough=False, spectral_minimum_dimension=1",
                "passthrough=False, spectral_minimum_dimension=1",
            ],
            "status": ["ok", "skip", "ok", "ok"],
            "ari": [0.8, None, 0.7, 0.7],
            "found_clusters": [3, 0, 2, 2],
            "true_clusters": [3, 3, 3, 2],
            "cluster_count_abs_error": [0.0, None, 1.0, 0.0],
            "over_split": [0.0, None, 0.0, 0.0],
            "under_split": [0.0, None, 1.0, 0.0],
        }
    )

    with_variants = add_phase1_variant_columns(results)
    summary = summarize_phase1_benchmark(results)
    distribution = cluster_count_distribution(results)

    assert with_variants["spectral_minimum_dimension"].tolist() == [0, 0, 1, 1]
    assert summary.iloc[0]["spectral_minimum_dimension"] == 1
    assert bool(summary.iloc[0]["passthrough"]) is False
    assert summary.iloc[0]["penalized_mean_ari"] == 0.7
    assert set(distribution["found_clusters"]) == {2, 3}


def test_q5_predictive_gain_uses_without_spectral_baseline() -> None:
    q5_summary = pd.DataFrame(
        {
            "model_id": [
                "q5_without_spectral_geometry",
                "q5_barycentric_edge_spectral",
            ],
            "median_tail_auc": [0.6, 0.75],
            "median_holdout_r_squared": [0.1, 0.25],
            "median_residual_tail_exceedance_absolute_error": [0.04, 0.015],
        }
    )

    gain = q5_predictive_gain_summary(q5_summary)
    best = gain.iloc[0]

    assert best["model_id"] == "q5_barycentric_edge_spectral"
    assert best["median_tail_auc_gain_vs_baseline"] == pytest.approx(0.15)
    assert best["median_r_squared_gain_vs_baseline"] == pytest.approx(0.15)
    assert best["median_tail_error_reduction_vs_baseline"] == pytest.approx(0.025)
