from __future__ import annotations

import math

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction import (
    CalibrationModel,
    PoolStats,
    compute_pool_stats,
    predict_local_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types import (
    SiblingPairRecord,
)


def _make_record(
    parent: str,
    stat: float,
    degrees_of_freedom: float,
    sibling_null_prior_from_edge_pvalue: float,
    structural_dimension: float,
) -> SiblingPairRecord:
    return SiblingPairRecord(
        parent=parent,
        left=f"{parent}L",
        right=f"{parent}R",
        stat=stat,
        degrees_of_freedom=degrees_of_freedom,
        p_value=0.5,
        branch_length_sum=0.1,
        n_parent=32,
        is_null_like=False,
        sibling_null_prior_from_edge_pvalue=sibling_null_prior_from_edge_pvalue,
        structural_dimension=structural_dimension,
    )


def test_compute_pool_stats_tracks_structural_k_center_and_bandwidth() -> None:
    records = [
        _make_record(
            "p0",
            stat=3.0,
            degrees_of_freedom=2.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            structural_dimension=2.0,
        ),
        _make_record(
            "p1",
            stat=10.0,
            degrees_of_freedom=4.0,
            sibling_null_prior_from_edge_pvalue=2.0,
            structural_dimension=4.0,
        ),
        _make_record(
            "p2",
            stat=12.0,
            degrees_of_freedom=6.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            structural_dimension=8.0,
        ),
    ]
    ratios = np.array([record.stat / record.degrees_of_freedom for record in records], dtype=float)
    model = CalibrationModel(
        method="weighted_mean",
        n_calibration=len(records),
        global_inflation_factor=float(np.mean(ratios)),
        max_observed_ratio=float(np.max(ratios)),
    )

    pool = compute_pool_stats(records, model)

    assert math.isclose(pool.geometric_mean_structural_dimension, 4.0, rel_tol=1e-9)
    assert pool.bandwidth_log_structural_dimension > 0.0
    assert pool.bandwidth_status == "weighted_log_k_std"


def test_predict_local_inflation_factor_tracks_nearby_structural_dimensions() -> None:
    pool = PoolStats(
        c_global=3.0,
        mean_log_structural_dimension=math.log(4.0),
        geometric_mean_structural_dimension=4.0,
        bandwidth_log_structural_dimension=0.5,
        bandwidth_status="weighted_log_k_std",
        max_ratio=4.0,
        n_records=3,
        calibration_log_structural_dimensions=np.log(np.array([2.0, 4.0, 16.0], dtype=float)),
        calibration_sibling_null_priors=np.array([1.0, 1.0, 1.0], dtype=float),
        calibration_stat_df_ratios=np.array([1.3, 3.5, 1.1], dtype=float),
    )
    model = CalibrationModel(
        method="weighted_mean",
        n_calibration=3,
        global_inflation_factor=3.0,
        max_observed_ratio=4.0,
    )

    near_center = predict_local_inflation_factor(model, pool, structural_dimension=4.0)
    toward_large = predict_local_inflation_factor(model, pool, structural_dimension=16.0)
    far_out = predict_local_inflation_factor(model, pool, structural_dimension=256.0)

    assert near_center > toward_large
    assert 1.0 <= far_out <= pool.max_ratio
    assert far_out < near_center


def test_predict_local_inflation_factor_falls_back_to_global_with_zero_log_k_spread() -> None:
    records = [
        _make_record(
            "p0",
            stat=4.0,
            degrees_of_freedom=4.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            structural_dimension=4.0,
        ),
        _make_record(
            "p1",
            stat=8.0,
            degrees_of_freedom=8.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            structural_dimension=4.0,
        ),
    ]
    model = CalibrationModel(
        method="weighted_mean",
        n_calibration=len(records),
        global_inflation_factor=2.5,
        max_observed_ratio=2.5,
    )

    pool = compute_pool_stats(records, model)
    predicted = predict_local_inflation_factor(model, pool, structural_dimension=32.0)

    assert pool.bandwidth_log_structural_dimension == 0.0
    assert pool.bandwidth_status == "global_fallback_zero_log_k_spread"
    assert predicted == model.global_inflation_factor
