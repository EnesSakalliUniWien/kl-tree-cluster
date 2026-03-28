from __future__ import annotations

import math

import numpy as np

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction import (
    CalibrationModel,
    SiblingLocalGaussianInflationCalibrator,
    fit_sibling_inflation_calibrator,
    predict_sibling_adjustment,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types import (
    SiblingPairRecord,
)


def _make_record(
    parent: str,
    stat: float,
    degrees_of_freedom: float,
    sibling_null_prior_from_edge_pvalue: float,
    sibling_scale: float,
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
        sibling_scale=sibling_scale,
    )


def test_fit_sibling_inflation_calibrator_tracks_scale_center_and_spread() -> None:
    records = [
        _make_record(
            "p0",
            stat=3.0,
            degrees_of_freedom=2.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            sibling_scale=2.0,
        ),
        _make_record(
            "p1",
            stat=10.0,
            degrees_of_freedom=4.0,
            sibling_null_prior_from_edge_pvalue=2.0,
            sibling_scale=4.0,
        ),
        _make_record(
            "p2",
            stat=12.0,
            degrees_of_freedom=6.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            sibling_scale=8.0,
        ),
    ]
    ratios = np.array([record.stat / record.degrees_of_freedom for record in records], dtype=float)
    model = CalibrationModel(
        method="weighted_mean",
        n_calibration=len(records),
        global_inflation_factor=float(np.mean(ratios)),
        max_observed_ratio=float(np.max(ratios)),
    )

    calibrator = fit_sibling_inflation_calibrator(records, model)

    assert math.isclose(calibrator.center, 4.0, rel_tol=1e-9)
    assert calibrator.spread > 0.0
    assert calibrator.spread_status == "weighted_log_scale_std"


def test_predict_sibling_adjustment_tracks_nearby_sibling_scales() -> None:
    calibrator = SiblingLocalGaussianInflationCalibrator(
        global_adjustment=3.0,
        log_center=math.log(4.0),
        center=4.0,
        spread=0.5,
        spread_status="weighted_log_scale_std",
        max_adjustment=4.0,
        record_count=3,
        sample_log_scales=np.log(np.array([2.0, 4.0, 16.0], dtype=float)),
        sample_weights=np.array([1.0, 1.0, 1.0], dtype=float),
        sample_adjustments=np.array([1.3, 3.5, 1.1], dtype=float),
    )
    near_center = predict_sibling_adjustment(calibrator, sibling_scale=4.0)
    toward_large = predict_sibling_adjustment(calibrator, sibling_scale=16.0)
    far_out = predict_sibling_adjustment(calibrator, sibling_scale=256.0)

    assert near_center > toward_large
    assert 1.0 <= far_out <= calibrator.max_adjustment
    assert far_out < near_center


def test_predict_sibling_adjustment_falls_back_to_global_with_zero_log_scale_spread() -> None:
    records = [
        _make_record(
            "p0",
            stat=4.0,
            degrees_of_freedom=4.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            sibling_scale=4.0,
        ),
        _make_record(
            "p1",
            stat=8.0,
            degrees_of_freedom=8.0,
            sibling_null_prior_from_edge_pvalue=1.0,
            sibling_scale=4.0,
        ),
    ]
    model = CalibrationModel(
        method="weighted_mean",
        n_calibration=len(records),
        global_inflation_factor=2.5,
        max_observed_ratio=2.5,
    )

    calibrator = fit_sibling_inflation_calibrator(records, model)
    predicted = predict_sibling_adjustment(calibrator, sibling_scale=32.0)

    assert calibrator.spread == 0.0
    assert calibrator.spread_status == "global_fallback_zero_log_scale_spread"
    assert predicted == model.global_inflation_factor


def test_fit_sibling_inflation_calibrator_falls_back_to_global_when_no_positive_weights() -> None:
    records = [
        _make_record(
            "p0",
            stat=4.0,
            degrees_of_freedom=2.0,
            sibling_null_prior_from_edge_pvalue=0.0,
            sibling_scale=2.0,
        ),
        _make_record(
            "p1",
            stat=12.0,
            degrees_of_freedom=4.0,
            sibling_null_prior_from_edge_pvalue=0.0,
            sibling_scale=16.0,
        ),
    ]
    model = CalibrationModel(
        method="weighted_mean",
        n_calibration=0,
        global_inflation_factor=1.0,
        max_observed_ratio=3.0,
        diagnostics={"fit_status": "neutral_no_positive_weights"},
    )

    calibrator = fit_sibling_inflation_calibrator(records, model)
    predicted = predict_sibling_adjustment(calibrator, sibling_scale=16.0)

    assert calibrator.record_count == 0
    assert calibrator.spread_status == "global_fallback_no_positive_weights"
    assert predicted == model.global_inflation_factor
