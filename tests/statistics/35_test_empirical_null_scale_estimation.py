from __future__ import annotations

import math

import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.scale_correction.empirical_null_scale_estimation import (
    fit_empirical_null_scale_model,
    predict_scale_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)


def _make_record(
    parent: str,
    *,
    stat: float,
    degrees_of_freedom: float,
    is_null_like: bool = True,
    is_gate2_blocked: bool = False,
    sibling_null_weight: float = 1.0,
    sibling_calibration_scale: float = 2.0,
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
        is_null_like=is_null_like,
        is_gate2_blocked=is_gate2_blocked,
        sibling_null_weight=sibling_null_weight,
        sibling_calibration_scale=sibling_calibration_scale,
    )


def test_fit_empirical_null_scale_model_rejects_empty_calibration_set() -> None:
    with pytest.raises(ValueError, match="no sibling calibration records"):
        fit_empirical_null_scale_model([])


def test_fit_empirical_null_scale_model_rejects_nonfinite_statistic() -> None:
    records = [_make_record("bad_stat", stat=float("nan"), degrees_of_freedom=2.0)]

    with pytest.raises(ValueError, match="finite statistics"):
        fit_empirical_null_scale_model(records)


def test_fit_empirical_null_scale_model_rejects_no_positive_degrees_of_freedom() -> None:
    records = [_make_record("bad_df", stat=0.0, degrees_of_freedom=0.0)]

    with pytest.raises(ValueError, match="no positive-degree calibration records"):
        fit_empirical_null_scale_model(records)


def test_fit_empirical_null_scale_model_rejects_negative_statistic() -> None:
    records = [_make_record("negative", stat=-1.0, degrees_of_freedom=2.0)]

    with pytest.raises(ValueError, match="non-negative statistics"):
        fit_empirical_null_scale_model(records)


def test_fit_empirical_null_scale_model_rejects_invalid_null_weight() -> None:
    records = [
        _make_record(
            "bad_weight",
            stat=1.0,
            degrees_of_freedom=1.0,
            sibling_null_weight=1.5,
        )
    ]

    with pytest.raises(ValueError, match="sibling_null_weight"):
        fit_empirical_null_scale_model(records)


def test_fit_empirical_null_scale_model_uses_all_weighted_records() -> None:
    records = [
        _make_record(
            "blocked",
            stat=4.0,
            degrees_of_freedom=1.0,
            is_gate2_blocked=True,
            sibling_null_weight=0.5,
        ),
        _make_record(
            "focal",
            stat=20.0,
            degrees_of_freedom=4.0,
            is_null_like=False,
            sibling_null_weight=0.25,
        ),
    ]

    model = fit_empirical_null_scale_model(records)

    assert model.n_calibration == 2
    assert math.isclose(
        model.baseline_scale_factor,
        ((0.5 * 4.0) + (0.25 * 20.0)) / ((0.5 * 1.0) + (0.25 * 4.0)),
    )


def test_fit_empirical_null_scale_model_uses_context_weighted_scale() -> None:
    records = [
        _make_record("p0", stat=2.0, degrees_of_freedom=1.0, sibling_calibration_scale=1.0),
        _make_record("p1", stat=9.0, degrees_of_freedom=3.0, sibling_calibration_scale=2.0),
        _make_record("p2", stat=20.0, degrees_of_freedom=5.0, sibling_calibration_scale=4.0),
    ]

    model = fit_empirical_null_scale_model(records)

    expected_scale = (2.0 + 9.0 + 20.0) / (1.0 + 3.0 + 5.0)
    expected_mean_ratio = (2.0 + 3.0 + 4.0) / 3.0

    assert math.isclose(model.baseline_scale_factor, expected_scale, rel_tol=1e-9)
    assert model.max_observed_ratio == 4.0
    assert model.n_calibration == 3
    assert model.method == "context_weighted_empirical_null_scale"
    assert model.diagnostics["fit_status"] == "context_weighted_empirical_null_scale"
    assert model.diagnostics["n_contributing"] == 3
    assert math.isclose(model.diagnostics["mean_ratio"], expected_mean_ratio, rel_tol=1e-9)
    assert model.diagnostics["weighted_sum_statistic"] == 31.0
    assert model.diagnostics["weighted_sum_degrees_of_freedom"] == 9.0
    assert predict_scale_factor(model, records[0]) >= 1.0


def test_fit_empirical_null_scale_model_keeps_zero_ratios_as_calibration_data() -> None:
    records = [
        _make_record("zero", stat=0.0, degrees_of_freedom=2.0),
        _make_record("positive", stat=4.0, degrees_of_freedom=2.0),
    ]

    model = fit_empirical_null_scale_model(records)

    assert model.n_calibration == 2
    assert model.baseline_scale_factor == 1.0


def test_fit_empirical_null_scale_model_enforces_one_sided_scale_floor() -> None:
    records = [
        _make_record("p0", stat=0.4, degrees_of_freedom=1.0),
        _make_record("p1", stat=1.2, degrees_of_freedom=2.0),
    ]

    model = fit_empirical_null_scale_model(records)

    assert model.baseline_scale_factor == 1.0
    assert model.max_observed_ratio == 0.6
