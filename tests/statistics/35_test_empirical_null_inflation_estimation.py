from __future__ import annotations

import math

import pytest
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    fit_empirical_null_inflation_model,
    predict_empirical_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)


def _make_record(
    parent: str,
    *,
    stat: float,
    degrees_of_freedom: float,
    reference_scale: float = 1.0,
    is_null_like: bool = True,
    is_edge_blocked: bool = False,
    sibling_null_weight: float = 1.0,
    sibling_projection_dimension: float = 2.0,
    n_parent: int = 32,
    feature_family: str = "bernoulli",
) -> SiblingPairRecord:
    return SiblingPairRecord(
        parent=parent,
        left=f"{parent}L",
        right=f"{parent}R",
        stat=stat,
        reference_scale=reference_scale,
        degrees_of_freedom=degrees_of_freedom,
        p_value=0.5,
        branch_length_sum=0.1,
        n_parent=n_parent,
        is_null_like=is_null_like,
        is_edge_blocked=is_edge_blocked,
        sibling_null_weight=sibling_null_weight,
        sibling_projection_dimension=sibling_projection_dimension,
        feature_family=feature_family,
    )


def test_fit_empirical_null_inflation_model_rejects_empty_calibration_set() -> None:
    with pytest.raises(ValueError, match="no sibling calibration records"):
        fit_empirical_null_inflation_model([])


def test_fit_empirical_null_inflation_model_rejects_nonfinite_statistic() -> None:
    records = [_make_record("bad_stat", stat=float("nan"), degrees_of_freedom=2.0)]

    with pytest.raises(ValueError, match="finite statistics"):
        fit_empirical_null_inflation_model(records)


def test_fit_empirical_null_inflation_model_rejects_no_positive_degrees_of_freedom() -> None:
    records = [_make_record("bad_df", stat=0.0, degrees_of_freedom=0.0)]

    with pytest.raises(ValueError, match="no positive-degree calibration records"):
        fit_empirical_null_inflation_model(records)


def test_fit_empirical_null_inflation_model_rejects_negative_statistic() -> None:
    records = [_make_record("negative", stat=-1.0, degrees_of_freedom=2.0)]

    with pytest.raises(ValueError, match="non-negative statistics"):
        fit_empirical_null_inflation_model(records)


def test_fit_empirical_null_inflation_model_rejects_invalid_null_weight() -> None:
    records = [
        _make_record(
            "bad_weight",
            stat=1.0,
            degrees_of_freedom=1.0,
            sibling_null_weight=1.5,
        )
    ]

    with pytest.raises(ValueError, match="sibling_null_weight"):
        fit_empirical_null_inflation_model(records)


def test_fit_empirical_null_inflation_model_excludes_selected_nonnull_records() -> None:
    records = [
        _make_record(
            "blocked",
            stat=4.0,
            degrees_of_freedom=1.0,
            is_edge_blocked=True,
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

    model = fit_empirical_null_inflation_model(records)

    assert model.n_calibration == 1
    assert model.n_stopped_or_null_calibration == 1
    assert math.isclose(
        model.baseline_empirical_inflation_factor,
        4.0 / 1.0,
    )


def test_fit_empirical_null_inflation_model_rejects_selected_nonnull_only_support() -> None:
    records = [
        _make_record(
            "selected",
            stat=20.0,
            degrees_of_freedom=4.0,
            is_null_like=False,
            sibling_null_weight=0.25,
        ),
    ]

    with pytest.raises(ValueError, match="selected non-null"):
        fit_empirical_null_inflation_model(records)


def test_fit_empirical_null_inflation_model_uses_reference_scale_in_inflation_denominator() -> None:
    records = [
        _make_record("p0", stat=4.0, reference_scale=2.0, degrees_of_freedom=1.0),
        _make_record("p1", stat=12.0, reference_scale=3.0, degrees_of_freedom=2.0),
    ]

    model = fit_empirical_null_inflation_model(records)

    assert math.isclose(
        model.baseline_empirical_inflation_factor,
        (4.0 + 12.0) / ((2.0 * 1.0) + (3.0 * 2.0)),
    )


def test_fit_empirical_null_inflation_model_uses_context_weighted_inflation() -> None:
    records = [
        _make_record("p0", stat=2.0, degrees_of_freedom=1.0, sibling_projection_dimension=1.0),
        _make_record("p1", stat=9.0, degrees_of_freedom=3.0, sibling_projection_dimension=2.0),
        _make_record("p2", stat=20.0, degrees_of_freedom=5.0, sibling_projection_dimension=4.0),
    ]

    model = fit_empirical_null_inflation_model(records)

    expected_scale = (2.0 + 9.0 + 20.0) / (1.0 + 3.0 + 5.0)

    assert math.isclose(
        model.baseline_empirical_inflation_factor,
        expected_scale,
        rel_tol=1e-9,
    )
    assert model.n_calibration == 3
    assert model.method == "context_weighted_supported_empirical_null_inflation"
    assert model.sample_statistics.tolist() == [2.0, 9.0, 20.0]
    assert model.sample_degrees_of_freedom.tolist() == [1.0, 3.0, 5.0]
    assert (
        predict_empirical_inflation_factor(
            model,
            records[0],
        )
        >= 1.0
    )


def test_predict_empirical_inflation_conditions_on_parent_sample_size() -> None:
    records = [
        _make_record(
            "small_0",
            stat=80.0,
            degrees_of_freedom=2.0,
            sibling_projection_dimension=5.0,
            n_parent=2,
            feature_family="categorical",
        ),
        _make_record(
            "small_1",
            stat=60.0,
            degrees_of_freedom=2.0,
            sibling_projection_dimension=5.0,
            n_parent=3,
            feature_family="categorical",
        ),
        _make_record(
            "large_0",
            stat=10.0,
            degrees_of_freedom=5.0,
            sibling_projection_dimension=5.0,
            n_parent=100,
            feature_family="categorical",
        ),
    ]

    model = fit_empirical_null_inflation_model(records)

    small_parent_inflation = predict_empirical_inflation_factor(
        model,
        records[0],
    )
    large_parent_inflation = predict_empirical_inflation_factor(
        model,
        records[2],
    )

    assert large_parent_inflation < small_parent_inflation
    assert large_parent_inflation < model.baseline_empirical_inflation_factor


def test_fit_empirical_null_inflation_model_keeps_zero_ratios_as_calibration_data() -> None:
    records = [
        _make_record("zero", stat=0.0, degrees_of_freedom=2.0),
        _make_record("positive", stat=4.0, degrees_of_freedom=2.0),
    ]

    model = fit_empirical_null_inflation_model(records)

    assert model.n_calibration == 2
    assert model.baseline_empirical_inflation_factor == 1.0


def test_fit_empirical_null_inflation_model_enforces_one_sided_inflation_floor() -> None:
    records = [
        _make_record("p0", stat=0.4, degrees_of_freedom=1.0),
        _make_record("p1", stat=1.2, degrees_of_freedom=2.0),
    ]

    model = fit_empirical_null_inflation_model(records)

    assert model.baseline_empirical_inflation_factor == 1.0


def test_fit_empirical_null_inflation_model_reports_underflow_stable_effective_sample_size() -> None:
    records = [
        _make_record("p0", stat=4.0, degrees_of_freedom=1.0, sibling_null_weight=1e-240),
        _make_record("p1", stat=8.0, degrees_of_freedom=2.0, sibling_null_weight=2e-240),
    ]

    model = fit_empirical_null_inflation_model(records)

    assert math.isclose(model.effective_sample_size, 1.8)
