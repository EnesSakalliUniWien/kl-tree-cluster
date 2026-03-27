from __future__ import annotations

import math

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.inflation_estimation import (
    fit_inflation_model,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types import (
    SiblingPairRecord,
)


def _make_record(
    parent: str,
    *,
    stat: float,
    degrees_of_freedom: float,
    sibling_null_prior_from_edge_pvalue: float,
    structural_dimension: float = 4.0,
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
        is_gate2_blocked=False,
        sibling_null_prior_from_edge_pvalue=sibling_null_prior_from_edge_pvalue,
        structural_dimension=structural_dimension,
    )


def test_fit_inflation_model_returns_neutral_model_when_no_valid_pairs() -> None:
    records = [
        _make_record("bad_stat", stat=float("nan"), degrees_of_freedom=2.0, sibling_null_prior_from_edge_pvalue=1.0),
        _make_record("bad_df", stat=3.0, degrees_of_freedom=0.0, sibling_null_prior_from_edge_pvalue=1.0),
    ]

    model = fit_inflation_model(records)

    assert model.method == "weighted_mean"
    assert model.n_calibration == 0
    assert model.global_inflation_factor == 1.0
    assert model.max_observed_ratio == 1.0
    assert model.diagnostics["fit_status"] == "neutral_no_data"


def test_fit_inflation_model_returns_neutral_model_when_only_positive_ratios_have_zero_weight() -> None:
    records = [
        _make_record("zero_ratio", stat=0.0, degrees_of_freedom=2.0, sibling_null_prior_from_edge_pvalue=1.0),
        _make_record("zero_weight", stat=4.0, degrees_of_freedom=2.0, sibling_null_prior_from_edge_pvalue=0.0),
    ]

    model = fit_inflation_model(records)

    assert model.method == "weighted_mean"
    assert model.n_calibration == 0
    assert model.global_inflation_factor == 1.0
    assert model.max_observed_ratio == 1.0
    assert model.diagnostics["fit_status"] == "neutral_no_positive_weights"


def test_fit_inflation_model_returns_neutral_model_when_no_positive_weights() -> None:
    records = [
        _make_record("p0", stat=4.0, degrees_of_freedom=2.0, sibling_null_prior_from_edge_pvalue=0.0),
        _make_record("p1", stat=9.0, degrees_of_freedom=3.0, sibling_null_prior_from_edge_pvalue=0.0),
    ]

    model = fit_inflation_model(records)

    assert model.method == "weighted_mean"
    assert model.n_calibration == 0
    assert model.global_inflation_factor == 1.0
    assert model.max_observed_ratio == 1.0
    assert model.diagnostics["fit_status"] == "neutral_no_positive_weights"


def test_fit_inflation_model_uses_weighted_mean_and_contributing_pair_count() -> None:
    records = [
        _make_record("p0", stat=2.0, degrees_of_freedom=1.0, sibling_null_prior_from_edge_pvalue=1.0),
        _make_record("p1", stat=9.0, degrees_of_freedom=3.0, sibling_null_prior_from_edge_pvalue=2.0),
        _make_record("p2", stat=20.0, degrees_of_freedom=5.0, sibling_null_prior_from_edge_pvalue=0.0),
    ]

    model = fit_inflation_model(records)

    expected_weighted_mean = (2.0 * 1.0 + 3.0 * 2.0 + 4.0 * 0.0) / (1.0 + 2.0 + 0.0)
    expected_effective_n = (3.0**2) / (1.0**2 + 2.0**2)

    assert math.isclose(model.global_inflation_factor, expected_weighted_mean, rel_tol=1e-9)
    assert model.max_observed_ratio == 4.0
    assert model.n_calibration == 2
    assert model.diagnostics["fit_status"] == "weighted_mean"
    assert model.diagnostics["n_valid_pairs"] == 3
    assert model.diagnostics["n_contributing"] == 2
    assert math.isclose(model.diagnostics["effective_n"], expected_effective_n, rel_tol=1e-9)


def test_fit_inflation_model_clamps_below_one_to_neutral() -> None:
    records = [
        _make_record("p0", stat=0.4, degrees_of_freedom=1.0, sibling_null_prior_from_edge_pvalue=1.0),
        _make_record("p1", stat=1.2, degrees_of_freedom=2.0, sibling_null_prior_from_edge_pvalue=1.0),
    ]

    model = fit_inflation_model(records)

    assert model.global_inflation_factor == 1.0
    assert model.max_observed_ratio == 1.0
