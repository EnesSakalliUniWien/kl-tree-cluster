from __future__ import annotations

import pytest
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from kl_clustering_analysis import config


@pytest.mark.slow
def test_global_sibling_calibration_restores_gauss_null_large_to_one_cluster() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_null_large")
    data_t, _, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])

    result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 1
    annotations = result.extra["annotations"]
    assert (
        "context_weighted_empirical_null_inflation"
        in set(annotations["Sibling_Test_Method"].dropna())
    )


@pytest.mark.slow
def test_inflation_calibration_keeps_cat_highcard_conservative() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "cat_highcard_20cat_4c")
    data_t, _, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])

    result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 1


@pytest.mark.slow
def test_strict_sibling_calibration_preserves_gauss_clear_small() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_clear_small")
    data_t, _, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])

    result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 3
