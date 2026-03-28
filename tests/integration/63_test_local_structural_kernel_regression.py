from __future__ import annotations

import pytest

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.kl_runner import _run_kl_method
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from kl_clustering_analysis import config


@pytest.mark.slow
def test_local_structural_kernel_restores_gauss_null_large_to_one_cluster() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_null_large")
    data_t, _, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])

    result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 1
    annotations = result.extra["annotations"]
    audit = annotations.attrs["sibling_divergence_audit"]
    assert audit["deflation_mode"] == "local_gaussian_adjuster"
    assert audit["local_adjuster_spread"] > 0.0


@pytest.mark.slow
def test_neighborhood_stable_blocked_weight_restores_cat_highcard_to_four_clusters() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "cat_highcard_20cat_4c")
    data_t, _, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])

    result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 4


@pytest.mark.slow
def test_neighborhood_stable_blocked_weight_preserves_gauss_clear_small() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_clear_small")
    data_t, _, _, _, distance_condensed, _, _ = prepare_case_inputs(case, ["kl"])

    result = _run_kl_method(data_t, distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 3
