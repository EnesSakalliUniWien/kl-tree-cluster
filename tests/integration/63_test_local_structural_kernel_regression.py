from __future__ import annotations

import pytest
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.kl_tree_context import build_kl_tree_context
from benchmarks.shared.runners.kl_runner import _run_kl_method
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)


@pytest.mark.slow
def test_strict_sibling_calibration_rejects_gauss_null_large_without_support() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_null_large")
    context = build_kl_tree_context(case, populate_node_distributions=False)

    with pytest.raises(ValueError, match="selected non-null"):
        _run_kl_method(context.data, context.distance_condensed, DEFAULT_SIBLING_ALPHA)


@pytest.mark.slow
def test_leaf_only_cat_highcard_requires_explicit_calibration_support() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "cat_highcard_20cat_4c")
    context = build_kl_tree_context(case, populate_node_distributions=False)

    with pytest.raises(ValueError, match="selected non-null"):
        _run_kl_method(
            context.data,
            context.distance_condensed,
            DEFAULT_SIBLING_ALPHA,
            feature_space=context.feature_space,
        )


@pytest.mark.slow
def test_strict_sibling_calibration_preserves_gauss_clear_small() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_clear_small")
    context = build_kl_tree_context(case, populate_node_distributions=False)

    result = _run_kl_method(context.data, context.distance_condensed, DEFAULT_SIBLING_ALPHA)

    assert result.found_clusters == 3
    stage_timings = result.extra["stage_timings"]
    for key in (
        "tree_build_sec",
        "populate_divergences_sec",
        "edge_gate_sec",
        "edge_gate_contrast_covariance_sec",
        "edge_gate_projection_sec",
        "edge_gate_wald_statistic_sec",
        "edge_gate_tree_bh_sec",
        "spectral_context_sec",
        "tangent_whitening_sec",
        "eigensolve_sec",
        "pca_projection_sec",
        "sibling_gate_sec",
        "sibling_gate_pair_record_collection_sec",
        "sibling_gate_inflation_fit_sec",
        "sibling_gate_adjusted_tests_sec",
        "sibling_gate_fdr_sec",
        "traversal_sec",
    ):
        assert stage_timings[key] >= 0.0
