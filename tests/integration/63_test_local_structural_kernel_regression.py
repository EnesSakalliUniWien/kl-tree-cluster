from __future__ import annotations

import pytest
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.kl_tree_context import build_kl_tree_context
from benchmarks.shared.runners.kl_runner import _run_kl_method
from kl_clustering_analysis import config


@pytest.mark.slow
def test_strict_sibling_calibration_rejects_gauss_null_large_without_support() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_null_large")
    context = build_kl_tree_context(case, populate_node_distributions=False)

    with pytest.raises(ValueError, match="selected non-null"):
        _run_kl_method(context.data, context.distance_condensed, config.SIBLING_ALPHA)


@pytest.mark.slow
def test_traversal_aligned_sibling_fdr_does_not_flat_penalize_cat_highcard_root() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "cat_highcard_20cat_4c")
    context = build_kl_tree_context(case, populate_node_distributions=False)

    result = _run_kl_method(
        context.data,
        context.distance_condensed,
        config.SIBLING_ALPHA,
        feature_space=context.feature_space,
    )

    annotations = result.extra["annotations"]
    root = result.extra["tree"].root()
    assert result.found_clusters == 2
    assert bool(annotations.loc[root, "Sibling_BH_Different"])
    assert (
        annotations.loc[root, "Sibling_Divergence_P_Value_Corrected"]
        == pytest.approx(annotations.loc[root, "Sibling_Divergence_P_Value"])
    )


@pytest.mark.slow
def test_strict_sibling_calibration_preserves_gauss_clear_small() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_clear_small")
    context = build_kl_tree_context(case, populate_node_distributions=False)

    result = _run_kl_method(context.data, context.distance_condensed, config.SIBLING_ALPHA)

    assert result.found_clusters == 3
