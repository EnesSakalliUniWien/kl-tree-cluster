from __future__ import annotations

import networkx as nx
import numpy as np
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util import method_execution
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.neighborhood_bandwidth import (
    CoherentSupportDecision,
    SupportRole,
    TauRegionKey,
    build_branch_length_distance_cache,
    classify_support_role,
    estimate_tau_region,
    role_allows_empirical_null_calibration,
    selected_neighborhood_kernel_weights,
)


def test_branch_length_distance_cache_uses_weighted_paths_and_mean_fallback() -> None:
    tree = nx.DiGraph()
    tree.add_edge("root", "left", branch_length=2.0)
    tree.add_edge("root", "right")

    cache = build_branch_length_distance_cache(tree)

    assert cache.status == "branch_length_with_mean_fallback"
    assert cache.fallback_edge_length == 2.0
    assert cache.distance("left", "right") == 4.0
    assert cache.distance("left", "left") == 0.0


def test_selected_signal_like_rows_do_not_calibrate_empirical_null() -> None:
    excluded = classify_support_role(
        {"topology_support_role": "selected_nonnull", "data_role": "signal"}
    )

    assert excluded == SupportRole.ALGORITHM_SELECTED_SIGNAL_LIKE_EXCLUDED
    assert not role_allows_empirical_null_calibration(excluded)
    assert role_allows_empirical_null_calibration(SupportRole.NULL_ANCHOR)
    assert role_allows_empirical_null_calibration(SupportRole.STOPPED_EDGE_NULL_ANCHOR)


def test_tau_region_shrinks_sparse_regions_to_parent() -> None:
    parent_region = TauRegionKey("*", "*", "*", "*", "*", "*")
    child_region = TauRegionKey(
        "nonroot",
        "non_direct",
        "binary",
        "k_low",
        "frontier",
        "short",
    )
    parent = estimate_tau_region(
        region=parent_region,
        stopping_edge_distances=[4.0],
        stable_neighbor_distances=[6.0],
        signal_neighbor_distances=[8.0],
        stable_log_ks=[1.0, 2.0],
        n_support=5,
        n_signal=2,
        effective_support=5.0,
    )
    child = estimate_tau_region(
        region=child_region,
        stopping_edge_distances=[1.0],
        stable_neighbor_distances=[1.0],
        signal_neighbor_distances=[1.0],
        stable_log_ks=[1.0],
        n_support=1,
        n_signal=1,
        effective_support=1.0,
        parent_estimate=parent,
        shrinkage_strength=3.0,
    )

    assert parent.tau_status == "local_stable"
    assert child.tau_status == "borrowed_or_unstable"
    assert child.borrowed_from_region == parent_region
    assert child.tau_t == 0.25 * 1.0 + 0.75 * 6.0
    assert not child.stable_for_promotion


def test_kernel_weights_combine_tree_distance_and_log_dimension() -> None:
    weights = selected_neighborhood_kernel_weights(
        distances=np.array([0.0, 2.0]),
        source_log_k=np.log(np.array([2.0, 4.0])),
        target_log_k=float(np.log(2.0)),
        tau=2.0,
        h_k=1.0,
    )

    assert weights[0] == 1.0
    assert 0.0 < weights[1] < weights[0]


def test_coherent_support_decision_fails_closed_on_root_invalid() -> None:
    decision = CoherentSupportDecision(
        root_usable_or_nonroot=False,
        topology_coherent=True,
        regional_tau_stable=True,
        spectral_flow_supported=True,
        empirical_null_admissible_or_not_required=True,
    )

    assert not decision.promotion_eligible
    assert decision.dominant_blocker == "root_invalid_or_unusable"
    assert decision.method_action == "fail_closed_root_invalid"


def test_internal_filter_hard_overlap_r1_fails_closed() -> None:
    case = next(
        case.copy() for case in get_default_test_cases() if case["name"] == "overlap_extreme_4c"
    )
    case["name"] = "overlap_extreme_4c__r1"
    case["seed"] = 9003
    data_df, labels, original, metadata = generate_case_data(case)

    params = METHOD_SPECS["tbs_internal_filter_v1"].param_grid[0]
    result_row, computed_result, method_audit = method_execution.run_single_method_once(
        method_id="tbs_internal_filter_v1",
        spec=METHOD_SPECS["tbs_internal_filter_v1"],
        params=params,
        case_idx=1,
        case_name=str(case["name"]),
        tc_seed=case["seed"],
        significance_level=0.01,
        edge_alpha=0.001,
        data_t=data_df,
        y_t=labels,
        x_original=original,
        meta=metadata,
        distance_matrix=None,
        distance_condensed=pdist(data_df.values, metric=params["tree_distance_metric"]),
        matrix_audit=False,
    )

    assert result_row.status.value == "skip"
    assert result_row.found_clusters == 0
    assert result_row.labels_length == 0
    assert result_row.skip_reason is not None
    assert "internal support" in result_row.skip_reason
    assert computed_result is None
    assert method_audit is None
