from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from benchmarks.shared.tbs_tree_context import build_tbs_tree_context
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.hierarchy_analysis.statistics.branch_length_utils import (
    EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
)


@pytest.mark.slow
def test_strict_sibling_calibration_fails_closed_gauss_null_large_without_support() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_null_large")
    context = build_tbs_tree_context(case, populate_node_distributions=False)

    result = run_tbs_on_distance(context.data, context.distance_condensed, DEFAULT_SIBLING_ALPHA)

    assert result.status == "ok"
    annotations = result.extra["annotations"]
    fail_closed = annotations[
        annotations["Sibling_Gate_P_Value_Calibration"].eq("undefined_no_internal_support")
    ]
    assert not fail_closed.empty
    assert fail_closed["Sibling_Test_Method"].eq("empirical_null_no_internal_support").all()
    assert fail_closed["Sibling_Gate_P_Value_Role"].eq("fail_closed_sibling_gate").all()
    assert fail_closed["Sibling_Divergence_Skipped"].eq(True).all()
    assert fail_closed["Sibling_Divergence_Invalid"].eq(True).all()
    assert not fail_closed["Sibling_BH_Different"].any()


@pytest.mark.slow
def test_leaf_only_cat_highcard_fails_closed_without_explicit_calibration_support() -> None:
    case = next(
        case for case in get_default_test_cases() if case["name"] == "cat_highcard_20cat_4c"
    )
    context = build_tbs_tree_context(case, populate_node_distributions=False)

    result = run_tbs_on_distance(
        context.data,
        context.distance_condensed,
        DEFAULT_SIBLING_ALPHA,
        feature_space=context.feature_space,
    )

    assert result.status == "ok"
    annotations = result.extra["annotations"]
    fail_closed = annotations[
        annotations["Sibling_Gate_P_Value_Calibration"].eq("undefined_no_internal_support")
    ]
    assert not fail_closed.empty
    assert fail_closed["Sibling_Test_Method"].eq("empirical_null_no_internal_support").all()
    assert fail_closed["Sibling_Gate_P_Value_Role"].eq("fail_closed_sibling_gate").all()
    assert fail_closed["Sibling_Divergence_Skipped"].eq(True).all()
    assert fail_closed["Sibling_Divergence_Invalid"].eq(True).all()
    assert not fail_closed["Sibling_BH_Different"].any()


@pytest.mark.slow
def test_default_sibling_calibration_marks_gauss_clear_small_boundary() -> None:
    case = next(case for case in get_default_test_cases() if case["name"] == "gauss_clear_small")
    context = build_tbs_tree_context(case, populate_node_distributions=False)

    result = run_tbs_on_distance(
        context.data,
        context.distance_condensed,
        DEFAULT_SIBLING_ALPHA,
        edge_branch_length_variance_policy=EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
        allow_linkage_ultrametric_branch_time=True,
        trace_level="full",
    )

    assert result.found_clusters == 2
    visited_internal_nodes = [
        node
        for node in result.extra["full_edge_traversal_trace"]
        if node["actual_visited"] and not node["is_leaf"]
    ]
    conservative_boundaries = [
        node
        for node in visited_internal_nodes
        if node["n_descendant_leaves"] == 20
        and node["edge_gate_open"] is True
        and node["sibling_gate_open"] is False
        and node["sibling_p_value"] > DEFAULT_SIBLING_ALPHA
    ]
    assert len(conservative_boundaries) == 1

    relaxed_result = run_tbs_on_distance(
        context.data,
        context.distance_condensed,
        0.03,
        edge_branch_length_variance_policy=EDGE_BRANCH_LENGTH_VARIANCE_POLICY_NORMALIZED,
        allow_linkage_ultrametric_branch_time=True,
        trace_level="full",
    )
    assert relaxed_result.found_clusters == 3

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


def test_tbs_runner_accepts_fixed_sibling_gate_profile() -> None:
    rng = np.random.default_rng(123)
    data = pd.DataFrame(
        rng.integers(0, 2, size=(16, 8)),
        index=[f"S{index}" for index in range(16)],
        columns=[f"F{index}" for index in range(8)],
    )
    result = run_tbs_on_distance(
        data,
        pdist(data.to_numpy(), metric="hamming"),
        DEFAULT_SIBLING_ALPHA,
        tree_linkage_method="average",
        sibling_gate_profile="fixed_global_guarded_v1",
        root_selective_permutation_guard_replicates=1,
        root_selective_permutation_guard_seed=19,
        root_selective_permutation_guard_alpha=0.01,
        trace_level="full",
    )

    metadata = result.extra["gate_bundle"].metadata.config
    assert result.status == "ok"
    assert metadata.sibling_gate_profile_id == "fixed_global_guarded_v1"
    assert metadata.sibling_gate_method == "fixed_global_chi_square"
    assert metadata.sibling_gate_alpha_penalty == 50.0
    assert metadata.root_stability_guard_threshold == 0.24
    assert metadata.root_stability_tree_distance_metric == "hamming"
    assert metadata.root_stability_tree_linkage_method == "average"
    assert metadata.root_selective_permutation_guard_replicates == 1
    assert metadata.root_selective_permutation_guard_seed == 19
    assert metadata.root_selective_permutation_guard_alpha == 0.01
    assert result.extra["sibling_gate_profile"] == "fixed_global_guarded_v1"
    assert result.extra["sibling_gate_method"] == "fixed_global_chi_square"
    assert result.extra["sibling_gate_alpha_penalty"] == 50.0
    assert result.extra["root_stability_subsample_replicates"] == 12
    assert result.extra["root_stability_tree_distance_metric"] == "hamming"
    assert result.extra["root_stability_tree_linkage_method"] == "average"
    assert result.extra["root_selective_permutation_guard_tree_distance_metric"] == ("hamming")
    assert result.extra["root_selective_permutation_guard_tree_linkage_method"] == ("average")
    assert result.extra["spectral_transport_passthrough_guard"] is False
    assert result.extra["spectral_transport_max_cost"] == 1.2
    assert result.extra["spectral_transport_require_mp_blocks"] is True


def test_tbs_runner_selected_root_guard_blocks_known_categorical_false_root() -> None:
    from benchmarks.diagnostics.calibration.sibling.gates.data_independent_sibling_gate_traversal_panel import (
        _generate_data_with_truth,
    )
    from benchmarks.validation.statistics.selected_edge_type1_geometry import (
        _case_contract,
        _select_cases,
    )
    from sklearn.metrics import adjusted_rand_score

    case = next(case for case in _select_cases(suite="full", case_names=("cat_clear_3cat_4c",)))
    (
        case_id,
        source_family,
        feature_representation,
        n_samples,
        n_features,
        n_categories,
    ) = _case_contract(case)
    seed = 20309045
    results = {}
    for role in ("null", "signal"):
        data, feature_space, truth, _true_clusters = _generate_data_with_truth(
            case=case,
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            data_role=role,
            seed=seed,
        )
        run = run_tbs_on_distance(
            data,
            pdist(data.to_numpy(dtype=float), metric="hamming"),
            0.01,
            tree_linkage_method="average",
            edge_alpha=0.001,
            feature_space=feature_space,
            sibling_gate_profile="fixed_coordinate_selective_root_v1",
            trace_level="full",
        )
        annotations = run.extra["annotations"]
        root = run.extra["tree"].root()
        metadata = run.extra["gate_bundle"].metadata.config
        assert metadata.sibling_gate_profile_id == "fixed_coordinate_selective_root_v1"
        assert metadata.root_selective_permutation_guard_replicates == 99
        assert metadata.root_selective_permutation_guard_seed == 0
        assert metadata.root_selective_permutation_guard_alpha == 0.01
        assert run.extra["root_selective_permutation_guard_replicates"] == 99
        results[role] = {
            "found_clusters": int(run.found_clusters),
            "ari": float(adjusted_rand_score(np.asarray(truth, dtype=int), np.asarray(run.labels))),
            "root_selective_p": float(annotations.at[root, "Root_Selective_Permutation_P_Value"]),
            "blocked": bool(annotations.at[root, "Root_Selective_Permutation_Guard_Blocked"]),
        }

    assert results["null"]["found_clusters"] == 1
    assert results["null"]["blocked"] is True
    assert results["null"]["root_selective_p"] == pytest.approx(0.17)
    assert results["signal"]["found_clusters"] == 6
    assert results["signal"]["blocked"] is False
    assert results["signal"]["root_selective_p"] == pytest.approx(0.01)
    assert results["signal"]["ari"] > 0.75
