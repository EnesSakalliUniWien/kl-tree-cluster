from __future__ import annotations

import json
import math

import pandas as pd
from benchmarks.diagnostics.calibration.selected_candidate_truth_law_panel import (
    SelectedCandidateTruthLawPanelConfig,
    build_generated_support_candidate_rows,
    build_selected_candidate_accepted_split_filter_rows,
    build_selected_candidate_frontier_law_rows,
    build_selected_candidate_frontier_witness_rows,
    build_selected_candidate_sibling_rescue_audit_rows,
    build_selected_candidate_truth_law_rows,
    run_selected_candidate_truth_law_panel,
    summarize_selected_candidate_accepted_split_filter,
    summarize_selected_candidate_accepted_split_frontiers,
    summarize_selected_candidate_accepted_split_pair_filter,
    summarize_selected_candidate_collision_law_components,
    summarize_selected_candidate_context_metrics_by_state,
    summarize_selected_candidate_family_likelihood,
    summarize_selected_candidate_family_likelihood_rows,
    summarize_selected_candidate_feature_metrics,
    summarize_selected_candidate_feature_metrics_by_state,
    summarize_selected_candidate_frontier_ablation,
    summarize_selected_candidate_frontier_law,
    summarize_selected_candidate_sibling_rescue_audit,
    summarize_selected_candidate_sibling_rescue_guard,
    summarize_selected_candidate_truth_law,
    summarize_selected_candidate_truth_law_states,
)
from benchmarks.diagnostics.calibration.selected_pass_through_branch_recovery_conditioning import (
    build_generated_branch_positive_support_fixture,
)


def _candidate_rows() -> pd.DataFrame:
    base = {
        "case_id": "case",
        "left_method_id": "method_a",
        "right_method_id": "method_b",
        "left_sibling_open": True,
        "right_sibling_open": True,
        "right_explicit_guard_blocked": False,
    }
    return pd.DataFrame.from_records(
        [
            {
                **base,
                "data_role": "signal",
                "replicate": 0,
                "node_id": "branch",
                "candidate_reason": "left_split|right_split",
                "left_traversal_state": "split",
                "right_traversal_state": "split",
                "left_traversal_decision": "split",
                "right_traversal_decision": "split",
            },
            {
                **base,
                "data_role": "signal",
                "replicate": 1,
                "node_id": "fragment",
                "candidate_reason": "left_pass_through|right_pass_through",
                "left_traversal_state": "pass_through",
                "right_traversal_state": "pass_through",
                "left_traversal_decision": "pass_through",
                "right_traversal_decision": "pass_through",
            },
            {
                **base,
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null_candidate",
                "candidate_reason": "left_pass_through|right_boundary",
                "left_traversal_state": "pass_through",
                "right_traversal_state": "boundary",
                "left_traversal_decision": "pass_through",
                "right_traversal_decision": "boundary",
                "right_sibling_open": False,
            },
        ]
    )


def _candidate_rows_with_context() -> pd.DataFrame:
    rows = _candidate_rows()
    rows["left_depth"] = [2, 5, 1]
    rows["right_depth"] = [2, 5, 1]
    rows["left_n_descendant_leaves"] = [80, 40, 120]
    rows["right_n_descendant_leaves"] = [80, 40, 120]
    rows["left_sibling_p_value"] = [0.001, 0.2, 0.05]
    rows["right_sibling_p_value"] = [0.001, 0.2, 0.05]
    rows["left_descendant_accepted_split_count"] = [2, 1, 0]
    rows["right_descendant_accepted_split_count"] = [1, 0, 0]
    rows["left_descendant_pass_through_count"] = [0, 2, 0]
    rows["right_descendant_pass_through_count"] = [0, 0, 0]
    rows["left_descendant_stable_boundary_count"] = [70, 35, 118]
    rows["right_descendant_stable_boundary_count"] = [70, 38, 119]
    return rows


def _gene_assignments() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sample_index in range(8):
        child = "left_child" if sample_index < 4 else "right_child"
        records.append(
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method_a",
                "replicate": 0,
                "sample_id": f"S{sample_index}",
                "path_node_ids": f"branch;{child};leaf_{sample_index}",
            }
        )
    for sample_index in range(8):
        child = "left_fragment" if sample_index < 4 else "right_fragment"
        records.append(
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method_a",
                "replicate": 1,
                "sample_id": f"S{sample_index}",
                "path_node_ids": f"fragment;{child};leaf_{sample_index}",
            }
        )
    for sample_index in range(8):
        child = "left_null" if sample_index < 4 else "right_null"
        records.append(
            {
                "case_id": "case",
                "data_role": "selected_null",
                "method_id": "method_a",
                "replicate": 0,
                "sample_id": f"S{sample_index}",
                "path_node_ids": f"null_candidate;{child};leaf_{sample_index}",
            }
        )
    return pd.DataFrame.from_records(records)


def _truth_maps() -> dict[tuple[str, str, str, int], tuple[int, dict[str, int]]]:
    branch_labels = {f"S{sample_index}": int(sample_index >= 4) for sample_index in range(8)}
    fragment_labels = {f"S{sample_index}": 0 for sample_index in range(8)}
    null_labels = {f"S{sample_index}": 0 for sample_index in range(8)}
    return {
        ("case", "signal", "method_a", 0): (11, branch_labels),
        ("case", "signal", "method_a", 1): (12, fragment_labels),
        ("case", "selected_null", "method_a", 0): (13, null_labels),
    }


def _feature_maps() -> dict[tuple[str, str, str, int], tuple[int, pd.DataFrame]]:
    branch = pd.DataFrame(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        index=[f"S{sample_index}" for sample_index in range(8)],
    )
    fragment = pd.DataFrame(
        [[1, 0, 1, 0, 1, 0, 1, 0] for _ in range(8)],
        index=[f"S{sample_index}" for sample_index in range(8)],
    )
    null = fragment.copy()
    return {
        ("case", "signal", "method_a", 0): (11, branch),
        ("case", "signal", "method_a", 1): (12, fragment),
        ("case", "selected_null", "method_a", 0): (13, null),
    }


def test_candidate_truth_rows_classify_branch_fragment_and_selected_null() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    by_node = {row["node_id"]: row for _, row in rows.iterrows()}

    assert by_node["branch"]["own_split_truth_class"] == "own_split_branch_recovery"
    assert by_node["fragment"]["own_split_truth_class"] == "own_split_false_fragment"
    assert by_node["null_candidate"]["own_split_truth_class"] == "selected_null_control"
    assert math.isclose(float(by_node["branch"]["own_split_ari"]), 1.0)
    assert by_node["branch"]["own_split_child_majority_distinct"] is True
    assert by_node["fragment"]["own_split_child_majority_distinct"] is False
    assert by_node["branch"]["feature_geometry_status"] == "feature_geometry_observed"
    assert by_node["fragment"]["feature_geometry_status"] == "feature_geometry_observed"
    assert float(by_node["branch"]["feature_branch_geometry_score"]) > 0.0
    assert float(by_node["fragment"]["feature_branch_geometry_score"]) == 0.0


def test_summary_separates_split_recovery_from_pass_through_fragment() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    summary = summarize_selected_candidate_truth_law(rows)
    row = summary.iloc[0]

    assert int(row["row_count"]) == 3
    assert int(row["branch_recovery_count"]) == 1
    assert int(row["false_fragment_count"]) == 1
    assert int(row["selected_null_control_count"]) == 1
    assert int(row["pass_through_branch_recovery_count"]) == 0
    assert int(row["split_branch_recovery_count"]) == 1
    assert row["diagnostic_status"] == "branch_recovery_observed_only_on_accepted_splits"
    assert row["production_action"] == "diagnostic_only_no_promotion"


def test_state_summary_groups_directed_traversal_states() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    states = summarize_selected_candidate_truth_law_states(rows)
    grouped = {
        (
            row["data_role"],
            row["left_traversal_state"],
            row["right_traversal_state"],
            row["own_split_truth_class"],
        ): int(row["row_count"])
        for _, row in states.iterrows()
    }

    assert grouped[("signal", "split", "split", "own_split_branch_recovery")] == 1
    assert grouped[
        ("signal", "pass_through", "pass_through", "own_split_false_fragment")
    ] == 1
    assert grouped[
        ("selected_null", "pass_through", "boundary", "selected_null_control")
    ] == 1


def test_feature_metric_summary_reports_non_oracle_separation() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    metrics = summarize_selected_candidate_feature_metrics(rows)
    by_metric = {row["metric"]: row for _, row in metrics.iterrows()}

    assert by_metric["feature_branch_geometry_score"]["best_direction"] == "high"
    assert by_metric["feature_branch_geometry_score"]["zero_negative_status"] == (
        "zero_negative_separates_branch_high"
    )


def test_feature_metric_state_summary_separates_scopes() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    scoped = summarize_selected_candidate_feature_metrics_by_state(rows)
    scopes = set(scoped["state_scope"].astype(str))
    split_branch = scoped[
        scoped["state_scope"].eq("split_split")
        & scoped["metric"].eq("feature_branch_geometry_score")
    ].iloc[0]
    pass_branch = scoped[
        scoped["state_scope"].eq("pass_through_any")
        & scoped["metric"].eq("feature_branch_geometry_score")
    ].iloc[0]

    assert {"all_candidates", "split_no_pass_through", "pass_through_any"} <= scopes
    assert int(split_branch["branch_count"]) == 1
    assert int(pass_branch["branch_count"]) == 0
    assert pass_branch["zero_negative_status"] == "insufficient_branch_or_negative_support"


def test_context_metric_state_summary_uses_optional_candidate_fields() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows_with_context(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    metrics = summarize_selected_candidate_context_metrics_by_state(rows)
    by_scope_metric = {
        (row["state_scope"], row["metric"]): row for _, row in metrics.iterrows()
    }

    assert rows["candidate_context_status"].eq("candidate_context_observed").all()
    assert rows.loc[rows["node_id"].eq("branch"), "descendant_accepted_split_delta"].iloc[
        0
    ] == 1
    assert (
        by_scope_metric[("all_candidates", "negative_log10_min_sibling_p_value")][
            "finite_branch_count"
        ]
        == 1
    )
    assert (
        by_scope_metric[("pass_through_any", "left_descendant_pass_through_count")][
            "finite_negative_count"
        ]
        == 2
    )


def test_family_likelihood_rows_expose_selected_null_collision() -> None:
    candidates = _candidate_rows_with_context()
    signal_branch = candidates["node_id"].eq("branch")
    selected_null = candidates["node_id"].eq("null_candidate")
    candidates.loc[
        signal_branch,
        [
            "candidate_reason",
            "left_traversal_state",
            "right_traversal_state",
            "left_traversal_decision",
            "right_traversal_decision",
            "right_sibling_open",
        ],
    ] = [
        "left_pass_through",
        "pass_through",
        "boundary",
        "pass_through",
        "boundary",
        False,
    ]
    candidates.loc[selected_null, "left_descendant_accepted_split_count"] = 1

    rows = build_selected_candidate_truth_law_rows(
        candidates,
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    family_rows = summarize_selected_candidate_family_likelihood_rows(rows)
    summary = summarize_selected_candidate_family_likelihood(family_rows)
    signal_family = family_rows.loc[
        family_rows["family_likelihood_status"].eq(
            "branch_family_collides_with_fragment_or_selected_null"
        )
    ].iloc[0]
    summary_row = summary.loc[
        summary["family_likelihood_status"].eq(
            "branch_family_collides_with_fragment_or_selected_null"
        )
    ].iloc[0]

    assert signal_family["ambiguity_bucket"] == "possible_signal_over_suppression"
    assert (
        signal_family["stop_rule_pattern"]
        == "left_pass_through_downstream_split_right_stops"
    )
    assert int(signal_family["branch_recovery_count"]) == 1
    assert int(signal_family["matched_selected_null_control_count"]) == 1
    assert bool(signal_family["matched_selected_null_collision"]) is True
    assert signal_family["production_action"] == (
        "fail_closed_until_collision_law_validated"
    )
    assert int(summary_row["colliding_branch_family_count"]) == 1
    assert summary_row["diagnostic_status"] == "branch_family_collision_observed"

    components = summarize_selected_candidate_collision_law_components(family_rows)
    pass_component = components.loc[
        components["component_id"].eq("pass_through_retention")
    ].iloc[0]

    assert int(pass_component["pass_through_branch_family_count"]) == 1
    assert int(pass_component["colliding_branch_family_count"]) == 1
    assert pass_component["component_status"] == (
        "pass_through_branch_families_all_collide"
    )
    assert pass_component["production_action"] == (
        "fail_closed_until_collision_law_validated"
    )


def test_accepted_split_filter_rows_report_fragment_overlap() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    branch = rows.loc[rows["node_id"].eq("branch")].iloc[0].copy()
    fragment = rows.loc[rows["node_id"].eq("fragment")].iloc[0].copy()
    clean = branch.copy()
    clean["replicate"] = 2
    clean["feature_homogeneity_gain_min"] = 0.50
    clean["feature_branch_geometry_score"] = 0.30
    branch["replicate"] = 0
    branch["feature_homogeneity_gain_min"] = 0.40
    branch["feature_branch_geometry_score"] = 0.25
    fragment["replicate"] = 0
    fragment["left_traversal_state"] = "split"
    fragment["right_traversal_state"] = "split"
    fragment["left_traversal_decision"] = "split"
    fragment["right_traversal_decision"] = "split"
    fragment["feature_homogeneity_gain_min"] = 0.10
    fragment["feature_branch_geometry_score"] = 0.05
    accepted_rows = pd.DataFrame.from_records([branch, fragment, clean])
    family_rows = summarize_selected_candidate_family_likelihood_rows(accepted_rows)
    filter_rows = build_selected_candidate_accepted_split_filter_rows(
        accepted_rows,
        family_rows,
    )
    summary = summarize_selected_candidate_accepted_split_filter(filter_rows)
    pair_summary = summarize_selected_candidate_accepted_split_pair_filter(filter_rows)
    by_metric = {row["metric"]: row for _, row in summary.iterrows()}
    best_pair = pair_summary.sort_values(
        ["zero_negative_positive_pass_count", "zero_negative_positive_recall"],
        ascending=False,
    ).iloc[0]

    assert set(filter_rows["accepted_split_filter_role"]) == {
        "clean_branch_family",
        "fragment_mixed_branch_family",
    }
    assert "median_negative_log10_min_sibling_p_value" in filter_rows.columns
    assert by_metric["max_feature_homogeneity_gain_min"]["zero_negative_status"] == (
        "zero_negative_separates_clean_family_high"
    )
    assert by_metric["max_feature_homogeneity_gain_min"]["production_action"] == (
        "diagnostic_only_no_promotion"
    )
    assert int(best_pair["zero_negative_positive_pass_count"]) == 1
    assert best_pair["zero_negative_status"] == "pairwise_full_zero_negative_filter"

    frontiers = summarize_selected_candidate_accepted_split_frontiers(filter_rows)
    strength = frontiers.loc[
        frontiers["frontier_id"].eq("feature_strength_high")
    ].iloc[0]
    law_rows = build_selected_candidate_frontier_law_rows(
        filter_rows,
        min_support=1,
    )
    law_summary = summarize_selected_candidate_frontier_law(
        law_rows,
        min_support=1,
    )
    law_strength = law_summary.loc[
        law_summary["frontier_id"].eq("feature_strength_high")
    ].iloc[0]
    supported_clean = law_rows.loc[
        law_rows["frontier_id"].eq("feature_strength_high")
        & law_rows["accepted_split_filter_role"].eq("clean_branch_family")
    ]

    assert int(strength["negative_dominated_positive_count"]) == 0
    assert strength["diagnostic_status"] == (
        "clean_families_not_negative_dominated_diagnostic"
    )
    assert supported_clean["frontier_non_dominated"].all()
    assert supported_clean["support_status"].eq(
        "frontier_law_support_sufficient"
    ).all()
    assert supported_clean["selected_family_frontier_law_status"].eq(
        "clean_family_frontier_supported_diagnostic"
    ).all()
    assert law_strength["diagnostic_status"] == (
        "selected_family_frontier_law_candidate_diagnostic"
    )
    assert float(law_strength["posterior_clean_frontier_mean"]) > 0.5


def test_accepted_split_frontier_reports_dominated_clean_family() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    branch = rows.loc[rows["node_id"].eq("branch")].iloc[0].copy()
    fragment = rows.loc[rows["node_id"].eq("fragment")].iloc[0].copy()
    clean = branch.copy()
    clean["replicate"] = 1
    clean["feature_homogeneity_gain_min"] = 0.10
    clean["feature_branch_geometry_score"] = 0.10
    clean["feature_child_contrast_norm"] = 0.10
    mixed_branch = branch.copy()
    mixed_branch["replicate"] = 0
    mixed_branch["feature_homogeneity_gain_min"] = 0.10
    mixed_branch["feature_branch_geometry_score"] = 0.10
    mixed_branch["feature_child_contrast_norm"] = 0.10
    fragment["replicate"] = 0
    fragment["left_traversal_state"] = "split"
    fragment["right_traversal_state"] = "split"
    fragment["left_traversal_decision"] = "split"
    fragment["right_traversal_decision"] = "split"
    fragment["feature_homogeneity_gain_min"] = 0.20
    fragment["feature_branch_geometry_score"] = 0.20
    fragment["feature_child_contrast_norm"] = 0.20
    accepted_rows = pd.DataFrame.from_records([mixed_branch, fragment, clean])
    family_rows = summarize_selected_candidate_family_likelihood_rows(accepted_rows)
    filter_rows = build_selected_candidate_accepted_split_filter_rows(
        accepted_rows,
        family_rows,
    )
    frontiers = summarize_selected_candidate_accepted_split_frontiers(filter_rows)
    strength = frontiers.loc[
        frontiers["frontier_id"].eq("feature_strength_high")
    ].iloc[0]
    law_rows = build_selected_candidate_frontier_law_rows(
        filter_rows,
        min_support=1,
    )
    law_summary = summarize_selected_candidate_frontier_law(
        law_rows,
        min_support=1,
    )
    law_strength = law_summary.loc[
        law_summary["frontier_id"].eq("feature_strength_high")
    ].iloc[0]
    dominated_clean = law_rows.loc[
        law_rows["frontier_id"].eq("feature_strength_high")
        & law_rows["accepted_split_filter_role"].eq("clean_branch_family")
    ].iloc[0]

    assert int(strength["negative_dominated_positive_count"]) == 1
    assert strength["diagnostic_status"] == "all_clean_families_negative_dominated"
    assert strength["production_action"] == (
        "fail_closed_until_fragment_filter_validated"
    )
    assert bool(dominated_clean["frontier_non_dominated"]) is False
    assert dominated_clean["selected_family_frontier_law_status"] == (
        "clean_family_negative_dominated_fail_closed"
    )
    assert law_strength["diagnostic_status"] == "selected_family_frontier_law_leaky"
    assert law_strength["production_action"] == (
        "fail_closed_until_fragment_filter_validated"
    )


def test_frontier_ablation_exposes_required_context_coordinate() -> None:
    rows = build_selected_candidate_truth_law_rows(
        _candidate_rows(),
        _gene_assignments(),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
        top_k=4,
    )
    branch = rows.loc[rows["node_id"].eq("branch")].iloc[0].copy()
    fragment = rows.loc[rows["node_id"].eq("fragment")].iloc[0].copy()
    clean = branch.copy()
    clean["replicate"] = 1
    clean["feature_homogeneity_gain_min"] = 0.10
    clean["feature_branch_geometry_score"] = 0.10
    clean["feature_child_contrast_norm"] = 1.00
    clean["feature_barycentric_geometry_score"] = 0.00
    clean["min_sibling_p_value"] = 0.001
    mixed_branch = branch.copy()
    mixed_branch["replicate"] = 0
    mixed_branch["feature_homogeneity_gain_min"] = 0.10
    mixed_branch["feature_branch_geometry_score"] = 0.10
    mixed_branch["feature_child_contrast_norm"] = 1.00
    mixed_branch["feature_barycentric_geometry_score"] = 0.00
    mixed_branch["min_sibling_p_value"] = 0.001
    fragment["replicate"] = 0
    fragment["left_traversal_state"] = "split"
    fragment["right_traversal_state"] = "split"
    fragment["left_traversal_decision"] = "split"
    fragment["right_traversal_decision"] = "split"
    fragment["feature_homogeneity_gain_min"] = 0.10
    fragment["feature_branch_geometry_score"] = 0.10
    fragment["feature_child_contrast_norm"] = 1.00
    fragment["feature_barycentric_geometry_score"] = 0.00
    fragment["min_sibling_p_value"] = 0.01
    accepted_rows = pd.DataFrame.from_records([mixed_branch, fragment, clean])
    family_rows = summarize_selected_candidate_family_likelihood_rows(accepted_rows)
    filter_rows = build_selected_candidate_accepted_split_filter_rows(
        accepted_rows,
        family_rows,
    )
    ablation = summarize_selected_candidate_frontier_ablation(
        filter_rows,
        min_support=1,
    )
    witness_rows = build_selected_candidate_frontier_witness_rows(filter_rows)
    sibling_audit = build_selected_candidate_sibling_rescue_audit_rows(witness_rows)
    sibling_summary = summarize_selected_candidate_sibling_rescue_audit(sibling_audit)
    sibling_guard = summarize_selected_candidate_sibling_rescue_guard(sibling_audit)
    full_context = ablation.loc[
        ablation["frontier_id"].eq("full_family_context_frontier")
    ]
    all_metrics = full_context.loc[full_context["ablation_id"].eq("all_metrics")].iloc[
        0
    ]
    without_sibling = full_context.loc[
        full_context["ablation_id"].eq("remove_median_min_sibling_p_value")
    ].iloc[0]
    full_witness = witness_rows.loc[
        witness_rows["frontier_id"].eq("full_family_context_frontier")
        & witness_rows["case_id"].eq("case")
        & witness_rows["replicate"].eq(1)
    ].iloc[0]
    full_sibling_audit = sibling_audit.loc[
        sibling_audit["frontier_id"].eq("full_family_context_frontier")
        & sibling_audit["case_id"].eq("case")
        & sibling_audit["replicate"].eq(1)
    ].iloc[0]
    full_sibling_summary = sibling_summary.loc[
        sibling_summary["frontier_id"].eq("full_family_context_frontier")
    ].iloc[0]
    full_sibling_guard = sibling_guard.loc[
        sibling_guard["frontier_id"].eq("full_family_context_frontier")
    ].iloc[0]
    log_sibling_witness = witness_rows.loc[
        witness_rows["frontier_id"].eq("full_family_log_sibling_context_frontier")
        & witness_rows["case_id"].eq("case")
        & witness_rows["replicate"].eq(1)
    ].iloc[0]
    log_sibling_audit = sibling_audit.loc[
        sibling_audit["frontier_id"].eq("full_family_log_sibling_context_frontier")
        & sibling_audit["case_id"].eq("case")
        & sibling_audit["replicate"].eq(1)
    ].iloc[0]

    assert int(all_metrics["clean_dominated_count"]) == 0
    assert all_metrics["fragility_status"] == (
        "frontier_ablation_zero_leakage_with_margin"
    )
    assert int(without_sibling["clean_dominated_count"]) == 1
    assert without_sibling["fragility_status"] == "frontier_ablation_leaky"
    assert without_sibling["production_action"] == (
        "fail_closed_until_fragment_filter_validated"
    )
    assert full_witness["witness_case_id"] == "case"
    assert int(full_witness["witness_replicate"]) == 0
    assert full_witness["active_metric"] == "median_min_sibling_p_value"
    assert full_witness["active_metric_direction"] == "low"
    assert float(full_witness["active_metric_gap"]) > 0.0
    assert full_sibling_audit["sibling_rescue_status"] == (
        "sibling_only_no_structural_support"
    )
    assert float(full_sibling_audit["structural_gap_max"]) == 0.0
    assert full_sibling_summary["summary_status"] == (
        "sibling_only_rescue_requires_law"
    )
    assert int(full_sibling_guard["retained_after_guard_count"]) == 0
    assert int(full_sibling_guard["blocked_by_guard_count"]) == 1
    assert full_sibling_guard["guard_status"] == (
        "sibling_rescue_guard_blocks_all_candidates"
    )
    assert log_sibling_witness["active_metric"] == (
        "median_negative_log10_min_sibling_p_value"
    )
    assert float(log_sibling_witness["active_metric_gap"]) > 0.0
    assert log_sibling_audit["sibling_rescue_status"] == (
        "sibling_only_no_structural_support"
    )
    assert float(log_sibling_audit["log_sibling_gap"]) > 0.0


def test_generated_support_candidate_rows_target_downstream_splits() -> None:
    support_rows, _assignments = build_generated_branch_positive_support_fixture()
    candidate_rows = build_generated_support_candidate_rows(support_rows)

    assert int(candidate_rows.shape[0]) == int(support_rows.shape[0])
    assert set(candidate_rows["node_id"]) == set(
        support_rows["truth_downstream_split_node_id"].astype(str)
    )
    assert candidate_rows["left_traversal_state"].eq("split").all()
    assert candidate_rows["right_traversal_state"].eq("split").all()
    assert (
        candidate_rows["candidate_reason"]
        .astype(str)
        .str.startswith("generated_")
        .all()
    )


def test_generated_support_runner_validates_feature_geometry(tmp_path) -> None:
    outputs = run_selected_candidate_truth_law_panel(
        SelectedCandidateTruthLawPanelConfig(
            output_dir=tmp_path / "generated",
            use_generated_support_fixture=True,
        )
    )

    for path in outputs.values():
        assert path.exists(), path
    rows = pd.read_csv(outputs["rows"])
    summary = pd.read_csv(outputs["summary"])
    feature_metrics = pd.read_csv(outputs["feature_metric_summary"])
    feature_state_metrics = pd.read_csv(outputs["feature_metric_state_summary"])
    context_state_metrics = pd.read_csv(outputs["context_metric_state_summary"])
    family_likelihood_rows = pd.read_csv(outputs["family_likelihood_rows"])
    family_likelihood_summary = pd.read_csv(outputs["family_likelihood_summary"])
    collision_components = pd.read_csv(outputs["collision_law_components"])
    accepted_filter = pd.read_csv(outputs["accepted_split_filter_summary"])
    pair_filter = pd.read_csv(outputs["accepted_split_pair_filter_summary"])
    frontier = pd.read_csv(outputs["accepted_split_frontier_summary"])
    frontier_law_rows = pd.read_csv(outputs["selected_family_frontier_law_rows"])
    frontier_law_summary = pd.read_csv(
        outputs["selected_family_frontier_law_summary"]
    )
    frontier_ablation = pd.read_csv(
        outputs["selected_family_frontier_ablation_summary"]
    )
    frontier_witness = pd.read_csv(outputs["selected_family_frontier_witness_rows"])
    sibling_audit = pd.read_csv(outputs["sibling_rescue_audit_rows"])
    sibling_summary = pd.read_csv(outputs["sibling_rescue_audit_summary"])
    sibling_guard = pd.read_csv(outputs["sibling_rescue_guard_summary"])
    by_metric = {row["metric"]: row for _, row in feature_metrics.iterrows()}
    accepted_component = collision_components.loc[
        collision_components["component_id"].eq("accepted_split_preservation")
    ].iloc[0]

    assert int(summary["branch_recovery_count"].iloc[0]) >= 3
    assert rows["feature_geometry_status"].eq("feature_geometry_observed").all()
    assert not feature_state_metrics.empty
    assert not context_state_metrics.empty
    assert not family_likelihood_rows.empty
    assert not family_likelihood_summary.empty
    assert not accepted_filter.empty
    assert not pair_filter.empty
    assert not frontier.empty
    assert not frontier_law_rows.empty
    assert not frontier_law_summary.empty
    assert not frontier_ablation.empty
    assert not frontier_witness.empty
    assert not sibling_audit.empty
    assert not sibling_summary.empty
    assert not sibling_guard.empty
    assert int(accepted_component["clean_branch_family_count"]) >= 3
    assert accepted_component["component_status"] == (
        "accepted_split_preservation_clean_diagnostic"
    )
    assert by_metric["feature_homogeneity_gain_min"]["zero_negative_status"] == (
        "zero_negative_separates_branch_high"
    )
    assert by_metric["feature_branch_geometry_score"]["zero_negative_status"] == (
        "zero_negative_separates_branch_high"
    )
    assert outputs["generated_support_candidate_rows"].exists()
    assert outputs["generated_support_gene_assignments"].exists()


def test_selected_candidate_truth_law_runner_writes_outputs(tmp_path) -> None:
    candidate_path = tmp_path / "candidate_rows.csv"
    assignments_path = tmp_path / "gene_assignments.csv"
    _candidate_rows().to_csv(candidate_path, index=False)
    _gene_assignments().to_csv(assignments_path, index=False)

    outputs = run_selected_candidate_truth_law_panel(
        SelectedCandidateTruthLawPanelConfig(
            candidate_rows_path=candidate_path,
            gene_assignments_path=assignments_path,
            output_dir=tmp_path / "out",
            top_k=4,
        ),
        truth_label_maps=_truth_maps(),
        feature_data_maps=_feature_maps(),
    )

    for path in outputs.values():
        assert path.exists(), path
    rows = pd.read_csv(outputs["rows"])
    summary = pd.read_csv(outputs["summary"])
    states = pd.read_csv(outputs["state_summary"])
    feature_metrics = pd.read_csv(outputs["feature_metric_summary"])
    feature_state_metrics = pd.read_csv(outputs["feature_metric_state_summary"])
    context_state_metrics = pd.read_csv(outputs["context_metric_state_summary"])
    family_likelihood_rows = pd.read_csv(outputs["family_likelihood_rows"])
    family_likelihood_summary = pd.read_csv(outputs["family_likelihood_summary"])
    collision_components = pd.read_csv(outputs["collision_law_components"])
    accepted_filter_rows = pd.read_csv(outputs["accepted_split_filter_rows"])
    accepted_filter_summary = pd.read_csv(outputs["accepted_split_filter_summary"])
    pair_filter_summary = pd.read_csv(outputs["accepted_split_pair_filter_summary"])
    frontier_summary = pd.read_csv(outputs["accepted_split_frontier_summary"])
    frontier_law_rows = pd.read_csv(outputs["selected_family_frontier_law_rows"])
    frontier_law_summary = pd.read_csv(
        outputs["selected_family_frontier_law_summary"]
    )
    frontier_ablation_summary = pd.read_csv(
        outputs["selected_family_frontier_ablation_summary"]
    )
    frontier_witness_rows = pd.read_csv(outputs["selected_family_frontier_witness_rows"])
    sibling_audit_rows = pd.read_csv(outputs["sibling_rescue_audit_rows"])
    sibling_audit_summary = pd.read_csv(outputs["sibling_rescue_audit_summary"])
    sibling_rescue_guard_summary = pd.read_csv(
        outputs["sibling_rescue_guard_summary"]
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert int(rows.shape[0]) == 3
    assert int(summary["selected_null_control_count"].iloc[0]) == 1
    assert not states.empty
    assert not feature_metrics.empty
    assert not feature_state_metrics.empty
    assert not context_state_metrics.empty
    assert not family_likelihood_rows.empty
    assert not family_likelihood_summary.empty
    assert not collision_components.empty
    assert not accepted_filter_rows.empty
    assert not accepted_filter_summary.empty
    assert not pair_filter_summary.empty
    assert not frontier_summary.empty
    assert not frontier_law_rows.empty
    assert not frontier_law_summary.empty
    assert not frontier_ablation_summary.empty
    assert not frontier_witness_rows.empty
    assert not sibling_audit_rows.empty
    assert not sibling_audit_summary.empty
    assert not sibling_rescue_guard_summary.empty
    assert manifest["row_count"] == 3
    assert manifest["family_count"] == 3
    assert "feature_metric_summary" in manifest["outputs"]
    assert "feature_metric_state_summary" in manifest["outputs"]
    assert "context_metric_state_summary" in manifest["outputs"]
    assert "family_likelihood_rows" in manifest["outputs"]
    assert "family_likelihood_summary" in manifest["outputs"]
    assert "collision_law_components" in manifest["outputs"]
    assert "accepted_split_filter_rows" in manifest["outputs"]
    assert "accepted_split_filter_summary" in manifest["outputs"]
    assert "accepted_split_pair_filter_summary" in manifest["outputs"]
    assert "accepted_split_frontier_summary" in manifest["outputs"]
    assert "selected_family_frontier_law_rows" in manifest["outputs"]
    assert "selected_family_frontier_law_summary" in manifest["outputs"]
    assert "selected_family_frontier_ablation_summary" in manifest["outputs"]
    assert "selected_family_frontier_witness_rows" in manifest["outputs"]
    assert "sibling_rescue_audit_rows" in manifest["outputs"]
    assert "sibling_rescue_audit_summary" in manifest["outputs"]
    assert "sibling_rescue_guard_summary" in manifest["outputs"]
    assert manifest["thresholds"]["min_frontier_law_support"] == 3
    assert manifest["thresholds"]["min_frontier_law_margin"] == 1e-08
    assert manifest["production_action"] == "diagnostic_only_no_promotion"
