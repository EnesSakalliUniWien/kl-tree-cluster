from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    _generate_data_with_truth,
)
from benchmarks.diagnostics.calibration.selected_pass_through_branch_recovery_conditioning import (
    SelectedPassThroughBranchRecoveryConditioningConfig,
    build_branch_recovery_conditioning_rows,
    build_feature_geometry_rows,
    build_focused_branch_recovery_fixture_rows,
    build_generated_branch_positive_support_fixture,
    run_selected_pass_through_branch_recovery_conditioning,
    summarize_branch_recovery_conditioning,
    summarize_branch_recovery_conditioning_metrics,
)
from benchmarks.validation.selected_edge_type1_geometry import (
    _case_contract,
    _select_cases,
)


def test_focused_fixture_contains_branch_barycentric_fragment_and_null_rows() -> None:
    source = build_focused_branch_recovery_fixture_rows()
    rows = build_branch_recovery_conditioning_rows(source)

    assert int(rows["full_branch_recovery_target"].sum()) >= 3
    assert int(rows["partial_branch_recovery_target"].sum()) == 1
    assert int(rows["barycentric_mixture_target"].sum()) == 1
    assert int(rows["fragment_target"].sum()) == 2
    assert int(rows["selected_null_control_target"].sum()) == 2


def test_feature_geometry_separates_when_balance_product_leaks() -> None:
    rows = build_branch_recovery_conditioning_rows(
        build_focused_branch_recovery_fixture_rows()
    )
    metrics = summarize_branch_recovery_conditioning_metrics(rows)
    by_metric = {row["metric"]: row for _, row in metrics.iterrows()}

    assert by_metric["feature_branch_geometry_score"][
        "zero_negative_status"
    ] == "zero_negative_separates_all_branch_recovery"
    assert by_metric["feature_homogeneity_gain_min"][
        "zero_negative_status"
    ] == "zero_negative_separates_all_branch_recovery"
    assert by_metric["branch_recovery_oracle_score"][
        "zero_negative_status"
    ] == "zero_negative_separates_all_branch_recovery"
    assert by_metric["completed_balance_product"]["zero_negative_status"] != (
        "zero_negative_separates_all_branch_recovery"
    )


def test_summary_reports_observable_fixture_but_keeps_production_fail_closed() -> None:
    rows = build_branch_recovery_conditioning_rows(
        build_focused_branch_recovery_fixture_rows()
    )
    metrics = summarize_branch_recovery_conditioning_metrics(rows)
    summary = summarize_branch_recovery_conditioning(
        rows,
        metrics,
        min_full_branch_recovery_count=2,
    )

    assert summary["diagnostic_status"].iloc[0] == (
        "branch_recovery_conditioning_fixture_observed_diagnostic_only"
    )
    assert summary["next_required_step"].iloc[0] == (
        "validate_branch_conditioning_on_selected_null_controls"
    )
    assert summary["production_action"].iloc[0] == (
        "fail_closed_until_branch_law_validated"
    )


def test_summary_reports_missing_full_branch_support_for_real_like_rows() -> None:
    source = build_focused_branch_recovery_fixture_rows()
    source = source[
        ~source["truth_geometry_role"].eq("truth_recovery_pass_through_positive")
    ].copy()
    rows = build_branch_recovery_conditioning_rows(source)
    metrics = summarize_branch_recovery_conditioning_metrics(rows)
    summary = summarize_branch_recovery_conditioning(
        rows,
        metrics,
        min_full_branch_recovery_count=2,
    )

    assert summary["full_branch_recovery_count"].iloc[0] == 0
    assert summary["diagnostic_status"].iloc[0] == (
        "full_branch_recovery_support_missing"
    )


def test_feature_geometry_builder_uses_paths_without_truth_predictors() -> None:
    case = _select_cases(suite="binary", case_names=["overlap_unbal_4c_small"])[0]
    (
        case_id,
        source_family,
        feature_representation,
        n_samples,
        n_features,
        n_categories,
    ) = _case_contract(case)
    seed = 20260615
    data, _feature_space, labels, _true_clusters = _generate_data_with_truth(
        case=case,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        data_role="signal",
        seed=seed,
    )
    label_values = pd.Series(labels, index=data.index.astype(str))
    child_by_sample = {
        sample_id: f"child_{int(label) % 2}"
        for sample_id, label in label_values.items()
    }
    assignments = pd.DataFrame.from_records(
        [
            {
                "case_id": case_id,
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "sample_id": str(sample_id),
                "path_node_ids": f"pass;split;{child_by_sample[str(sample_id)]};leaf_{sample_id}",
            }
            for sample_id in data.index.astype(str)
        ]
    )
    node_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": case_id,
                "data_role": "signal",
                "replicate": 0,
                "node_id": "pass",
                "left_method_id": "method",
                "data_seed": seed,
                "truth_downstream_split_node_id": "split",
            }
        ]
    )

    geometry = build_feature_geometry_rows(
        node_rows,
        assignments,
        suite="binary",
        top_k=8,
        max_pairwise_samples=120,
    )

    row = geometry.iloc[0]
    assert row["feature_geometry_status"] == "feature_geometry_observed"
    assert int(row["feature_geometry_node_sample_count"]) == int(data.shape[0])
    assert int(row["feature_geometry_left_child_count"]) > 0
    assert int(row["feature_geometry_right_child_count"]) > 0
    assert pd.notna(row["feature_branch_geometry_score"])


def test_generated_branch_positive_support_fixture_has_path_geometry_support() -> None:
    node_rows, assignments = build_generated_branch_positive_support_fixture()
    feature_rows = build_feature_geometry_rows(
        node_rows,
        assignments,
        suite="binary",
    )
    merged = node_rows.merge(
        feature_rows,
        on=["case_id", "data_role", "replicate", "node_id"],
        how="left",
    )
    rows = build_branch_recovery_conditioning_rows(merged)
    metrics = summarize_branch_recovery_conditioning_metrics(rows)
    summary = summarize_branch_recovery_conditioning(
        rows,
        metrics,
        min_full_branch_recovery_count=2,
    )
    by_metric = {row["metric"]: row for _, row in metrics.iterrows()}

    assert int(rows["full_branch_recovery_target"].sum()) == 3
    assert int(rows["partial_branch_recovery_target"].sum()) == 1
    assert int(rows["selected_null_control_target"].sum()) == 3
    assert rows["feature_geometry_status"].eq("feature_geometry_observed").all()
    assert by_metric["feature_branch_geometry_score"]["zero_negative_status"] == (
        "zero_negative_separates_all_branch_recovery"
    )
    assert by_metric["completed_balance_product"]["zero_negative_status"] != (
        "zero_negative_separates_all_branch_recovery"
    )
    assert summary["diagnostic_status"].iloc[0] == (
        "branch_recovery_conditioning_fixture_observed_diagnostic_only"
    )


def test_branch_recovery_conditioning_runner_writes_outputs(tmp_path) -> None:
    outputs = run_selected_pass_through_branch_recovery_conditioning(
        SelectedPassThroughBranchRecoveryConditioningConfig(
            output_dir=tmp_path,
            use_focused_fixture=True,
        )
    )

    for path in outputs.values():
        assert path.exists(), path
    rows = pd.read_csv(outputs["rows"])
    summary = pd.read_csv(outputs["summary"])
    assert int(rows["full_branch_recovery_target"].sum()) >= 3
    assert summary["production_action"].iloc[0] == (
        "fail_closed_until_branch_law_validated"
    )


def test_branch_recovery_runner_writes_generated_support_inputs(tmp_path) -> None:
    outputs = run_selected_pass_through_branch_recovery_conditioning(
        SelectedPassThroughBranchRecoveryConditioningConfig(
            output_dir=tmp_path,
            use_generated_support_fixture=True,
        )
    )

    for path in outputs.values():
        assert path.exists(), path
    rows = pd.read_csv(outputs["rows"])
    summary = pd.read_csv(outputs["summary"])
    assert rows["feature_geometry_status"].eq("feature_geometry_observed").all()
    assert int(summary["full_branch_recovery_count"].iloc[0]) == 3
    assert outputs["generated_support_node_rows"].exists()
    assert outputs["generated_support_gene_assignments"].exists()
