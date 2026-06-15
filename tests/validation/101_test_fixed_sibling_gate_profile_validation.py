from __future__ import annotations

import json

import benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation as profile_validation
import pandas as pd
from benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation import (
    FixedSiblingGateProfileValidationConfig,
    build_method_constant_evidence_fields,
    build_profile_production_components,
    run_fixed_sibling_gate_profile_validation,
    summarize_profile_validation_rows,
    summarize_profile_validation_transfer,
    validate_profiles,
)


def test_validate_profiles_rejects_unknown_profile() -> None:
    try:
        validate_profiles(("unknown_profile",))
    except ValueError as exc:
        assert "Unknown sibling-gate profile" in str(exc)
    else:
        raise AssertionError("validate_profiles accepted an unknown profile")


def test_summarize_profile_validation_rows_detects_adaptive_projection() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "profile_id": "fixed_global_guarded_v1",
                "adaptive_projection_avoided": False,
                "adaptive_projection_detected": True,
                "false_split": False,
                "ari": 1.0,
                "found_clusters": 1,
                "exact_cluster_count": True,
                "sibling_open_count": 0,
                "root_stability_guard_block_count": 0,
                "observed_sibling_gate_method": "projected_wald_inflation",
            }
        ]
    )

    summary = summarize_profile_validation_rows(rows)

    assert summary.iloc[0]["profile_validation_status"] == (
        "fixed_profile_adaptive_projection_detected"
    )
    assert summary.iloc[0]["adaptive_projection_avoided_rate"] == 0.0


def test_profile_transfer_components_fail_closed_when_coverage_is_missing() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "profile_id": "fixed_global_guarded_v1",
                "n_replicates": 1,
                "adaptive_projection_avoided_rate": 1.0,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.95,
                "mean_ari": 1.0,
                "mean_ari_lower_confidence": 1.0,
                "profile_validation_status": "fixed_profile_null_candidate",
            }
        ]
    )

    transfer = summarize_profile_validation_transfer(summary)
    components, production = build_profile_production_components(transfer)

    assert transfer.iloc[0]["profile_transfer_status"] == (
        "fixed_profile_insufficient_coverage"
    )
    assert "fixed_profile_insufficient_coverage" in set(
        components["component_status"]
    )
    assert production.iloc[0]["production_decision"] == "fail_closed_undefined"


def test_profile_transfer_summary_reports_null_support_sizing() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "profile_id": "fixed_global_guarded_v1",
                "n_replicates": 2,
                "adaptive_projection_avoided_rate": 1.0,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.65762,
                "mean_ari": 1.0,
                "mean_ari_lower_confidence": 1.0,
                "profile_validation_status": "fixed_profile_null_candidate",
            },
            {
                "case_id": "unit",
                "data_role": "signal",
                "source_family": "binary_template",
                "profile_id": "fixed_global_guarded_v1",
                "n_replicates": 2,
                "adaptive_projection_avoided_rate": 1.0,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.65762,
                "mean_ari": 0.9,
                "mean_ari_lower_confidence": 0.8,
                "profile_validation_status": "fixed_profile_signal_retained",
            },
        ]
    )

    transfer = summarize_profile_validation_transfer(summary)

    assert transfer.iloc[0][
        "required_zero_false_split_null_replicates_per_case"
    ] == 73
    assert transfer.iloc[0][
        "additional_zero_false_split_null_replicates_per_case"
    ] == 71
    assert transfer.iloc[0]["max_observed_null_false_split_count"] == 0
    assert transfer.iloc[0][
        "max_required_null_replicates_given_observed_false_splits"
    ] == 73
    assert transfer.iloc[0][
        "max_additional_zero_false_null_replicates_given_observed_false_splits"
    ] == 71


def test_profile_transfer_summary_sizes_observed_false_split_support() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "profile_id": "fixed_global_guarded_v1",
                "n_replicates": 73,
                "adaptive_projection_avoided_rate": 1.0,
                "false_split_rate": 1.0 / 73.0,
                "false_split_rate_upper_confidence": 0.073597,
                "mean_ari": 1.0,
                "mean_ari_lower_confidence": 1.0,
                "profile_validation_status": "fixed_profile_null_candidate",
            },
            {
                "case_id": "unit",
                "data_role": "signal",
                "source_family": "binary_template",
                "profile_id": "fixed_global_guarded_v1",
                "n_replicates": 73,
                "adaptive_projection_avoided_rate": 1.0,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.049,
                "mean_ari": 0.9,
                "mean_ari_lower_confidence": 0.8,
                "profile_validation_status": "fixed_profile_signal_retained",
            },
        ]
    )

    transfer = summarize_profile_validation_transfer(summary)

    assert transfer.iloc[0]["max_observed_null_false_split_count"] == 1
    assert transfer.iloc[0][
        "max_required_null_replicates_given_observed_false_splits"
    ] == 110
    assert transfer.iloc[0][
        "max_additional_zero_false_null_replicates_given_observed_false_splits"
    ] == 37


def test_run_fixed_profile_validation_writes_outputs_and_evidence(
    tmp_path,
) -> None:
    config = FixedSiblingGateProfileValidationConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        profiles=("fixed_global_guarded_v1",),
        data_roles=("null",),
        sibling_alpha=0.01,
        edge_alpha=0.001,
        replicates=1,
        base_seed=20260613,
        root_selective_permutation_guard_replicates=1,
        root_selective_permutation_guard_seed=7,
        root_selective_permutation_guard_alpha=0.01,
    )

    outputs = run_fixed_sibling_gate_profile_validation(config)

    for path in outputs.values():
        assert path.exists(), path
    assert outputs["checkpoint_rows_dir"].is_dir()

    rows = pd.read_csv(outputs["rows"])
    summary = pd.read_csv(outputs["summary"])
    evidence = json.loads(outputs["method_constant_evidence_fields"].read_text())
    manifest = json.loads(outputs["manifest"].read_text())

    assert bool(rows["adaptive_projection_avoided"].all())
    assert rows["observed_sibling_gate_method"].eq("fixed_global_chi_square").all()
    assert "root_sibling_p_value" in rows.columns
    assert "root_sibling_open" in rows.columns
    assert "root_stability_subsample_mean_ari" in rows.columns
    assert "root_selective_permutation_p_value" in rows.columns
    assert "root_selective_permutation_guard_blocked" in rows.columns
    assert "selective_permutation_guard_tested_count" in rows.columns
    assert "selective_permutation_guard_block_count" in rows.columns
    assert "mean_root_stability_subsample_mean_ari" in summary.columns
    assert "mean_root_selective_permutation_p_value" in summary.columns
    assert "mean_selective_permutation_guard_tested_count" in summary.columns
    assert summary["profile_validation_status"].isin(
        {"fixed_profile_null_candidate"}
    ).all()
    assert "sibling_gate_profile" in evidence["constants"]
    assert "adaptive_projection_avoidance_check" in evidence["constants"][
        "sibling_gate_profile"
    ]
    assert evidence["constants"]["sibling_gate_profile"]["profile_grid"] == [
        "fixed_global_guarded_v1"
    ]
    assert evidence["constants"]["sibling_gate_profile"][
        "root_selective_permutation_guard"
    ]["status"] == "enabled"
    assert "root_selective_permutation_guard_replicates" in evidence["constants"]
    assert "root_selective_permutation_guard_scope" in evidence["constants"]
    assert evidence["constants"]["root_selective_permutation_guard_replicates"][
        "permutation_replicate_grid"
    ] == [1]
    assert evidence["constants"]["root_selective_permutation_guard_scope"][
        "scope_grid"
    ] == ["root"]
    null_confidence = evidence["constants"]["sibling_gate_profile"][
        "null_false_split_confidence"
    ]
    assert "required_zero_false_split_null_replicates_per_case" in null_confidence[0]
    assert "max_required_null_replicates_given_observed_false_splits" in null_confidence[
        0
    ]
    assert len(manifest["outputs"]["checkpoint_rows"]) == 1


def test_run_fixed_profile_validation_resumes_from_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    config = FixedSiblingGateProfileValidationConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        profiles=("fixed_global_guarded_v1",),
        data_roles=("null",),
        sibling_alpha=0.01,
        edge_alpha=0.001,
        replicates=1,
        base_seed=20260613,
    )
    run_fixed_sibling_gate_profile_validation(config)

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("resume should read the checkpoint row")

    monkeypatch.setattr(
        profile_validation,
        "_rows_for_replicate",
        fail_if_recomputed,
    )
    resumed = run_fixed_sibling_gate_profile_validation(
        FixedSiblingGateProfileValidationConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            profiles=("fixed_global_guarded_v1",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
            resume_from_checkpoints=True,
        )
    )

    rows = pd.read_csv(resumed["rows"], keep_default_na=False)
    manifest = json.loads(resumed["manifest"].read_text())
    assert rows.shape[0] == 1
    assert rows.iloc[0]["profile_id"] == "fixed_global_guarded_v1"
    assert rows.iloc[0]["data_role"] == "null"
    assert manifest["config"]["resume_from_checkpoints"] is True
    assert len(manifest["outputs"]["checkpoint_rows"]) == 1


def test_evidence_uses_profile_owned_root_selective_guard_constants(tmp_path) -> None:
    config = FixedSiblingGateProfileValidationConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        profiles=("fixed_coordinate_selective_root_v1",),
        data_roles=("null",),
        sibling_alpha=0.01,
        edge_alpha=0.001,
        replicates=1,
        base_seed=20260613,
    )
    outputs = {
        "rows": tmp_path / "rows.csv",
        "summary": tmp_path / "summary.csv",
        "transfer_summary": tmp_path / "transfer.csv",
    }
    summary = pd.DataFrame.from_records(
        [{"adaptive_projection_avoided_rate": 1.0}]
    )
    transfer = pd.DataFrame.from_records(
        [
            {
                "source_family": "binary_template",
                "profile_id": "fixed_coordinate_selective_root_v1",
                "max_null_false_split_rate": 0.0,
                "max_null_false_split_rate_upper_confidence": 0.05,
                "required_zero_false_split_null_replicates_per_case": 73,
                "additional_zero_false_split_null_replicates_per_case": 0,
                "max_observed_null_false_split_count": 0,
                "max_required_null_replicates_given_observed_false_splits": 73,
                "max_additional_zero_false_null_replicates_given_observed_false_splits": 0,
                "min_signal_mean_ari": 0.9,
                "min_signal_mean_ari_lower_confidence": 0.8,
            }
        ]
    )

    evidence = build_method_constant_evidence_fields(
        config=config,
        outputs=outputs,
        summary=summary,
        transfer_summary=transfer,
    )

    profile_evidence = evidence["constants"]["sibling_gate_profile"]
    guard_evidence = profile_evidence["root_selective_permutation_guard"]
    assert guard_evidence["status"] == "enabled"
    assert guard_evidence["by_profile"]["fixed_coordinate_selective_root_v1"][
        "replicates"
    ] == 99
    assert guard_evidence["by_profile"]["fixed_coordinate_selective_root_v1"][
        "alpha"
    ] == 0.01
    assert evidence["constants"]["root_selective_permutation_guard_replicates"][
        "permutation_replicate_grid"
    ] == [99]
    assert evidence["constants"]["root_selective_permutation_guard_alpha"][
        "alpha_grid"
    ] == [0.01]
    assert evidence["constants"]["root_selective_permutation_guard_scope"][
        "scope_grid"
    ] == ["root"]


def test_evidence_reports_global_passthrough_scope(tmp_path) -> None:
    config = FixedSiblingGateProfileValidationConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        profiles=("fixed_coordinate_global_passthrough_v1",),
        data_roles=("null",),
        sibling_alpha=0.01,
        edge_alpha=0.001,
        replicates=1,
        base_seed=20260613,
    )
    outputs = {
        "rows": tmp_path / "rows.csv",
        "summary": tmp_path / "summary.csv",
        "transfer_summary": tmp_path / "transfer.csv",
    }
    summary = pd.DataFrame.from_records(
        [{"adaptive_projection_avoided_rate": 1.0}]
    )
    transfer = pd.DataFrame.from_records(
        [
            {
                "source_family": "binary_template",
                "profile_id": "fixed_coordinate_global_passthrough_v1",
                "max_null_false_split_rate": 0.0,
                "max_null_false_split_rate_upper_confidence": 0.05,
                "required_zero_false_split_null_replicates_per_case": 73,
                "additional_zero_false_split_null_replicates_per_case": 0,
                "max_observed_null_false_split_count": 0,
                "max_required_null_replicates_given_observed_false_splits": 73,
                "max_additional_zero_false_null_replicates_given_observed_false_splits": 0,
                "min_signal_mean_ari": 0.9,
                "min_signal_mean_ari_lower_confidence": 0.8,
            }
        ]
    )

    evidence = build_method_constant_evidence_fields(
        config=config,
        outputs=outputs,
        summary=summary,
        transfer_summary=transfer,
    )

    guard_evidence = evidence["constants"]["sibling_gate_profile"][
        "root_selective_permutation_guard"
    ]
    assert guard_evidence["by_profile"]["fixed_coordinate_global_passthrough_v1"][
        "scope"
    ] == "global_sibling_min_passthrough_descendant"
    assert evidence["constants"]["root_selective_permutation_guard_scope"][
        "scope_grid"
    ] == ["global_sibling_min_passthrough_descendant"]
