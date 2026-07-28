from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.selected.tail.selected_tail_admissibility_domain import (
    SelectedTailRun,
)
from benchmarks.diagnostics.calibration.selected.tail.selected_tail_promotion_gate import (
    DEFAULT_Q5_MODEL_ID,
    STUDY_ROLE,
    evaluate_q5_promotion_gate,
    evaluate_selected_tail_promotion_gate,
    run_selected_tail_promotion_gate,
)


def _tail_law_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_family": "gaussian_blobs",
                "feature_family": "bernoulli",
                "parent_size_bin": "small_0_0.25",
                "sibling_projection_dimension": 1,
                "edge_action_bin": "edge_action_ge8",
                "tail_law_role": "descriptive_selected_ratio_tail_law_not_calibration",
                "alpha": 0.01,
                "n_records": 1200,
                "n_matching_simulations": 600,
                "required_min_matching_simulations": 499,
                "required_min_matched_records": 499,
                "max_exceedance_standard_error": 0.002,
                "selected_ratio_mean": 10.0,
                "selected_ratio_median": 9.0,
                "selected_ratio_trainless_q90": 15.0,
                "selected_ratio_trainless_q95": 18.0,
                "selected_ratio_trainless_q99": 21.0,
                "production_tail_law_admissible": True,
                "tail_law_admissibility_failure_reasons": "",
                "n_train_rows_across_folds": 1000,
                "n_test_rows_across_folds": 200,
                "n_used_folds": 5,
                "tail_threshold_mean": 20.0,
                "tail_threshold_median": 20.0,
                "heldout_exceedance_rate": 0.011,
                "heldout_exceedance_absolute_error": 0.001,
                "heldout_exceedance_standard_error": 0.001,
                "tail_law_status": "descriptive_holdout_tail_law",
                "tail_law_failure_reasons": "",
            },
            {
                "source_family": "binary_template",
                "feature_family": "bernoulli",
                "parent_size_bin": "large_0.5_0.75",
                "sibling_projection_dimension": 1,
                "edge_action_bin": "edge_action_2_4",
                "tail_law_role": "descriptive_selected_ratio_tail_law_not_calibration",
                "alpha": 0.01,
                "n_records": 2,
                "n_matching_simulations": 2,
                "required_min_matching_simulations": 499,
                "required_min_matched_records": 499,
                "max_exceedance_standard_error": 0.002,
                "selected_ratio_mean": 2.0,
                "selected_ratio_median": 2.0,
                "selected_ratio_trainless_q90": 2.0,
                "selected_ratio_trainless_q95": 2.0,
                "selected_ratio_trainless_q99": 2.0,
                "production_tail_law_admissible": False,
                "tail_law_admissibility_failure_reasons": (
                    "matching_simulations_below_tail_resolution_contract;"
                    "matched_records_below_tail_resolution_contract"
                ),
                "n_train_rows_across_folds": 0,
                "n_test_rows_across_folds": 2,
                "n_used_folds": 0,
                "tail_threshold_mean": None,
                "tail_threshold_median": None,
                "heldout_exceedance_rate": None,
                "heldout_exceedance_absolute_error": None,
                "heldout_exceedance_standard_error": None,
                "tail_law_status": "no_valid_tail_law_folds",
                "tail_law_failure_reasons": "insufficient_train_simulations",
            },
        ]
    )


def _q5_validation_table(*, parent_size_failure: bool) -> pd.DataFrame:
    rows = []
    for split in (
        "replicate_modulo",
        "leave_one_case_out",
        "leave_one_feature_family_out",
        "leave_one_parent_size_bin_out",
    ):
        error = 0.001
        if parent_size_failure and split == "leave_one_parent_size_bin_out":
            error = 0.25
        rows.append(
            {
                "model_id": DEFAULT_Q5_MODEL_ID,
                "split_strategy": split,
                "model_status": f"diagnostic_holdout_{split}",
                "residual_tail_exceedance_rate": 0.01 + error,
                "residual_tail_exceedance_absolute_error": error,
                "residual_tail_exceedance_standard_error": 0.001,
                "holdout_tail_auc_from_linear_score": 0.99,
                "holdout_log_ratio_r_squared": 0.1,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_q5_promotion_gate_blocks_parent_size_transfer_failure() -> None:
    gate_table, gate = evaluate_q5_promotion_gate(_q5_validation_table(parent_size_failure=True))

    assert not gate.q5_global_gate_pass
    assert "leave_one_parent_size_bin_out:residual_tail_abs_error_above_gate" in (
        gate.failure_reasons
    )
    parent_row = gate_table[gate_table["split_strategy"].eq("leave_one_parent_size_bin_out")].iloc[
        0
    ]
    assert not bool(parent_row["q5_split_gate_pass"])


def test_promotion_gate_returns_diagnostic_only_when_q5_fails() -> None:
    promotion, _q5, summary = evaluate_selected_tail_promotion_gate(
        _tail_law_table(),
        _q5_validation_table(parent_size_failure=True),
        require_c_hat_precision=False,
    )

    gaussian = promotion[promotion["source_family"].eq("gaussian_blobs")].iloc[0]
    assert gaussian["promotion_decision"] == "external_diagnostic_only"
    assert "q5_global_gate_failed" in gaussian["promotion_failure_reasons"]
    binary = promotion[promotion["source_family"].eq("binary_template")].iloc[0]
    assert binary["promotion_decision"] == "undefined_support_failure"
    assert summary["study_role"].eq(STUDY_ROLE).all()


def test_promotion_gate_can_mark_external_admissible_when_all_gates_pass() -> None:
    promotion, _q5, _summary = evaluate_selected_tail_promotion_gate(
        _tail_law_table(),
        _q5_validation_table(parent_size_failure=False),
        require_c_hat_precision=False,
    )

    gaussian = promotion[promotion["source_family"].eq("gaussian_blobs")].iloc[0]
    assert gaussian["promotion_decision"] == "external_admissible"


def test_run_selected_tail_promotion_gate_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        records_path = Path(tmpdir) / "selected_ratio_tail_law.csv"
        q5_path = Path(tmpdir) / "q5.csv"
        output_dir = Path(tmpdir) / "out"
        _tail_law_table().to_csv(records_path, index=False)
        _q5_validation_table(parent_size_failure=True).to_csv(q5_path, index=False)

        outputs = run_selected_tail_promotion_gate(
            runs=(SelectedTailRun(run_id="test_run", path=records_path),),
            q5_validation_path=q5_path,
            output_dir=output_dir,
            require_c_hat_precision=False,
        )

        assert set(outputs) == {"contexts", "q5_gate", "summary", "manifest"}
        assert (output_dir / "selected_tail_promotion_contexts.csv").exists()
        assert (output_dir / "selected_tail_q5_promotion_gate.csv").exists()
        assert (output_dir / "selected_tail_promotion_summary.csv").exists()
        assert (output_dir / "manifest.json").exists()
