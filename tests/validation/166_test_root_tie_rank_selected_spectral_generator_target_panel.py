from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    root_tie_rank_selected_spectral_generator_target_panel as panel,
)


def _row(
    *,
    case_id: str,
    proposal_family: str,
    data_role: str,
    calibration_role: str,
    selected_ratio: float,
    tie_fraction: float,
    edge_margin: float,
    spectral_ratio: float,
    bandwidth_band: str = "bandwidth_root_reopen_observed",
    conditioning_target_case_id: str = "",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "proposal_family": proposal_family,
        "data_role": data_role,
        "calibration_role": calibration_role,
        "root_sibling_selected_ratio": selected_ratio,
        "root_tie_rank_median_fraction": tie_fraction,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
        "root_bandwidth_reopen_band": bandwidth_band,
        "conditioning_target_case_id": conditioning_target_case_id,
        "alpha_resolution_required_null_count": 99,
        "tail_precision_required_null_count": 1584,
        "additional_null_count_for_alpha_resolution": 99,
        "additional_null_count_for_tail_precision": 1584,
    }


def _target(
    case_id: str = "target",
    spectral_ratio: float = 5.0,
    selected_ratio: float = 100.0,
    edge_margin: float = 1000.0,
) -> dict[str, object]:
    return _row(
        case_id=case_id,
        proposal_family="observed_target",
        data_role="observed_target",
        calibration_role="observed_target_not_null_support",
        selected_ratio=selected_ratio,
        tie_fraction=0.80,
        edge_margin=edge_margin,
        spectral_ratio=spectral_ratio,
    )


def test_generator_targets_measure_required_spectral_lift() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(spectral_ratio=8.0),
            _row(
                case_id="coupled_low",
                proposal_family="coupled_edge_spectral_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=1000.0,
                tie_fraction=0.85,
                edge_margin=2000.0,
                spectral_ratio=2.0,
            ),
        ]
    )

    targets = panel.build_selected_spectral_generator_target_rows(rows)
    row = targets.iloc[0]

    assert row["proposal_family"] == "coupled_edge_spectral_proposal"
    assert row["eligible_generated_count"] == 1
    assert row["eligible_calibration_support_count"] == 0
    assert not bool(row["spectral_reach_after_current_generator"])
    assert row["spectral_lift_multiplier_required"] == pytest.approx(4.0)
    assert row["generator_target_status"] == (
        "diagnostic_generator_needs_spectral_lift_and_external_null_support"
    )
    assert row["next_generator_step"] == (
        "derive_spectral_lift_generator_with_external_null_semantics"
    )


def test_generator_targets_distinguish_calibration_reach_and_missing_stratum() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(case_id="easy", spectral_ratio=4.0),
            _target(case_id="hard", spectral_ratio=4.0, selected_ratio=1_000_000.0),
            _row(
                case_id="iid_reach",
                proposal_family="iid_marginal_bernoulli",
                data_role="selected_null",
                calibration_role="selected_null_candidate_support",
                selected_ratio=200.0,
                tie_fraction=0.90,
                edge_margin=3000.0,
                spectral_ratio=5.0,
            ),
            _row(
                case_id="diagnostic_low_action",
                proposal_family="sparse_block_spike_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=2.0,
                tie_fraction=0.90,
                edge_margin=2.0,
                spectral_ratio=10.0,
            ),
        ]
    )

    targets = panel.build_selected_spectral_generator_target_rows(rows)
    indexed = targets.set_index(["target_case_id", "proposal_family"])

    iid_easy = indexed.loc[("easy", "iid_marginal_bernoulli")]
    sparse_easy = indexed.loc[("easy", "sparse_block_spike_proposal")]
    iid_hard = indexed.loc[("hard", "iid_marginal_bernoulli")]

    assert iid_easy["generator_target_status"] == (
        "calibration_generator_reaches_spectral_target"
    )
    assert iid_easy["conditioning_stratum_status"] == (
        "family_stratum_has_selected_null_support"
    )
    assert sparse_easy["generator_target_status"] == (
        "conditioning_stratum_missing_for_family"
    )
    assert iid_hard["generator_target_status"] == (
        "conditioning_stratum_missing_for_family"
    )


def test_conditioned_rows_only_support_matching_target() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(case_id="target_a", spectral_ratio=10.0),
            _target(case_id="target_b", spectral_ratio=10.0),
            _row(
                case_id="reach_a",
                proposal_family="conditioned_coherent_rank_one_spike_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=1000.0,
                tie_fraction=0.90,
                edge_margin=3000.0,
                spectral_ratio=12.0,
                conditioning_target_case_id="target_a",
            ),
            _row(
                case_id="low_b",
                proposal_family="conditioned_coherent_rank_one_spike_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=1000.0,
                tie_fraction=0.90,
                edge_margin=3000.0,
                spectral_ratio=2.0,
                conditioning_target_case_id="target_b",
            ),
        ]
    )

    targets = panel.build_selected_spectral_generator_target_rows(rows)
    indexed = targets.set_index("target_case_id")

    assert indexed.loc["target_a", "eligible_generated_count"] == 1
    assert indexed.loc["target_a", "spectral_reach_after_current_generator"]
    assert indexed.loc["target_b", "eligible_generated_count"] == 1
    assert not indexed.loc["target_b", "spectral_reach_after_current_generator"]


def test_generator_target_summary_and_runner_write_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "combined.csv"
    rows = pd.DataFrame.from_records(
        [
            _target(spectral_ratio=8.0),
            _row(
                case_id="coupled_low",
                proposal_family="coupled_edge_spectral_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=1000.0,
                tie_fraction=0.85,
                edge_margin=2000.0,
                spectral_ratio=2.0,
            ),
        ]
    )
    rows.to_csv(input_path, index=False)

    targets = panel.build_selected_spectral_generator_target_rows(rows)
    summary = panel.summarize_selected_spectral_generator_target_rows(targets)

    assert summary.iloc[0]["summary_status"] == (
        "diagnostic_family_requires_spectral_lift_all_targets"
    )
    assert summary.iloc[0]["minimum_null_roots_for_alpha_resolution_total"] == 99
    assert summary.iloc[0]["minimum_null_roots_for_tail_precision_total"] == 1584

    outputs = panel.run_selected_spectral_generator_target_panel(
        panel.RootTieRankSelectedSpectralGeneratorTargetConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
