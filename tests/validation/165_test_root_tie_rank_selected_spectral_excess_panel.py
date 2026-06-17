from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    root_tie_rank_selected_spectral_excess_panel as panel,
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


def test_selected_spectral_excess_identifies_diagnostic_reach_not_calibration() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(),
            _row(
                case_id="diagnostic_reach",
                proposal_family="coupled_edge_spectral_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=2000.0,
                spectral_ratio=6.0,
            ),
            _row(
                case_id="selected_null_low",
                proposal_family="iid_marginal_bernoulli",
                data_role="selected_null",
                calibration_role="selected_null_candidate_support",
                selected_ratio=180.0,
                tie_fraction=0.85,
                edge_margin=1800.0,
                spectral_ratio=2.0,
            ),
        ]
    )

    spectral = panel.build_selected_spectral_excess_rows(rows)
    row = spectral.iloc[0]

    assert row["eligible_generated_count"] == 2
    assert row["eligible_calibration_support_count"] == 1
    assert row["spectral_exceedance_count"] == 1
    assert row["spectral_calibration_exceedance_count"] == 0
    assert row["empirical_conservative_spectral_tail_p_value"] == pytest.approx(0.5)
    assert row["best_generated_case_id"] == "diagnostic_reach"
    assert row["spectral_excess_status"] == (
        "spectral_excess_reached_by_diagnostic_proposal_not_calibration"
    )
    assert row["next_mathematical_step"] == (
        "convert_diagnostic_spectral_family_to_external_null_support_or_reject"
    )


def test_selected_spectral_excess_separates_hard_and_missing_support() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(case_id="hard", spectral_ratio=8.0),
            _row(
                case_id="eligible_low",
                proposal_family="coupled_edge_spectral_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=500.0,
                tie_fraction=0.80,
                edge_margin=5000.0,
                spectral_ratio=1.5,
            ),
            _target(
                case_id="missing",
                spectral_ratio=4.0,
                selected_ratio=1_000_000.0,
                edge_margin=1_000_000.0,
            ),
            _row(
                case_id="ineligible_low_action",
                proposal_family="sparse_block_spike_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=2.0,
                tie_fraction=0.80,
                edge_margin=2.0,
                spectral_ratio=10.0,
            ),
        ]
    )

    spectral = panel.build_selected_spectral_excess_rows(rows).set_index(
        "target_case_id"
    )

    assert spectral.loc["hard", "spectral_excess_status"] == (
        "spectral_excess_hard_residual"
    )
    assert spectral.loc["hard", "conditioning_support_status"] == (
        "diagnostic_only_support_external_null_missing"
    )
    assert spectral.loc["hard", "next_mathematical_step"] == (
        "derive_external_selected_spectral_excess_law_for_diagnostic_stratum"
    )
    assert spectral.loc["missing", "spectral_excess_status"] == (
        "spectral_excess_conditioning_support_missing"
    )
    assert spectral.loc["missing", "next_mathematical_step"] == (
        "generate_high_action_edge_tie_measured_selected_null_roots"
    )


def test_selected_spectral_excess_summary_and_runner_write_outputs(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "combined.csv"
    rows = pd.DataFrame.from_records(
        [
            _target(),
            _row(
                case_id="diagnostic_reach",
                proposal_family="coupled_edge_spectral_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=2000.0,
                spectral_ratio=6.0,
            ),
        ]
    )
    rows.to_csv(input_path, index=False)

    spectral = panel.build_selected_spectral_excess_rows(rows)
    summary = panel.summarize_selected_spectral_excess_rows(spectral)
    assert summary.iloc[0]["summary_status"] == (
        "diagnostic_spectral_reach_requires_external_null_support"
    )

    outputs = panel.run_selected_spectral_excess_panel(
        panel.RootTieRankSelectedSpectralExcessConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
