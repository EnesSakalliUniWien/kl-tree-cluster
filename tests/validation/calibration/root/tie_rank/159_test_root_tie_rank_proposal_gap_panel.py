from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_proposal_gap_panel as panel,
)


def _key(*, tie: str, edge: str, spectral: str, bandwidth: str) -> str:
    return "|".join(
        (
            "component=discrete_tie_rank_region",
            f"tie={tie}",
            f"edge={edge}",
            f"spectral={spectral}",
            f"bandwidth={bandwidth}",
        )
    )


def _row(
    *,
    case_id: str,
    proposal_family: str,
    calibration_role: str,
    tie_band: str,
    edge_band: str,
    spectral_band: str,
    bandwidth_band: str,
    selected_ratio: float,
    tie_fraction: float,
    edge_margin: float,
    spectral_ratio: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "observed_target"
        if proposal_family == "observed_target"
        else "diagnostic_proposal",
        "calibration_role": calibration_role,
        "proposal_family": proposal_family,
        "root_conditioning_stratum_key": _key(
            tie=tie_band,
            edge=edge_band,
            spectral=spectral_band,
            bandwidth=bandwidth_band,
        ),
        "root_tie_rank_band": tie_band,
        "root_edge_margin_band": edge_band,
        "root_spectral_ratio_band": spectral_band,
        "root_bandwidth_reopen_band": bandwidth_band,
        "root_sibling_selected_ratio": selected_ratio,
        "root_tie_rank_median_fraction": tie_fraction,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
    }


def _combined_rows() -> pd.DataFrame:
    target = _row(
        case_id="target",
        proposal_family="observed_target",
        calibration_role="observed_target_not_null_support",
        tie_band="tie_rank_high_0_75_0_90",
        edge_band="edge_log_margin_high_ge_7",
        spectral_band="spectral_ratio_gt_4",
        bandwidth_band="bandwidth_root_reopen_observed",
        selected_ratio=100.0,
        tie_fraction=0.82,
        edge_margin=1500.0,
        spectral_ratio=5.0,
    )
    return pd.DataFrame.from_records(
        [
            target,
            _row(
                case_id="exact",
                proposal_family="exact_family",
                calibration_role="diagnostic_proposal_not_null_support",
                tie_band="tie_rank_high_0_75_0_90",
                edge_band="edge_log_margin_high_ge_7",
                spectral_band="spectral_ratio_gt_4",
                bandwidth_band="bandwidth_root_reopen_observed",
                selected_ratio=120.0,
                tie_fraction=0.84,
                edge_margin=1700.0,
                spectral_ratio=5.5,
            ),
            _row(
                case_id="dense",
                proposal_family="dense_family",
                calibration_role="diagnostic_proposal_not_null_support",
                tie_band="tie_rank_high_0_75_0_90",
                edge_band="edge_log_margin_high_ge_7",
                spectral_band="spectral_ratio_1_2",
                bandwidth_band="bandwidth_reopen_missing",
                selected_ratio=150.0,
                tie_fraction=0.80,
                edge_margin=2000.0,
                spectral_ratio=1.5,
            ),
            _row(
                case_id="sparse",
                proposal_family="sparse_family",
                calibration_role="diagnostic_proposal_not_null_support",
                tie_band="tie_rank_high_0_75_0_90",
                edge_band="edge_log_margin_low_lt_5",
                spectral_band="spectral_ratio_gt_4",
                bandwidth_band="bandwidth_reopen_missing",
                selected_ratio=5.0,
                tie_fraction=0.83,
                edge_margin=10.0,
                spectral_ratio=6.0,
            ),
        ]
    )


def test_gap_rows_classify_exact_action_and_spectral_failure_modes() -> None:
    rows = panel.build_root_tie_rank_proposal_gap_rows(_combined_rows())
    by_family = rows.set_index("proposal_family")

    assert by_family.loc["exact_family", "exact_stratum_hit"]
    assert by_family.loc["exact_family", "gap_pattern"] == "exact_target_stratum_hit"

    dense = by_family.loc["dense_family"]
    assert dense["selected_ratio_exceeds_target"]
    assert dense["edge_band_match"]
    assert not dense["spectral_band_match"]
    assert dense["gap_pattern"] == "action_edge_without_spectral"

    sparse = by_family.loc["sparse_family"]
    assert sparse["spectral_band_match"]
    assert not sparse["edge_band_match"]
    assert not sparse["selected_ratio_exceeds_target"]
    assert sparse["gap_pattern"] == "spectral_without_edge_action"


def test_gap_summary_reports_family_failure_modes() -> None:
    rows = panel.build_root_tie_rank_proposal_gap_rows(_combined_rows())
    summary = panel.summarize_root_tie_rank_proposal_gap_rows(rows).set_index("proposal_family")

    assert summary.loc["exact_family", "summary_status"] == ("proposal_hits_observed_target_strata")
    assert summary.loc["dense_family", "summary_status"] == ("action_edge_without_spectral")
    assert summary.loc["sparse_family", "summary_status"] == ("spectral_without_edge_action")


def test_gap_panel_runner_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "combined.csv"
    _combined_rows().to_csv(input_path, index=False)

    outputs = panel.run_root_tie_rank_proposal_gap_panel(
        panel.RootTieRankProposalGapPanelConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
