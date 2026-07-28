from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_coupling_equation_panel as panel,
)


def _row(
    *,
    case_id: str,
    proposal_family: str,
    calibration_role: str,
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
        "root_bandwidth_reopen_band": bandwidth_band,
        "root_sibling_selected_ratio": selected_ratio,
        "root_tie_rank_median_fraction": tie_fraction,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
    }


def _combined_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            _row(
                case_id="target",
                proposal_family="observed_target",
                calibration_role="observed_target_not_null_support",
                bandwidth_band="bandwidth_root_reopen_observed",
                selected_ratio=100.0,
                tie_fraction=0.80,
                edge_margin=1000.0,
                spectral_ratio=5.0,
            ),
            _row(
                case_id="full",
                proposal_family="full_family",
                calibration_role="diagnostic_proposal_not_null_support",
                bandwidth_band="bandwidth_root_reopen_observed",
                selected_ratio=140.0,
                tie_fraction=0.85,
                edge_margin=1400.0,
                spectral_ratio=6.0,
            ),
            _row(
                case_id="dense",
                proposal_family="dense_family",
                calibration_role="diagnostic_proposal_not_null_support",
                bandwidth_band="bandwidth_reopen_missing",
                selected_ratio=3000.0,
                tie_fraction=0.82,
                edge_margin=5000.0,
                spectral_ratio=1.2,
            ),
            _row(
                case_id="sparse",
                proposal_family="sparse_family",
                calibration_role="diagnostic_proposal_not_null_support",
                bandwidth_band="bandwidth_reopen_missing",
                selected_ratio=8.0,
                tie_fraction=0.83,
                edge_margin=15.0,
                spectral_ratio=8.0,
            ),
        ]
    )


def test_coupling_rows_classify_joint_failure_modes() -> None:
    rows = panel.build_root_tie_rank_coupling_equation_rows(_combined_rows())
    by_family = rows.set_index("proposal_family")

    full = by_family.loc["full_family"]
    assert full["bottleneck_coupling_dominates"]
    assert full["measured_neighborhood_coupling_dominates"]
    assert full["coupling_pattern"] == "measured_neighborhood_coupling_dominates"

    dense = by_family.loc["dense_family"]
    assert dense["action_edge_bottleneck_dominates"]
    assert not dense["spectral_excess_dominates"]
    assert not dense["bottleneck_coupling_dominates"]
    assert dense["coupling_pattern"] == "action_edge_high_spectral_low"

    sparse = by_family.loc["sparse_family"]
    assert sparse["spectral_excess_dominates"]
    assert not sparse["action_edge_bottleneck_dominates"]
    assert not sparse["bottleneck_coupling_dominates"]
    assert sparse["coupling_pattern"] == "spectral_high_action_edge_low"


def test_coupling_summary_reports_family_status() -> None:
    rows = panel.build_root_tie_rank_coupling_equation_rows(_combined_rows())
    summary = panel.summarize_root_tie_rank_coupling_equation_rows(rows).set_index(
        "proposal_family"
    )

    assert summary.loc["full_family", "summary_status"] == (
        "measured_neighborhood_coupling_reached"
    )
    assert summary.loc["dense_family", "summary_status"] == (
        "action_edge_bottleneck_spectral_deficit"
    )
    assert summary.loc["sparse_family", "summary_status"] == ("spectral_excess_action_edge_deficit")


def test_coupling_panel_runner_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "combined.csv"
    _combined_rows().to_csv(input_path, index=False)

    outputs = panel.run_root_tie_rank_coupling_equation_panel(
        panel.RootTieRankCouplingEquationConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
