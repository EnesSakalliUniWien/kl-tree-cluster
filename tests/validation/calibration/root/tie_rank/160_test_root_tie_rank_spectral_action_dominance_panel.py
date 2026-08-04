from __future__ import annotations

from pathlib import Path

from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_spectral_action_dominance_panel as panel,
)
from tests.validation.calibration.root.tie_rank.fixtures import feasibility_rows


def _combined_rows():
    return feasibility_rows(
        full=(120.0, 0.85, 1200.0, 6.0),
        dense=(150.0, 0.82, 2000.0, 1.5),
        sparse=(5.0, 0.83, 20.0, 6.0),
    )


def test_dominance_rows_classify_continuous_failure_modes() -> None:
    rows = panel.build_root_tie_rank_spectral_action_dominance_rows(_combined_rows())
    by_family = rows.set_index("proposal_family")

    full = by_family.loc["full_family"]
    assert full["full_continuous_dominates"]
    assert full["bandwidth_measured_match"]
    assert full["dominance_pattern"] == "full_continuous_and_bandwidth_dominance"

    dense = by_family.loc["dense_family"]
    assert dense["action_edge_dominates"]
    assert not dense["spectral_dominates"]
    assert dense["dominance_pattern"] == "action_edge_dominance_spectral_deficit"

    sparse = by_family.loc["sparse_family"]
    assert sparse["spectral_dominates"]
    assert not sparse["selected_ratio_dominates"]
    assert not sparse["edge_dominates"]
    assert sparse["dominance_pattern"] == "spectral_dominance_action_edge_deficit"


def test_dominance_summary_reports_family_status() -> None:
    rows = panel.build_root_tie_rank_spectral_action_dominance_rows(_combined_rows())
    summary = panel.summarize_root_tie_rank_spectral_action_dominance_rows(rows).set_index(
        "proposal_family"
    )

    assert summary.loc["full_family", "summary_status"] == (
        "full_continuous_and_bandwidth_dominance_observed"
    )
    assert summary.loc["dense_family", "summary_status"] == (
        "action_edge_dominance_spectral_deficit"
    )
    assert summary.loc["sparse_family", "summary_status"] == (
        "spectral_dominance_action_edge_deficit"
    )


def test_dominance_panel_runner_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "combined.csv"
    _combined_rows().to_csv(input_path, index=False)

    outputs = panel.run_root_tie_rank_spectral_action_dominance_panel(
        panel.RootTieRankSpectralActionDominanceConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
