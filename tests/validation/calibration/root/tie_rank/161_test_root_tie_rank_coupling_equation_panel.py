from __future__ import annotations

from pathlib import Path

from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_coupling_equation_panel as panel,
)
from tests.validation.calibration.root.tie_rank.fixtures import feasibility_rows


def _combined_rows():
    return feasibility_rows(
        full=(140.0, 0.85, 1400.0, 6.0),
        dense=(3000.0, 0.82, 5000.0, 1.2),
        sparse=(8.0, 0.83, 15.0, 8.0),
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
