from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_tie_rank_neighborhood_join_audit as panel,
)


def _feasibility_row(
    *,
    case_id: str,
    proposal_family: str,
    bandwidth_band: str,
    bandwidth_count: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "base_case_id": case_id.split("__")[0],
        "data_role": "observed_target"
        if proposal_family == "observed_target"
        else "diagnostic_proposal",
        "calibration_role": "observed_target_not_null_support"
        if proposal_family == "observed_target"
        else "diagnostic_proposal_not_null_support",
        "proposal_family": proposal_family,
        "root_bandwidth_reopen_band": bandwidth_band,
        "root_bandwidth_reopen_count": bandwidth_count,
    }


def _mixed_row(
    *,
    case_id: str,
    frontier_count: int,
    bandwidth_count: int,
    locality_status: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "root_frontier_row_count": frontier_count,
        "root_bandwidth_reopen_count": bandwidth_count,
        "root_hybrid_strict_support_count": 0,
        "root_bandwidth_locality_status": locality_status,
    }


def _topology_row(
    *,
    case_id: str,
    reopens: bool,
    hybrid: bool = False,
    candidate_scope: str = "root_non_direct",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "candidate_scope": candidate_scope,
        "parent_id": "",
        "depth": 0,
        "bandwidth_reference_reopens": reopens,
        "bandwidth_reference_direct_positive_reopens": reopens,
        "interpolation_effective_support_pass": True,
        "hybrid_strict_support": hybrid,
    }


def test_neighborhood_join_audit_labels_missing_replay_and_join_states(
    tmp_path: Path,
) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    (matrix_dir / "generated_missing_frontier.csv").write_text("x\n1\n", encoding="utf-8")
    (matrix_dir / "generated_join_needed.csv").write_text("x\n1\n", encoding="utf-8")

    feasibility = pd.DataFrame.from_records(
        [
            _feasibility_row(
                case_id="observed",
                proposal_family="observed_target",
                bandwidth_band="bandwidth_root_reopen_observed",
                bandwidth_count=1.0,
            ),
            _feasibility_row(
                case_id="generated_missing_frontier",
                proposal_family="dense_family",
                bandwidth_band="bandwidth_reopen_missing",
                bandwidth_count=float("nan"),
            ),
            _feasibility_row(
                case_id="generated_join_needed",
                proposal_family="join_family",
                bandwidth_band="bandwidth_reopen_missing",
                bandwidth_count=float("nan"),
            ),
            _feasibility_row(
                case_id="generated_matrix_missing",
                proposal_family="missing_family",
                bandwidth_band="bandwidth_reopen_missing",
                bandwidth_count=float("nan"),
            ),
        ]
    )
    mixed = pd.DataFrame.from_records(
        [
            _mixed_row(
                case_id="generated_missing_frontier",
                frontier_count=0,
                bandwidth_count=0,
                locality_status="topology_frontier_not_joined",
            ),
            _mixed_row(
                case_id="generated_join_needed",
                frontier_count=0,
                bandwidth_count=0,
                locality_status="topology_frontier_not_joined",
            ),
        ]
    )
    topology = pd.DataFrame.from_records(
        [_topology_row(case_id="generated_join_needed", reopens=True)]
    )

    rows = panel.build_root_tie_rank_neighborhood_join_audit_rows(
        proposal_feasibility_rows=feasibility,
        proposal_mixed_rows=mixed,
        topology_frontier_rows=topology,
        generated_matrix_dir=matrix_dir,
    )
    by_case = rows.set_index("case_id")

    assert by_case.loc["observed", "bandwidth_join_status"] == (
        "bandwidth_measured_available"
    )
    assert by_case.loc["generated_missing_frontier", "bandwidth_join_status"] == (
        "generated_matrix_available_topology_frontier_absent_"
        "after_mixed_frontier_not_joined"
    )
    assert by_case.loc[
        "generated_missing_frontier",
        "neighborhood_replay_next_action",
    ] == "run_generated_measurability_and_topology_frontier_replay"
    assert by_case.loc["generated_join_needed", "bandwidth_join_status"] == (
        "topology_frontier_rows_present_reopen_not_joined"
    )
    assert by_case.loc[
        "generated_join_needed",
        "neighborhood_replay_next_action",
    ] == "join_topology_frontier_rows_before_feasibility"
    assert by_case.loc["generated_matrix_missing", "bandwidth_join_status"] == (
        "generated_matrix_missing_topology_frontier_absent"
    )


def test_neighborhood_join_audit_counts_direct_measurable_root_frontier(
    tmp_path: Path,
) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    (matrix_dir / "generated_direct_root.csv").write_text("x\n1\n", encoding="utf-8")

    feasibility = pd.DataFrame.from_records(
        [
            _feasibility_row(
                case_id="generated_direct_root",
                proposal_family="direct_family",
                bandwidth_band="bandwidth_reopen_missing",
                bandwidth_count=float("nan"),
            )
        ]
    )
    topology = pd.DataFrame.from_records(
        [
            _topology_row(
                case_id="generated_direct_root",
                reopens=False,
                candidate_scope="direct_measurable",
            )
        ]
    )

    rows = panel.build_root_tie_rank_neighborhood_join_audit_rows(
        proposal_feasibility_rows=feasibility,
        topology_frontier_rows=topology,
        generated_matrix_dir=matrix_dir,
    )
    row = rows.iloc[0]

    assert row["topology_frontier_root_row_count"] == 1
    assert row["bandwidth_join_status"] == (
        "topology_frontier_rows_present_no_reopen_not_joined"
    )
    assert row["neighborhood_replay_next_action"] == (
        "join_topology_frontier_rows_before_feasibility"
    )


def test_neighborhood_join_summary_reports_next_required_work(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    (matrix_dir / "generated_missing_frontier.csv").write_text("x\n1\n", encoding="utf-8")

    feasibility = pd.DataFrame.from_records(
        [
            _feasibility_row(
                case_id="generated_missing_frontier",
                proposal_family="dense_family",
                bandwidth_band="bandwidth_reopen_missing",
                bandwidth_count=float("nan"),
            )
        ]
    )
    mixed = pd.DataFrame.from_records(
        [
            _mixed_row(
                case_id="generated_missing_frontier",
                frontier_count=0,
                bandwidth_count=0,
                locality_status="topology_frontier_not_joined",
            )
        ]
    )
    rows = panel.build_root_tie_rank_neighborhood_join_audit_rows(
        proposal_feasibility_rows=feasibility,
        proposal_mixed_rows=mixed,
        generated_matrix_dir=matrix_dir,
    )
    summary = panel.summarize_root_tie_rank_neighborhood_join_audit_rows(rows)

    assert summary.iloc[0]["generated_replay_needed_count"] == 1
    assert summary.iloc[0]["summary_status"] == (
        "generated_topology_frontier_replay_needed"
    )


def test_neighborhood_join_audit_runner_writes_outputs(tmp_path: Path) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    (matrix_dir / "generated.csv").write_text("x\n1\n", encoding="utf-8")

    feasibility_path = tmp_path / "feasibility.csv"
    mixed_path = tmp_path / "mixed.csv"
    pd.DataFrame.from_records(
        [
            _feasibility_row(
                case_id="generated",
                proposal_family="dense_family",
                bandwidth_band="bandwidth_reopen_missing",
                bandwidth_count=float("nan"),
            )
        ]
    ).to_csv(feasibility_path, index=False)
    pd.DataFrame.from_records(
        [
            _mixed_row(
                case_id="generated",
                frontier_count=0,
                bandwidth_count=0,
                locality_status="topology_frontier_not_joined",
            )
        ]
    ).to_csv(mixed_path, index=False)

    outputs = panel.run_root_tie_rank_neighborhood_join_audit(
        panel.RootTieRankNeighborhoodJoinAuditConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=feasibility_path,
            proposal_mixed_rows_path=mixed_path,
            topology_frontier_rows_path=None,
            generated_matrix_dir=matrix_dir,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
