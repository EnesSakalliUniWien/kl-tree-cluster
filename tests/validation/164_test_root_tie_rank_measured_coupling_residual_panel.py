from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    root_tie_rank_measured_coupling_residual_panel as panel,
)


def _row(
    *,
    target: str,
    family: str,
    generated: str,
    measured_ratio: float,
    measured_dominates: bool,
    target_tie: float = 0.8,
    generated_tie: float = 0.8,
    target_action: float = 5.0,
    generated_action: float = 5.0,
    target_edge: float = 5.0,
    generated_edge: float = 5.0,
    target_spectral: float = 1.5,
    generated_spectral: float = 1.5,
    action_edge_deficit: float = 0.0,
    measured_deficit: float = 0.0,
    pattern: str = "partial_coupling_match",
    bandwidth: str = "bandwidth_band_match",
) -> dict[str, object]:
    target_action_edge = target_tie * min(target_action, target_edge)
    generated_action_edge = generated_tie * min(generated_action, generated_edge)
    target_bottleneck = target_action_edge * target_spectral
    generated_bottleneck = generated_action_edge * generated_spectral
    return {
        "target_case_id": target,
        "proposal_family": family,
        "best_generated_case_id": generated,
        "target_tie_fraction": target_tie,
        "generated_tie_fraction": generated_tie,
        "target_action_log": target_action,
        "generated_action_log": generated_action,
        "target_edge_log": target_edge,
        "generated_edge_log": generated_edge,
        "target_spectral_excess_log": target_spectral,
        "generated_spectral_excess_log": generated_spectral,
        "target_action_edge_bottleneck": target_action_edge,
        "generated_action_edge_bottleneck": generated_action_edge,
        "target_bottleneck_coupling": target_bottleneck,
        "generated_bottleneck_coupling": generated_bottleneck,
        "bottleneck_coupling_ratio": generated_bottleneck / target_bottleneck,
        "target_measured_neighborhood_coupling": target_bottleneck,
        "generated_measured_neighborhood_coupling": target_bottleneck
        * measured_ratio,
        "measured_neighborhood_coupling_deficit": measured_deficit,
        "measured_neighborhood_coupling_ratio": measured_ratio,
        "measured_neighborhood_coupling_dominates": measured_dominates,
        "generated_neighborhood_measured": True,
        "bandwidth_gap_status": bandwidth,
        "coupling_deficit_score": measured_deficit + action_edge_deficit,
        "coupling_pattern": pattern,
    }


def test_residual_rows_pick_best_measured_proposal_and_label_diagnostic_reach() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                target="easy",
                family="iid_marginal_bernoulli",
                generated="iid",
                measured_ratio=0.4,
                measured_dominates=False,
                measured_deficit=2.0,
            ),
            _row(
                target="easy",
                family="two_block_tilt_proposal",
                generated="tilt",
                measured_ratio=1.2,
                measured_dominates=True,
                measured_deficit=0.0,
                pattern="measured_neighborhood_coupling_dominates",
            ),
        ]
    )

    residual = panel.build_measured_coupling_residual_rows(rows)
    row = residual.iloc[0]

    assert row["best_proposal_family"] == "two_block_tilt_proposal"
    assert bool(row["measured_neighborhood_coupling_reached"])
    assert row["dominant_residual_axis"] == "none_measured_coupling_reached"
    assert row["target_resolution_status"] == (
        "diagnostic_measured_coupling_reaches_target_not_calibration"
    )
    assert row["next_mathematical_step"] == (
        "convert_diagnostic_family_to_external_null_support_or_reject"
    )


def test_residual_rows_localize_spectral_and_action_edge_deficits() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                target="spectral_hard",
                family="coupled_edge_spectral_proposal",
                generated="dense",
                measured_ratio=0.35,
                measured_dominates=False,
                generated_spectral=0.3,
                measured_deficit=4.0,
                pattern="action_edge_high_spectral_low",
            ),
            _row(
                target="action_hard",
                family="sparse_block_spike_proposal",
                generated="sparse",
                measured_ratio=0.30,
                measured_dominates=False,
                generated_action=1.0,
                generated_edge=1.0,
                generated_spectral=1.8,
                action_edge_deficit=3.2,
                measured_deficit=4.2,
                pattern="spectral_high_action_edge_low",
            ),
        ]
    )

    residual = panel.build_measured_coupling_residual_rows(rows).set_index(
        "target_case_id"
    )

    assert residual.loc["spectral_hard", "dominant_residual_axis"] == (
        "spectral_excess"
    )
    assert residual.loc["spectral_hard", "next_mathematical_step"] == (
        "derive_selected_spectral_excess_given_high_action_edge_tie_rank"
    )
    assert residual.loc["action_hard", "dominant_residual_axis"] == (
        "action_edge_bottleneck"
    )
    assert residual.loc["action_hard", "dominant_action_edge_axis"] == (
        "action_and_edge_joint_deficit"
    )
    assert residual.loc["action_hard", "next_mathematical_step"] == (
        "derive_joint_action_edge_generator_with_spectral_persistence"
    )


def test_residual_summary_and_runner_write_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "coupling.csv"
    coupling = pd.DataFrame.from_records(
        [
            _row(
                target="easy",
                family="two_block_tilt_proposal",
                generated="tilt",
                measured_ratio=1.1,
                measured_dominates=True,
            ),
            _row(
                target="hard",
                family="coupled_edge_spectral_proposal",
                generated="dense",
                measured_ratio=0.4,
                measured_dominates=False,
                generated_spectral=0.2,
                measured_deficit=5.0,
            ),
        ]
    )
    coupling.to_csv(input_path, index=False)

    rows = panel.build_measured_coupling_residual_rows(coupling)
    summary = panel.summarize_measured_coupling_residual_rows(rows)
    assert set(summary["target_resolution_status"]) == {
        "diagnostic_measured_coupling_reaches_target_not_calibration",
        "hard_measured_coupling_residual",
    }
    assert (
        summary.loc[
            summary["target_resolution_status"].eq("hard_measured_coupling_residual"),
            "summary_status",
        ].iloc[0]
        == "hard_roots_require_new_selected_coupling_law"
    )

    outputs = panel.run_measured_coupling_residual_panel(
        panel.RootTieRankMeasuredCouplingResidualConfig(
            output_dir=tmp_path / "out",
            coupling_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()


def test_relative_deficit_handles_zero_target() -> None:
    assert panel._relative_deficit(0.0, 0.0) == pytest.approx(0.0)
