from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_spectral_lift_parameter_sweep as sweep,
)


def _target(
    *,
    case_id: str = "target",
    selected_ratio: float = 100.0,
    tie_fraction: float = 0.8,
    edge_margin: float = 1000.0,
    spectral_ratio: float = 8.0,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "root_sibling_selected_ratio": selected_ratio,
        "root_tie_rank_median_fraction": tie_fraction,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
    }


def _generated(
    *,
    case_id: str,
    setting: str,
    selected_ratio: float,
    tie_fraction: float,
    edge_margin: float,
    spectral_ratio: float,
    family: str = sweep.COUPLED_EDGE_SPECTRAL_PROPOSAL,
    base_case_id: str = "base",
    two_block_delta: float = 0.45,
    spike_fraction: float = 0.2,
    spike_delta: float = 0.65,
    conditioning_target_case_id: str = "",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "sweep_setting_id": setting,
        "proposal_family": family,
        "proposal_two_block_delta": two_block_delta,
        "proposal_spike_feature_fraction": spike_fraction,
        "proposal_spike_delta": spike_delta,
        "conditioning_target_case_id": conditioning_target_case_id,
        "root_sibling_selected_ratio": selected_ratio,
        "root_tie_rank_median_fraction": tie_fraction,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
    }


def test_spectral_lift_target_rows_find_replay_candidate_and_lift_gap() -> None:
    observed = pd.DataFrame.from_records([_target(spectral_ratio=8.0)])
    generated = pd.DataFrame.from_records(
        [
            _generated(
                case_id="low",
                setting="setting_a",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=3000.0,
                spectral_ratio=2.0,
            ),
            _generated(
                case_id="reach",
                setting="setting_b",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=3000.0,
                spectral_ratio=10.0,
                two_block_delta=0.55,
            ),
            _generated(
                case_id="missing_action",
                setting="setting_c",
                selected_ratio=1.0,
                tie_fraction=0.85,
                edge_margin=1.0,
                spectral_ratio=100.0,
                two_block_delta=0.65,
            ),
        ]
    )

    rows = sweep.build_spectral_lift_sweep_target_rows(
        observed_mixed_rows=observed,
        generated_mixed_rows=generated,
    ).set_index("sweep_setting_id")

    assert rows.loc["setting_a", "root_metric_screen_status"] == (
        "root_metric_spectral_lift_still_required"
    )
    assert rows.loc["setting_a", "spectral_lift_multiplier_required"] == pytest.approx(4.0)
    assert rows.loc["setting_b", "root_metric_screen_status"] == (
        "root_metric_spectral_reach_replay_needed"
    )
    assert rows.loc["setting_b", "next_replay_step"] == (
        "replay_setting_through_selected_neighborhood"
    )
    assert rows.loc["setting_c", "root_metric_screen_status"] == (
        "root_metric_conditioning_missing"
    )


def test_spectral_lift_summary_reports_setting_statuses() -> None:
    observed = pd.DataFrame.from_records([_target(spectral_ratio=8.0)])
    generated = pd.DataFrame.from_records(
        [
            _generated(
                case_id="low",
                setting="setting_a",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=3000.0,
                spectral_ratio=2.0,
            ),
            _generated(
                case_id="reach",
                setting="setting_b",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=3000.0,
                spectral_ratio=10.0,
                two_block_delta=0.55,
            ),
        ]
    )
    targets = sweep.build_spectral_lift_sweep_target_rows(
        observed_mixed_rows=observed,
        generated_mixed_rows=generated,
    )
    summary = sweep.summarize_spectral_lift_sweep_targets(
        target_rows=targets,
        generated_mixed_rows=generated,
    ).set_index("sweep_setting_id")

    assert summary.loc["setting_a", "summary_status"] == ("setting_requires_more_spectral_lift")
    assert summary.loc["setting_b", "summary_status"] == (
        "setting_reaches_all_targets_root_metrics_replay_needed"
    )
    assert summary.loc["setting_b", "generated_root_count"] == 1


def test_setting_records_include_coherent_rank_one_spike_proposal() -> None:
    settings = sweep._setting_records(
        proposal_families=(sweep.COHERENT_RANK_ONE_SPIKE_PROPOSAL,),
        two_block_delta_grid=(0.45,),
        spike_feature_fraction_grid=(0.25,),
        spike_delta_grid=(0.3,),
    )

    assert len(settings) == 1
    setting = settings[0]
    assert setting["sweep_setting_id"] == (
        f"{sweep.COHERENT_RANK_ONE_SPIKE_PROPOSAL}__sf0_250__sd0_300"
    )
    assert setting["proposal_family"] == sweep.COHERENT_RANK_ONE_SPIKE_PROPOSAL
    assert math.isnan(float(setting["proposal_two_block_delta"]))
    assert setting["proposal_spike_feature_fraction"] == pytest.approx(0.25)
    assert setting["proposal_spike_delta"] == pytest.approx(0.3)


def test_generate_coherent_rank_one_spike_matrix_has_expected_shape_and_metadata() -> None:
    matrix, metadata = sweep.generate_coherent_rank_one_spike_matrix(
        base_case={
            "name": "probe",
            "n_samples": 20,
            "n_features": 40,
            "n_clusters": 4,
            "feature_sparsity": 0.05,
        },
        seed=123,
        spike_feature_fraction=0.25,
        spike_delta=0.3,
    )

    assert matrix.shape == (20, 40)
    assert metadata["proposal_spike_feature_count"] == 10
    assert math.isnan(float(metadata["proposal_two_block_delta"]))
    assert metadata["proposal_spike_feature_fraction"] == pytest.approx(0.25)
    assert metadata["proposal_spike_delta"] == pytest.approx(0.3)
    assert matrix.sum(axis=0).min() >= 1
    assert matrix.mean() == pytest.approx(metadata["generated_matrix_density"])


def test_conditioned_coherent_settings_use_target_geometry_without_spectral_target() -> None:
    observed = pd.DataFrame.from_records(
        [
            _target(
                case_id="easy",
                selected_ratio=10.0,
                tie_fraction=0.5,
                edge_margin=20.0,
                spectral_ratio=100.0,
            ),
            _target(
                case_id="hard",
                selected_ratio=1000.0,
                tie_fraction=1.0,
                edge_margin=2000.0,
                spectral_ratio=2.0,
            ),
        ]
    )

    settings = sweep._conditioned_coherent_spike_setting_records(observed)
    by_target = {str(row["conditioning_target_case_id"]): row for row in settings}

    assert set(by_target) == {"easy", "hard"}
    assert (
        by_target["hard"]["proposal_spike_feature_fraction"]
        < by_target["easy"]["proposal_spike_feature_fraction"]
    )
    assert by_target["hard"]["proposal_spike_delta"] > by_target["easy"]["proposal_spike_delta"]
    assert by_target["easy"]["conditioning_geometry_score"] == pytest.approx(0.0)
    assert by_target["hard"]["conditioning_geometry_score"] == pytest.approx(1.0)


def test_conditioned_generated_rows_only_support_matching_target() -> None:
    observed = pd.DataFrame.from_records(
        [
            _target(case_id="target_a", spectral_ratio=10.0),
            _target(case_id="target_b", spectral_ratio=10.0),
        ]
    )
    generated = pd.DataFrame.from_records(
        [
            _generated(
                case_id="a_reach",
                setting="conditioned_a",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=3000.0,
                spectral_ratio=12.0,
                conditioning_target_case_id="target_a",
            ),
            _generated(
                case_id="b_low",
                setting="conditioned_b",
                selected_ratio=200.0,
                tie_fraction=0.85,
                edge_margin=3000.0,
                spectral_ratio=2.0,
                conditioning_target_case_id="target_b",
            ),
        ]
    )

    rows = sweep.build_spectral_lift_sweep_target_rows(
        observed_mixed_rows=observed,
        generated_mixed_rows=generated,
    )

    assert set(rows["target_case_id"]) == {"target_a", "target_b"}
    assert set(rows["sweep_setting_id"]) == {"conditioned_a", "conditioned_b"}
    row_map = rows.set_index(["target_case_id", "sweep_setting_id"])
    assert row_map.loc[
        ("target_a", "conditioned_a"),
        "spectral_reach_after_setting",
    ]
    assert not row_map.loc[
        ("target_b", "conditioned_b"),
        "spectral_reach_after_setting",
    ]


def test_spectral_lift_runner_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    def fake_evaluate(
        config: sweep.RootTieRankSpectralLiftParameterSweepConfig,
    ) -> dict[str, pd.DataFrame]:
        generated = pd.DataFrame.from_records(
            [
                _generated(
                    case_id="reach",
                    setting="setting_b",
                    selected_ratio=200.0,
                    tie_fraction=0.85,
                    edge_margin=3000.0,
                    spectral_ratio=10.0,
                )
            ]
        )
        target_rows = sweep.build_spectral_lift_sweep_target_rows(
            observed_mixed_rows=pd.DataFrame.from_records([_target()]),
            generated_mixed_rows=generated,
        )
        summary = sweep.summarize_spectral_lift_sweep_targets(
            target_rows=target_rows,
            generated_mixed_rows=generated,
        )
        return {
            "generated_rows": generated,
            "target_rows": target_rows,
            "summary": summary,
            "failures": pd.DataFrame(columns=sweep.FAILURE_COLUMNS),
        }

    monkeypatch.setattr(sweep, "evaluate_spectral_lift_parameter_sweep", fake_evaluate)

    outputs = sweep.run_spectral_lift_parameter_sweep(
        sweep.RootTieRankSpectralLiftParameterSweepConfig(output_dir=tmp_path / "out")
    )

    assert set(outputs) == {
        "generated_rows",
        "target_rows",
        "summary",
        "failures",
        "manifest",
    }
    for path in outputs.values():
        assert path.exists()
