from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.selected import (
    root_selected_spectral_tail_law_panel as panel,
)


def _root_row(
    *,
    case_id: str,
    proposal_family: str,
    data_role: str,
    calibration_role: str,
    spectral_ratio: float,
    tie_fraction: float = 0.8,
    selected_ratio: float = 100.0,
    edge_margin: float = 1000.0,
    bandwidth_band: str = "bandwidth_root_reopen_observed",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "proposal_family": proposal_family,
        "data_role": data_role,
        "calibration_role": calibration_role,
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": tie_fraction,
        "root_sibling_selected_ratio": selected_ratio,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
        "root_bandwidth_reopen_band": bandwidth_band,
    }


def _target(case_id: str = "target", spectral_ratio: float = 5.0) -> dict[str, object]:
    return _root_row(
        case_id=case_id,
        proposal_family="observed_target",
        data_role="observed_target",
        calibration_role="observed_target_not_null_support",
        spectral_ratio=spectral_ratio,
    )


def test_root_spectral_tail_fails_closed_when_support_missing() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(spectral_ratio=5.0),
            _root_row(
                case_id="diagnostic",
                proposal_family="conditioned_coherent_rank_one_spike_proposal",
                data_role="diagnostic_proposal",
                calibration_role="diagnostic_proposal_not_null_support",
                spectral_ratio=10.0,
            ),
        ]
    )

    tail = panel.build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=rows,
    ).iloc[0]

    assert tail["selected_null_support_count"] == 0
    assert pd.isna(tail["conservative_spectral_tail_p_value"])
    assert tail["root_tail_inference_status"] == (
        "fail_closed_selected_root_spectral_tail_support_missing"
    )
    assert tail["next_mathematical_step"] == (
        "generate_selected_null_roots_in_same_root_tail_stratum"
    )


def test_root_spectral_tail_uses_conservative_empirical_support() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(case_id="overlap_mod_4c_small", spectral_ratio=4.0),
            _root_row(
                case_id="support_low",
                proposal_family="iid_marginal_bernoulli",
                data_role="selected_null",
                calibration_role="selected_null_candidate_support",
                spectral_ratio=2.0,
            ),
            _root_row(
                case_id="support_high",
                proposal_family="iid_marginal_bernoulli",
                data_role="selected_null",
                calibration_role="selected_null_candidate_support",
                spectral_ratio=5.0,
            ),
        ]
    )

    tail = panel.build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=rows,
    ).iloc[0]

    assert tail["selected_null_support_count"] == 2
    assert tail["selected_null_exceedance_count"] == 1
    assert tail["conservative_spectral_tail_p_value"] == pytest.approx(2 / 3)
    assert tail["root_tail_inference_status"] == (
        "calibrated_selected_root_spectral_tail_available"
    )


def test_root_spectral_tail_uses_importance_weighted_external_support() -> None:
    low_support = _root_row(
        case_id="external_low",
        proposal_family="importance_coupled_external_null",
        data_role="external_selected_null",
        calibration_role="external_null_support",
        spectral_ratio=2.0,
    )
    low_support["importance_log_weight"] = 0.0
    high_support = _root_row(
        case_id="external_high",
        proposal_family="importance_coupled_external_null",
        data_role="external_selected_null",
        calibration_role="external_null_support",
        spectral_ratio=5.0,
    )
    high_support["importance_log_weight"] = -1.3862943611198906
    rows = pd.DataFrame.from_records(
        [_target(case_id="target", spectral_ratio=4.0), low_support, high_support]
    )

    tail = panel.build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=rows,
    ).iloc[0]

    assert tail["selected_null_support_count"] == 2
    assert tail["selected_null_exceedance_count"] == 1
    assert tail["selected_null_importance_effective_sample_size"] == pytest.approx(
        1.4705882352941178
    )
    assert tail["selected_null_importance_weighted_exceedance_fraction"] == pytest.approx(0.2)
    assert tail["conservative_spectral_tail_p_value"] == pytest.approx(0.5238095238095238)
    assert tail["spectral_tail_p_value_status"] == (
        "importance_weighted_external_selected_null_tail"
    )


def test_root_spectral_tail_can_use_deformed_h_u_excess() -> None:
    rows = pd.DataFrame.from_records(
        [
            _target(case_id="target", spectral_ratio=10.0),
            _root_row(
                case_id="support_low_identity_high_deformed",
                proposal_family="importance_coupled_external_null",
                data_role="external_selected_null",
                calibration_role="external_null_support",
                spectral_ratio=1.1,
            ),
            _root_row(
                case_id="support_high_identity_low_deformed",
                proposal_family="importance_coupled_external_null",
                data_role="external_selected_null",
                calibration_role="external_null_support",
                spectral_ratio=20.0,
            ),
        ]
    )
    target_deformed = pd.DataFrame.from_records(
        [
            {
                "target_case_id": "target",
                "s_root_deformed_excess_log": 0.5,
            }
        ]
    )
    support_deformed = pd.DataFrame.from_records(
        [
            {
                "case_id": "support_low_identity_high_deformed",
                "s_root_deformed_excess_log": 0.6,
            },
            {
                "case_id": "support_high_identity_low_deformed",
                "s_root_deformed_excess_log": 0.1,
            },
        ]
    )

    tail = panel.build_root_selected_spectral_tail_law_rows(
        joined_feasibility_rows=rows,
        deformed_mp_edge_rows=target_deformed,
        deformed_mp_edge_support_rows=support_deformed,
        h_u_population_law_status="deformed_mp_edge_measured_support_side",
    ).iloc[0]

    assert tail["spectral_tail_variable"] == "deformed_mp_s_h_u"
    assert tail["s_root_spectral_excess_log"] == pytest.approx(0.5)
    assert tail["s_root_deformed_excess_log"] == pytest.approx(0.5)
    assert tail["selected_null_support_count"] == 2
    assert tail["selected_null_exceedance_count"] == 1
    assert tail["conservative_spectral_tail_p_value"] == pytest.approx(2 / 3)


def test_root_spectral_tail_runner_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.csv"
    pd.DataFrame.from_records([_target()]).to_csv(input_path, index=False)

    outputs = panel.run_root_selected_spectral_tail_law_panel(
        panel.RootSelectedSpectralTailLawConfig(
            output_dir=tmp_path / "out",
            joined_feasibility_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
