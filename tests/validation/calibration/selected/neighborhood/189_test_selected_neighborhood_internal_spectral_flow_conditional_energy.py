from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.selected.neighborhood import (
    selected_neighborhood_internal_spectral_flow_conditional_energy as panel,
)


def _energy_row(**overrides: object) -> dict[str, object]:
    row = {
        "case_id": "overlap_unbal_6c_med",
        "data_role": "signal",
        "method_id": "method",
        "replicate": 0,
        "edge_count": 10,
        "leaf_mp_supported_edge_count": 3,
        "internal_mp_supported_edge_count": 5,
        "strict_shared_mp_supported_edge_count": 3,
        "internal_only_mp_supported_edge_count": 2,
        "delta_strict_shared_mp_joint_transport_energy": -0.10,
        "internal_only_mp_joint_transport_energy": 1.0,
        "neighborhood_energy_status": ("diagnostic_internal_smooths_strict_shared_transport"),
    }
    row.update(overrides)
    return row


def _validity(
    case_id: str,
    *,
    validity_status: str = "root_validity_supported_by_stability_and_selection",
    tail_status: str = "calibrated_selected_root_spectral_tail_available",
    usability_status: str = "usable_selected_root_tail_after_validity_replay",
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "root_validity_status": validity_status,
        "root_tail_inference_status": tail_status,
        "selected_root_usability_status": usability_status,
    }


def test_paired_selected_null_warning_blocks_signal_rescue() -> None:
    energy = pd.DataFrame.from_records(
        [
            _energy_row(),
            _energy_row(
                data_role="selected_null",
                internal_only_mp_supported_edge_count=1,
            ),
        ]
    )
    root_validity = pd.DataFrame.from_records([_validity("overlap_unbal_6c_med")])

    rows = panel.build_conditional_internal_energy_rows(
        energy,
        root_validity_rows=root_validity,
    )
    by_role = rows.set_index("data_role")

    assert bool(by_role.loc["selected_null", "selected_null_internal_energy_warning"])
    assert not bool(by_role.loc["signal", "conditional_internal_energy_rescue_candidate"])
    assert by_role.loc["signal", "conditional_energy_status"] == (
        "fail_closed_paired_selected_null_internal_energy_warning"
    )


def test_valid_signal_without_null_warning_becomes_diagnostic_candidate() -> None:
    energy = pd.DataFrame.from_records(
        [
            _energy_row(),
            _energy_row(
                data_role="selected_null",
                internal_only_mp_supported_edge_count=0,
                neighborhood_energy_status="selected_null_energy_clean",
            ),
        ]
    )
    root_validity = pd.DataFrame.from_records([_validity("overlap_unbal_6c_med")])

    rows = panel.build_conditional_internal_energy_rows(
        energy,
        root_validity_rows=root_validity,
    )
    signal = rows[rows["data_role"].eq("signal")].iloc[0]

    assert bool(signal["conditional_internal_energy_rescue_candidate"])
    assert signal["conditional_energy_status"] == ("conditional_internal_energy_rescue_candidate")


def test_root_context_blocks_before_energy_candidate() -> None:
    energy = pd.DataFrame.from_records(
        [
            _energy_row(case_id="overlap_mod_4c_small"),
            _energy_row(
                case_id="overlap_mod_4c_small",
                data_role="selected_null",
                internal_only_mp_supported_edge_count=0,
            ),
        ]
    )
    root_validity = pd.DataFrame.from_records(
        [
            _validity(
                "overlap_mod_4c_small",
                validity_status="root_validity_failed_feature_subsample_replay",
                tail_status="calibrated_selected_root_spectral_tail_available",
                usability_status="fail_closed_root_validity_failed",
            )
        ]
    )

    rows = panel.build_conditional_internal_energy_rows(
        energy,
        root_validity_rows=root_validity,
    )
    signal = rows[rows["data_role"].eq("signal")].iloc[0]

    assert not bool(signal["conditional_internal_energy_rescue_candidate"])
    assert signal["conditional_energy_status"] == ("fail_closed_root_validity_missing_or_failed")


def test_strict_shared_energy_must_smooth() -> None:
    energy = pd.DataFrame.from_records(
        [
            _energy_row(
                delta_strict_shared_mp_joint_transport_energy=0.10,
                neighborhood_energy_status=("diagnostic_internal_degrades_strict_shared_transport"),
            ),
            _energy_row(
                data_role="selected_null",
                internal_only_mp_supported_edge_count=0,
            ),
        ]
    )
    root_validity = pd.DataFrame.from_records([_validity("overlap_unbal_6c_med")])

    rows = panel.build_conditional_internal_energy_rows(
        energy,
        root_validity_rows=root_validity,
    )
    signal = rows[rows["data_role"].eq("signal")].iloc[0]

    assert signal["conditional_energy_status"] == ("fail_closed_strict_shared_energy_not_smoothing")


def test_conditional_energy_panel_writes_outputs(tmp_path: Path) -> None:
    energy_path = tmp_path / "energy.csv"
    validity_path = tmp_path / "validity.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame.from_records(
        [
            _energy_row(),
            _energy_row(data_role="selected_null", internal_only_mp_supported_edge_count=0),
        ]
    ).to_csv(energy_path, index=False)
    pd.DataFrame.from_records([_validity("overlap_unbal_6c_med")]).to_csv(
        validity_path,
        index=False,
    )

    outputs = panel.run_conditional_internal_energy_panel(
        panel.ConditionalInternalEnergyConfig(
            output_dir=output_dir,
            internal_energy_rows_path=energy_path,
            root_validity_rows_path=validity_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "manifest"}
    assert all(path.exists() for path in outputs.values())

    summary = pd.read_csv(outputs["summary"]).iloc[0]
    assert int(summary["rescue_candidate_count"]) == 1
