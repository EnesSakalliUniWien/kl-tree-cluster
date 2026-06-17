from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration import (
    spectral_transport_threshold_calibration_panel as panel,
)


def _rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for data_role, baseline_false, baseline_ari in (
        (panel.NULL_OUTPUT_ROLE, True, 0.0),
        (panel.SIGNAL_OUTPUT_ROLE, False, 0.8),
    ):
        records.append(
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": f"{data_role}_case",
                "data_role": data_role,
                "variant_id": panel.BASELINE_VARIANT,
                "spectral_transport_max_cost": float("nan"),
                "found_clusters": 3 if baseline_false else 2,
                "ari": baseline_ari,
                "false_split": baseline_false,
                "status": "ok",
                "spectral_transport_blocked_count": float("nan"),
            }
        )
        records.append(
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": f"{data_role}_case",
                "data_role": data_role,
                "variant_id": panel.SPECTRAL_VARIANT,
                "spectral_transport_max_cost": 0.75,
                "found_clusters": 1 if baseline_false else 1,
                "ari": 1.0 if baseline_false else 0.0,
                "false_split": False,
                "status": "ok",
                "spectral_transport_blocked_count": 1.0,
            }
        )
        records.append(
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": f"{data_role}_case",
                "data_role": data_role,
                "variant_id": panel.SPECTRAL_VARIANT,
                "spectral_transport_max_cost": 1.2,
                "found_clusters": 3 if baseline_false else 2,
                "ari": baseline_ari,
                "false_split": baseline_false,
                "status": "ok",
                "spectral_transport_blocked_count": 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def test_threshold_summary_separates_signal_regression_from_null_failure() -> None:
    pairwise = panel.build_threshold_pairwise_rows(_rows())
    summary = panel.summarize_thresholds(pairwise)

    by_threshold = summary.set_index("spectral_transport_max_cost")
    assert by_threshold.loc[0.75, "false_split_reduction"] == 1
    assert by_threshold.loc[0.75, "signal_regression_count"] == 1
    assert by_threshold.loc[0.75, "operating_point_status"] == "signal_regression"
    assert by_threshold.loc[1.2, "false_split_reduction"] == 0
    assert by_threshold.loc[1.2, "signal_regression_count"] == 0
    assert by_threshold.loc[1.2, "operating_point_status"] == (
        "null_false_split_not_reduced"
    )


def test_threshold_panel_writes_outputs_with_mocked_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(panel, "run_threshold_rows", lambda _config: _rows())

    outputs = panel.run_spectral_transport_threshold_calibration_panel(
        panel.SpectralTransportThresholdCalibrationConfig(output_dir=tmp_path)
    )

    assert set(outputs) == {"rows", "pairwise", "summary", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["pairwise"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
    summary = pd.read_csv(outputs["summary"])
    assert set(summary["operating_point_status"]) == {
        "signal_regression",
        "null_false_split_not_reduced",
    }
