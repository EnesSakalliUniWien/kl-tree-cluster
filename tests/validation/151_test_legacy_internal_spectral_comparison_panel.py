from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    legacy_internal_spectral_comparison_panel as panel,
)


def _comparison_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "overlap_case",
                "data_role": "selected_null",
                "variant_id": panel.CURRENT_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 4,
                "ari": 0.20,
                "false_split": False,
                "spectral_raw_mp_signal_count_sum": 3.0,
                "spectral_mp_threshold_rows_sum": 10.0,
                "spectral_test_dimension_sum": 5.0,
            },
            {
                "case_id": "overlap_case",
                "data_role": "selected_null",
                "variant_id": panel.LEGACY_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 6,
                "ari": 0.10,
                "false_split": True,
                "spectral_raw_mp_signal_count_sum": 4.0,
                "spectral_mp_threshold_rows_sum": 14.0,
                "spectral_test_dimension_sum": 6.0,
            },
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "variant_id": panel.CURRENT_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 4,
                "ari": 0.60,
                "false_split": False,
                "spectral_raw_mp_signal_count_sum": 2.0,
                "spectral_mp_threshold_rows_sum": 8.0,
                "spectral_test_dimension_sum": 4.0,
            },
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "variant_id": panel.LEGACY_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 4,
                "ari": 0.80,
                "false_split": False,
                "spectral_raw_mp_signal_count_sum": 3.0,
                "spectral_mp_threshold_rows_sum": 11.0,
                "spectral_test_dimension_sum": 5.0,
            },
        ]
    )


def _labels_by_key() -> dict[tuple[str, str, int, str], np.ndarray]:
    return {
        ("overlap_case", "selected_null", 0, panel.CURRENT_VARIANT): np.array(
            [0, 0, 1, 1]
        ),
        ("overlap_case", "selected_null", 0, panel.LEGACY_VARIANT): np.array(
            [0, 1, 2, 3]
        ),
        ("signal_case", "signal", 0, panel.CURRENT_VARIANT): np.array([0, 0, 1, 1]),
        ("signal_case", "signal", 0, panel.LEGACY_VARIANT): np.array([0, 0, 1, 1]),
    }


def test_legacy_internal_spectral_pairwise_summary_tracks_regressions() -> None:
    pairwise = panel.build_pairwise_rows(_comparison_rows(), _labels_by_key())
    summary = panel.summarize_comparison(pairwise)

    selected = summary.set_index("data_role").loc["selected_null"]
    assert selected["legacy_signal_regression_count"] == 1
    assert selected["legacy_signal_improvement_count"] == 0
    assert selected["mean_delta_clusters_legacy_minus_current"] == 2.0
    assert selected["current_false_split_count"] == 0
    assert selected["legacy_false_split_count"] == 1
    assert selected["mean_delta_mp_threshold_rows_sum"] == 4.0

    signal = summary.set_index("data_role").loc["signal"]
    assert signal["legacy_signal_regression_count"] == 0
    assert signal["legacy_signal_improvement_count"] == 1
    assert signal["mean_delta_ari_legacy_minus_current"] == pytest.approx(0.20)


def test_legacy_internal_spectral_panel_writes_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = _comparison_rows()
    pairwise = panel.build_pairwise_rows(rows, _labels_by_key())

    monkeypatch.setattr(
        panel,
        "run_comparison_rows",
        lambda _config: (rows, pairwise),
    )

    outputs = panel.run_legacy_internal_spectral_comparison_panel(
        panel.LegacyInternalSpectralComparisonConfig(output_dir=tmp_path)
    )

    assert set(outputs) == {"rows", "pairwise", "summary", "manifest"}
    assert all(path.exists() for path in outputs.values())
    summary = pd.read_csv(outputs["summary"])
    assert set(summary["data_role"]) == {"selected_null", "signal"}
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["variants"][panel.CURRENT_VARIANT][
        "spectral_include_internal_barycenters"
    ] is False
    assert manifest["variants"][panel.LEGACY_VARIANT][
        "spectral_include_internal_barycenters"
    ] is True
