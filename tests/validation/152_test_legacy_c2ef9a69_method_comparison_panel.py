from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    legacy_c2ef9a69_method_comparison_panel as panel,
)


def _comparison_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "small_null",
                "data_role": "selected_null",
                "variant_id": panel.CURRENT_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 1,
                "ari": 1.00,
                "false_split": False,
                "under_split": False,
                "n_singleton_clusters": 0,
                "largest_cluster_fraction": 1.0,
            },
            {
                "case_id": "small_null",
                "data_role": "selected_null",
                "variant_id": panel.LEGACY_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 3,
                "ari": 0.00,
                "false_split": True,
                "under_split": False,
                "n_singleton_clusters": 2,
                "largest_cluster_fraction": 0.5,
            },
            {
                "case_id": "small_signal",
                "data_role": "signal",
                "variant_id": panel.CURRENT_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 4,
                "ari": 0.60,
                "false_split": False,
                "under_split": False,
                "n_singleton_clusters": 0,
                "largest_cluster_fraction": 0.25,
            },
            {
                "case_id": "small_signal",
                "data_role": "signal",
                "variant_id": panel.LEGACY_VARIANT,
                "replicate": 0,
                "status": "ok",
                "found_clusters": 2,
                "ari": 0.40,
                "false_split": False,
                "under_split": True,
                "n_singleton_clusters": 0,
                "largest_cluster_fraction": 0.5,
            },
        ]
    )


def _labels_by_key() -> dict[tuple[str, str, int, str], np.ndarray]:
    return {
        ("small_null", "selected_null", 0, panel.CURRENT_VARIANT): np.array(
            [0, 0, 0, 0]
        ),
        ("small_null", "selected_null", 0, panel.LEGACY_VARIANT): np.array(
            [0, 1, 2, 2]
        ),
        ("small_signal", "signal", 0, panel.CURRENT_VARIANT): np.array(
            [0, 1, 2, 3]
        ),
        ("small_signal", "signal", 0, panel.LEGACY_VARIANT): np.array(
            [0, 0, 1, 1]
        ),
    }


def test_legacy_c2ef9a69_pairwise_summary_tracks_method_deltas() -> None:
    pairwise = panel.build_pairwise_rows(_comparison_rows(), _labels_by_key())
    summary = panel.summarize_comparison(pairwise)

    selected = summary.set_index("data_role").loc["selected_null"]
    assert selected["legacy_regression_count"] == 1
    assert selected["legacy_false_split_count"] == 1
    assert selected["mean_delta_clusters_legacy_minus_current"] == 2.0
    assert selected["mean_delta_singletons_legacy_minus_current"] == 2.0

    signal = summary.set_index("data_role").loc["signal"]
    assert signal["legacy_regression_count"] == 1
    assert signal["legacy_under_split_count"] == 1
    assert signal["mean_delta_ari_legacy_minus_current"] == pytest.approx(-0.20)
    assert signal["mean_delta_largest_cluster_fraction_legacy_minus_current"] == (
        pytest.approx(0.25)
    )


def test_legacy_c2ef9a69_panel_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _comparison_rows()
    pairwise = panel.build_pairwise_rows(rows, _labels_by_key())

    monkeypatch.setattr(
        panel,
        "run_comparison_rows",
        lambda _config: (rows, pairwise),
    )

    outputs = panel.run_legacy_c2ef9a69_method_comparison_panel(
        panel.LegacyC2ef9a69MethodComparisonConfig(output_dir=tmp_path)
    )

    assert set(outputs) == {"rows", "pairwise", "summary", "manifest"}
    assert all(path.exists() for path in outputs.values())
    summary = pd.read_csv(outputs["summary"])
    assert set(summary["data_role"]) == {"selected_null", "signal"}
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["variants"][panel.CURRENT_VARIANT]["method_id"] == "tbs"
    assert manifest["variants"][panel.LEGACY_VARIANT]["method_id"] == (
        "tbs_legacy_c2ef9a69"
    )
