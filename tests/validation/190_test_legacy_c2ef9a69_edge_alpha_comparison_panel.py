from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    legacy_c2ef9a69_edge_alpha_comparison_panel as panel,
)


def _pairwise_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "edge_alpha": 0.0001,
                "case_id": "small_null",
                "data_role": "selected_null",
                "replicate": 0,
                "current_status": "ok",
                "legacy_status": "ok",
                "current_found_clusters": 1,
                "legacy_found_clusters": 1,
                "delta_clusters_legacy_minus_current": 0,
                "current_ari": 1.0,
                "legacy_ari": 1.0,
                "delta_ari_legacy_minus_current": 0.0,
                "current_false_split": False,
                "legacy_false_split": False,
                "current_under_split": False,
                "legacy_under_split": False,
                "partition_ari_between_variants": 1.0,
            },
            {
                "edge_alpha": 0.0001,
                "case_id": "small_signal",
                "data_role": "signal",
                "replicate": 0,
                "current_status": "ok",
                "legacy_status": "ok",
                "current_found_clusters": 4,
                "legacy_found_clusters": 4,
                "delta_clusters_legacy_minus_current": 0,
                "current_ari": 0.8,
                "legacy_ari": 0.8,
                "delta_ari_legacy_minus_current": 0.0,
                "current_false_split": False,
                "legacy_false_split": False,
                "current_under_split": False,
                "legacy_under_split": False,
                "partition_ari_between_variants": 1.0,
            },
            {
                "edge_alpha": 0.001,
                "case_id": "small_null",
                "data_role": "selected_null",
                "replicate": 0,
                "current_status": "ok",
                "legacy_status": "ok",
                "current_found_clusters": 1,
                "legacy_found_clusters": 3,
                "delta_clusters_legacy_minus_current": 2,
                "current_ari": 1.0,
                "legacy_ari": 0.0,
                "delta_ari_legacy_minus_current": -1.0,
                "current_false_split": False,
                "legacy_false_split": True,
                "current_under_split": False,
                "legacy_under_split": False,
                "partition_ari_between_variants": 0.0,
            },
            {
                "edge_alpha": 0.001,
                "case_id": "small_signal",
                "data_role": "signal",
                "replicate": 0,
                "current_status": "ok",
                "legacy_status": "ok",
                "current_found_clusters": 3,
                "legacy_found_clusters": 4,
                "delta_clusters_legacy_minus_current": 1,
                "current_ari": 0.4,
                "legacy_ari": 0.7,
                "delta_ari_legacy_minus_current": 0.3,
                "current_false_split": False,
                "legacy_false_split": False,
                "current_under_split": True,
                "legacy_under_split": False,
                "partition_ari_between_variants": 0.5,
            },
        ]
    )


def test_edge_alpha_tradeoff_keeps_signal_gain_and_null_leak_separate() -> None:
    pairwise = panel.annotate_pairwise_rows(_pairwise_rows())
    alpha_summary = panel.summarize_by_alpha(pairwise)
    tradeoff = panel.summarize_alpha_tradeoff(alpha_summary).set_index("edge_alpha")

    assert pairwise["legacy_extra_selected_null_false_split"].sum() == 1
    assert pairwise["legacy_signal_gain"].sum() == 1
    assert tradeoff.loc[0.001, "selected_null_legacy_extra_false_split_count"] == 1
    assert tradeoff.loc[0.001, "signal_legacy_gain_count"] == 1
    assert tradeoff.loc[0.001, "tradeoff_status"] == (
        "legacy_power_not_admissible_extra_null_leak"
    )
    assert tradeoff.loc[0.0001, "tradeoff_status"] == (
        "no_legacy_edge_alpha_power_gain"
    )


def test_edge_alpha_panel_writes_no_shortcut_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": "small_null",
                "data_role": "selected_null",
                "variant_id": "current_kl",
                "grid_edge_alpha": 0.001,
            }
        ]
    )
    pairwise = panel.annotate_pairwise_rows(_pairwise_rows())

    monkeypatch.setattr(
        panel,
        "run_edge_alpha_comparison_rows",
        lambda _config: (rows, pairwise),
    )

    outputs = panel.run_legacy_c2ef9a69_edge_alpha_comparison_panel(
        panel.LegacyC2ef9a69EdgeAlphaComparisonConfig(
            output_dir=tmp_path,
            edge_alphas=(0.0001, 0.001),
        )
    )

    assert set(outputs) == {
        "rows",
        "pairwise",
        "alpha_summary",
        "tradeoff_summary",
        "manifest",
    }
    assert all(path.exists() for path in outputs.values())
    tradeoff = pd.read_csv(outputs["tradeoff_summary"])
    assert "legacy_power_not_admissible_extra_null_leak" in set(
        tradeoff["tradeoff_status"]
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["parameters"]["edge_alphas"] == [0.0001, 0.001]
