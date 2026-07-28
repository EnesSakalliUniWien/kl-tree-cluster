from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.spectral.mp.mp_projection_dimension_behavior_sweep import (
    RULE_IDS,
    STUDY_ROLE,
    candidate_projection_dimensions,
    evaluate_projection_behavior_summary,
    run_mp_projection_dimension_behavior_sweep,
)


def test_candidate_projection_dimensions_caps_raw_rules_to_parent_basis() -> None:
    dimensions = candidate_projection_dimensions(
        current_edge_derived_dimension=1,
        parent_test_projection_dimension=2,
        raw_mp_signal_count=4,
    )

    assert set(dimensions) == set(RULE_IDS)
    assert dimensions["current_edge_derived_rule"] == 1
    assert dimensions["parent_test_projection_dimension"] == 2
    assert dimensions["raw_mp_parent_signal_count"] == 2
    assert dimensions["raw_mp_parent_signal_count_floor1"] == 2
    assert dimensions["raw_mp_parent_signal_count_floor2"] == 2


def test_evaluate_projection_behavior_summary_reports_null_and_signal_endpoints() -> None:
    records = pd.DataFrame(
        {
            "rule_id": [
                "current_edge_derived_rule",
                "current_edge_derived_rule",
                "raw_mp_parent_signal_count",
                "raw_mp_parent_signal_count",
            ],
            "projection_dimension": [1, 1, 0, 2],
            "current_edge_derived_dimension": [1, 1, 1, 1],
            "test_status": [
                "ok",
                "ok",
                "zero_dimensional_projection",
                "ok",
            ],
            "raw_p_value": [0.005, 0.50, 1.0, 0.001],
            "sibling_alpha": [0.01, 0.01, 0.01, 0.01],
            "is_null_context": [True, False, True, False],
            "is_signal_context": [False, True, False, True],
        }
    )

    summary = evaluate_projection_behavior_summary(records)

    assert summary["study_role"].eq(STUDY_ROLE).all()
    current = summary[summary["rule_id"].eq("current_edge_derived_rule")].iloc[0]
    raw = summary[summary["rule_id"].eq("raw_mp_parent_signal_count")].iloc[0]
    assert current["null_false_split_rate"] == 1.0
    assert current["signal_retention_rate"] == 0.0
    assert raw["frequency_k0"] == 0.5
    assert raw["null_false_split_rate"] == 0.0
    assert raw["signal_retention_rate"] == 1.0


def test_run_mp_projection_dimension_behavior_sweep_smoke_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "out"

        outputs = run_mp_projection_dimension_behavior_sweep(
            suite="method_proof",
            case_names=("mp_spike_below_bbp_continuous",),
            n_replicates=1,
            output_dir=output_dir,
        )

        assert set(outputs) == {
            "records",
            "summary",
            "case_summary",
            "status",
            "manifest",
        }
        assert (output_dir / "mp_projection_dimension_behavior_records.csv").exists()
        assert (output_dir / "mp_projection_dimension_behavior_summary.csv").exists()
