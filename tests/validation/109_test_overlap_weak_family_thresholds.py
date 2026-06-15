from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.overlap_weak_family_thresholds import (
    OverlapWeakFamilyThresholdConfig,
    build_weak_family_rows,
    build_weak_family_thresholds,
    run_overlap_weak_family_thresholds,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "case_id": "case_a",
                "data_role": "selected_null",
                "replicate": 0,
                "structural_truth_role": "null_like",
                "sibling_p_value": 0.01,
                "homogeneity_gain_min": 0.01,
                "subspace_consensus_jaccard_topk": 0.30,
                "continuous_homogeneity_threshold": 0.02,
                "depth": 1,
                "n_parent": 120,
                "barycentric_balance": 0.40,
            },
            {
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "case_id": "case_b",
                "data_role": "signal",
                "replicate": 0,
                "structural_truth_role": "signal_truth_aligned",
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.05,
                "subspace_consensus_jaccard_topk": 0.50,
                "continuous_homogeneity_threshold": 0.02,
                "depth": 3,
                "n_parent": 100,
                "barycentric_balance": 0.45,
            },
            {
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "case_id": "case_b",
                "data_role": "signal",
                "replicate": 0,
                "structural_truth_role": "signal_truth_misaligned",
                "sibling_p_value": 0.0001,
                "homogeneity_gain_min": 0.02,
                "subspace_consensus_jaccard_topk": 0.40,
                "continuous_homogeneity_threshold": 0.02,
                "depth": 2,
                "n_parent": 150,
                "barycentric_balance": 0.35,
            },
            {
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "case_id": "case_c",
                "data_role": "signal",
                "replicate": 0,
                "structural_truth_role": "signal_truth_misaligned",
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.03,
                "subspace_consensus_jaccard_topk": 0.60,
                "continuous_homogeneity_threshold": 0.02,
                "depth": 1,
                "n_parent": 200,
                "barycentric_balance": 0.20,
            },
            {
                "structural_decision_zone": "stable_structural_accept",
                "case_id": "case_d",
                "data_role": "signal",
                "replicate": 0,
                "structural_truth_role": "signal_truth_aligned",
                "sibling_p_value": 1e-6,
                "homogeneity_gain_min": 0.10,
                "subspace_consensus_jaccard_topk": 0.80,
                "continuous_homogeneity_threshold": 0.02,
                "depth": 4,
                "n_parent": 90,
                "barycentric_balance": 0.50,
            },
        ]
    )


def test_family_rows_group_by_case_data_role_and_replicate() -> None:
    family_rows = build_weak_family_rows(_rows())

    roles = dict(zip(family_rows["case_id"], family_rows["family_truth_role"], strict=True))
    assert roles == {
        "case_a": "null_like_family",
        "case_b": "signal_truth_aligned_family",
        "case_c": "signal_truth_misaligned_family",
    }
    case_b = family_rows[family_rows["case_id"].eq("case_b")].iloc[0]
    assert int(case_b["family_size"]) == 2
    assert int(case_b["truth_aligned_row_count"]) == 1
    assert int(case_b["truth_misaligned_row_count"]) == 1
    assert float(case_b["max_context_homogeneity_margin"]) == pytest.approx(0.03)


def test_family_threshold_summary_keeps_positive_family_at_zero_negative() -> None:
    _family_rows, summary, scan = build_weak_family_thresholds(
        _rows(),
        metrics=("max_homogeneity_gain_min",),
    )

    combined = summary[
        summary["comparison"].eq("aligned_family_vs_null_or_misaligned")
    ].iloc[0]
    assert combined["best_direction"] == "greater_equal"
    assert int(combined["zero_negative_positive_count"]) == 1
    assert float(combined["zero_negative_positive_retention"]) == 1.0
    assert combined["zero_negative_status"] == "zero_negative_separates_all_positives"
    assert set(scan["direction"]) == {"greater_equal", "less_equal"}


def test_run_weak_family_thresholds_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "zones.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_weak_family_thresholds(
        OverlapWeakFamilyThresholdConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
            metrics=("max_homogeneity_gain_min", "max_context_homogeneity_margin"),
        )
    )

    for path in outputs.values():
        assert path.exists()
