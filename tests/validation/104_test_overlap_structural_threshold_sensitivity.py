from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_structural_threshold_sensitivity import (
    OverlapStructuralThresholdSensitivityConfig,
    build_threshold_sensitivity,
    run_overlap_structural_threshold_sensitivity,
    summarize_threshold_recommendations,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "replicate": 0,
                "decision_class": "accepted_internal_split",
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.001,
                "heterogeneity_gain_max": -0.001,
                "subspace_consensus_jaccard_topk": 0.80,
                "heterogeneity_subspace_consensus_jaccard_topk": 0.10,
                "truth_split_ari": pd.NA,
            },
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "replicate": 0,
                "decision_class": "accepted_internal_split",
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.03,
                "heterogeneity_gain_max": -0.03,
                "subspace_consensus_jaccard_topk": 0.70,
                "heterogeneity_subspace_consensus_jaccard_topk": 0.10,
                "truth_split_ari": 0.90,
            },
            {
                "case_id": "signal_case",
                "data_role": "signal",
                "replicate": 0,
                "decision_class": "accepted_internal_split",
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": -0.03,
                "heterogeneity_gain_max": 0.03,
                "subspace_consensus_jaccard_topk": 0.10,
                "heterogeneity_subspace_consensus_jaccard_topk": 0.70,
                "truth_split_ari": 0.10,
            },
        ]
    )


def test_threshold_sensitivity_separates_null_signal_and_warnings() -> None:
    sensitivity = build_threshold_sensitivity(
        _rows(),
        homogeneity_thresholds=(0.02,),
        heterogeneity_thresholds=(0.02,),
        subspace_thresholds=(0.25,),
        sibling_p_thresholds=(0.01,),
    )

    null_row = sensitivity[sensitivity["data_role"].eq("selected_null")].iloc[0]
    signal_row = sensitivity[sensitivity["data_role"].eq("signal")].iloc[0]

    assert int(null_row["null_false_accept_count"]) == 0
    assert int(null_row["blocked_null_false_split_count"]) == 1
    assert int(signal_row["signal_truth_aligned_accept_count"]) == 1
    assert int(signal_row["same_subspace_heterogeneous_warning_count"]) == 1
    assert signal_row["threshold_status"] == "signal_retained_with_heterogeneity_warnings"


def test_recommendations_score_threshold_tradeoff() -> None:
    sensitivity = build_threshold_sensitivity(
        _rows(),
        homogeneity_thresholds=(0.0, 0.02),
        heterogeneity_thresholds=(0.02,),
        subspace_thresholds=(0.25,),
        sibling_p_thresholds=(0.01,),
    )
    recommendations = summarize_threshold_recommendations(sensitivity)

    strict = recommendations[
        recommendations["homogeneity_gain_threshold"].eq(0.02)
    ].iloc[0]
    permissive = recommendations[
        recommendations["homogeneity_gain_threshold"].eq(0.0)
    ].iloc[0]

    assert strict["threshold_recommendation_status"] in {
        "threshold_warns_same_subspace_heterogeneity",
        "threshold_candidate_diagnostic",
    }
    assert strict["threshold_stability_status"] in {
        "threshold_stable_with_heterogeneity_warnings",
        "threshold_stable_candidate",
    }
    assert int(strict["null_case_replicates_with_false_accept"]) == 0
    assert int(strict["signal_truth_aligned_case_replicates_retained"]) == 1
    assert permissive["null_false_accept_count"] > strict["null_false_accept_count"]
    assert (
        permissive["threshold_stability_status"]
        == "threshold_unstable_null_false_accepts"
    )


def test_run_threshold_sensitivity_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_structural_threshold_sensitivity(
        OverlapStructuralThresholdSensitivityConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
            homogeneity_thresholds=(0.02,),
            heterogeneity_thresholds=(0.02,),
            subspace_thresholds=(0.25,),
            sibling_p_thresholds=(0.01,),
        )
    )

    for path in outputs.values():
        assert path.exists()
