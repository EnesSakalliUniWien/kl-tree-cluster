from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_structural_context_thresholds import (
    OverlapStructuralContextThresholdConfig,
    add_context_bins,
    build_context_threshold_sensitivity,
    run_overlap_structural_context_thresholds,
    summarize_context_thresholds,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "data_role": "selected_null",
                "decision_class": "accepted_internal_split",
                "depth": 1,
                "n_parent": 120,
                "barycentric_balance": 0.45,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.004,
                "subspace_consensus_jaccard_topk": 0.60,
                "truth_split_ari": pd.NA,
            },
            {
                "data_role": "signal",
                "decision_class": "accepted_internal_split",
                "depth": 3,
                "n_parent": 210,
                "barycentric_balance": 0.48,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.026,
                "subspace_consensus_jaccard_topk": 0.40,
                "truth_split_ari": 0.95,
            },
            {
                "data_role": "signal",
                "decision_class": "accepted_internal_split",
                "depth": 0,
                "n_parent": 400,
                "barycentric_balance": 0.20,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.006,
                "subspace_consensus_jaccard_topk": 0.10,
                "truth_split_ari": 0.20,
            },
        ]
    )


def test_context_bins_capture_depth_size_balance_and_consensus() -> None:
    enriched = add_context_bins(_rows())

    assert list(enriched["depth_bin"]) == ["shallow_1_2", "deep_3_plus", "root"]
    assert list(enriched["parent_size_bin"]) == [
        "small_parent_lt150",
        "medium_parent_150_299",
        "large_parent_ge300",
    ]
    assert list(enriched["balance_bin"]) == [
        "balanced_ge0.4",
        "balanced_ge0.4",
        "unbalanced_lt0.25",
    ]
    assert list(enriched["subspace_consensus_bin"]) == [
        "high_consensus_ge0.5",
        "mid_consensus_0.25_0.5",
        "low_consensus_lt0.25",
    ]


def test_context_threshold_summary_identifies_context_specific_cutoffs() -> None:
    sensitivity = build_context_threshold_sensitivity(
        _rows(),
        homogeneity_thresholds=(0.005, 0.02),
        subspace_thresholds=(0.15, 0.25),
        sibling_p_thresholds=(0.01,),
    )
    summary = summarize_context_thresholds(sensitivity)

    shallow = summary[
        summary["context_axis"].eq("depth_bin")
        & summary["context_bin"].eq("shallow_1_2")
    ].iloc[0]
    deep = summary[
        summary["context_axis"].eq("depth_bin")
        & summary["context_bin"].eq("deep_3_plus")
    ].iloc[0]

    assert shallow["context_threshold_status"] == "context_no_truth_aligned_signal"
    assert float(shallow["best_homogeneity_gain_threshold"]) == 0.005
    assert deep["context_threshold_status"] == "context_threshold_candidate"
    assert int(deep["signal_truth_aligned_accept_count"]) == 1


def test_run_context_thresholds_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_structural_context_thresholds(
        OverlapStructuralContextThresholdConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
            homogeneity_thresholds=(0.005, 0.02),
            subspace_thresholds=(0.15, 0.25),
            sibling_p_thresholds=(0.01,),
        )
    )

    for path in outputs.values():
        assert path.exists()
