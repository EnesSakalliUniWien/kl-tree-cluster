from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_structural_continuous_rule import (
    ContinuousRuleParameters,
    OverlapStructuralContinuousRuleConfig,
    apply_continuous_rule,
    continuous_homogeneity_threshold,
    run_overlap_structural_continuous_rule,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "null_overlap",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "root",
                "decision_class": "accepted_internal_split",
                "depth": 0,
                "n_parent": 400,
                "barycentric_balance": 0.45,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.006,
                "subspace_consensus_jaccard_topk": 0.50,
                "truth_split_ari": pd.NA,
            },
            {
                "case_id": "signal_overlap",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "deep",
                "decision_class": "accepted_internal_split",
                "depth": 4,
                "n_parent": 100,
                "barycentric_balance": 0.49,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.011,
                "subspace_consensus_jaccard_topk": 0.30,
                "truth_split_ari": 0.90,
            },
            {
                "case_id": "signal_overlap",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "misaligned",
                "decision_class": "accepted_internal_split",
                "depth": 1,
                "n_parent": 350,
                "barycentric_balance": 0.20,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.006,
                "subspace_consensus_jaccard_topk": 0.30,
                "truth_split_ari": 0.10,
            },
        ]
    )


def _parameters() -> ContinuousRuleParameters:
    return ContinuousRuleParameters(
        base_threshold=0.005,
        shallow_penalty=0.01,
        parent_size_penalty=0.005,
        imbalance_penalty=0.0,
        subspace_consensus_threshold=0.15,
        sibling_p_threshold=0.01,
    )


def test_threshold_surface_is_stricter_for_root_large_parents() -> None:
    thresholds = continuous_homogeneity_threshold(_rows(), _parameters())

    assert thresholds.iloc[0] > thresholds.iloc[1]
    assert thresholds.iloc[2] > thresholds.iloc[1]


def test_continuous_rule_separates_synthetic_null_and_deep_signal() -> None:
    decisions, summary = apply_continuous_rule(_rows(), _parameters())

    accepted_by_node = dict(
        zip(decisions["node_id"], decisions["continuous_rule_accept"], strict=True)
    )
    assert accepted_by_node == {
        "root": False,
        "deep": True,
        "misaligned": False,
    }
    assert summary["rule_status"] == "continuous_rule_candidate"
    assert summary["null_structural_accept_count"] == 0
    assert summary["signal_truth_aligned_accept_count"] == 1


def test_run_continuous_rule_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_structural_continuous_rule(
        OverlapStructuralContinuousRuleConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
            base_thresholds=(0.005,),
            shallow_penalties=(0.01,),
            parent_size_penalties=(0.005,),
            imbalance_penalties=(0.0,),
            subspace_thresholds=(0.15,),
            sibling_p_thresholds=(0.01,),
        )
    )

    for path in outputs.values():
        assert path.exists()
