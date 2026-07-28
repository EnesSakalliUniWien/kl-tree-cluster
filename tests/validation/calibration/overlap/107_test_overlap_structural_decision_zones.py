from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_structural_continuous_rule import (
    ContinuousRuleParameters,
)
from benchmarks.diagnostics.calibration.overlap.overlap_structural_decision_zones import (
    OverlapStructuralDecisionZoneConfig,
    assign_structural_decision_zones,
    continuous_context_margins,
    run_overlap_structural_decision_zones,
    summarize_structural_decision_zones,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "signal_overlap",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "stable",
                "decision_class": "accepted_internal_split",
                "depth": 4,
                "n_parent": 100,
                "barycentric_balance": 0.49,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.020,
                "subspace_consensus_jaccard_topk": 0.40,
                "truth_split_ari": 0.90,
                "structural_sibling_status": "structural_same_subspace_supported",
                "structural_change_mode": "homogeneous_same_subspace",
            },
            {
                "case_id": "null_overlap",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "weak_null",
                "decision_class": "accepted_internal_split",
                "depth": 1,
                "n_parent": 180,
                "barycentric_balance": 0.40,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.004,
                "subspace_consensus_jaccard_topk": 0.50,
                "truth_split_ari": pd.NA,
                "structural_sibling_status": "weak_homogeneity_gain",
                "structural_change_mode": "weak_or_mixed_structural_change",
            },
            {
                "case_id": "signal_overlap",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "context_blocked",
                "decision_class": "accepted_internal_split",
                "depth": 0,
                "n_parent": 500,
                "barycentric_balance": 0.49,
                "sibling_p_value": 0.001,
                "homogeneity_gain_min": 0.008,
                "subspace_consensus_jaccard_topk": 0.60,
                "truth_split_ari": 0.80,
                "structural_sibling_status": "structural_same_subspace_supported",
                "structural_change_mode": "homogeneous_same_subspace",
            },
            {
                "case_id": "signal_overlap",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "leaf",
                "decision_class": "leaf_fragment",
                "depth": 2,
                "n_parent": 80,
                "barycentric_balance": 0.30,
                "sibling_p_value": 1.0,
                "homogeneity_gain_min": 0.0,
                "subspace_consensus_jaccard_topk": 0.0,
                "truth_split_ari": pd.NA,
                "structural_sibling_status": "weak_homogeneity_gain",
                "structural_change_mode": "weak_or_mixed_structural_change",
            },
        ]
    )


def _parameters() -> ContinuousRuleParameters:
    return ContinuousRuleParameters(
        base_threshold=0.005,
        shallow_penalty=0.015,
        parent_size_penalty=0.0,
        imbalance_penalty=0.005,
        subspace_consensus_threshold=0.15,
        sibling_p_threshold=0.01,
    )


def test_decision_zones_separate_stable_weak_and_context_blocked_rows() -> None:
    zones = assign_structural_decision_zones(_rows(), _parameters())

    by_node = dict(zip(zones["node_id"], zones["structural_decision_zone"], strict=True))
    assert by_node == {
        "stable": "stable_structural_accept",
        "weak_null": "unstable_weak_homogeneity_zone",
        "context_blocked": "stable_structure_context_blocked",
        "leaf": "nonaccepted_or_leaf",
    }


def test_continuous_context_margins_explain_context_blocks() -> None:
    margins = continuous_context_margins(_rows(), _parameters())
    rows = _rows()
    by_node = margins.set_index(rows["node_id"])

    assert by_node.loc["stable", "continuous_homogeneity_margin"] > 0
    assert by_node.loc["stable", "continuous_subspace_margin"] > 0
    assert by_node.loc["stable", "continuous_log_p_margin"] > 0
    assert by_node.loc["stable", "continuous_context_min_margin"] > 0

    assert by_node.loc["context_blocked", "continuous_homogeneity_margin"] < 0
    assert by_node.loc["context_blocked", "continuous_context_min_margin"] < 0


def test_decision_zone_summary_counts_truth_roles() -> None:
    zones = assign_structural_decision_zones(_rows(), _parameters())
    summary = summarize_structural_decision_zones(zones)

    stable = summary[summary["structural_decision_zone"].eq("stable_structural_accept")].iloc[0]
    weak = summary[summary["structural_decision_zone"].eq("unstable_weak_homogeneity_zone")].iloc[0]

    assert int(stable["signal_truth_aligned_count"]) == 1
    assert int(stable["null_count"]) == 0
    assert int(weak["null_count"]) == 1


def test_run_decision_zones_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_structural_decision_zones(
        OverlapStructuralDecisionZoneConfig(
            rows_path=rows_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
