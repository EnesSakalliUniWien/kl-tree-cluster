from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_fragment_risk_guard import (
    OverlapFragmentRiskGuardConfig,
    build_fragment_guard_rows,
    run_overlap_fragment_risk_guard,
    scan_fragment_guard_thresholds,
)


def _structural_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_null",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null",
                "left_pairwise_jaccard": 0.30,
                "right_pairwise_jaccard": 0.20,
                "homogeneity_gain_left": 0.02,
                "homogeneity_gain_right": 0.01,
                "homogeneity_gain_min": 0.01,
                "left_edge_norm": 1.0,
                "right_edge_norm": 0.8,
                "n_left": 40,
                "n_right": 60,
                "n_parent": 100,
                "barycentric_balance": 0.40,
                "subspace_consensus_jaccard_topk": 0.30,
            },
            {
                "case_id": "case_recovery",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "left_pairwise_jaccard": 0.70,
                "right_pairwise_jaccard": 0.68,
                "homogeneity_gain_left": 0.04,
                "homogeneity_gain_right": 0.03,
                "homogeneity_gain_min": 0.03,
                "left_edge_norm": 1.0,
                "right_edge_norm": 0.9,
                "n_left": 48,
                "n_right": 52,
                "n_parent": 100,
                "barycentric_balance": 0.48,
                "subspace_consensus_jaccard_topk": 0.50,
            },
            {
                "case_id": "case_fragment",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "left_pairwise_jaccard": 0.90,
                "right_pairwise_jaccard": 0.20,
                "homogeneity_gain_left": 0.06,
                "homogeneity_gain_right": -0.01,
                "homogeneity_gain_min": -0.01,
                "left_edge_norm": 1.5,
                "right_edge_norm": 0.3,
                "n_left": 15,
                "n_right": 85,
                "n_parent": 100,
                "barycentric_balance": 0.15,
                "subspace_consensus_jaccard_topk": 0.20,
            },
        ]
    )


def _zone_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_null",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "null",
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "structural_truth_role": "null_like",
            },
            {
                "case_id": "case_recovery",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "structural_truth_role": "signal_truth_aligned",
            },
            {
                "case_id": "case_fragment",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "structural_decision_zone": "unstable_weak_homogeneity_zone",
                "structural_truth_role": "signal_truth_misaligned",
            },
        ]
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_recovery",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "truth_geometry_mode": "balanced_truth_recovery",
            },
            {
                "case_id": "case_fragment",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "truth_geometry_mode": "one_sided_pure_fragment",
            },
        ]
    )


def test_fragment_guard_rows_include_null_and_signal_roles() -> None:
    rows = build_fragment_guard_rows(_structural_rows(), _zone_rows(), _truth_rows())

    roles = dict(zip(rows["node_id"], rows["guard_truth_role"], strict=True))
    assert roles == {
        "null": "null_like",
        "recovery": "truth_recovery",
        "fragment": "fragment_like",
    }
    fragment = rows[rows["node_id"].eq("fragment")].iloc[0]
    recovery = rows[rows["node_id"].eq("recovery")].iloc[0]
    assert float(fragment["fragment_risk_proxy_score"]) > float(
        recovery["fragment_risk_proxy_score"]
    )


def test_scan_fragment_guard_thresholds_finds_candidate() -> None:
    rows = build_fragment_guard_rows(_structural_rows(), _zone_rows(), _truth_rows())
    scan = scan_fragment_guard_thresholds(
        rows,
        guard_specs=(("fragment_risk_proxy_score", "greater_equal"),),
    )

    candidates = scan[scan["guard_status"].eq("fragment_guard_candidate")]
    assert not candidates.empty
    best = candidates.sort_values("guard_score", ascending=False).iloc[0]
    assert int(best["truth_recovery_retained"]) == 1
    assert int(best["fragment_like_blocked"]) == 1


def test_run_fragment_guard_writes_outputs(tmp_path) -> None:
    structural_path = tmp_path / "structural.csv"
    zone_path = tmp_path / "zones.csv"
    truth_path = tmp_path / "truth.csv"
    _structural_rows().to_csv(structural_path, index=False)
    _zone_rows().to_csv(zone_path, index=False)
    _truth_rows().to_csv(truth_path, index=False)

    outputs = run_overlap_fragment_risk_guard(
        OverlapFragmentRiskGuardConfig(
            structural_rows_path=structural_path,
            decision_zone_rows_path=zone_path,
            truth_geometry_rows_path=truth_path,
            output_dir=tmp_path / "out",
            guard_specs=(("fragment_risk_proxy_score", "greater_equal"),),
        )
    )

    for path in outputs.values():
        assert path.exists()
