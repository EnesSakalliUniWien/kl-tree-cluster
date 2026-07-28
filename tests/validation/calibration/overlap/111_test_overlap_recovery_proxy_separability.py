from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.overlap.overlap_recovery_proxy_separability import (
    OverlapRecoveryProxySeparabilityConfig,
    build_recovery_proxy_rows,
    build_recovery_proxy_separability,
    run_overlap_recovery_proxy_separability,
)


def _structural_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "depth": 2,
                "n_parent": 100,
                "n_left": 48,
                "n_right": 52,
                "barycentric_balance": 0.48,
                "left_pairwise_jaccard": 0.70,
                "right_pairwise_jaccard": 0.68,
                "homogeneity_gain_left": 0.04,
                "homogeneity_gain_right": 0.03,
                "homogeneity_gain_min": 0.03,
                "left_edge_norm": 1.0,
                "right_edge_norm": 0.9,
                "subspace_consensus_jaccard_topk": 0.50,
            },
            {
                "case_id": "case_b",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "depth": 1,
                "n_parent": 100,
                "n_left": 15,
                "n_right": 85,
                "barycentric_balance": 0.15,
                "left_pairwise_jaccard": 0.90,
                "right_pairwise_jaccard": 0.20,
                "homogeneity_gain_left": 0.06,
                "homogeneity_gain_right": -0.01,
                "homogeneity_gain_min": -0.01,
                "left_edge_norm": 1.5,
                "right_edge_norm": 0.3,
                "subspace_consensus_jaccard_topk": 0.20,
            },
            {
                "case_id": "case_c",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "diffuse",
                "depth": 1,
                "n_parent": 100,
                "n_left": 40,
                "n_right": 60,
                "barycentric_balance": 0.40,
                "left_pairwise_jaccard": 0.30,
                "right_pairwise_jaccard": 0.35,
                "homogeneity_gain_left": 0.01,
                "homogeneity_gain_right": 0.00,
                "homogeneity_gain_min": 0.00,
                "left_edge_norm": 1.0,
                "right_edge_norm": 0.8,
                "subspace_consensus_jaccard_topk": 0.30,
            },
        ]
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "recovery",
                "truth_geometry_mode": "balanced_truth_recovery",
            },
            {
                "case_id": "case_b",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "fragment",
                "truth_geometry_mode": "one_sided_pure_fragment",
            },
            {
                "case_id": "case_c",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "diffuse",
                "truth_geometry_mode": "diffuse_truth_mismatch",
            },
        ]
    )


def test_proxy_rows_derive_balances_and_scores() -> None:
    rows = build_recovery_proxy_rows(_structural_rows(), _truth_rows())

    recovery = rows[rows["node_id"].eq("recovery")].iloc[0]
    fragment = rows[rows["node_id"].eq("fragment")].iloc[0]
    assert recovery["proxy_truth_role"] == "truth_recovery"
    assert fragment["proxy_truth_role"] == "fragment_like"
    assert float(recovery["size_balance"]) == 0.48
    assert float(fragment["edge_norm_balance"]) == pytest.approx(0.2)
    assert float(recovery["balanced_recovery_proxy_score"]) > float(
        fragment["balanced_recovery_proxy_score"]
    )
    assert float(fragment["fragment_risk_proxy_score"]) > float(
        recovery["fragment_risk_proxy_score"]
    )


def test_proxy_summary_separates_recovery_from_fragment_like() -> None:
    _proxy_rows, summary, scan = build_recovery_proxy_separability(
        _structural_rows(),
        _truth_rows(),
        metrics=("balanced_recovery_proxy_score", "fragment_risk_proxy_score"),
    )

    balanced = summary[
        summary["metric"].eq("balanced_recovery_proxy_score")
        & summary["comparison"].eq("recovery_vs_fragment_like")
    ].iloc[0]
    fragment = summary[
        summary["metric"].eq("fragment_risk_proxy_score")
        & summary["comparison"].eq("recovery_vs_fragment_like")
    ].iloc[0]
    assert balanced["best_direction"] == "greater_equal"
    assert float(balanced["best_auc"]) == 1.0
    assert fragment["best_direction"] == "less_equal"
    assert float(fragment["best_auc"]) == 1.0
    assert set(scan["direction"]) == {"greater_equal", "less_equal"}


def test_run_recovery_proxy_writes_outputs(tmp_path) -> None:
    structural_path = tmp_path / "structural.csv"
    truth_path = tmp_path / "truth.csv"
    _structural_rows().to_csv(structural_path, index=False)
    _truth_rows().to_csv(truth_path, index=False)

    outputs = run_overlap_recovery_proxy_separability(
        OverlapRecoveryProxySeparabilityConfig(
            structural_rows_path=structural_path,
            truth_geometry_rows_path=truth_path,
            output_dir=tmp_path / "out",
            metrics=("balanced_recovery_proxy_score", "fragment_risk_proxy_score"),
        )
    )

    for path in outputs.values():
        assert path.exists()
