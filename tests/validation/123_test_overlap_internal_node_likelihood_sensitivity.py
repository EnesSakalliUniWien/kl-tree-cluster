from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_internal_node_likelihood_sensitivity import (
    InternalNodeLikelihoodThresholds,
    OverlapInternalNodeLikelihoodSensitivityConfig,
    apply_internal_node_thresholds,
    build_internal_node_likelihood_sensitivity,
    run_overlap_internal_node_likelihood_sensitivity,
    summarize_internal_node_likelihood_sensitivity,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "guard_truth_role": "truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": 0.002,
                "subspace_consensus_jaccard_topk": 0.20,
                "size_balance": 0.34,
                "edge_norm_balance": 0.50,
                "fragment_risk_proxy_score": 1.20,
            },
            {
                "guard_truth_role": "truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": -0.001,
                "subspace_consensus_jaccard_topk": 0.25,
                "size_balance": 0.40,
                "edge_norm_balance": 0.70,
                "fragment_risk_proxy_score": 0.80,
            },
            {
                "guard_truth_role": "null_like",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": -0.001,
                "subspace_consensus_jaccard_topk": 0.50,
                "size_balance": 0.40,
                "edge_norm_balance": 0.70,
                "fragment_risk_proxy_score": 0.80,
            },
        ]
    )


def test_apply_internal_node_thresholds_classifies_recovery_and_leakage() -> None:
    strict = InternalNodeLikelihoodThresholds(
        context_margin_floor=0.0,
        soft_subspace_floor=0.15,
        size_balance_floor=0.33,
        edge_norm_balance_floor=0.49,
        fragment_risk_ceiling=1.25,
    )
    relaxed = InternalNodeLikelihoodThresholds(
        context_margin_floor=-0.002,
        soft_subspace_floor=0.15,
        size_balance_floor=0.33,
        edge_norm_balance_floor=0.49,
        fragment_risk_ceiling=1.25,
    )

    strict_result = apply_internal_node_thresholds(_rows(), strict)
    relaxed_result = apply_internal_node_thresholds(_rows(), relaxed)

    assert strict_result["candidate_status"] == "zero_negative_low_recovery"
    assert strict_result["negative_candidate_count"] == 0
    assert relaxed_result["candidate_status"] == "negative_leakage"
    assert relaxed_result["negative_candidate_count"] == 1


def test_sensitivity_summary_reports_default_status() -> None:
    grid = [
        InternalNodeLikelihoodThresholds(
            context_margin_floor=0.0,
            soft_subspace_floor=0.15,
            size_balance_floor=0.33,
            edge_norm_balance_floor=0.49,
            fragment_risk_ceiling=1.25,
        )
    ]
    rows = build_internal_node_likelihood_sensitivity(_rows(), grid=grid)
    summary = summarize_internal_node_likelihood_sensitivity(
        rows,
        default_rule_id=grid[0].rule_id,
    )

    assert int(summary["grid_count"].iloc[0]) == 1
    assert summary["default_rule_status"].iloc[0] == "zero_negative_low_recovery"


def test_run_internal_node_likelihood_sensitivity_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_internal_node_likelihood_sensitivity(
        OverlapInternalNodeLikelihoodSensitivityConfig(
            likelihood_rows_path=rows_path,
            output_dir=tmp_path / "out",
            context_margin_floors=(0.0,),
            subspace_floors=(0.15,),
            size_balance_floors=(0.33,),
            edge_norm_balance_floors=(0.49,),
            fragment_risk_ceilings=(1.25,),
        )
    )

    for path in outputs.values():
        assert path.exists()
