from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_internal_node_likelihood_sensitivity import (
    InternalNodeLikelihoodThresholds,
)
from benchmarks.diagnostics.calibration.overlap_internal_node_likelihood_transfer import (
    OverlapInternalNodeLikelihoodTransferConfig,
    build_internal_node_likelihood_transfer_rows,
    run_overlap_internal_node_likelihood_transfer,
    summarize_internal_node_likelihood_transfer,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "truth_a",
                "replicate": 0,
                "guard_truth_role": "truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": 0.002,
                "subspace_consensus_jaccard_topk": 0.20,
                "size_balance": 0.34,
                "edge_norm_balance": 0.50,
                "fragment_risk_proxy_score": 1.20,
            },
            {
                "case_id": "truth_b",
                "replicate": 1,
                "guard_truth_role": "truth_recovery",
                "selected_family_log_bayes_factor_lower": 8.0,
                "continuous_context_min_margin": 0.003,
                "subspace_consensus_jaccard_topk": 0.25,
                "size_balance": 0.40,
                "edge_norm_balance": 0.70,
                "fragment_risk_proxy_score": 0.80,
            },
            {
                "case_id": "null_holdout",
                "replicate": 2,
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


def _grid() -> list[InternalNodeLikelihoodThresholds]:
    return [
        InternalNodeLikelihoodThresholds(
            context_margin_floor=-0.002,
            soft_subspace_floor=0.15,
            size_balance_floor=0.33,
            edge_norm_balance_floor=0.49,
            fragment_risk_ceiling=1.25,
        ),
        InternalNodeLikelihoodThresholds(
            context_margin_floor=0.0,
            soft_subspace_floor=0.15,
            size_balance_floor=0.33,
            edge_norm_balance_floor=0.49,
            fragment_risk_ceiling=1.25,
        ),
    ]


def test_transfer_rows_expose_relaxed_context_leakage() -> None:
    rows = build_internal_node_likelihood_transfer_rows(
        _rows(),
        grid=_grid(),
        split_columns=("case_id",),
    )

    null_holdout = rows[rows["holdout_value"].eq("null_holdout")]
    assert set(null_holdout["transfer_status"]) == {
        "transfer_leakage",
        "transfer_no_truth_recovery_in_holdout",
    }
    relaxed = null_holdout[null_holdout["rule_id"].str.startswith("ctx=-0.002")]
    assert int(relaxed["test_null_like_candidate_count"].iloc[0]) == 1


def test_transfer_summary_reports_leakage() -> None:
    rows = build_internal_node_likelihood_transfer_rows(
        _rows(),
        grid=_grid(),
        split_columns=("case_id",),
    )
    summary = summarize_internal_node_likelihood_transfer(rows)

    assert summary["transfer_status"].iloc[0] == "transfer_leakage"
    assert int(summary["leakage_rule_count"].iloc[0]) >= 1


def test_run_internal_node_likelihood_transfer_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "rows.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_internal_node_likelihood_transfer(
        OverlapInternalNodeLikelihoodTransferConfig(
            likelihood_rows_path=rows_path,
            output_dir=tmp_path / "out",
            split_columns=("case_id",),
            context_margin_floors=(-0.002, 0.0),
            subspace_floors=(0.15,),
            size_balance_floors=(0.33,),
            edge_norm_balance_floors=(0.49,),
            fragment_risk_ceilings=(1.25,),
        )
    )

    for path in outputs.values():
        assert path.exists()
