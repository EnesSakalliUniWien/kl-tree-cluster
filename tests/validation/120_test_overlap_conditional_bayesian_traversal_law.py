from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_conditional_bayesian_traversal_law import (
    OverlapConditionalBayesianTraversalLawConfig,
    build_conditional_bayesian_traversal_rows,
    p_value_log_bayes_factor_lower_bound,
    run_overlap_conditional_bayesian_traversal_law,
    summarize_conditional_bayesian_rows,
)


def _family_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "coherent_case",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_family_size": 1,
                "residual_min_sibling_p_value": 1e-8,
                "residual_neg_log10_min_sibling_p_value": 8.0,
                "residual_max_homogeneity_gain_min": 0.030,
                "residual_max_continuous_context_margin": 0.020,
                "residual_max_subspace_consensus_jaccard_topk": 0.80,
                "residual_max_depth": 4,
                "residual_median_parent_size": 80,
                "residual_max_barycentric_balance": 0.49,
                "residual_min_fragment_risk_proxy_score": 0.50,
                "residual_max_balanced_recovery_proxy_score": 2.60,
                "residual_min_size_balance": 0.45,
                "residual_min_edge_norm_balance": 0.90,
            },
            {
                "case_id": "fragment_case",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_nonrecovery_family",
                "residual_family_size": 2,
                "residual_min_sibling_p_value": 1e-6,
                "residual_neg_log10_min_sibling_p_value": 6.0,
                "residual_max_homogeneity_gain_min": -0.010,
                "residual_max_continuous_context_margin": -0.020,
                "residual_max_subspace_consensus_jaccard_topk": 0.10,
                "residual_max_depth": 0,
                "residual_median_parent_size": 800,
                "residual_max_barycentric_balance": 0.10,
                "residual_min_fragment_risk_proxy_score": 2.00,
                "residual_max_balanced_recovery_proxy_score": 0.50,
                "residual_min_size_balance": 0.10,
                "residual_min_edge_norm_balance": 0.20,
            },
            {
                "case_id": "null_case",
                "data_role": "selected_null",
                "replicate": 0,
                "residual_family_truth_role": "residual_null_like_family",
                "residual_family_size": 1,
                "residual_min_sibling_p_value": 0.50,
                "residual_neg_log10_min_sibling_p_value": 0.30103,
                "residual_max_homogeneity_gain_min": 0.000,
                "residual_max_continuous_context_margin": -0.010,
                "residual_max_subspace_consensus_jaccard_topk": 0.30,
                "residual_max_depth": 1,
                "residual_median_parent_size": 200,
                "residual_max_barycentric_balance": 0.30,
                "residual_min_fragment_risk_proxy_score": 1.10,
                "residual_max_balanced_recovery_proxy_score": 1.20,
                "residual_min_size_balance": 0.30,
                "residual_min_edge_norm_balance": 0.50,
            },
        ]
    )


def test_p_value_bayes_factor_lower_bound_is_monotone() -> None:
    values = p_value_log_bayes_factor_lower_bound(
        pd.Series([1e-8, 1e-4, 0.5]),
    )

    assert values.iloc[0] > values.iloc[1]
    assert values.iloc[1] > values.iloc[2]
    assert values.iloc[2] == 0.0


def test_conditional_status_requires_structural_neighborhood_support() -> None:
    rows = build_conditional_bayesian_traversal_rows(_family_rows())
    status_by_case = dict(
        zip(rows["case_id"], rows["conditional_bayesian_status"], strict=True)
    )

    assert status_by_case["coherent_case"] == "conditional_coherent_candidate"
    assert (
        status_by_case["fragment_case"]
        == "p_value_extreme_structurally_incoherent"
    )
    coherent = rows[rows["case_id"].eq("coherent_case")].iloc[0]
    fragment = rows[rows["case_id"].eq("fragment_case")].iloc[0]
    assert coherent["neighborhood_log_bayes_factor"] > 0.0
    assert fragment["neighborhood_log_bayes_factor"] < 0.0


def test_summary_and_runner_write_outputs(tmp_path) -> None:
    rows = build_conditional_bayesian_traversal_rows(_family_rows())
    summary = summarize_conditional_bayesian_rows(rows)

    assert set(summary["residual_family_truth_role"]) == {
        "residual_nonrecovery_family",
        "residual_null_like_family",
        "residual_truth_recovery_family",
    }
    family_path = tmp_path / "families.csv"
    _family_rows().to_csv(family_path, index=False)
    outputs = run_overlap_conditional_bayesian_traversal_law(
        OverlapConditionalBayesianTraversalLawConfig(
            residual_family_rows_path=family_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
