from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_bayesian_neighborhood_component_audit import (
    OverlapBayesianNeighborhoodComponentAuditConfig,
    build_bayesian_neighborhood_component_rows,
    run_overlap_bayesian_neighborhood_component_audit,
    summarize_bayesian_neighborhood_components,
)


def _bayesian_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "coherent",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_family_size": 1,
                "selected_family_log_bayes_factor_lower": 10.0,
                "selection_context_log_penalty": 0.5,
                "context_prior_log_odds": -2.0,
                "homogeneity_log_bayes_factor": 1.0,
                "context_margin_log_bayes_factor": 0.6,
                "subspace_log_bayes_factor": 0.4,
                "balance_log_bayes_factor": 0.3,
                "balanced_recovery_log_bayes_factor": 0.2,
                "fragment_risk_log_penalty": 0.0,
                "neighborhood_log_bayes_factor": 2.5,
                "conditional_coherent_posterior": 0.99,
                "conditional_bayesian_status": "conditional_coherent_candidate",
            },
            {
                "case_id": "subspace_blocked",
                "data_role": "signal",
                "replicate": 0,
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_family_size": 1,
                "selected_family_log_bayes_factor_lower": 12.0,
                "selection_context_log_penalty": 0.5,
                "context_prior_log_odds": -2.0,
                "homogeneity_log_bayes_factor": 1.0,
                "context_margin_log_bayes_factor": 0.5,
                "subspace_log_bayes_factor": -0.8,
                "balance_log_bayes_factor": 0.2,
                "balanced_recovery_log_bayes_factor": 0.1,
                "fragment_risk_log_penalty": 0.0,
                "neighborhood_log_bayes_factor": 1.0,
                "conditional_coherent_posterior": 0.95,
                "conditional_bayesian_status": ("p_value_extreme_neighborhood_insufficient"),
            },
            {
                "case_id": "fragment_blocked",
                "data_role": "selected_null",
                "replicate": 0,
                "residual_family_truth_role": "residual_null_like_family",
                "residual_family_size": 1,
                "selected_family_log_bayes_factor_lower": 8.0,
                "selection_context_log_penalty": 0.5,
                "context_prior_log_odds": -2.0,
                "homogeneity_log_bayes_factor": 0.1,
                "context_margin_log_bayes_factor": -0.2,
                "subspace_log_bayes_factor": 0.0,
                "balance_log_bayes_factor": 0.1,
                "balanced_recovery_log_bayes_factor": 0.0,
                "fragment_risk_log_penalty": 0.6,
                "neighborhood_log_bayes_factor": -0.6,
                "conditional_coherent_posterior": 0.80,
                "conditional_bayesian_status": ("p_value_extreme_structurally_incoherent"),
            },
        ]
    )


def test_component_audit_identifies_limiting_component() -> None:
    rows = build_bayesian_neighborhood_component_rows(_bayesian_rows())
    by_case = rows.set_index("case_id")

    assert by_case.loc["coherent", "component_audit_status"] == (
        "component_audit_coherent_candidate"
    )
    assert by_case.loc["subspace_blocked", "limiting_component"] == ("subspace_log_bayes_factor")
    assert by_case.loc["subspace_blocked", "component_audit_status"] == (
        "component_audit_p_value_evidence_without_strong_neighborhood"
    )
    assert by_case.loc["fragment_blocked", "limiting_component"] == (
        "fragment_risk_log_contribution"
    )
    assert by_case.loc["fragment_blocked", "component_audit_status"] == (
        "component_audit_structurally_incoherent"
    )


def test_component_summary_counts_statuses() -> None:
    rows = build_bayesian_neighborhood_component_rows(_bayesian_rows())
    summary = summarize_bayesian_neighborhood_components(rows)
    recovery = summary[
        summary["residual_family_truth_role"].eq("residual_truth_recovery_family")
    ].iloc[0]

    assert int(recovery["family_count"]) == 2
    assert int(recovery["coherent_candidate_count"]) == 1
    assert int(recovery["p_value_evidence_without_strong_neighborhood_count"]) == 1


def test_component_audit_runner_writes_outputs(tmp_path) -> None:
    bayesian_path = tmp_path / "bayesian.csv"
    _bayesian_rows().to_csv(bayesian_path, index=False)

    outputs = run_overlap_bayesian_neighborhood_component_audit(
        OverlapBayesianNeighborhoodComponentAuditConfig(
            bayesian_rows_path=bayesian_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
