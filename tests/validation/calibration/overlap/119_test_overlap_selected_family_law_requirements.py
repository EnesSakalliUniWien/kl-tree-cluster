from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_selected_family_law_requirements import (
    OverlapSelectedFamilyLawRequirementsConfig,
    build_conditioning_envelope,
    build_selected_family_law_requirements,
    run_overlap_selected_family_law_requirements,
)


def _family_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "residual_family_truth_role": "residual_null_like_family",
                "residual_family_size": 1,
                "residual_neg_log10_min_sibling_p_value": 5.0,
                "residual_max_homogeneity_gain_min": 0.01,
                "residual_max_continuous_context_margin": -0.01,
                "residual_max_subspace_consensus_jaccard_topk": 0.4,
                "residual_max_depth": 1,
                "residual_median_parent_size": 100,
                "residual_max_barycentric_balance": 0.4,
                "residual_min_fragment_risk_proxy_score": 0.6,
                "residual_max_balanced_recovery_proxy_score": 1.0,
                "residual_min_size_balance": 0.3,
                "residual_min_edge_norm_balance": 0.5,
            },
            {
                "residual_family_truth_role": "residual_truth_recovery_family",
                "residual_family_size": 2,
                "residual_neg_log10_min_sibling_p_value": 10.0,
                "residual_max_homogeneity_gain_min": 0.02,
                "residual_max_continuous_context_margin": 0.01,
                "residual_max_subspace_consensus_jaccard_topk": 0.6,
                "residual_max_depth": 2,
                "residual_median_parent_size": 150,
                "residual_max_barycentric_balance": 0.45,
                "residual_min_fragment_risk_proxy_score": 0.8,
                "residual_max_balanced_recovery_proxy_score": 2.0,
                "residual_min_size_balance": 0.4,
                "residual_min_edge_norm_balance": 0.7,
            },
            {
                "residual_family_truth_role": "residual_nonrecovery_family",
                "residual_family_size": 2,
                "residual_neg_log10_min_sibling_p_value": 12.0,
                "residual_max_homogeneity_gain_min": 0.015,
                "residual_max_continuous_context_margin": 0.005,
                "residual_max_subspace_consensus_jaccard_topk": 0.3,
                "residual_max_depth": 1,
                "residual_median_parent_size": 120,
                "residual_max_barycentric_balance": 0.48,
                "residual_min_fragment_risk_proxy_score": 0.5,
                "residual_max_balanced_recovery_proxy_score": 1.2,
                "residual_min_size_balance": 0.35,
                "residual_min_edge_norm_balance": 0.55,
            },
        ]
    )


def _stability_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "stage_name": "residual_selected_family_null_evidence",
                "threshold_stability_status": "nontransferable_focused_cutpoint",
                "requires_selected_family_law": True,
            },
            {
                "stage_name": "residual_null_evidence_only_unresolved",
                "threshold_stability_status": "selected_family_law_required",
                "requires_selected_family_law": True,
            },
        ]
    )


def test_conditioning_envelope_summarizes_roles() -> None:
    envelope = build_conditioning_envelope(_family_rows())

    recovery_p = envelope[
        envelope["residual_family_truth_role"].eq("residual_truth_recovery_family")
        & envelope["metric"].eq("residual_neg_log10_min_sibling_p_value")
    ].iloc[0]
    assert int(recovery_p["family_count"]) == 1
    assert float(recovery_p["median_value"]) == 10.0


def test_selected_family_law_requirements_include_core_components() -> None:
    requirements = build_selected_family_law_requirements(
        _family_rows(),
        _stability_rows(),
    )

    assert set(requirements["requirement_id"]) == {
        "selected_family_null_evidence_law",
        "selected_family_structural_recovery_condition",
        "threshold_transfer_validation",
        "multiscale_unstable_zone_reporting",
    }
    structural = requirements[
        requirements["requirement_id"].eq("selected_family_structural_recovery_condition")
    ].iloc[0]
    assert structural["current_status"] == "law_required_structural_target_missing"


def test_run_selected_family_law_requirements_writes_outputs(tmp_path) -> None:
    family_path = tmp_path / "families.csv"
    stability_path = tmp_path / "stability.csv"
    _family_rows().to_csv(family_path, index=False)
    _stability_rows().to_csv(stability_path, index=False)

    outputs = run_overlap_selected_family_law_requirements(
        OverlapSelectedFamilyLawRequirementsConfig(
            residual_family_rows_path=family_path,
            stability_contract_rows_path=stability_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
