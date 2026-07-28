from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.traversal.production_admissibility_contract import (
    STUDY_ROLE,
    evaluate_production_admissibility_components,
    run_production_admissibility_contract,
    summarize_production_admissibility_contracts,
)


def _components() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "contract_id": "ready_contract",
                "component_id": "edge_null",
                "component_type": "edge_null_calibration",
                "component_status": "within_nominal_tolerance",
                "required_for_production": True,
            },
            {
                "contract_id": "ready_contract",
                "component_id": "sibling_null",
                "component_type": "sibling_null_calibration",
                "component_status": "production_ready",
                "required_for_production": True,
            },
            {
                "contract_id": "diagnostic_contract",
                "component_id": "traversal_guard",
                "component_type": "traversal_guard",
                "component_status": "diagnostic_only_guard",
                "required_for_production": True,
            },
            {
                "contract_id": "equation_diagnostic_contract",
                "component_id": "selected_edge_sibling_equation",
                "component_type": "selected_edge_sibling_null_equation",
                "component_status": "conditional_empirical_p_value",
                "required_for_production": True,
            },
            {
                "contract_id": "distribution_diagnostic_contract",
                "component_id": "statistic_distribution_shape",
                "component_type": "statistic_distribution_shape_panel",
                "component_status": "chi_square_shape_candidate",
                "required_for_production": True,
            },
            {
                "contract_id": "laplacian_diagnostic_contract",
                "component_id": "covariance_laplacian",
                "component_type": "covariance_laplacian_panel",
                "component_status": "laplacian_skipped_high_dimension",
                "required_for_production": True,
            },
            {
                "contract_id": "differential_diagnostic_contract",
                "component_id": "differential_validity",
                "component_type": "differential_statistic_validity_panel",
                "component_status": "fixed_subspace_candidate",
                "required_for_production": True,
            },
            {
                "contract_id": "regularized_diagnostic_contract",
                "component_id": "regularized_wald",
                "component_type": "regularized_wald_statistic_panel",
                "component_status": "regularized_fixed_tree_candidate",
                "required_for_production": True,
            },
            {
                "contract_id": "data_independent_gate_diagnostic_contract",
                "component_id": "data_independent_gate",
                "component_type": "data_independent_sibling_gate_panel",
                "component_status": "data_independent_gate_null_candidate",
                "required_for_production": True,
            },
            {
                "contract_id": "data_independent_gate_transfer_diagnostic_contract",
                "component_id": "data_independent_gate_transfer",
                "component_type": "data_independent_sibling_gate_transfer",
                "component_status": "data_independent_gate_penalty_transfer_candidate",
                "required_for_production": True,
            },
            {
                "contract_id": "fail_closed_contract",
                "component_id": "external_tail",
                "component_type": "external_selected_tail",
                "component_status": "undefined_external_not_admissible",
                "required_for_production": True,
            },
            {
                "contract_id": "equation_fail_closed_contract",
                "component_id": "selected_edge_sibling_equation",
                "component_type": "selected_edge_sibling_null_equation",
                "component_status": "insufficient_null_support",
                "required_for_production": True,
            },
            {
                "contract_id": "distribution_fail_closed_contract",
                "component_id": "statistic_distribution_shape",
                "component_type": "statistic_distribution_shape_panel",
                "component_status": "skew_exceeds_df_reference",
                "required_for_production": True,
            },
            {
                "contract_id": "differential_fail_closed_contract",
                "component_id": "differential_validity",
                "component_type": "differential_statistic_validity_panel",
                "component_status": "projection_unstable",
                "required_for_production": True,
            },
            {
                "contract_id": "regularized_fail_closed_contract",
                "component_id": "regularized_wald",
                "component_type": "regularized_wald_statistic_panel",
                "component_status": "regularized_tail_misaligned",
                "required_for_production": True,
            },
            {
                "contract_id": "data_independent_gate_fail_closed_contract",
                "component_id": "data_independent_gate",
                "component_type": "data_independent_sibling_gate_panel",
                "component_status": "data_independent_gate_null_inflated",
                "required_for_production": True,
            },
            {
                "contract_id": "data_independent_gate_transfer_fail_closed_contract",
                "component_id": "data_independent_gate_transfer",
                "component_type": "data_independent_sibling_gate_transfer",
                "component_status": "data_independent_gate_penalty_signal_weak",
                "required_for_production": True,
            },
        ]
    )


def test_production_admissibility_contract_summarizes_decisions() -> None:
    rows = evaluate_production_admissibility_components(_components())
    summary = summarize_production_admissibility_contracts(rows)

    assert rows["study_role"].eq(STUDY_ROLE).all()
    decisions = dict(zip(summary["contract_id"], summary["production_decision"]))

    assert decisions["ready_contract"] == "production_admissible"
    assert decisions["diagnostic_contract"] == "diagnostic_only"
    assert decisions["equation_diagnostic_contract"] == "diagnostic_only"
    assert decisions["distribution_diagnostic_contract"] == "diagnostic_only"
    assert decisions["laplacian_diagnostic_contract"] == "diagnostic_only"
    assert decisions["differential_diagnostic_contract"] == "diagnostic_only"
    assert decisions["regularized_diagnostic_contract"] == "diagnostic_only"
    assert decisions["data_independent_gate_diagnostic_contract"] == "diagnostic_only"
    assert decisions["data_independent_gate_transfer_diagnostic_contract"] == "diagnostic_only"
    assert decisions["fail_closed_contract"] == "fail_closed_undefined"
    assert decisions["equation_fail_closed_contract"] == "fail_closed_undefined"
    assert decisions["distribution_fail_closed_contract"] == "fail_closed_undefined"
    assert decisions["differential_fail_closed_contract"] == "fail_closed_undefined"
    assert decisions["regularized_fail_closed_contract"] == "fail_closed_undefined"
    assert decisions["data_independent_gate_fail_closed_contract"] == "fail_closed_undefined"
    assert (
        decisions["data_independent_gate_transfer_fail_closed_contract"] == "fail_closed_undefined"
    )

    blocked = summary[summary["contract_id"].eq("fail_closed_contract")].iloc[0]
    assert blocked["blocking_component_ids"] == "external_tail"
    equation_blocked = summary[summary["contract_id"].eq("equation_fail_closed_contract")].iloc[0]
    assert equation_blocked["blocking_component_ids"] == "selected_edge_sibling_equation"


def test_production_admissibility_contract_rejects_unknown_status() -> None:
    components = _components()
    components.loc[0, "component_status"] = "looks_good"

    with pytest.raises(ValueError, match="unknown statuses"):
        evaluate_production_admissibility_components(components)


def test_run_production_admissibility_contract_writes_outputs(tmp_path: Path) -> None:
    components_path = tmp_path / "components.csv"
    output_dir = tmp_path / "out"
    _components().to_csv(components_path, index=False)

    outputs = run_production_admissibility_contract(
        components_path=components_path,
        output_dir=output_dir,
    )

    assert set(outputs) == {"components", "summary", "manifest"}
    assert (output_dir / "production_admissibility_components.csv").exists()
    assert (output_dir / "production_admissibility_summary.csv").exists()
    assert (output_dir / "manifest.json").exists()
