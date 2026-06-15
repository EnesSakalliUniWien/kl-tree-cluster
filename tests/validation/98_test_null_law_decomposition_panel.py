from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.null_law_decomposition_panel import (
    NullLawDecompositionConfig,
    build_null_law_production_components,
    projection_operator_eigenvalues,
    projection_orthonormality_error,
    run_null_law_decomposition_panel,
    summarize_null_law_decomposition_rows,
)
from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)


def test_projection_operator_eigenvalues_detect_nonorthonormal_weights() -> None:
    projection = np.array([[2.0, 0.0], [0.0, 1.0]])

    weights = projection_operator_eigenvalues(projection)

    assert weights.tolist() == pytest.approx([4.0, 1.0])
    assert projection_orthonormality_error(projection) > 0.0


def test_null_law_summary_marks_adaptive_tail_inflation_fail_closed() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "mode": "fixed_topology",
                "projection_source": "adaptive_same_sample",
                "source_family": "binary_template",
                "df_bin": "df_0_1",
                "p_value": 0.001,
                "tail_reject_at_0.05": True,
                "test_statistic": 12.0,
                "projection_orthonormality_error": 0.0,
                "operator_weight_min": 1.0,
                "operator_weight_max": 1.0,
                "operator_satterthwaite_scale": 1.0,
                "operator_satterthwaite_df": 1.0,
            },
            {
                "case_id": "unit",
                "mode": "fixed_topology",
                "projection_source": "adaptive_same_sample",
                "source_family": "binary_template",
                "df_bin": "df_0_1",
                "p_value": 0.002,
                "tail_reject_at_0.05": True,
                "test_statistic": 11.0,
                "projection_orthonormality_error": 0.0,
                "operator_weight_min": 1.0,
                "operator_weight_max": 1.0,
                "operator_satterthwaite_scale": 1.0,
                "operator_satterthwaite_df": 1.0,
            },
        ]
    )

    summary = summarize_null_law_decomposition_rows(rows, min_rows=2)
    components = build_null_law_production_components(summary)
    contract_rows = evaluate_production_admissibility_components(components)
    contract = summarize_production_admissibility_contracts(contract_rows)

    assert summary.iloc[0]["null_law_status"] == "null_law_adaptive_projection_tail_inflated"
    assert contract.iloc[0]["production_decision"] == "fail_closed_undefined"


def test_null_law_summary_marks_fixed_projection_candidate_diagnostic_only() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "mode": "fixed_topology",
                "projection_source": "independent_tree_sample",
                "source_family": "binary_template",
                "df_bin": "df_0_1",
                "p_value": 0.4,
                "tail_reject_at_0.05": False,
                "test_statistic": 0.7,
                "projection_orthonormality_error": 0.0,
                "operator_weight_min": 1.0,
                "operator_weight_max": 1.0,
                "operator_satterthwaite_scale": 1.0,
                "operator_satterthwaite_df": 1.0,
            },
            {
                "case_id": "unit",
                "mode": "fixed_topology",
                "projection_source": "independent_tree_sample",
                "source_family": "binary_template",
                "df_bin": "df_0_1",
                "p_value": 0.6,
                "tail_reject_at_0.05": False,
                "test_statistic": 0.3,
                "projection_orthonormality_error": 0.0,
                "operator_weight_min": 1.0,
                "operator_weight_max": 1.0,
                "operator_satterthwaite_scale": 1.0,
                "operator_satterthwaite_df": 1.0,
            },
        ]
    )

    summary = summarize_null_law_decomposition_rows(rows, min_rows=2)
    components = build_null_law_production_components(summary)
    contract_rows = evaluate_production_admissibility_components(components)
    contract = summarize_production_admissibility_contracts(contract_rows)

    assert summary.iloc[0]["null_law_status"] == "null_law_fixed_projection_candidate"
    assert contract.iloc[0]["production_decision"] == "diagnostic_only"


def test_run_null_law_decomposition_panel_writes_outputs(tmp_path: Path) -> None:
    outputs = run_null_law_decomposition_panel(
        NullLawDecompositionConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            edge_alphas=(0.001,),
            sibling_alpha=0.01,
            replicates=1,
            base_seed=20260613,
        )
    )

    assert set(outputs) == {
        "rows",
        "summary",
        "production_components",
        "production_summary",
        "manifest",
    }
    rows = pd.read_csv(tmp_path / "null_law_decomposition_rows.csv")
    summary = pd.read_csv(tmp_path / "null_law_decomposition_summary.csv")

    assert {
        "adaptive_same_sample",
        "independent_tree_sample",
        "random_fixed_orthonormal",
    } <= set(rows["projection_source"])
    assert not summary.empty
    assert set(summary["null_law_status"]) <= {
        "null_law_adaptive_projection_tail_inflated",
        "null_law_fixed_projection_candidate",
        "null_law_fixed_projection_tail_misaligned",
        "null_law_insufficient_rows",
    }
