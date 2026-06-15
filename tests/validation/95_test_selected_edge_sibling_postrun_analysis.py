from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.production_admissibility_contract import (
    evaluate_production_admissibility_components,
    summarize_production_admissibility_contracts,
)
from benchmarks.diagnostics.calibration.selected_edge_sibling_postrun_analysis import (
    build_distribution_shape_records,
    build_postrun_production_admissibility_components,
    build_selected_edge_sibling_equation_records,
    run_selected_edge_sibling_postrun_analysis,
)
from benchmarks.validation.selected_edge_type1_geometry import run_selected_edge_replicate


def _sibling_artifact() -> pd.DataFrame:
    _edge_rows, sibling_rows, _final_rows = run_selected_edge_replicate(
        case_id="binary_2clusters",
        source_family="binary_template",
        feature_representation="binary",
        n_samples=12,
        n_features=5,
        replicate=0,
        data_seed=11,
        tree_seed=11,
        mode="selected_tree",
        edge_alpha=0.001,
        sibling_alpha=0.01,
        run_id="postrun",
    )
    return pd.DataFrame.from_records(sibling_rows)


def test_build_distribution_shape_records_from_enriched_sibling_artifact() -> None:
    records = build_distribution_shape_records(_sibling_artifact())

    assert not records.empty
    assert records["test_statistic"].notna().all()
    assert records["degrees_of_freedom"].notna().all()
    assert "alternate_degrees_of_freedom" in records.columns
    assert "alternate_reference_scale" in records.columns


def test_build_equation_records_from_enriched_sibling_artifact() -> None:
    records = build_selected_edge_sibling_equation_records(_sibling_artifact())

    assert not records.empty
    assert records["sibling_test_statistic"].notna().all()
    assert records["left_edge_p_value"].notna().all()
    assert records["is_null_context"].eq(True).all()


def test_postrun_production_components_fail_closed_on_shape_mismatch() -> None:
    siblings = _sibling_artifact()
    distribution_summary = pd.DataFrame.from_records(
        [
            {
                "test_family": "binary_template",
                "statistic_context_role": "selected_tree__raw_only_pre_calibration",
                "df_bin": "df_0_1",
                "distribution_shape_status": "skew_exceeds_df_reference",
            }
        ]
    )
    equation_summary = pd.DataFrame.from_records(
        [
            {
                "selected_edge_sibling_status": "conditional_empirical_p_value",
            }
        ]
    )

    components = build_postrun_production_admissibility_components(
        siblings=siblings,
        distribution_summary=distribution_summary,
        equation_summary=equation_summary,
    )
    rows = evaluate_production_admissibility_components(components)
    summary = summarize_production_admissibility_contracts(rows)

    assert not components.empty
    assert summary.iloc[0]["production_decision"] == "fail_closed_undefined"
    assert "distribution_shape:" in summary.iloc[0]["blocking_component_ids"]


def test_run_selected_edge_sibling_postrun_analysis_writes_outputs(tmp_path: Path) -> None:
    siblings_path = tmp_path / "selected_edge_geometry_siblings.csv"
    output_dir = tmp_path / "analysis"
    _sibling_artifact().to_csv(siblings_path, index=False)

    outputs = run_selected_edge_sibling_postrun_analysis(
        siblings_path=siblings_path,
        output_dir=output_dir,
        min_null_records=3,
    )

    assert set(outputs) == {
        "distribution_rows",
        "distribution_summary",
        "equation_rows",
        "equation_contexts",
        "equation_summary",
        "production_components",
        "production_summary",
        "manifest",
    }
    assert (output_dir / "sibling_distribution_shape_summary.csv").exists()
    assert (output_dir / "selected_edge_sibling_equation_summary.csv").exists()
    production_summary = pd.read_csv(output_dir / "production_admissibility_summary.csv")
    assert production_summary["production_decision"].isin(
        {"fail_closed_undefined", "diagnostic_only", "production_admissible"}
    ).all()
