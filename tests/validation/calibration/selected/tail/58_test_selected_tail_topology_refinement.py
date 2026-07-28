from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.selected.tail.selected_tail_topology_refinement import (
    DATA_ADAPTIVE_CONTEXT_MODE,
    DATA_ADAPTIVE_PRODUCTION_BLOCK_REASON,
    PREDECLARED_CONTEXT_MODE,
    STUDY_ROLE,
    RecordsInput,
    add_refinement_bins,
    compare_refinements_to_base,
    evaluate_topology_refined_tail_laws,
    load_selected_geometry_records,
    parse_records_specs,
    run_topology_refinement_diagnostic,
    summarize_refinement_families,
)


def _records() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(80):
        low_balance = index < 40
        ratio = 10.0 + (index % 40) if low_balance else 100.0 + 2.0 * (index % 40)
        rows.append(
            {
                "case_id": "case",
                "source_family": "gaussian_blobs",
                "feature_family": "bernoulli",
                "parent_size_bin": "medium_0.25_0.5",
                "sibling_projection_dimension": 2,
                "edge_action_bin": "edge_action_ge8",
                "selected_hierarchy_simulation_id": f"case:{index}",
                "replicate_index": index,
                "feature_dimension": 40,
                "parent_sample_size": 80,
                "left_child_sample_size": 16 if low_balance else 38,
                "right_child_sample_size": 64 if low_balance else 42,
                "selected_hierarchy_ratio": ratio,
                "child_balance": 0.2 if low_balance else 0.48,
                "subtree_colless_normalized": 0.8 if low_balance else 0.05,
                "subtree_sackin_mean_depth": 5.0 if low_balance else 2.0,
                "subtree_branch_length_condition_ratio": 20.0 if low_balance else 2.0,
                "subtree_branch_length_cv": 1.5 if low_balance else 0.2,
                "branch_length_asymmetry": 0.7 if low_balance else 0.1,
                "negative_log10_min_child_edge_bh_p_value": 8.5 + index / 100.0,
                "eigenvalue_effective_rank": 1.5 if low_balance else 3.5,
                "selected_eigenvalue_over_mp_upper_bound": (0.9 if low_balance else 2.5),
                "selected_subspace_cos2": 0.2 if low_balance else 0.8,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_parse_and_load_records_specs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "records.csv"
        _records().to_csv(path, index=False)

        specs = parse_records_specs([f"run={path}"])
        loaded = load_selected_geometry_records(specs)

    assert specs == (RecordsInput(run_id="run", path=path),)
    assert "run_id" in loaded.columns
    assert loaded["run_id"].eq("run").all()


def test_load_records_prefixes_simulation_ids_by_run() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        first_path = Path(tmpdir) / "first.csv"
        second_path = Path(tmpdir) / "second.csv"
        records = _records().head(4)
        records.to_csv(first_path, index=False)
        records.to_csv(second_path, index=False)

        loaded = load_selected_geometry_records(
            (
                RecordsInput(run_id="first", path=first_path),
                RecordsInput(run_id="second", path=second_path),
            )
        )

    assert loaded["selected_hierarchy_simulation_id"].nunique() == 8
    assert set(loaded["selected_hierarchy_simulation_id"]) == {
        "first:case:0",
        "first:case:1",
        "first:case:2",
        "first:case:3",
        "second:case:0",
        "second:case:1",
        "second:case:2",
        "second:case:3",
    }


def test_load_records_rejects_stale_edge_action_bins() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "records.csv"
        records = _records().head(1).copy()
        records["edge_action_bin"] = "edge_action_0_2"
        records.to_csv(path, index=False)

        try:
            load_selected_geometry_records((RecordsInput(run_id="run", path=path),))
        except ValueError as exc:
            assert "stale edge_action_bin" in str(exc)
        else:
            raise AssertionError("stale edge_action_bin should be rejected")


def test_refinement_bins_are_added_within_base_context() -> None:
    binned = add_refinement_bins(_records())

    assert "child_balance_bin" in binned.columns
    assert set(binned["child_balance_bin"]) == {"value_1_0_2", "value_2_0_48"}
    assert "selected_subspace_cos2_bin" in binned.columns


def test_topology_refined_tail_laws_report_base_and_refinements() -> None:
    tail_laws = evaluate_topology_refined_tail_laws(
        _records(),
        n_folds=4,
        min_train_simulations=5,
        min_train_records=5,
        required_min_matching_simulations=20,
        required_min_matched_records=20,
        max_exceedance_standard_error=0.2,
    )

    assert {"base", "topology", "spectral_alignment"} <= set(tail_laws["context_family"])
    assert tail_laws["study_role"].eq(STUDY_ROLE).all()
    assert tail_laws["max_exceedance_standard_error"].eq(0.2).all()

    base_rows = tail_laws[tail_laws["context_family"].eq("base")]
    refined_rows = tail_laws[~tail_laws["context_family"].eq("base")]
    assert base_rows["context_definition_mode"].eq(PREDECLARED_CONTEXT_MODE).all()
    assert refined_rows["context_definition_mode"].eq(DATA_ADAPTIVE_CONTEXT_MODE).all()
    assert refined_rows["production_tail_law_admissible"].eq(False).all()
    diagnostic_passed_refined = refined_rows[refined_rows["diagnostic_tail_law_contract_passed"]]
    assert not diagnostic_passed_refined.empty
    assert (
        diagnostic_passed_refined["production_tail_law_failure_reasons"]
        .str.contains(
            DATA_ADAPTIVE_PRODUCTION_BLOCK_REASON,
            regex=False,
        )
        .all()
    )

    summary = summarize_refinement_families(tail_laws)
    assert "n_precision_pass_contexts" in summary.columns
    assert "n_diagnostic_tail_law_contract_passed_contexts" in summary.columns
    assert summary["n_precision_pass_contexts"].sum() == summary["n_descriptive_contexts"].sum()
    assert (
        summary.loc[
            summary["context_family"].ne("base"),
            "n_production_admissible_contexts",
        ]
        == 0
    ).all()
    comparison = compare_refinements_to_base(tail_laws)
    assert "interpretation" in comparison.columns
    assert "n_refined_diagnostic_tail_law_contract_passed_contexts" in comparison.columns
    assert "n_refined_production_admissible_contexts" in comparison.columns
    assert comparison["n_refined_production_admissible_contexts"].eq(0).all()
    assert comparison["n_refined_contexts"].max() >= 1


def test_run_topology_refinement_diagnostic_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "records.csv"
        output_dir = Path(tmpdir) / "out"
        _records().to_csv(input_path, index=False)

        outputs = run_topology_refinement_diagnostic(
            records_inputs=(RecordsInput(run_id="synthetic", path=input_path),),
            output_dir=output_dir,
            n_folds=4,
            min_train_simulations=5,
            min_train_records=5,
            required_min_matching_simulations=20,
            required_min_matched_records=20,
            max_exceedance_standard_error=0.2,
        )

        assert set(outputs) == {
            "refined_tail_law",
            "refinement_summary",
            "base_context_refinement_comparison",
        }
        assert (output_dir / "refined_tail_law.csv").exists()
        assert (output_dir / "refinement_summary.csv").exists()
        assert (output_dir / "base_context_refinement_comparison.csv").exists()
        assert (output_dir / "manifest.json").exists()
