from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.selected.tail.selected_tail_admissibility_domain import (
    SelectedTailRun,
    build_admissibility_domain_table,
    nearest_boundary_contexts,
    parse_run_specs,
    run_selected_tail_admissibility_domain,
    summarize_admissibility_domain,
)


def _tail_law_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_family": "gaussian_blobs",
                "feature_family": "bernoulli",
                "parent_size_bin": "small_0_0.25",
                "sibling_projection_dimension": 1,
                "edge_action_bin": "edge_action_ge8",
                "n_records": 1000,
                "n_matching_simulations": 600,
                "required_min_matching_simulations": 499,
                "required_min_matched_records": 499,
                "max_exceedance_standard_error": 0.002,
                "production_tail_law_admissible": True,
                "tail_law_admissibility_failure_reasons": "",
                "heldout_exceedance_rate": 0.01,
                "heldout_exceedance_standard_error": 0.001,
            },
            {
                "source_family": "categorical_multinomial",
                "feature_family": "categorical",
                "parent_size_bin": "medium_0.25_0.5",
                "sibling_projection_dimension": 2,
                "edge_action_bin": "edge_action_ge8",
                "n_records": 1200,
                "n_matching_simulations": 499,
                "required_min_matching_simulations": 499,
                "required_min_matched_records": 499,
                "max_exceedance_standard_error": 0.002,
                "production_tail_law_admissible": False,
                "tail_law_admissibility_failure_reasons": ("heldout_exceedance_se_above_contract"),
                "heldout_exceedance_rate": 0.012,
                "heldout_exceedance_standard_error": 0.003,
            },
            {
                "source_family": "binary_template",
                "feature_family": "bernoulli",
                "parent_size_bin": "small_0_0.25",
                "sibling_projection_dimension": 1,
                "edge_action_bin": "edge_action_ge8",
                "n_records": 900,
                "n_matching_simulations": 300,
                "required_min_matching_simulations": 499,
                "required_min_matched_records": 499,
                "max_exceedance_standard_error": 0.002,
                "production_tail_law_admissible": False,
                "tail_law_admissibility_failure_reasons": (
                    "matching_simulations_below_tail_resolution_contract"
                ),
                "heldout_exceedance_rate": 0.01,
                "heldout_exceedance_standard_error": 0.001,
            },
        ]
    )


def test_parse_run_specs_requires_named_csv_paths() -> None:
    runs = parse_run_specs(["focused=/tmp/focused.csv", "boundary=/tmp/boundary.csv"])

    assert runs == (
        SelectedTailRun(run_id="focused", path=Path("/tmp/focused.csv")),
        SelectedTailRun(run_id="boundary", path=Path("/tmp/boundary.csv")),
    )


def test_admissibility_domain_classifies_contexts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "selected_ratio_tail_law.csv"
        _tail_law_table().to_csv(path, index=False)

        domain = build_admissibility_domain_table((SelectedTailRun(run_id="test_run", path=path),))

    by_source = domain.set_index("source_family")
    assert by_source.loc["gaussian_blobs", "admissibility_class"] == (
        "production_admissible_context"
    )
    assert by_source.loc["categorical_multinomial", "admissibility_class"] == (
        "support_met_tail_precision_failed"
    )
    assert by_source.loc["binary_template", "admissibility_class"] == (
        "support_failed_tail_precision_met_or_unchecked"
    )


def test_admissibility_domain_uses_recorded_precision_contract() -> None:
    table = _tail_law_table()
    table.loc[0, "max_exceedance_standard_error"] = 0.01
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "selected_ratio_tail_law.csv"
        table.to_csv(path, index=False)

        domain = build_admissibility_domain_table((SelectedTailRun(run_id="test_run", path=path),))

    gaussian = domain[domain["source_family"].eq("gaussian_blobs")].iloc[0]
    assert abs(float(gaussian["tail_precision_margin"]) - 0.009) < 1e-12


def test_summary_and_boundary_outputs_are_written() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "selected_ratio_tail_law.csv"
        output_dir = Path(tmpdir) / "out"
        _tail_law_table().to_csv(input_path, index=False)

        outputs = run_selected_tail_admissibility_domain(
            runs=(SelectedTailRun(run_id="test_run", path=input_path),),
            output_dir=output_dir,
            top_n=2,
        )

        assert (output_dir / "context_admissibility_domain.csv").exists()
        assert (output_dir / "admissibility_summary_by_run_family.csv").exists()
        assert (output_dir / "nearest_boundary_contexts.csv").exists()
        assert (output_dir / "manifest.json").exists()
        assert set(outputs) == {
            "context_admissibility_domain",
            "admissibility_summary_by_run_family",
            "nearest_boundary_contexts",
        }

        summary = summarize_admissibility_domain(outputs["context_admissibility_domain"])
        assert int(summary["n_admissible_contexts"].sum()) == 1

        boundary = nearest_boundary_contexts(
            outputs["context_admissibility_domain"],
            top_n=2,
        )
        assert boundary.shape[0] == 2
