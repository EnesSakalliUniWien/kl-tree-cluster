from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.validation.statistics.traversal_sibling_fdr_null import (
    FdrLayer,
    TraversalSiblingFdrConfig,
    classify_fdr_outcome,
    estimate_fdr,
    parse_fdr_layers,
    run_traversal_sibling_fdr_layer,
    run_traversal_sibling_fdr_layers,
)


def test_parse_fdr_layers_accepts_named_layers() -> None:
    assert parse_fdr_layers("synthetic_valid_p,fixed_tree_wald") == (
        FdrLayer.SYNTHETIC_VALID_P,
        FdrLayer.FIXED_TREE_WALD,
    )


def test_estimate_fdr_uses_false_discoveries_over_rejections() -> None:
    summary = estimate_fdr(
        [
            {"status": "ok", "n_false_rejections": 0, "n_rejections": 0},
            {"status": "ok", "n_false_rejections": 1, "n_rejections": 2},
        ]
    )
    assert summary["mean_fdp"] == 0.25
    assert summary["false_rejection_rate"] == 0.5
    assert summary["n_support_failures"] == 0.0


def test_estimate_fdr_reports_support_failures_without_neutral_fdp() -> None:
    summary = estimate_fdr(
        [
            {"status": "support_failure", "n_false_rejections": 0, "n_rejections": 0},
        ]
    )
    assert pd.isna(summary["mean_fdp"])
    assert summary["n_support_failures"] == 1.0


def test_classify_fdr_outcome_separates_algorithm_from_calibration() -> None:
    assert (
        classify_fdr_outcome(layer=FdrLayer.SYNTHETIC_VALID_P, mean_fdp=0.008, alpha=0.01)
        == "algorithmic_fdr_control"
    )
    assert (
        classify_fdr_outcome(layer=FdrLayer.FIXED_TREE_WALD, mean_fdp=0.08, alpha=0.01)
        == "fixed_tree_calibration_failure"
    )
    assert (
        classify_fdr_outcome(
            layer=FdrLayer.SELECTED_TREE_INFLATED,
            mean_fdp=float("nan"),
            alpha=0.01,
            n_ok=0,
            n_support_failures=2,
        )
        == "inflation_support_failure"
    )
    assert (
        classify_fdr_outcome(
            layer=FdrLayer.SELECTED_TREE_INFLATED,
            mean_fdp=0.0,
            alpha=0.01,
            n_ok=2,
            n_support_failures=18,
        )
        == "inflation_support_failure"
    )


def test_config_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        TraversalSiblingFdrConfig(
            layer=FdrLayer.SYNTHETIC_VALID_P,
            case_names=("binary_2clusters",),
            replicates=10,
            alpha=0.0,
            base_seed=1,
        )


def test_synthetic_valid_p_layer_measures_algorithmic_family_error() -> None:
    config = TraversalSiblingFdrConfig(
        layer=FdrLayer.SYNTHETIC_VALID_P,
        case_names=("synthetic_balanced_binary_tree",),
        replicates=200,
        alpha=0.05,
        base_seed=123,
    )
    outputs = run_traversal_sibling_fdr_layer(config)

    assert outputs["summary"]["n_simulations"] == 200.0
    assert outputs["summary"]["n_support_failures"] == 0.0
    assert outputs["summary"]["outcome"] in {
        "algorithmic_fdr_control",
        "algorithmic_fdr_failure",
    }


def test_fixed_tree_wald_layer_reports_calibration_status() -> None:
    config = TraversalSiblingFdrConfig(
        layer=FdrLayer.FIXED_TREE_WALD,
        case_names=("binary_2clusters",),
        replicates=3,
        alpha=0.01,
        base_seed=20260604,
    )
    outputs = run_traversal_sibling_fdr_layer(config)

    assert outputs["summary"]["n_simulations"] == 3.0
    assert "outcome" in outputs["summary"]
    assert all("n_rejections" in row for row in outputs["simulation_rows"])


def test_selected_tree_layers_report_strict_support_failures_or_decisions() -> None:
    for layer in (FdrLayer.SELECTED_TREE_WALD, FdrLayer.SELECTED_TREE_INFLATED):
        config = TraversalSiblingFdrConfig(
            layer=layer,
            case_names=("binary_2clusters",),
            replicates=2,
            alpha=0.01,
            base_seed=20260604,
        )
        outputs = run_traversal_sibling_fdr_layer(config)

        assert outputs["summary"]["n_simulations"] == 2.0
        assert "n_support_failures" in outputs["summary"]


def test_runner_writes_combined_outputs(tmp_path: Path) -> None:
    config = TraversalSiblingFdrConfig(
        layer=FdrLayer.SYNTHETIC_VALID_P,
        case_names=("synthetic_balanced_binary_tree",),
        replicates=5,
        alpha=0.01,
        base_seed=20260604,
    )

    manifest = run_traversal_sibling_fdr_layers(
        [config],
        output_dir=tmp_path,
    )

    assert manifest["n_simulation_rows"] == 5
    assert (tmp_path / "traversal_sibling_fdr_simulations.csv").exists()
    assert (tmp_path / "traversal_sibling_fdr_summary.csv").exists()
    assert (tmp_path / "traversal_sibling_fdr_manifest.json").exists()
