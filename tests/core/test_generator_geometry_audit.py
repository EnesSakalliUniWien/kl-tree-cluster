from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from benchmarks.diagnostics.generators.case_geometry_audit import (
    audit_case,
    build_generator_geometry_audit,
    write_generator_geometry_audit,
)
from benchmarks.shared.cases import get_default_test_cases


def _case(name: str) -> dict[str, object]:
    return next(case for case in get_default_test_cases() if case["name"] == name)


def test_generator_geometry_audit_records_low_signal_dimensional_case() -> None:
    row = audit_case(_case("dim_diffuse_6c_536f"))

    assert "low_geometry_signal" in row.geometry_flags
    assert row.nearest_neighbor_same_label_fraction < 0.55
    assert row.recommended_simulation_family == (
        "explicit_bernoulli_threshold_model_or_continuous_gaussian_variant"
    )
    assert row.observation_model == "per_feature_median_thresholded_continuous_matrix"
    assert row.benchmark_intent == "discretized_continuous_stress"


def test_generator_geometry_audit_records_sparse_high_dimensional_case_without_promising_recovery() -> None:
    row = audit_case(_case("gauss_sparse_signal_highd_noise"))

    assert row.n_samples == 40
    assert row.n_features == 20_000
    assert row.true_clusters == 4
    assert row.benchmark_intent == "irrelevant_feature_robustness_stress"
    assert row.metadata_scientific_caution == (
        "known_labels_do_not_imply_tree_recoverability_geometry_audit_is_authoritative"
    )
    assert np.isfinite(row.nearest_neighbor_same_label_fraction)


def test_generator_geometry_audit_records_duplicate_heavy_binary_case() -> None:
    row = audit_case(_case("binary_perfect_4c"))

    assert "duplicate_heavy" in row.geometry_flags
    assert row.max_duplicate_count >= 8
    assert row.mixed_label_duplicate_blocks == 0


def test_generator_geometry_audit_records_graph_assumption_warning() -> None:
    row = audit_case(_case("sbm_hard"))

    assert "graph_rows_as_feature_matrix" in row.geometry_flags
    assert row.simulation_model == "stochastic_block_model_graph"
    assert row.recommended_simulation_family == "graph_sbm_or_lfr_with_graph_native_distances"


def test_build_generator_geometry_audit_has_one_row_per_default_case() -> None:
    frame = build_generator_geometry_audit()

    assert len(frame) == len(get_default_test_cases())
    assert {
        "case_id",
        "simulation_model",
        "observation_model",
        "benchmark_intent",
        "resolved_feature_family",
        "nearest_neighbor_same_label_fraction",
        "geometry_flags",
        "recommended_simulation_family",
    }.issubset(frame.columns)


def test_write_generator_geometry_audit_outputs_csv_and_markdown(tmp_path: Path) -> None:
    frame = pd.DataFrame([audit_case(_case("binary_perfect_4c")).__dict__])

    output = tmp_path / "generator.csv"
    markdown = tmp_path / "generator.md"
    write_generator_geometry_audit(frame, output=output, markdown_output=markdown)

    assert output.exists()
    assert markdown.exists()
    assert "binary_perfect_4c" in markdown.read_text(encoding="utf-8")
