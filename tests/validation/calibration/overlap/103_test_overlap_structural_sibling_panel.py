from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.overlap.overlap_structural_sibling_panel import (
    OverlapStructuralSiblingPanelConfig,
    classify_structural_change_mode,
    classify_structural_sibling_row,
    cosine_similarity,
    jaccard_index,
    pairwise_binary_jaccard_similarity,
    run_overlap_structural_sibling_panel,
    top_coordinate_set,
)


def test_top_coordinate_and_similarity_helpers_are_stable() -> None:
    assert top_coordinate_set([0.0, -4.0, 2.0, 0.0], top_k=2) == frozenset({1, 2})
    assert jaccard_index(frozenset({1, 2}), frozenset({2, 3})) == pytest.approx(1 / 3)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_pairwise_binary_jaccard_similarity_detects_homogeneity() -> None:
    homogeneous = pairwise_binary_jaccard_similarity([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    mixed = pairwise_binary_jaccard_similarity([[1, 0, 1], [0, 1, 0], [1, 1, 0]])

    assert homogeneous == pytest.approx(1.0)
    assert mixed < homogeneous


def test_structural_status_distinguishes_supported_from_mismatch() -> None:
    assert (
        classify_structural_sibling_row(
            homogeneity_gain_min=0.10,
            heterogeneity_gain_max=-0.10,
            delta_homogeneity_jaccard_topk=0.50,
            max_delta_edge_jaccard_topk=0.50,
            max_edge_homogeneity_jaccard_topk=0.50,
            heterogeneity_subspace_consensus_jaccard_topk=0.0,
            max_sibling_edge_alignment=0.90,
            sibling_contrast_norm=1.0,
        )
        == "structural_same_subspace_supported"
    )
    assert (
        classify_structural_sibling_row(
            homogeneity_gain_min=0.00,
            heterogeneity_gain_max=0.00,
            delta_homogeneity_jaccard_topk=0.90,
            max_delta_edge_jaccard_topk=0.90,
            max_edge_homogeneity_jaccard_topk=0.90,
            heterogeneity_subspace_consensus_jaccard_topk=0.90,
            max_sibling_edge_alignment=0.90,
            sibling_contrast_norm=1.0,
        )
        == "weak_homogeneity_gain"
    )
    assert (
        classify_structural_sibling_row(
            homogeneity_gain_min=0.10,
            heterogeneity_gain_max=-0.10,
            delta_homogeneity_jaccard_topk=0.0,
            max_delta_edge_jaccard_topk=0.10,
            max_edge_homogeneity_jaccard_topk=0.90,
            heterogeneity_subspace_consensus_jaccard_topk=0.0,
            max_sibling_edge_alignment=0.20,
            sibling_contrast_norm=1.0,
        )
        == "barycentric_focus_mismatch"
    )
    assert (
        classify_structural_sibling_row(
            homogeneity_gain_min=0.10,
            heterogeneity_gain_max=-0.10,
            delta_homogeneity_jaccard_topk=0.70,
            max_delta_edge_jaccard_topk=0.70,
            max_edge_homogeneity_jaccard_topk=0.10,
            heterogeneity_subspace_consensus_jaccard_topk=0.0,
            max_sibling_edge_alignment=0.90,
            sibling_contrast_norm=1.0,
        )
        == "structural_homogeneity_subspace_mismatch"
    )
    assert (
        classify_structural_sibling_row(
            homogeneity_gain_min=-0.03,
            heterogeneity_gain_max=0.05,
            delta_homogeneity_jaccard_topk=0.10,
            max_delta_edge_jaccard_topk=0.80,
            max_edge_homogeneity_jaccard_topk=0.10,
            heterogeneity_subspace_consensus_jaccard_topk=0.70,
            max_sibling_edge_alignment=0.90,
            sibling_contrast_norm=1.0,
        )
        == "same_subspace_heterogeneity_increase"
    )
    assert (
        classify_structural_sibling_row(
            homogeneity_gain_min=-0.03,
            heterogeneity_gain_max=0.05,
            delta_homogeneity_jaccard_topk=0.10,
            max_delta_edge_jaccard_topk=0.80,
            max_edge_homogeneity_jaccard_topk=0.10,
            heterogeneity_subspace_consensus_jaccard_topk=0.10,
            max_sibling_edge_alignment=0.90,
            sibling_contrast_norm=1.0,
        )
        == "unrelated_subspace_heterogeneity_signal"
    )


def test_structural_change_mode_separates_heterogeneous_subspace_signal() -> None:
    assert (
        classify_structural_change_mode(
            homogeneity_gain_min=0.04,
            heterogeneity_gain_max=-0.04,
            subspace_consensus_jaccard_topk=0.60,
            heterogeneity_subspace_consensus_jaccard_topk=0.0,
            sibling_contrast_norm=1.0,
        )
        == "homogeneous_same_subspace"
    )
    assert (
        classify_structural_change_mode(
            homogeneity_gain_min=-0.04,
            heterogeneity_gain_max=0.04,
            subspace_consensus_jaccard_topk=0.10,
            heterogeneity_subspace_consensus_jaccard_topk=0.60,
            sibling_contrast_norm=1.0,
        )
        == "heterogeneous_same_subspace"
    )
    assert (
        classify_structural_change_mode(
            homogeneity_gain_min=-0.04,
            heterogeneity_gain_max=0.04,
            subspace_consensus_jaccard_topk=0.10,
            heterogeneity_subspace_consensus_jaccard_topk=0.10,
            sibling_contrast_norm=1.0,
        )
        == "unrelated_subspace_signal"
    )


def test_run_overlap_structural_sibling_panel_writes_outputs(tmp_path) -> None:
    outputs = run_overlap_structural_sibling_panel(
        OverlapStructuralSiblingPanelConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
            top_k=4,
            max_pairwise_samples=50,
        )
    )

    for path in outputs.values():
        assert path.exists()

    rows = pd.read_csv(outputs["rows"])
    assert set(rows.columns) >= {
        "case_id",
        "structural_sibling_status",
        "delta_homogeneity_jaccard_topk",
        "max_edge_homogeneity_jaccard_topk",
        "subspace_consensus_jaccard_topk",
        "heterogeneity_subspace_consensus_jaccard_topk",
        "homogeneity_gain_min",
        "heterogeneity_gain_max",
        "structural_change_mode",
    }
