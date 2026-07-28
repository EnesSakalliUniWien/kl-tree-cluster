from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_spectral_flow import (
    build_multiplicity_spectral_panels,
    build_spectral_flow_panels,
    compare_row_basis_subspaces,
    match_spectral_mode_blocks,
    normalized_characteristic_polynomial,
    spectral_barrier,
    spectral_mode_blocks,
    spectral_mode_distance,
    summarize_spectral_flow_edges,
    summarize_spectral_flow_separation,
)


def test_subspace_comparison_is_sign_invariant_for_mp_eigenvector() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    right = np.asarray([[-1.0, 0.0], [0.0, 1.0]])

    comparison = compare_row_basis_subspaces(left, right, dimension=1)

    assert comparison.common_dimension == 1
    assert comparison.largest_cosine == pytest.approx(1.0)
    assert comparison.chordal_distance == pytest.approx(0.0)


def test_subspace_comparison_detects_top_mp_eigenvector_rotation() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    right = np.asarray([[0.0, 1.0], [1.0, 0.0]])

    comparison = compare_row_basis_subspaces(left, right, dimension=1)

    assert comparison.common_dimension == 1
    assert comparison.largest_cosine == pytest.approx(0.0)
    assert comparison.chordal_distance == pytest.approx(1.0)


def test_spectral_barrier_is_lower_for_coherent_modes() -> None:
    coherent = spectral_barrier(
        subspace_chordal_distance=0.05,
        log_eigenvalue_delta_value=0.10,
        mp_dimension_gap=0.0,
        parent_gap_ratio=5.0,
        child_gap_ratio=4.0,
    )
    rotated = spectral_barrier(
        subspace_chordal_distance=0.90,
        log_eigenvalue_delta_value=0.80,
        mp_dimension_gap=0.5,
        parent_gap_ratio=1.1,
        child_gap_ratio=1.0,
    )

    assert coherent < rotated


def test_build_spectral_flow_panels_separates_mp_supported_and_floor_only() -> None:
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "left"), ("root", "right")])
    annotations = pd.DataFrame(
        {
            "Sibling_Divergence_P_Value_Corrected": [0.50, 0.01, 0.20],
            "Sibling_BH_Different": [False, True, False],
        },
        index=["root", "left", "right"],
    )
    spectral_context = SimpleNamespace(
        principal_component_projections_by_node={
            "root": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "left": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "right": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        },
        principal_component_eigenvalues_by_node={
            "root": np.asarray([4.0, 1.0]),
            "left": np.asarray([3.8, 1.0]),
            "right": np.asarray([3.5, 1.0]),
        },
        test_projection_dimensions_by_node={"root": 2, "left": 2, "right": 2},
        raw_mp_signal_counts_by_node={"root": 1, "left": 1, "right": 0},
        mp_threshold_rows_by_node={"root": 20, "left": 10, "right": 10},
        effective_independent_rows_by_node={"root": 20, "left": 10, "right": 10},
    )

    nodes, edges = build_spectral_flow_panels(
        case_id="case",
        data_role="signal",
        method_id="method",
        replicate=0,
        tree=tree,
        annotations=annotations,
        spectral_context=spectral_context,
    )
    by_child = {row["child_id"]: row for _, row in edges.iterrows()}

    assert not nodes.empty
    assert by_child["left"]["flow_status"] == "mp_supported_subspace_compared"
    assert by_child["left"]["mp_subspace_chordal_distance"] == pytest.approx(0.0)
    assert by_child["right"]["flow_status"] == "floor_only_no_mp_certified_mode"
    assert by_child["right"]["mp_pair_supported"] is False


def test_summaries_report_signal_vs_selected_null_auc() -> None:
    edges = pd.DataFrame.from_records(
        [
            {
                "schema_version": "x",
                "study_role": "x",
                "data_role": "signal",
                "method_id": "method",
                "flow_status": "mp_supported_subspace_compared",
                "mp_pair_supported": True,
                "mp_subspace_chordal_distance": 0.1,
                "mp_log_eigenvalue_delta": 0.1,
                "spectral_barrier": 0.2,
                "spectral_flow_affinity": 0.8,
                "parent_top_eigenvalue_over_mp": 2.0,
                "child_top_eigenvalue_over_mp": 1.8,
            },
            {
                "schema_version": "x",
                "study_role": "x",
                "data_role": "selected_null",
                "method_id": "method",
                "flow_status": "mp_supported_subspace_compared",
                "mp_pair_supported": True,
                "mp_subspace_chordal_distance": 0.9,
                "mp_log_eigenvalue_delta": 0.7,
                "spectral_barrier": 1.5,
                "spectral_flow_affinity": 0.2,
                "parent_top_eigenvalue_over_mp": 1.0,
                "child_top_eigenvalue_over_mp": 0.9,
            },
        ]
    )

    summary = summarize_spectral_flow_edges(edges)
    separation = summarize_spectral_flow_separation(edges)
    affinity = separation.loc[separation["metric"].eq("spectral_flow_affinity")].iloc[0]

    assert not summary.empty
    assert affinity["auc_signal_vs_selected_null"] == pytest.approx(1.0)


def test_repeated_mp_eigenvalues_form_one_projector_block() -> None:
    projection = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    blocks = spectral_mode_blocks(
        np.asarray([5.0, 4.98, 2.0]),
        projection,
        raw_mp_signal_count=3,
        eigenvalue_block_log_tolerance=0.01,
    )

    assert [block.multiplicity for block in blocks] == [2, 1]
    assert blocks[0].projector_rank == 2


def test_block_distance_is_invariant_to_rotation_inside_repeated_eigenspace() -> None:
    left = spectral_mode_blocks(
        np.asarray([5.0, 5.0]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        raw_mp_signal_count=2,
        eigenvalue_block_log_tolerance=0.01,
    )[0]
    right = spectral_mode_blocks(
        np.asarray([5.0, 5.0]),
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        raw_mp_signal_count=2,
        eigenvalue_block_log_tolerance=0.01,
    )[0]

    distance = spectral_mode_distance(left, right)

    assert distance.projector_chordal_distance == pytest.approx(0.0)
    assert distance.multiplicity_distance == pytest.approx(0.0)
    assert distance.polynomial_distance == pytest.approx(0.0)


def test_mode_matching_penalizes_multiplicity_and_polynomial_changes() -> None:
    left = spectral_mode_blocks(
        np.asarray([5.0, 5.0]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        raw_mp_signal_count=2,
        eigenvalue_block_log_tolerance=0.01,
    )
    right = spectral_mode_blocks(
        np.asarray([5.0, 3.0]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        raw_mp_signal_count=2,
        eigenvalue_block_log_tolerance=0.01,
    )

    matching = match_spectral_mode_blocks(left, right)

    assert matching.matched_block_count == 1
    assert matching.unmatched_block_count == 1
    assert matching.mode_transport_cost > 0.0


def test_normalized_characteristic_polynomial_is_scale_invariant() -> None:
    first = normalized_characteristic_polynomial(np.asarray([2.0, 2.0]))
    second = normalized_characteristic_polynomial(np.asarray([8.0, 8.0]))

    assert np.allclose(first, second)


def test_build_multiplicity_spectral_panels_emits_mode_transport_rows() -> None:
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "left"), ("root", "right")])
    annotations = pd.DataFrame(
        {
            "Sibling_Divergence_P_Value_Corrected": [0.50, 0.01, 0.20],
            "Sibling_BH_Different": [False, True, False],
        },
        index=["root", "left", "right"],
    )
    spectral_context = SimpleNamespace(
        principal_component_projections_by_node={
            "root": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "left": np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
            "right": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        },
        principal_component_eigenvalues_by_node={
            "root": np.asarray([5.0, 5.0]),
            "left": np.asarray([5.1, 5.1]),
            "right": np.asarray([5.0, 2.0]),
        },
        raw_mp_signal_counts_by_node={"root": 2, "left": 2, "right": 0},
    )

    blocks, mode_edges = build_multiplicity_spectral_panels(
        case_id="case",
        data_role="signal",
        method_id="method",
        replicate=0,
        tree=tree,
        annotations=annotations,
        spectral_context=spectral_context,
        eigenvalue_block_log_tolerance=0.01,
    )
    by_child = {row["child_id"]: row for _, row in mode_edges.iterrows()}

    assert not blocks.empty
    assert by_child["left"]["mode_flow_status"] == "mp_blocks_compared"
    assert by_child["left"]["mean_block_projector_chordal_distance"] == pytest.approx(0.0)
    assert by_child["right"]["mode_flow_status"] == "mp_block_missing_on_one_side"
