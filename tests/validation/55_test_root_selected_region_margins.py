from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root_selected_region_margins import (
    _root_spectral_summary,
    annotate_root_child_construction_roles,
    euclidean_average_linkage_inequality_geometry,
    fixed_projection_edge_conditioned_sibling_tail,
    projected_wald_edge_opening_geometry,
    replay_average_linkage_margins,
    root_edge_sibling_wald_relationship,
    summarize_root_child_margin_geometry,
    summarize_root_selected_region_relationships,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _four_leaf_condensed_distances() -> np.ndarray:
    square = np.asarray(
        [
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 5.5, 6.5],
            [5.0, 5.5, 0.0, 1.2],
            [6.0, 6.5, 1.2, 0.0],
        ],
        dtype=float,
    )
    return squareform(square)


def test_replay_average_linkage_margins_exposes_root_child_inequalities() -> None:
    distances = _four_leaf_condensed_distances()
    linkage_matrix = linkage(distances, method="average")

    replay = replay_average_linkage_margins(linkage_matrix, distances)
    margins = annotate_root_child_construction_roles(
        replay.merge_margins,
        root_children=replay.root_children,
    )

    assert list(margins["root_region_role"]) == [
        "root_child_construction",
        "root_child_construction",
        "root_final_merge",
    ]
    assert list(margins["merge_margin_status"]) == [
        "defined",
        "defined",
        "final_root_merge_no_competitor",
    ]
    assert margins.loc[0, "merge_margin_to_nearest_competitor"] == pytest.approx(0.2)
    assert margins.loc[0, "selected_tie_rank_lexicographic"] == 1
    assert margins.loc[0, "selected_tie_rank_fraction"] == pytest.approx(1.0)
    assert margins.loc[0, "selected_tie_break_status"] == "unique_minimum"

    summary = summarize_root_child_margin_geometry(margins)
    assert summary["root_child_margin_status"] == "defined"
    assert summary["root_child_construction_merge_count"] == 2
    assert summary["root_child_defined_margin_count"] == 2
    assert summary["root_child_min_merge_margin"] == pytest.approx(0.2)
    assert summary["root_selected_region_law_status"] == "nonsmooth_geometry_not_defined"


def test_euclidean_average_linkage_signed_distance_uses_inequality_gradient() -> None:
    leaf_matrix = np.asarray([[0.0], [1.0], [4.0]], dtype=float)

    geometry = euclidean_average_linkage_inequality_geometry(
        leaf_matrix,
        selected_left_members=(0,),
        selected_right_members=(1,),
        competitor_left_members=(1,),
        competitor_right_members=(2,),
        margin=2.0,
        null_feature_covariance=np.asarray([[4.0]], dtype=float),
    )

    assert geometry["selected_region_geometry_status"] == (
        "smooth_first_order_geometry_defined"
    )
    assert geometry["merge_inequality_gradient_norm"] == pytest.approx(np.sqrt(6.0))
    assert geometry["merge_first_order_signed_distance"] == pytest.approx(
        2.0 / np.sqrt(6.0)
    )
    assert geometry["merge_null_quadratic_gradient_scale"] == pytest.approx(
        np.sqrt(24.0)
    )
    assert geometry["merge_null_whitened_first_order_signed_distance"] == pytest.approx(
        2.0 / np.sqrt(24.0)
    )
    assert geometry["null_whitened_geometry_status"] == (
        "continuous_iid_empirical_gaussian_root_null"
    )


def test_replay_average_linkage_margins_rejects_nonminimal_linkage() -> None:
    distances = _four_leaf_condensed_distances()
    invalid_linkage = np.asarray(
        [
            [0, 2, 5.0, 2],
            [1, 3, 6.5, 2],
            [4, 5, 3.425, 4],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="not minimal"):
        replay_average_linkage_margins(invalid_linkage, distances)


def test_root_child_margin_summary_marks_leaf_children_without_fallback() -> None:
    distances = np.asarray([1.0], dtype=float)
    linkage_matrix = np.asarray([[0.0, 1.0, 1.0, 2.0]], dtype=float)

    replay = replay_average_linkage_margins(linkage_matrix, distances)
    margins = annotate_root_child_construction_roles(
        replay.merge_margins,
        root_children=replay.root_children,
    )
    summary = summarize_root_child_margin_geometry(margins)

    assert summary["root_child_margin_status"] == "root_children_are_leaves"
    assert summary["root_child_construction_merge_count"] == 0
    assert summary["root_child_defined_margin_count"] == 0


def test_replay_average_linkage_margins_adds_smooth_geometry_for_euclidean_data() -> None:
    leaf_matrix = np.asarray([[0.0], [1.0], [4.0], [8.0]], dtype=float)
    distances = squareform(np.abs(leaf_matrix - leaf_matrix.T))
    linkage_matrix = linkage(distances, method="average")

    replay = replay_average_linkage_margins(
        linkage_matrix,
        distances,
        leaf_matrix=leaf_matrix,
        geometry_metric="euclidean",
        null_feature_covariance=np.asarray([[2.0]], dtype=float),
    )
    margins = annotate_root_child_construction_roles(
        replay.merge_margins,
        root_children=replay.root_children,
    )
    construction = margins[margins["root_region_role"].eq("root_child_construction")]

    assert set(construction["selected_region_geometry_status"]) == {
        "smooth_first_order_geometry_defined"
    }
    assert construction["merge_first_order_signed_distance"].notna().all()
    assert construction["merge_null_whitened_first_order_signed_distance"].notna().all()
    assert summarize_root_child_margin_geometry(margins)[
        "root_selected_region_law_status"
    ] == "null_whitened_first_order_signed_distance_defined"


def test_replay_average_linkage_records_selected_rank_inside_tie_cell() -> None:
    square = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    distances = squareform(square)
    linkage_matrix = np.asarray(
        [
            [1.0, 2.0, 1.0, 2.0],
            [0.0, 3.0, 1.0, 3.0],
        ],
        dtype=float,
    )

    replay = replay_average_linkage_margins(linkage_matrix, distances)
    first = replay.merge_margins.iloc[0]

    assert first["tied_minimum_pair_count"] == 3
    assert first["selected_pair_tied_for_minimum"]
    assert first["selected_tie_rank_lexicographic"] == 3
    assert first["selected_tie_rank_fraction"] == pytest.approx(1.0)
    assert first["selected_tie_log_rank"] == pytest.approx(np.log(3.0))
    assert first["selected_tie_break_status"] == "tied_minimum_lexicographic_rank"


def test_projected_wald_edge_opening_geometry_uses_chi_square_radial_boundary() -> None:
    threshold = float(stats.chi2.isf(0.05, df=2.0))
    statistic = float((np.sqrt(threshold) + 1.5) ** 2)

    geometry = projected_wald_edge_opening_geometry(
        statistic=statistic,
        degrees_of_freedom=2.0,
        alpha=0.05,
    )

    assert geometry["edge_opening_boundary_status"] == (
        "fixed_subspace_chi_square_radial_boundary"
    )
    assert geometry["edge_chi_square_threshold"] == pytest.approx(threshold)
    assert geometry["edge_statistic_margin"] == pytest.approx(statistic - threshold)
    assert geometry["edge_statistic_over_threshold"] == pytest.approx(
        statistic / threshold
    )
    assert geometry["edge_radial_distance"] == pytest.approx(1.5)


def test_fixed_projection_edge_conditioned_tail_matches_truncated_chi_square() -> None:
    threshold = float(stats.chi2.isf(0.001, df=2))
    statistic = threshold + 2.0

    tail = fixed_projection_edge_conditioned_sibling_tail(
        sibling_statistic=statistic,
        sibling_degrees_of_freedom=2,
        edge_degrees_of_freedom=2,
        edge_alpha=0.001,
    )

    assert tail["edge_conditioned_sibling_law_status"] == (
        "evaluated_equal_projection_truncated_chi_square"
    )
    assert tail["edge_conditioning_threshold"] == pytest.approx(threshold)
    assert tail["edge_conditioning_probability"] == pytest.approx(0.001)
    assert tail["edge_conditioned_sibling_p_value"] == pytest.approx(
        stats.chi2.sf(statistic, df=2) / stats.chi2.sf(threshold, df=2)
    )


def test_fixed_projection_edge_conditioned_tail_integrates_extra_edge_dimensions() -> None:
    at_zero = fixed_projection_edge_conditioned_sibling_tail(
        sibling_statistic=0.0,
        sibling_degrees_of_freedom=2,
        edge_degrees_of_freedom=4,
        edge_alpha=0.001,
    )
    at_threshold = fixed_projection_edge_conditioned_sibling_tail(
        sibling_statistic=float(stats.chi2.isf(0.001, df=4)),
        sibling_degrees_of_freedom=2,
        edge_degrees_of_freedom=4,
        edge_alpha=0.001,
    )

    assert at_zero["edge_conditioned_sibling_law_status"] == (
        "evaluated_edge_projection_superset_truncated_chi_square"
    )
    assert at_zero["edge_conditioned_sibling_p_value"] == pytest.approx(1.0)
    assert 0.0 < at_threshold["edge_conditioned_sibling_p_value"] < 1.0


def test_root_edge_sibling_wald_relationship_verifies_barycentric_z_identity() -> None:
    tree = nx.DiGraph()
    tree.add_edge("root", "L")
    tree.add_edge("root", "R")
    left = np.asarray([0.2, 0.7], dtype=float)
    right = np.asarray([0.8, 0.4], dtype=float)
    left_n = 3
    right_n = 5
    parent = (left_n * left + right_n * right) / float(left_n + right_n)
    tree.nodes["root"]["distribution"] = parent
    tree.nodes["root"]["leaf_count"] = left_n + right_n
    tree.nodes["L"]["distribution"] = left
    tree.nodes["L"]["leaf_count"] = left_n
    tree.nodes["R"]["distribution"] = right
    tree.nodes["R"]["leaf_count"] = right_n
    spectral_context = SpectralContext(
        test_projection_dimensions_by_node={"root": 2},
        raw_mp_signal_counts_by_node={"root": 2},
        effective_independent_rows_by_node={"root": 8},
        mp_threshold_rows_by_node={"root": 8},
        principal_component_projections_by_node={
            "root": np.eye(2, dtype=float),
        },
        principal_component_eigenvalues_by_node={
            "root": np.ones(2, dtype=float),
        },
    )

    relationship = root_edge_sibling_wald_relationship(
        tree,
        parent="root",
        left_child="L",
        right_child="R",
        sibling_projection_dimension=1,
        spectral_context=spectral_context,
    )

    assert relationship["root_edge_sibling_relationship_status"] == (
        "barycentric_z_identity_verified"
    )
    assert relationship["root_edge_sibling_z_max_relative_residual"] == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert relationship["root_edge_parent_projection_dimension"] == 2
    assert relationship["root_sibling_projection_dimension_for_relationship"] == 1
    assert relationship["root_edge_equivalent_parent_projection_statistic"] >= (
        relationship["root_sibling_recomputed_statistic_from_parent_projection"]
    )
    assert relationship["root_edge_extra_parent_projection_energy"] >= 0.0


def test_root_spectral_summary_captures_full_h_u_spectrum() -> None:
    spectral_context = SpectralContext(
        test_projection_dimensions_by_node={"root": 2},
        raw_mp_signal_counts_by_node={"root": 1},
        effective_independent_rows_by_node={"root": 8},
        mp_threshold_rows_by_node={"root": 8},
        principal_component_projections_by_node={"root": np.eye(2, dtype=float)},
        principal_component_eigenvalues_by_node={
            "root": np.asarray([5.0, 2.0], dtype=float),
        },
        full_component_eigenvalues_by_node={
            "root": np.asarray([5.0, 2.0, 1.0, 0.5], dtype=float),
        },
        active_feature_counts_by_node={"root": 4},
    )

    summary = _root_spectral_summary(
        root="root",
        sibling_projection_dimension=2,
        spectral_context=spectral_context,
    )

    assert summary["root_active_feature_count"] == 4
    assert summary["root_full_eigenvalue_count"] == 4
    assert json.loads(summary["root_full_component_eigenvalues_json"]) == [
        5.0,
        2.0,
        1.0,
        0.5,
    ]
    assert json.loads(summary["root_projected_eigenvalues_json"]) == [5.0, 2.0]
    assert summary["root_mp_upper_bound"] == pytest.approx(
        (1.0 + np.sqrt(4.0 / 8.0)) ** 2
    )
    assert summary["root_selected_eigenvalue_over_mp_upper_bound"] == pytest.approx(
        2.0 / ((1.0 + np.sqrt(4.0 / 8.0)) ** 2)
    )


def test_root_spectral_summary_requires_full_h_u_spectrum() -> None:
    spectral_context = SpectralContext(
        test_projection_dimensions_by_node={"root": 2},
        raw_mp_signal_counts_by_node={"root": 1},
        effective_independent_rows_by_node={"root": 8},
        mp_threshold_rows_by_node={"root": 8},
        principal_component_projections_by_node={"root": np.eye(2, dtype=float)},
        principal_component_eigenvalues_by_node={
            "root": np.asarray([5.0, 2.0], dtype=float),
        },
    )

    with pytest.raises(KeyError, match="Missing root spectral context"):
        _root_spectral_summary(
            root="root",
            sibling_projection_dimension=2,
            spectral_context=spectral_context,
        )


def test_root_selected_region_relationships_report_statuses() -> None:
    root_table = pd.DataFrame(
        {
            "root_sibling_selected_ratio": [1.0, 3.0, 9.0],
            "root_child_min_merge_margin": [0.1, 0.2, 0.3],
            "root_child_min_first_order_signed_distance": [0.1, 0.1, 0.1],
            "root_child_min_null_whitened_first_order_signed_distance": [
                np.nan,
                np.nan,
                np.nan,
            ],
            "root_edge_path_radial_distance": [0.0, 1.0, 2.0],
            "root_edge_path_statistic_margin": [0.0, 2.0, 6.0],
            "root_edge_path_bh_action": [0.0, 1.0, 2.0],
            "root_edge_extra_parent_projection_energy": [0.5, 1.0, 1.5],
            "root_edge_to_sibling_projection_energy_ratio": [1.2, 1.5, 2.0],
            "root_selected_eigenvalue_over_mp_upper_bound": [2.0, 1.0, 0.5],
        }
    )

    relationships = summarize_root_selected_region_relationships(root_table)
    status_by_covariate = dict(
        zip(
            relationships["covariate"],
            relationships["relationship_status"],
            strict=False,
        )
    )

    assert status_by_covariate["root_child_min_merge_margin"] == "evaluated"
    assert status_by_covariate["root_child_min_first_order_signed_distance"] == (
        "constant_covariate"
    )
    assert status_by_covariate[
        "root_child_min_null_whitened_first_order_signed_distance"
    ] == "insufficient_valid_pairs"
    assert status_by_covariate["root_edge_path_radial_distance"] == "evaluated"
    assert status_by_covariate["root_edge_path_statistic_margin"] == "evaluated"
    assert status_by_covariate["root_edge_path_bh_action"] == "evaluated"
    assert status_by_covariate["root_edge_extra_parent_projection_energy"] == "evaluated"
    assert status_by_covariate["root_edge_to_sibling_projection_energy_ratio"] == (
        "evaluated"
    )
