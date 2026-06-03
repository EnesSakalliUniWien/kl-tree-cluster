from __future__ import annotations

import numpy as np
import pytest
from benchmarks.diagnostics.calibration.root_selected_region_margins import (
    annotate_root_child_construction_roles,
    euclidean_average_linkage_inequality_geometry,
    replay_average_linkage_margins,
    summarize_root_child_margin_geometry,
)
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
    )

    assert geometry["selected_region_geometry_status"] == (
        "smooth_first_order_geometry_defined"
    )
    assert geometry["merge_inequality_gradient_norm"] == pytest.approx(np.sqrt(6.0))
    assert geometry["merge_first_order_signed_distance"] == pytest.approx(
        2.0 / np.sqrt(6.0)
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
    assert summarize_root_child_margin_geometry(margins)[
        "root_selected_region_law_status"
    ] == "smooth_first_order_signed_distance_defined"
