from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration import (
    root_tree_geometry_hard_negative_replay_panel as panel,
)
from tree_break_selection.tree.poset_tree import PosetTree


def test_tree_geometry_parser_requires_builder_rooting_distance_linkage() -> None:
    geometries = panel.parse_tree_geometries(
        "linkage:linkage_root:hamming:average,"
        "neighbor_joining:mad:jaccard:average"
    )

    assert geometries[0].tree_builder == "linkage"
    assert geometries[0].tree_distance_metric == "hamming"
    assert geometries[1].tree_builder == "neighbor_joining"
    assert geometries[1].tree_rooting == "mad"


def test_hard_negative_blocks_unstable_root_even_when_open() -> None:
    status, action, supported = panel.classify_hard_negative_row(
        {
            "run_status": "ok",
            "root_sibling_open": True,
            "root_stability_guard_blocked": True,
            "root_stability_subsample_mean_ari": 0.001,
            "root_selective_permutation_p_value": 0.001,
        }
    )

    assert status == "hard_negative_control_blocked_root_unstable"
    assert action == "keep_fail_closed_root_unstable"
    assert supported is False


def test_hard_negative_leaks_only_when_validity_supported_and_open() -> None:
    status, action, supported = panel.classify_hard_negative_row(
        {
            "run_status": "ok",
            "root_sibling_open": True,
            "root_stability_guard_blocked": False,
            "root_stability_subsample_mean_ari": 0.8,
            "root_selective_permutation_p_value": 0.005,
            "root_selective_permutation_guard_blocked": False,
            "root_selective_permutation_guard_would_block": False,
        }
    )

    assert status == "hard_negative_control_leaked_root_validity_supported"
    assert action == "block_promotion_and_inspect_geometry_rescue"
    assert supported is True


def _toy_tree_for_rootless_cut() -> PosetTree:
    tree = PosetTree()
    tree.add_edges_from(
        [
            ("root", "x"),
            ("x", "a"),
            ("x", "b"),
            ("a", "L0"),
            ("a", "L1"),
            ("b", "L2"),
            ("b", "L3"),
        ]
    )
    tree.graph["root"] = "root"
    for node in tree.nodes:
        tree.nodes[node]["is_leaf"] = node.startswith("L")
        if tree.nodes[node]["is_leaf"]:
            tree.nodes[node]["label"] = node
    return tree


def test_unrooted_edge_cut_can_find_truth_aligned_non_root_split() -> None:
    summary = panel._unrooted_edge_cut_summary(
        tree=_toy_tree_for_rootless_cut(),
        sample_ids=("L0", "L1", "L2", "L3"),
        truth_labels=pd.Series([0, 0, 1, 1]).to_numpy(),
    )

    assert summary["best_unrooted_edge_cut_truth_ari"] == pytest.approx(1.0)
    assert summary["best_unrooted_edge_cut_is_root_edge"] is False
    assert summary["unrooted_geometry_status"] == (
        "unrooted_edge_cut_truth_aligned_bipartition_available"
    )
    assert summary["rootless_method_action"] == (
        "inspect_rootless_edge_tail_law_before_root_rescue"
    )


def test_hard_negative_summary_fails_when_any_geometry_leaks() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": "overlap_extreme_4c",
                "data_role": "signal",
                "run_status": "ok",
                "root_validity_supported": False,
                "hard_negative_control_status": (
                    "hard_negative_control_blocked_root_unstable"
                ),
                "root_stability_subsample_mean_ari": 0.001,
                "root_selective_permutation_p_value": 0.23,
            },
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": "overlap_extreme_4c",
                "data_role": "signal",
                "run_status": "ok",
                "root_validity_supported": True,
                "hard_negative_control_status": (
                    "hard_negative_control_leaked_root_validity_supported"
                ),
                "root_stability_subsample_mean_ari": 0.8,
                "root_selective_permutation_p_value": 0.001,
            },
        ]
    )

    summary = panel.summarize_root_tree_geometry_hard_negative_replay_rows(rows)

    assert summary.iloc[0]["summary_status"] == "hard_negative_control_failed"
    assert summary.iloc[0]["hard_negative_leak_count"] == 1


def test_hard_negative_summary_counts_rootless_truth_aligned_geometries() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": "overlap_extreme_4c",
                "data_role": "signal",
                "run_status": "ok",
                "root_validity_supported": False,
                "hard_negative_control_status": (
                    "hard_negative_control_blocked_root_unstable"
                ),
                "root_stability_subsample_mean_ari": 0.001,
                "root_selective_permutation_p_value": 0.23,
                "unrooted_geometry_status": (
                    "unrooted_edge_cut_truth_aligned_bipartition_available"
                ),
                "best_unrooted_edge_cut_truth_ari": 0.72,
            },
            {
                "schema_version": panel.SCHEMA_VERSION,
                "study_role": panel.STUDY_ROLE,
                "case_id": "overlap_extreme_4c",
                "data_role": "signal",
                "run_status": "ok",
                "root_validity_supported": False,
                "hard_negative_control_status": (
                    "hard_negative_control_blocked_root_unstable"
                ),
                "root_stability_subsample_mean_ari": 0.002,
                "root_selective_permutation_p_value": 0.2,
                "unrooted_geometry_status": (
                    "unrooted_edge_cut_no_truth_aligned_bipartition"
                ),
                "best_unrooted_edge_cut_truth_ari": 0.01,
            },
        ]
    )

    summary = panel.summarize_root_tree_geometry_hard_negative_replay_rows(rows)

    assert summary.iloc[0]["rootless_truth_aligned_geometry_count"] == 1
    assert summary.iloc[0]["max_best_unrooted_edge_cut_truth_ari"] == pytest.approx(
        0.72
    )
