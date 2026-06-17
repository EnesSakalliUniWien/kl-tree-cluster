from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_signal_suppression_localizer import (
    build_case_summary,
    build_localization_rows,
    run_analysis,
)


def _pairwise_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "overlap_helpful",
                "data_role": "signal",
                "replicate": 0,
                "partition_ari_between_methods": 0.8,
                "delta_n_clusters_left_minus_right": 2.0,
                "delta_singletons_left_minus_right": 0.0,
                "delta_singleton_fraction_left_minus_right": 0.0,
                "delta_largest_cluster_fraction_left_minus_right": -0.2,
                "delta_effective_cluster_count_left_minus_right": 1.5,
                "delta_run_row_ari_left_minus_right": 0.05,
            },
            {
                "case_id": "overlap_harmful",
                "data_role": "signal",
                "replicate": 0,
                "partition_ari_between_methods": 0.7,
                "delta_n_clusters_left_minus_right": 3.0,
                "delta_singletons_left_minus_right": 0.0,
                "delta_singleton_fraction_left_minus_right": 0.0,
                "delta_largest_cluster_fraction_left_minus_right": -0.3,
                "delta_effective_cluster_count_left_minus_right": 2.0,
                "delta_run_row_ari_left_minus_right": -0.10,
            },
            {
                "case_id": "overlap_same",
                "data_role": "signal",
                "replicate": 0,
                "partition_ari_between_methods": 1.0,
                "delta_n_clusters_left_minus_right": 0.0,
                "delta_singletons_left_minus_right": 0.0,
                "delta_singleton_fraction_left_minus_right": 0.0,
                "delta_largest_cluster_fraction_left_minus_right": 0.0,
                "delta_effective_cluster_count_left_minus_right": 0.0,
                "delta_run_row_ari_left_minus_right": 0.0,
            },
        ]
    )


def _candidate_rows() -> pd.DataFrame:
    base = {
        "data_role": "signal",
        "left_method_id": "left",
        "right_method_id": "right",
        "left_traversal_decision": "split",
        "right_traversal_decision": "not_visited",
        "left_traversal_state": "split",
        "right_traversal_state": "boundary",
        "left_traversal_stop_reason": "accepted_split",
        "right_traversal_stop_reason": "explicit_guard_blocked",
        "left_neighborhood_evidence_family": "traversal_only",
        "right_neighborhood_evidence_family": "traversal_only",
        "left_explicit_guard_blocked": False,
        "right_explicit_guard_blocked": True,
        "left_depth": 2.0,
        "right_depth": 2.0,
        "left_n_descendant_leaves": 10.0,
        "right_n_descendant_leaves": 10.0,
        "left_child_parent_edge_open": True,
        "right_child_parent_edge_open": True,
        "left_sibling_open": True,
        "right_sibling_open": False,
        "left_sibling_p_value": 0.001,
        "right_sibling_p_value": 0.001,
        "left_sibling_projection_dimension": pd.NA,
        "right_sibling_projection_dimension": pd.NA,
        "left_topology_pass_through_candidate": False,
        "right_topology_pass_through_candidate": False,
        "left_balance_product": pd.NA,
        "right_balance_product": pd.NA,
        "left_outgoing_edge_norm_balance": pd.NA,
        "right_outgoing_edge_norm_balance": pd.NA,
        "left_descendant_accepted_split_count": 1,
        "right_descendant_accepted_split_count": 0,
        "left_descendant_pass_through_count": 0,
        "right_descendant_pass_through_count": 0,
        "left_descendant_guard_blocked_count": 0,
        "right_descendant_guard_blocked_count": 1,
        "left_descendant_stable_boundary_count": 8,
        "right_descendant_stable_boundary_count": 8,
        "traversal_decision_agrees": False,
        "traversal_state_agrees": False,
        "traversal_stop_reason_agrees": False,
        "candidate_contrast_status": "candidate_decision_diverges",
    }
    rows = []
    for case_id, reason in [
        ("overlap_helpful", "left_split|right_guard_blocked"),
        ("overlap_helpful", "left_pass_through"),
        ("overlap_harmful", "left_pass_through"),
        ("overlap_same", "left_split|right_split"),
    ]:
        row = base.copy()
        row.update(
            {
                "case_id": case_id,
                "replicate": 0,
                "node_id": f"{case_id}_{reason}",
                "candidate_reason": reason,
            }
        )
        if reason == "left_split|right_split":
            row["right_explicit_guard_blocked"] = False
            row["candidate_contrast_status"] = "candidate_decision_agrees"
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_localization_rows_classifies_signal_suppression() -> None:
    localized = build_localization_rows(_pairwise_rows(), _candidate_rows())

    statuses = dict(zip(localized["case_id"], localized["suppression_localization_status"]))
    assert statuses["overlap_helpful"] == "useful_movement_suppressed"
    assert statuses["overlap_harmful"] == "extra_movement_hurts_or_overfragments"
    assert statuses["overlap_same"] == "no_suppressed_signal_candidate"

    helpful = localized[localized["case_id"] == "overlap_helpful"].iloc[0]
    assert helpful["suppressed_candidate_count"] == 2
    assert helpful["split_suppressed_by_guard_count"] == 1
    assert helpful["pass_through_suppressed_count"] == 1
    assert "left_split|right_guard_blocked" in helpful["suppressed_node_examples"]


def test_case_summary_and_run_analysis_outputs(tmp_path: Path) -> None:
    pairwise = _pairwise_rows()
    candidates = _candidate_rows()
    summary = build_case_summary(build_localization_rows(pairwise, candidates))
    helpful_summary = summary[summary["case_id"] == "overlap_helpful"].iloc[0]
    assert helpful_summary["useful_suppressed_replicate_count"] == 1
    assert helpful_summary["suppressed_candidate_count"] == 2

    pairwise_path = tmp_path / "pairwise.csv"
    candidate_path = tmp_path / "candidates.csv"
    pairwise.to_csv(pairwise_path, index=False)
    candidates.to_csv(candidate_path, index=False)

    localized, case_summary, status_summary = run_analysis(
        pairwise_path=pairwise_path,
        candidate_rows_path=candidate_path,
        output_dir=tmp_path / "out",
    )
    assert len(localized) == 3
    assert len(case_summary) == 3
    assert not status_summary.empty
    assert (tmp_path / "out" / "overlap_signal_suppression_localization_rows.csv").exists()
    assert (tmp_path / "out" / "overlap_signal_suppression_case_summary.csv").exists()
    assert (tmp_path / "out" / "overlap_signal_suppression_status_summary.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
