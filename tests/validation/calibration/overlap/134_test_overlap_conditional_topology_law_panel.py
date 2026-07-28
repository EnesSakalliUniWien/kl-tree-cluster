from __future__ import annotations

import json
import math

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_conditional_topology_law_panel import (
    OverlapConditionalTopologyLawPanelConfig,
    build_cached_tree_distances,
    build_conditional_topology_analytical_cases,
    build_conditional_topology_law_rows,
    infer_directed_incidence,
    run_overlap_conditional_topology_law_panel,
    summarize_conditional_topology_law_rows,
)


def _row(
    *,
    role: str = "truth_recovery",
    depth: int = 1,
    decision_class: str = "accepted_internal_split",
    traversal_decision: str = "split",
    incoming: float = 0.45,
    outgoing: float = 0.48,
    edge_norm: float = 0.95,
    fragment: float = 0.60,
    selected: float = 8.0,
    context: float = -0.01,
    neighborhood_scale: float | None = None,
    n_left: float = 50.0,
    n_right: float = 50.0,
    node_id: str = "N1",
    parent_id: str = "root",
    replicate: int = 0,
    topology_support_role: str | None = None,
    topology_signal_role: str | None = None,
    balance_product: float | None = None,
    branch_length_to_parent: float | None = None,
) -> dict[str, object]:
    record = {
        "case_id": "case",
        "data_role": "signal",
        "replicate": replicate,
        "node_id": node_id,
        "parent_id": parent_id,
        "guard_truth_role": role,
        "depth": depth,
        "decision_class": decision_class,
        "traversal_decision": traversal_decision,
        "n_parent_context": 200,
        "n_node": 100,
        "n_incoming_sibling": 100,
        "n_left": n_left,
        "n_right": n_right,
        "incoming_branch_balance": incoming,
        "outgoing_balance": outgoing,
        "outgoing_edge_norm_balance": edge_norm,
        "outgoing_fragment_risk_proxy_score": fragment,
        "selected_family_log_bayes_factor_lower": selected,
        "continuous_context_min_margin": context,
    }
    if neighborhood_scale is not None:
        record["neighborhood_scale"] = neighborhood_scale
    if topology_support_role is not None:
        record["topology_support_role"] = topology_support_role
    if topology_signal_role is not None:
        record["topology_signal_role"] = topology_signal_role
    if balance_product is not None:
        record["balance_product"] = balance_product
    if branch_length_to_parent is not None:
        record["branch_length_to_parent"] = branch_length_to_parent
    return record


def test_cached_tree_distances_materializes_all_pairs_once() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="root", parent_id="", depth=0),
            _row(node_id="left", parent_id="root", depth=1),
            _row(node_id="right", parent_id="root", depth=1),
            _row(node_id="leaf", parent_id="left", depth=2),
        ]
    )

    cache = build_cached_tree_distances(rows)

    assert cache.status == "cached_all_pairs_tree_distances"
    assert cache.computed_pair_count == 6
    assert cache.distance("left", "right") == 2.0
    assert cache.distance("leaf", "right") == 3.0
    assert cache.distance("leaf", "right") == 3.0
    assert cache.computed_pair_count == 6


def test_cached_tree_distances_use_parent_branch_lengths_when_available() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="root", parent_id="", depth=0),
            _row(
                node_id="left",
                parent_id="root",
                depth=1,
                branch_length_to_parent=0.25,
            ),
            _row(
                node_id="right",
                parent_id="root",
                depth=1,
                branch_length_to_parent=1.75,
            ),
            _row(
                node_id="leaf",
                parent_id="left",
                depth=2,
                branch_length_to_parent=0.50,
            ),
        ]
    )

    cache = build_cached_tree_distances(rows)

    assert cache.status == "cached_all_pairs_branch_length_tree_distances"
    assert cache.distance_metric == "branch_length"
    assert cache.edge_count == 3
    assert cache.branch_length_edge_count == 3
    assert cache.distances_available
    assert cache.distance("left", "right") == 2.0
    assert cache.distance("leaf", "right") == 2.5


def test_cached_tree_distances_report_mixed_branch_length_coverage() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="left", parent_id="root", depth=1, branch_length_to_parent=0.25),
            _row(node_id="right", parent_id="root", depth=1),
        ]
    )

    cache = build_cached_tree_distances(rows)

    assert cache.status == "cached_all_pairs_mixed_branch_length_tree_distances"
    assert cache.distance_metric == "mixed_branch_length_and_hop_count"
    assert cache.edge_count == 2
    assert cache.branch_length_edge_count == 1
    assert cache.distance("left", "right") == 1.25


def test_cached_tree_distances_reports_missing_parent_edges() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="left", parent_id="", depth=1),
            _row(node_id="right", parent_id="", depth=1),
        ]
    )

    cache = build_cached_tree_distances(rows)

    assert cache.status == "tree_distance_parent_edges_unavailable"
    assert cache.computed_pair_count == 1
    assert math.isinf(cache.distance("left", "right"))


def test_cached_tree_distances_uses_auxiliary_parent_nodes() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="left", parent_id="root", depth=1),
            _row(node_id="right", parent_id="root", depth=1),
        ]
    )

    cache = build_cached_tree_distances(rows)

    assert cache.status == "cached_all_pairs_tree_distances"
    assert set(cache.nodes) == {"left", "right", "root"}
    assert cache.distance("left", "right") == 2.0
    assert cache.computed_pair_count == 3


def test_directed_incidence_distinguishes_root_internal_and_leaf() -> None:
    root = infer_directed_incidence(
        depth=0,
        decision_class="accepted_internal_split",
        traversal_decision="split",
        n_left=50,
        n_right=50,
    )
    internal = infer_directed_incidence(
        depth=2,
        decision_class="accepted_internal_split",
        traversal_decision="split",
        n_left=25,
        n_right=25,
        parent_id="N0",
    )
    leaf = infer_directed_incidence(
        depth=3,
        decision_class="leaf_fragment",
        traversal_decision="boundary",
        n_children=0,
        parent_id="N1",
    )

    assert root.incidence_role == "root"
    assert not root.has_incoming_edge
    assert root.has_outgoing_test
    assert root.directed_degree == 2
    assert internal.incidence_role == "internal"
    assert internal.has_incoming_edge
    assert internal.has_outgoing_test
    assert internal.directed_degree == 3
    assert leaf.incidence_role == "leaf"
    assert leaf.has_incoming_edge
    assert not leaf.has_outgoing_test
    assert leaf.directed_degree == 1


def test_root_has_no_fake_incoming_component() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(depth=0, incoming=math.nan, node_id="root"),
                _row(role="null_like", depth=0, incoming=math.nan, node_id="root2"),
            ]
        ),
        min_truth_support_per_stratum=1,
    )

    root = rows.loc[rows["node_id"].eq("root")].iloc[0]
    assert root["incidence_role"] == "root"
    assert not bool(root["has_incoming_edge"])
    assert float(root["incoming_balance_log_lr"]) == 0.0
    assert root["support_status"] == "support_observed_diagnostic_only"


def test_leaf_rows_are_not_sibling_test_candidates() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    depth=2,
                    decision_class="leaf_fragment",
                    traversal_decision="boundary",
                    n_left=math.nan,
                    n_right=math.nan,
                    node_id="leaf",
                ),
                _row(role="null_like", node_id="internal"),
            ]
        ),
        min_truth_support_per_stratum=1,
    )

    leaf = rows.loc[rows["node_id"].eq("leaf")].iloc[0]
    assert leaf["incidence_role"] == "leaf"
    assert not bool(leaf["has_outgoing_test"])
    assert leaf["support_status"] == "leaf_no_outgoing_test_fail_closed"
    assert leaf["conditional_topology_status"] == "leaf_no_outgoing_test_fail_closed"


def test_outgoing_balance_and_edge_norm_increase_posterior() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    role="truth_recovery",
                    outgoing=0.49,
                    edge_norm=0.96,
                    fragment=0.55,
                    node_id="coherent",
                ),
                _row(
                    role="null_like",
                    outgoing=0.25,
                    edge_norm=0.50,
                    fragment=1.25,
                    node_id="incoherent",
                ),
            ]
        ),
        min_truth_support_per_stratum=1,
    )

    coherent = rows.loc[rows["node_id"].eq("coherent")].iloc[0]
    incoherent = rows.loc[rows["node_id"].eq("incoherent")].iloc[0]
    assert float(coherent["conditional_log_odds"]) > float(incoherent["conditional_log_odds"])
    assert int(coherent["conditional_rank"]) == 1


def test_selected_family_context_alone_does_not_promote_weak_topology() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    role="truth_recovery",
                    outgoing=0.24,
                    edge_norm=0.45,
                    fragment=1.40,
                    selected=40.0,
                    context=0.02,
                    node_id="selected_only",
                ),
                _row(
                    role="truth_recovery",
                    outgoing=0.49,
                    edge_norm=0.96,
                    fragment=0.55,
                    selected=4.0,
                    context=-0.01,
                    node_id="topology",
                    replicate=1,
                ),
            ]
        ),
        min_truth_support_per_stratum=1,
    )

    selected_only = rows.loc[rows["node_id"].eq("selected_only")].iloc[0]
    assert not bool(selected_only["topology_core_supported"])
    assert selected_only["conditional_topology_status"] == ("selected_context_only_not_promoted")


def test_neighborhood_scale_component_is_explicit_and_support_gated() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(node_id="truth_a", neighborhood_scale=10.0),
                _row(
                    role="truth_recovery",
                    node_id="truth_b",
                    replicate=1,
                    neighborhood_scale=12.0,
                ),
                _row(
                    role="null_like",
                    node_id="far_negative",
                    neighborhood_scale=80.0,
                ),
            ]
        ),
        min_truth_support_per_stratum=2,
    )

    truth = rows.loc[rows["node_id"].eq("truth_a")].iloc[0]
    negative = rows.loc[rows["node_id"].eq("far_negative")].iloc[0]
    assert truth["neighborhood_scale_support_status"] == (
        "neighborhood_scale_support_observed_diagnostic_only"
    )
    assert float(truth["neighborhood_scale_log_component"]) > float(
        negative["neighborhood_scale_log_component"]
    )
    assert int(truth["neighborhood_scale_support_truth_count"]) == 2


def test_neighborhood_scale_support_is_reported_without_silent_promotion() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(node_id="truth", neighborhood_scale=10.0),
                _row(role="null_like", node_id="negative", neighborhood_scale=11.0),
            ]
        ),
        min_truth_support_per_stratum=2,
    )
    summary = summarize_conditional_topology_law_rows(rows)

    truth = rows.loc[rows["node_id"].eq("truth")].iloc[0]
    assert truth["neighborhood_scale_support_status"] == (
        "neighborhood_scale_support_insufficient_fail_closed"
    )
    assert summary["neighborhood_scale_support_insufficient_row_count"].iloc[0] == 2
    assert summary["production_status"].iloc[0] == (
        "diagnostic_only_support_insufficient_fail_closed"
    )


def test_topology_neighborhood_bandwidth_excludes_selected_nonnull_support() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    role="null_like",
                    node_id="stable_null",
                    parent_id="root",
                    topology_support_role="strict_null",
                    neighborhood_scale=10.0,
                ),
                _row(
                    role="diffuse_or_wrong",
                    node_id="selected_nonnull",
                    parent_id="root",
                    topology_support_role="selected_nonnull",
                    neighborhood_scale=11.0,
                ),
                _row(
                    node_id="candidate",
                    parent_id="selected_nonnull",
                    topology_signal_role="signal",
                    neighborhood_scale=12.0,
                ),
            ]
        ),
        min_truth_support_per_stratum=2,
    )

    candidate = rows.loc[rows["node_id"].eq("candidate")].iloc[0]
    assert int(candidate["topology_neighborhood_support_count"]) == 1
    assert int(candidate["topology_neighborhood_selected_nonnull_excluded_count"]) == 1
    assert candidate["topology_neighborhood_support_status"] == (
        "topology_neighborhood_support_insufficient_fail_closed"
    )
    assert float(candidate["topology_neighborhood_log_component"]) == 0.0


def test_topology_neighborhood_bandwidth_component_uses_cached_tree_context() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    role="null_like",
                    node_id="stable_a",
                    parent_id="root",
                    topology_support_role="strict_null",
                    neighborhood_scale=10.0,
                ),
                _row(
                    role="null_like",
                    node_id="stable_b",
                    parent_id="root",
                    topology_support_role="edge_blocked",
                    neighborhood_scale=12.0,
                    replicate=1,
                ),
                _row(
                    role="truth_recovery",
                    node_id="signal",
                    parent_id="stable_a",
                    topology_signal_role="signal",
                    neighborhood_scale=11.0,
                ),
                _row(
                    role="truth_recovery",
                    node_id="near_signal",
                    parent_id="signal",
                    neighborhood_scale=11.5,
                    replicate=2,
                ),
                _row(
                    role="truth_recovery",
                    node_id="near_stable",
                    parent_id="stable_b",
                    neighborhood_scale=11.5,
                    replicate=3,
                ),
            ]
        ),
        min_truth_support_per_stratum=1,
    )

    near_signal = rows.loc[rows["node_id"].eq("near_signal")].iloc[0]
    near_stable = rows.loc[rows["node_id"].eq("near_stable")].iloc[0]
    assert near_signal["topology_neighborhood_support_status"] == (
        "topology_neighborhood_support_observed_diagnostic_only"
    )
    assert float(near_signal["topology_neighborhood_log_component"]) > float(
        near_stable["topology_neighborhood_log_component"]
    )
    assert float(near_signal["conditional_log_odds"]) > float(near_stable["conditional_log_odds"])


def test_guarded_recovery_requires_support_guards_and_topology_evidence() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    node_id="candidate",
                    parent_id="parent",
                    topology_signal_role="signal",
                    outgoing=0.47,
                    edge_norm=0.96,
                    balance_product=0.23,
                ),
                _row(
                    node_id="truth_support",
                    parent_id="parent",
                    topology_signal_role="signal",
                    outgoing=0.46,
                    edge_norm=0.955,
                    balance_product=0.225,
                    replicate=1,
                ),
                _row(
                    role="null_like",
                    node_id="strict_null",
                    parent_id="parent",
                    topology_support_role="strict_null",
                    outgoing=0.49,
                    edge_norm=0.98,
                    balance_product=0.24,
                    replicate=2,
                ),
                _row(
                    role="diffuse_or_wrong",
                    node_id="low_product",
                    parent_id="parent",
                    topology_support_role="selected_nonnull",
                    outgoing=0.49,
                    edge_norm=0.98,
                    balance_product=0.20,
                    replicate=3,
                ),
                _row(
                    role="diffuse_or_wrong",
                    node_id="low_edge",
                    parent_id="parent",
                    topology_support_role="selected_nonnull",
                    outgoing=0.49,
                    edge_norm=0.94,
                    balance_product=0.23,
                    replicate=4,
                ),
            ]
        ),
        min_truth_support_per_stratum=2,
    )
    by_node = {row["node_id"]: row for _, row in rows.iterrows()}
    summary = summarize_conditional_topology_law_rows(
        rows,
        min_truth_support_per_stratum=2,
    ).iloc[0]

    assert bool(by_node["candidate"]["recover_internal_split"])
    assert by_node["candidate"]["guarded_recovery_status"] == (
        "guarded_internal_recovery_candidate_diagnostic_only"
    )
    assert not bool(by_node["strict_null"]["recover_internal_split"])
    assert by_node["strict_null"]["guarded_recovery_status"] == ("root_or_null_guard_blocked")
    assert by_node["low_product"]["guarded_recovery_status"] == ("balance_product_below_floor")
    assert by_node["low_edge"]["guarded_recovery_status"] == ("outgoing_edge_below_floor")
    assert int(summary["guarded_recovery_candidate_count"]) == 2
    assert int(summary["guarded_recovery_guard_blocked_row_count"]) == 1
    assert int(summary["guarded_recovery_evidence_blocked_row_count"]) == 2


def test_guarded_recovery_fails_closed_when_support_is_thin() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    node_id="single_truth",
                    parent_id="parent",
                    topology_signal_role="signal",
                    outgoing=0.47,
                    edge_norm=0.96,
                    balance_product=0.23,
                ),
                _row(
                    role="null_like",
                    node_id="strict_null",
                    parent_id="parent",
                    topology_support_role="strict_null",
                    outgoing=0.40,
                    edge_norm=0.90,
                    balance_product=0.18,
                    replicate=1,
                ),
            ]
        ),
        min_truth_support_per_stratum=2,
    )
    truth = rows.loc[rows["node_id"].eq("single_truth")].iloc[0]

    assert not bool(truth["recover_internal_split"])
    assert truth["guarded_recovery_status"] == "support_insufficient_fail_closed"


def test_missing_topology_features_fail_closed_not_positive() -> None:
    rows = build_conditional_topology_law_rows(
        pd.DataFrame.from_records(
            [
                _row(edge_norm=math.nan, fragment=math.nan, node_id="missing"),
                _row(role="null_like", node_id="null"),
            ]
        ),
        min_truth_support_per_stratum=1,
    )

    missing = rows.loc[rows["node_id"].eq("missing")].iloc[0]
    assert bool(missing["missing_topology_feature"])
    assert missing["support_status"] == "topology_features_missing_fail_closed"
    assert math.isfinite(float(missing["conditional_log_odds"]))


def test_analytical_cases_separate_emergent_truth_from_false_modes() -> None:
    rows = build_conditional_topology_law_rows(
        build_conditional_topology_analytical_cases(),
        min_truth_support_per_stratum=1,
    )
    by_case = {row["analytical_case"]: row for _, row in rows.iterrows()}

    assert float(by_case["context_negative_emergent_true_split"]["conditional_log_odds"]) > float(
        by_case["context_negative_fragment_false_split"]["conditional_log_odds"]
    )
    assert float(
        by_case["closed_root_passthrough_true_many_cluster_split"]["topology_core_log_odds"]
    ) > float(by_case["closed_root_passthrough_false_split"]["topology_core_log_odds"])
    assert (
        by_case["closed_root_passthrough_false_split"]["conditional_topology_status"]
        != "conditional_topology_candidate_diagnostic_only"
    )


def test_focused_real_slice_is_signal_detected_but_support_insufficient() -> None:
    source = pd.DataFrame.from_records(
        [
            _row(
                role="truth_recovery",
                incoming=0.4725,
                outgoing=0.492891,
                edge_norm=0.971963,
                fragment=0.571760,
                selected=18.626437,
                context=-0.005752,
                node_id="truth",
            ),
            _row(
                role="diffuse_or_wrong",
                incoming=0.425,
                outgoing=0.491304,
                edge_norm=0.965812,
                fragment=0.565164,
                selected=12.869595,
                context=-0.005258,
                node_id="negative",
            ),
        ]
    )
    rows = build_conditional_topology_law_rows(source)
    summary = summarize_conditional_topology_law_rows(rows)

    truth = rows.loc[rows["node_id"].eq("truth")].iloc[0]
    assert int(truth["conditional_rank"]) == 1
    assert truth["support_status"] == "support_insufficient_fail_closed"
    assert summary["diagnostic_status"].iloc[0] == (
        "conditional_topology_signal_detected_support_insufficient"
    )
    assert summary["production_status"].iloc[0] == (
        "diagnostic_only_support_insufficient_fail_closed"
    )


def test_run_conditional_topology_law_panel_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "topology.csv"
    pd.DataFrame.from_records(
        [
            _row(node_id="truth"),
            _row(role="truth_recovery", node_id="truth2", replicate=1),
            _row(role="null_like", outgoing=0.25, edge_norm=0.50, node_id="null"),
        ]
    ).to_csv(rows_path, index=False)

    outputs = run_overlap_conditional_topology_law_panel(
        OverlapConditionalTopologyLawPanelConfig(
            topology_rows_path=rows_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()

    summary = pd.read_csv(outputs["summary"])
    benchmark = pd.read_csv(outputs["benchmark_summary"])
    analytical = pd.read_csv(outputs["analytical_cases"])
    manifest = json.loads(outputs["manifest"].read_text())
    assert summary["diagnostic_status"].iloc[0] in {
        "conditional_topology_candidate_diagnostic_only",
        "conditional_topology_partial_truth_separation",
    }
    assert manifest["diagnostic_status"] == summary["diagnostic_status"].iloc[0]
    assert manifest["production_status"] == summary["production_status"].iloc[0]
    assert not benchmark.empty
    assert set(analytical["analytical_case"]) >= {
        "context_negative_emergent_true_split",
        "closed_root_passthrough_false_split",
    }
