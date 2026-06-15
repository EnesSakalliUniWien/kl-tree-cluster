from __future__ import annotations

import math

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_conditional_topology_law_panel import (
    OverlapConditionalTopologyLawPanelConfig,
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
    n_left: float = 50.0,
    n_right: float = 50.0,
    node_id: str = "N1",
    replicate: int = 0,
) -> dict[str, object]:
    return {
        "case_id": "case",
        "data_role": "signal",
        "replicate": replicate,
        "node_id": node_id,
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
    assert float(coherent["conditional_log_odds"]) > float(
        incoherent["conditional_log_odds"]
    )
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
    assert selected_only["conditional_topology_status"] == (
        "selected_context_only_not_promoted"
    )


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
    by_case = {
        row["analytical_case"]: row
        for _, row in rows.iterrows()
    }

    assert float(
        by_case["context_negative_emergent_true_split"]["conditional_log_odds"]
    ) > float(
        by_case["context_negative_fragment_false_split"]["conditional_log_odds"]
    )
    assert float(
        by_case["closed_root_passthrough_true_many_cluster_split"][
            "topology_core_log_odds"
        ]
    ) > float(
        by_case["closed_root_passthrough_false_split"]["topology_core_log_odds"]
    )
    assert by_case["closed_root_passthrough_false_split"][
        "conditional_topology_status"
    ] != "conditional_topology_candidate_diagnostic_only"


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
    assert summary["diagnostic_status"].iloc[0] in {
        "conditional_topology_candidate_diagnostic_only",
        "conditional_topology_partial_truth_separation",
    }
    assert not benchmark.empty
    assert set(analytical["analytical_case"]) >= {
        "context_negative_emergent_true_split",
        "closed_root_passthrough_false_split",
    }
