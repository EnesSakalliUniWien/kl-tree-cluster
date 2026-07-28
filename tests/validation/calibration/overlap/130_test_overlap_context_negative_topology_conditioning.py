from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap.overlap_context_negative_topology_conditioning import (
    OverlapContextNegativeTopologyConditioningConfig,
    build_context_negative_topology_conditioning_rows,
    run_overlap_context_negative_topology_conditioning,
    summarize_context_negative_topology_conditioning,
)


def _mode_rows() -> pd.DataFrame:
    base = {
        "bayesian_incidence_mode_status": "context_negative_emergent_mode_ambiguous",
        "selected_family_log_bayes_factor_lower": 8.0,
        "continuous_context_min_margin": -0.005,
        "branch_alignment_score": 0.10,
    }
    return pd.DataFrame.from_records(
        [
            {
                **base,
                "case_id": "truth",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "T",
                "guard_truth_role": "truth_recovery",
            },
            {
                **base,
                "case_id": "negative_low",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "N1",
                "guard_truth_role": "null_like",
            },
            {
                **base,
                "case_id": "negative_mid",
                "data_role": "selected_null",
                "replicate": 1,
                "node_id": "N2",
                "guard_truth_role": "null_like",
            },
            {
                **base,
                "case_id": "other_mode",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "O",
                "guard_truth_role": "truth_recovery",
                "bayesian_incidence_mode_status": "local_outcome_mode_candidate",
            },
        ]
    )


def _branch_row(
    *,
    case_id: str,
    data_role: str,
    replicate: int,
    node_id: str,
    outgoing_balance: float,
    sibling_projection_dimension: float,
    branch_length_to_parent: float | None = None,
) -> dict[str, object]:
    record = {
        "case_id": case_id,
        "data_role": data_role,
        "replicate": replicate,
        "node_id": node_id,
        "incoming_parent_id": f"P_{replicate}",
        "depth": 1,
        "decision_class": "accepted_internal_split",
        "traversal_decision": "split",
        "sibling_open": True,
        "sibling_p_value": 0.0001,
        "sibling_projection_dimension": sibling_projection_dimension,
        "selected_family_guard_blocked": False,
        "selected_family_p_value": 0.0001,
        "n_parent_context": 400,
        "n_node": 200,
        "n_incoming_sibling": 200,
        "n_left": int(round(200 * outgoing_balance)),
        "n_right": 200 - int(round(200 * outgoing_balance)),
        "incoming_branch_balance": 0.50,
        "outgoing_balance": outgoing_balance,
        "metric_family_alignment_score": 0.10,
    }
    if branch_length_to_parent is not None:
        record["branch_length_to_parent"] = branch_length_to_parent
    return record


def _gap_row(
    *,
    case_id: str,
    data_role: str,
    replicate: int,
    node_id: str,
    outgoing_balance: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": data_role,
        "replicate": replicate,
        "node_id": node_id,
        "subspace_consensus_jaccard_topk": 0.4,
        "size_balance": outgoing_balance,
        "edge_norm_balance": 0.9,
        "fragment_risk_proxy_score": 0.7,
        "soft_structure_pass": True,
        "default_internal_node_candidate": False,
        "blocking_components": "local_context_margin",
    }


def _income_row(
    *,
    case_id: str,
    data_role: str,
    replicate: int,
    node_id: str,
    outgoing_balance: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": data_role,
        "replicate": replicate,
        "node_id": node_id,
        "incoming_parent_depth": 0,
        "incoming_parent_context_margin": -0.01,
        "incoming_parent_homogeneity_gain_min": 0.002,
        "incoming_parent_subspace_consensus_jaccard_topk": 0.3,
        "outgoing_depth": 1,
        "outgoing_homogeneity_gain_min": 0.005,
        "outgoing_subspace_consensus_jaccard_topk": 0.4,
        "outgoing_size_balance": outgoing_balance,
        "outgoing_edge_norm_balance": 0.9,
        "outgoing_fragment_risk_proxy_score": 0.7,
        "context_transition_delta": 0.004,
        "subspace_transition_delta": 0.1,
        "homogeneity_transition_delta": 0.003,
    }


def _joined_inputs(truth_outgoing_balance: float = 0.49) -> tuple[pd.DataFrame, ...]:
    specs = [
        ("truth", "signal", 0, "T", truth_outgoing_balance, 7.0),
        ("negative_low", "selected_null", 0, "N1", 0.38, 11.0),
        ("negative_mid", "selected_null", 1, "N2", 0.45, 13.0),
        ("other_mode", "signal", 0, "O", 0.50, 17.0),
    ]
    branch = pd.DataFrame.from_records(
        [
            _branch_row(
                case_id=case_id,
                data_role=data_role,
                replicate=replicate,
                node_id=node_id,
                outgoing_balance=outgoing_balance,
                sibling_projection_dimension=sibling_projection_dimension,
                branch_length_to_parent=0.25 + replicate,
            )
            for (
                case_id,
                data_role,
                replicate,
                node_id,
                outgoing_balance,
                sibling_projection_dimension,
            ) in specs
        ]
    )
    gap = pd.DataFrame.from_records(
        [
            _gap_row(
                case_id=case_id,
                data_role=data_role,
                replicate=replicate,
                node_id=node_id,
                outgoing_balance=outgoing_balance,
            )
            for case_id, data_role, replicate, node_id, outgoing_balance, _ in specs
        ]
    )
    income = pd.DataFrame.from_records(
        [
            _income_row(
                case_id=case_id,
                data_role=data_role,
                replicate=replicate,
                node_id=node_id,
                outgoing_balance=outgoing_balance,
            )
            for case_id, data_role, replicate, node_id, outgoing_balance, _ in specs
        ]
    )
    return branch, _mode_rows(), gap, income


def test_topology_conditioning_finds_single_truth_separator_candidate() -> None:
    branch, mode, gap, income = _joined_inputs()

    rows = build_context_negative_topology_conditioning_rows(
        branch_rows=branch,
        mode_rows=mode,
        transfer_gap_rows=gap,
        income_outcome_rows=income,
    )
    summary, metric_summary, _category_summary = summarize_context_negative_topology_conditioning(
        rows,
        metrics=("outgoing_balance",),
    )

    assert rows.shape[0] == 3
    truth = rows.loc[rows["node_id"].eq("T")].iloc[0]
    null_like = rows.loc[rows["node_id"].eq("N1")].iloc[0]
    assert truth["neighborhood_scale"] == 7.0
    assert truth["neighborhood_scale_source"] == "sibling_projection_dimension"
    assert truth["parent_id"] == "P_0"
    assert truth["branch_length_to_parent"] == 0.25
    assert truth["topology_signal_role"] == "signal"
    assert truth["topology_support_role"] == ""
    assert null_like["topology_signal_role"] == ""
    assert null_like["topology_support_role"] == "strict_null"
    assert summary["diagnostic_status"].iloc[0] == (
        "topology_conditioning_single_truth_separator_candidate"
    )
    assert metric_summary["zero_negative_status"].iloc[0] == ("zero_negative_separates_all_truth")


def test_topology_conditioning_marks_signal_negatives_selected_nonnull() -> None:
    branch, mode, gap, income = _joined_inputs()
    mode = pd.concat(
        [
            mode,
            pd.DataFrame.from_records(
                [
                    {
                        "case_id": "signal_negative",
                        "data_role": "signal",
                        "replicate": 2,
                        "node_id": "S",
                        "guard_truth_role": "diffuse_or_wrong",
                        "bayesian_incidence_mode_status": (
                            "context_negative_emergent_mode_ambiguous"
                        ),
                        "selected_family_log_bayes_factor_lower": 6.0,
                        "continuous_context_min_margin": -0.004,
                        "branch_alignment_score": 0.10,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    branch = pd.concat(
        [
            branch,
            pd.DataFrame.from_records(
                [
                    _branch_row(
                        case_id="signal_negative",
                        data_role="signal",
                        replicate=2,
                        node_id="S",
                        outgoing_balance=0.42,
                        sibling_projection_dimension=9.0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    gap = pd.concat(
        [
            gap,
            pd.DataFrame.from_records(
                [
                    _gap_row(
                        case_id="signal_negative",
                        data_role="signal",
                        replicate=2,
                        node_id="S",
                        outgoing_balance=0.42,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    income = pd.concat(
        [
            income,
            pd.DataFrame.from_records(
                [
                    _income_row(
                        case_id="signal_negative",
                        data_role="signal",
                        replicate=2,
                        node_id="S",
                        outgoing_balance=0.42,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    rows = build_context_negative_topology_conditioning_rows(
        branch_rows=branch,
        mode_rows=mode,
        transfer_gap_rows=gap,
        income_outcome_rows=income,
    )

    signal_negative = rows.loc[rows["node_id"].eq("S")].iloc[0]
    assert signal_negative["topology_signal_role"] == ""
    assert signal_negative["topology_support_role"] == "selected_nonnull"


def test_topology_conditioning_tolerates_missing_projection_dimension() -> None:
    branch, mode, gap, income = _joined_inputs()
    branch = branch.drop(columns=["sibling_projection_dimension"])

    rows = build_context_negative_topology_conditioning_rows(
        branch_rows=branch,
        mode_rows=mode,
        transfer_gap_rows=gap,
        income_outcome_rows=income,
    )

    truth = rows.loc[rows["node_id"].eq("T")].iloc[0]
    assert pd.isna(truth["neighborhood_scale"])
    assert truth["neighborhood_scale_source"] == "missing_sibling_projection_dimension"
    assert truth["topology_signal_role"] == "signal"


def test_topology_conditioning_reports_no_separator_when_metric_overlaps() -> None:
    branch, mode, gap, income = _joined_inputs(truth_outgoing_balance=0.40)

    rows = build_context_negative_topology_conditioning_rows(
        branch_rows=branch,
        mode_rows=mode,
        transfer_gap_rows=gap,
        income_outcome_rows=income,
    )
    summary, metric_summary, _category_summary = summarize_context_negative_topology_conditioning(
        rows,
        metrics=("outgoing_balance",),
    )

    assert summary["diagnostic_status"].iloc[0] == (
        "topology_conditioning_no_zero_negative_separator"
    )
    assert metric_summary["zero_negative_status"].iloc[0] == ("zero_negative_no_truth_retention")


def test_run_topology_conditioning_writes_outputs(tmp_path) -> None:
    branch, mode, gap, income = _joined_inputs()
    branch_path = tmp_path / "branch.csv"
    mode_path = tmp_path / "mode.csv"
    gap_path = tmp_path / "gap.csv"
    income_path = tmp_path / "income.csv"
    branch.to_csv(branch_path, index=False)
    mode.to_csv(mode_path, index=False)
    gap.to_csv(gap_path, index=False)
    income.to_csv(income_path, index=False)

    outputs = run_overlap_context_negative_topology_conditioning(
        OverlapContextNegativeTopologyConditioningConfig(
            branch_rows_path=branch_path,
            mode_rows_path=mode_path,
            transfer_gap_rows_path=gap_path,
            income_outcome_rows_path=income_path,
            output_dir=tmp_path / "out",
            metrics=("outgoing_balance",),
        )
    )

    for path in outputs.values():
        assert path.exists()
