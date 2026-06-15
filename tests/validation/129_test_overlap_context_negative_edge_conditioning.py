from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_context_negative_edge_conditioning import (
    OverlapContextNegativeEdgeConditioningConfig,
    build_context_negative_edge_conditioning_rows,
    run_overlap_context_negative_edge_conditioning,
    summarize_context_negative_edge_conditioning,
)


def _mode_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "truth",
                "data_role": "signal",
                "replicate": 0,
                "node_id": "T",
                "guard_truth_role": "truth_recovery",
                "bayesian_incidence_mode_status": (
                    "context_negative_emergent_mode_ambiguous"
                ),
            },
            {
                "case_id": "negative_low",
                "data_role": "selected_null",
                "replicate": 0,
                "node_id": "N1",
                "guard_truth_role": "null_like",
                "bayesian_incidence_mode_status": (
                    "context_negative_emergent_mode_ambiguous"
                ),
            },
            {
                "case_id": "negative_mid",
                "data_role": "selected_null",
                "replicate": 1,
                "node_id": "N2",
                "guard_truth_role": "null_like",
                "bayesian_incidence_mode_status": (
                    "context_negative_emergent_mode_ambiguous"
                ),
            },
            {
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
    incoming: float,
    incoming_sibling: float,
    outgoing_left: float,
    outgoing_right: float,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": data_role,
        "replicate": replicate,
        "node_id": node_id,
        "incoming_edge_neglog10_bh_p_value": incoming,
        "incoming_sibling_edge_neglog10_bh_p_value": incoming_sibling,
        "outgoing_left_edge_neglog10_bh_p_value": outgoing_left,
        "outgoing_right_edge_neglog10_bh_p_value": outgoing_right,
        "min_outgoing_edge_neglog10_bh_p_value": min(outgoing_left, outgoing_right),
        "max_outgoing_edge_neglog10_bh_p_value": max(outgoing_left, outgoing_right),
        "outgoing_edge_neglog10_balance": min(outgoing_left, outgoing_right)
        / max(outgoing_left, outgoing_right),
        "incoming_edge_rejected": incoming > 3.0,
        "incoming_sibling_edge_rejected": incoming_sibling > 3.0,
        "outgoing_left_edge_rejected": outgoing_left > 3.0,
        "outgoing_right_edge_rejected": outgoing_right > 3.0,
        "outgoing_edges_both_rejected": outgoing_left > 3.0 and outgoing_right > 3.0,
        "incoming_edges_both_rejected": incoming > 3.0 and incoming_sibling > 3.0,
    }


def _separable_branch_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            _branch_row(
                case_id="truth",
                data_role="signal",
                replicate=0,
                node_id="T",
                incoming=1.0,
                incoming_sibling=1.2,
                outgoing_left=8.0,
                outgoing_right=7.0,
            ),
            _branch_row(
                case_id="negative_low",
                data_role="selected_null",
                replicate=0,
                node_id="N1",
                incoming=1.0,
                incoming_sibling=1.2,
                outgoing_left=2.0,
                outgoing_right=1.5,
            ),
            _branch_row(
                case_id="negative_mid",
                data_role="selected_null",
                replicate=1,
                node_id="N2",
                incoming=2.0,
                incoming_sibling=2.1,
                outgoing_left=3.0,
                outgoing_right=2.5,
            ),
            _branch_row(
                case_id="other_mode",
                data_role="signal",
                replicate=0,
                node_id="O",
                incoming=1.0,
                incoming_sibling=1.2,
                outgoing_left=9.0,
                outgoing_right=8.0,
            ),
        ]
    )


def test_edge_conditioning_finds_zero_negative_separator() -> None:
    rows = build_context_negative_edge_conditioning_rows(
        branch_rows=_separable_branch_rows(),
        mode_rows=_mode_rows(),
    )
    summary, metric_summary = summarize_context_negative_edge_conditioning(
        rows,
        metrics=("min_outgoing_edge_neglog10_bh_p_value",),
    )

    assert rows.shape[0] == 3
    assert summary["diagnostic_status"].iloc[0] == (
        "edge_conditioning_separator_found"
    )
    assert metric_summary["zero_negative_status"].iloc[0] == (
        "zero_negative_separates_all_truth"
    )


def test_edge_conditioning_reports_no_separator_when_truth_overlaps_negatives() -> None:
    branch = _separable_branch_rows().copy()
    branch.loc[branch["case_id"].eq("truth"), "min_outgoing_edge_neglog10_bh_p_value"] = 2.0
    branch.loc[branch["case_id"].eq("truth"), "outgoing_left_edge_neglog10_bh_p_value"] = 2.0
    branch.loc[branch["case_id"].eq("truth"), "outgoing_right_edge_neglog10_bh_p_value"] = 2.0

    rows = build_context_negative_edge_conditioning_rows(
        branch_rows=branch,
        mode_rows=_mode_rows(),
    )
    summary, metric_summary = summarize_context_negative_edge_conditioning(
        rows,
        metrics=("min_outgoing_edge_neglog10_bh_p_value",),
    )

    assert summary["diagnostic_status"].iloc[0] == (
        "edge_conditioning_no_zero_negative_separator"
    )
    assert metric_summary["zero_negative_status"].iloc[0] == (
        "zero_negative_no_truth_retention"
    )


def test_run_edge_conditioning_writes_outputs(tmp_path) -> None:
    branch_path = tmp_path / "branch.csv"
    mode_path = tmp_path / "mode.csv"
    _separable_branch_rows().to_csv(branch_path, index=False)
    _mode_rows().to_csv(mode_path, index=False)

    outputs = run_overlap_context_negative_edge_conditioning(
        OverlapContextNegativeEdgeConditioningConfig(
            branch_rows_path=branch_path,
            mode_rows_path=mode_path,
            output_dir=tmp_path / "out",
            metrics=("min_outgoing_edge_neglog10_bh_p_value",),
        )
    )

    for path in outputs.values():
        assert path.exists()
