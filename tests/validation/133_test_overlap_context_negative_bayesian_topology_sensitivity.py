from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.overlap_context_negative_bayesian_topology_sensitivity import (
    OverlapContextNegativeBayesianTopologySensitivityConfig,
    build_bayesian_topology_sensitivity_rows,
    run_overlap_context_negative_bayesian_topology_sensitivity,
    summarize_bayesian_topology_sensitivity,
)


def _row(
    *,
    role: str,
    incoming: float,
    outgoing: float,
    edge_norm: float,
    fragment: float,
    node_id: str,
) -> dict[str, object]:
    return {
        "case_id": "case",
        "data_role": "signal",
        "replicate": 0,
        "node_id": node_id,
        "guard_truth_role": role,
        "incoming_branch_balance": incoming,
        "outgoing_balance": outgoing,
        "outgoing_edge_norm_balance": edge_norm,
        "outgoing_fragment_risk_proxy_score": fragment,
        "selected_family_log_bayes_factor_lower": 8.0,
        "continuous_context_min_margin": -0.01,
    }


def _rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            _row(
                role="truth_recovery",
                incoming=0.48,
                outgoing=0.49,
                edge_norm=0.97,
                fragment=0.55,
                node_id="truth",
            ),
            _row(
                role="null_like",
                incoming=0.25,
                outgoing=0.25,
                edge_norm=0.50,
                fragment=1.20,
                node_id="null",
            ),
            _row(
                role="diffuse_or_wrong",
                incoming=0.30,
                outgoing=0.30,
                edge_norm=0.55,
                fragment=1.00,
                node_id="wrong",
            ),
        ]
    )


def test_sensitivity_marks_structural_signal_not_selected_context() -> None:
    rows = build_bayesian_topology_sensitivity_rows(
        _rows(),
        context_penalty_weights=(0.0, 50.0),
    )
    summary = summarize_bayesian_topology_sensitivity(rows)

    assert summary["diagnostic_status"].iloc[0] == (
        "bayesian_topology_structural_signal_robust_not_selected_context"
    )
    assert int(summary["topology_only_separates_count"].iloc[0]) == 2
    assert int(summary["outgoing_topology_only_separates_count"].iloc[0]) == 2
    assert int(summary["selected_context_only_separates_count"].iloc[0]) == 0


def test_sensitivity_rows_keep_selected_context_profile_fail_closed() -> None:
    rows = build_bayesian_topology_sensitivity_rows(
        _rows(),
        context_penalty_weights=(50.0,),
    )
    selected_context = rows.loc[rows["profile"].eq("selected_context_only")].iloc[0]
    topology_only = rows.loc[rows["profile"].eq("topology_only")].iloc[0]

    assert selected_context["profile_status"] == "sensitivity_not_separating"
    assert topology_only["profile_status"] == "sensitivity_truth_top_rank_separates"


def test_run_bayesian_topology_sensitivity_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "topology.csv"
    _rows().to_csv(rows_path, index=False)

    outputs = run_overlap_context_negative_bayesian_topology_sensitivity(
        OverlapContextNegativeBayesianTopologySensitivityConfig(
            topology_rows_path=rows_path,
            output_dir=tmp_path / "out",
            context_penalty_weights=(0.0, 50.0),
        )
    )

    for path in outputs.values():
        assert path.exists()
