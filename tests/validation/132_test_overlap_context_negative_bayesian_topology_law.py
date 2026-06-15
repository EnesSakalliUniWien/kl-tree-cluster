from __future__ import annotations

import math

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration.overlap_context_negative_bayesian_topology_law import (
    OverlapContextNegativeBayesianTopologyLawConfig,
    beta_log_likelihood_ratio,
    build_context_negative_bayesian_topology_rows,
    run_overlap_context_negative_bayesian_topology_law,
    summarize_bayesian_topology_rows,
)


def _base_row(
    *,
    role: str,
    incoming: float,
    outgoing: float,
    edge_norm: float,
    fragment: float,
    selected_family: float = 8.0,
    context: float = -0.01,
    case_id: str = "case",
    replicate: int = 0,
    node_id: str = "N1",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "signal",
        "replicate": replicate,
        "node_id": node_id,
        "guard_truth_role": role,
        "incoming_branch_balance": incoming,
        "outgoing_balance": outgoing,
        "outgoing_edge_norm_balance": edge_norm,
        "outgoing_fragment_risk_proxy_score": fragment,
        "selected_family_log_bayes_factor_lower": selected_family,
        "continuous_context_min_margin": context,
    }


def test_beta_log_likelihood_ratio_favors_high_topology_values() -> None:
    low, high = beta_log_likelihood_ratio(np.array([0.25, 0.90]))

    assert high > low


def test_bayesian_topology_law_ranks_balanced_income_outcome_truth() -> None:
    source = pd.DataFrame.from_records(
        [
            _base_row(
                role="truth_recovery",
                incoming=0.48,
                outgoing=0.49,
                edge_norm=0.97,
                fragment=0.55,
                node_id="truth",
            ),
            _base_row(
                role="diffuse_or_wrong",
                incoming=0.42,
                outgoing=0.40,
                edge_norm=0.70,
                fragment=0.90,
                node_id="wrong",
            ),
            _base_row(
                role="fragment_like",
                incoming=0.45,
                outgoing=0.22,
                edge_norm=0.60,
                fragment=1.30,
                node_id="fragment",
            ),
        ]
    )

    rows = build_context_negative_bayesian_topology_rows(source)
    summary = summarize_bayesian_topology_rows(rows)

    truth = rows.loc[rows["guard_truth_role"].eq("truth_recovery")].iloc[0]
    assert int(truth["posterior_rank"]) == 1
    assert summary["diagnostic_status"].iloc[0] == (
        "bayesian_topology_score_separates_focused_slice"
    )
    assert float(summary["posterior_log_odds_margin"].iloc[0]) > 0.0


def test_bayesian_topology_law_keeps_missing_components_neutral() -> None:
    source = pd.DataFrame.from_records(
        [
            _base_row(
                role="truth_recovery",
                incoming=0.48,
                outgoing=0.49,
                edge_norm=0.97,
                fragment=0.55,
                node_id="truth",
            ),
            _base_row(
                role="null_like",
                incoming=math.nan,
                outgoing=math.nan,
                edge_norm=math.nan,
                fragment=math.nan,
                selected_family=math.nan,
                context=math.nan,
                node_id="missing",
            ),
        ]
    )

    rows = build_context_negative_bayesian_topology_rows(source)

    missing = rows.loc[rows["node_id"].eq("missing")].iloc[0]
    assert np.isfinite(rows["posterior_log_odds"]).all()
    assert float(missing["incoming_balance_log_lr"]) == 0.0
    assert float(missing["outgoing_balance_log_lr"]) == 0.0
    assert float(missing["outgoing_edge_norm_log_lr"]) == 0.0
    assert float(missing["anti_fragment_log_lr"]) == 0.0


def test_run_bayesian_topology_law_writes_outputs(tmp_path) -> None:
    rows_path = tmp_path / "topology.csv"
    pd.DataFrame.from_records(
        [
            _base_row(
                role="truth_recovery",
                incoming=0.48,
                outgoing=0.49,
                edge_norm=0.97,
                fragment=0.55,
                node_id="truth",
            ),
            _base_row(
                role="null_like",
                incoming=0.30,
                outgoing=0.30,
                edge_norm=0.50,
                fragment=1.00,
                node_id="null",
            ),
        ]
    ).to_csv(rows_path, index=False)

    outputs = run_overlap_context_negative_bayesian_topology_law(
        OverlapContextNegativeBayesianTopologyLawConfig(
            topology_rows_path=rows_path,
            output_dir=tmp_path / "out",
        )
    )

    for path in outputs.values():
        assert path.exists()
