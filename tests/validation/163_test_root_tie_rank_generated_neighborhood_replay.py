from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_tie_rank_generated_neighborhood_replay as replay,
)


def _proposal_row(
    *,
    case_id: str,
    proposal_family: str = "iid_marginal_bernoulli",
    data_role: str = "selected_null",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "base_case_id": case_id.split("__")[0],
        "data_role": data_role,
        "calibration_role": "selected_null_candidate_support",
        "proposal_family": proposal_family,
        "proposal_calibration_status": "diagnostic",
        "root_bandwidth_reopen_band": "bandwidth_reopen_missing",
        "root_sibling_selected_ratio": 10.0,
        "root_tie_rank_median_fraction": 0.5,
        "root_edge_path_statistic_margin": 1.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 2.0,
        "root_bandwidth_reopen_count": float("nan"),
        "replicate": 0,
        "null_seed": 123,
        "generated_matrix_density": 0.5,
    }


def _node_decisions(case_id: str, data_role: str, method_id: str) -> pd.DataFrame:
    base = {
        "schema_version": "test",
        "case_id": case_id,
        "data_role": data_role,
        "method_id": method_id,
        "replicate": 0,
        "data_seed": 123,
        "visited": True,
        "child_parent_edge_open": True,
        "sibling_open": False,
        "sibling_projection_dimension": 2.0,
        "root_stability_guard_blocked": False,
        "root_selective_guard_blocked": False,
        "selected_family_guard_blocked": False,
        "selected_family_p_value": np.nan,
        "selected_family_base_p_value": np.nan,
        "selected_family_guard_refined": False,
        "selected_family_scope": "",
        "topology_has_incoming_edge": False,
        "topology_has_outgoing_test": True,
        "topology_directed_degree": 2,
        "topology_pass_through_candidate": False,
        "conditional_topology_status": "not_evaluated_missing_topology_features",
        "conditional_topology_log_odds": np.nan,
        "conditional_topology_probability": np.nan,
        "spectral_transport_node_mp_block_count": np.nan,
        "spectral_transport_node_mp_total_multiplicity": np.nan,
        "spectral_transport_child_best_mode_cost": np.nan,
        "spectral_transport_child_best_mode_affinity": np.nan,
        "spectral_transport_child_best_mode_status": "",
        "spectral_transport_best_descendant_split_path_cost": np.nan,
        "spectral_transport_best_descendant_split_path_has_mp_evidence": False,
        "spectral_transport_passthrough_supported": False,
        "spectral_transport_passthrough_blocked": False,
        "spectral_transport_bottleneck": "",
        "study_role": "test",
    }
    return pd.DataFrame.from_records(
        [
            {
                **base,
                "node_id": "root",
                "parent_id": "",
                "depth": 0,
                "traversal_decision": "boundary",
                "decision_class": "stable_boundary",
                "n_children": 2,
                "n_descendant_leaves": 4,
                "sibling_p_value": np.nan,
                "topology_incidence_role": "root",
            },
            {
                **base,
                "node_id": "left",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "boundary",
                "decision_class": "stable_boundary",
                "n_children": 0,
                "n_descendant_leaves": 2,
                "sibling_p_value": 0.55,
                "topology_incidence_role": "leaf",
            },
            {
                **base,
                "node_id": "right",
                "parent_id": "root",
                "depth": 1,
                "traversal_decision": "boundary",
                "decision_class": "stable_boundary",
                "n_children": 0,
                "n_descendant_leaves": 2,
                "sibling_p_value": 0.60,
                "topology_incidence_role": "leaf",
            },
        ]
    )


def test_generated_replay_filters_observed_targets() -> None:
    rows = pd.DataFrame.from_records(
        [
            _proposal_row(case_id="observed", proposal_family="observed_target"),
            _proposal_row(case_id="generated"),
        ]
    )

    selected = replay._generated_proposal_rows(rows)

    assert selected["case_id"].tolist() == ["generated"]


def test_generated_replay_builds_topology_frontier_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    case_id = "case__iid_marginal_bernoulli_r0000"
    pd.DataFrame(
        [[0, 1], [0, 1], [1, 0], [1, 0]],
        index=["L1", "L2", "L3", "L4"],
        columns=["F0", "F1"],
    ).to_csv(matrix_dir / f"{case_id}.csv")

    monkeypatch.setattr(
        replay,
        "_run_kl_method",
        lambda *args, **kwargs: SimpleNamespace(
            status="ok",
            skip_reason=None,
            found_clusters=2,
            labels=np.array([0, 0, 1, 1]),
        ),
    )
    monkeypatch.setattr(
        replay,
        "_build_node_decisions",
        lambda *, case_id, data_role, method_id, replicate, data_seed, result: (
            _node_decisions(case_id, data_role, method_id)
        ),
    )

    tables = replay.build_generated_neighborhood_replay_tables(
        proposal_rows=pd.DataFrame.from_records([_proposal_row(case_id=case_id)]),
        generated_matrix_dir=matrix_dir,
        method_id=replay.DEFAULT_METHOD_ID,
    )

    assert tables["run_rows"].iloc[0]["replay_status"] == "kl_replay_completed"
    assert tables["node_decisions"].shape[0] == 3
    assert not tables["measurability_rows"].empty
    root = tables["topology_rows"].loc[
        tables["topology_rows"]["candidate_scope"].eq("root_non_direct")
    ].iloc[0]
    assert root["case_id"] == case_id
    assert root["proposal_family"] == "iid_marginal_bernoulli"
    assert root["root_margin_evidence_status"] == (
        "root_margin_joined_case_family_diagnostic_only"
    )


def test_generated_replay_runner_writes_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    case_id = "case__iid_marginal_bernoulli_r0000"
    pd.DataFrame(
        [[0, 1], [0, 1], [1, 0], [1, 0]],
        index=["L1", "L2", "L3", "L4"],
        columns=["F0", "F1"],
    ).to_csv(matrix_dir / f"{case_id}.csv")
    proposal_path = tmp_path / "proposal.csv"
    pd.DataFrame.from_records([_proposal_row(case_id=case_id)]).to_csv(
        proposal_path,
        index=False,
    )

    monkeypatch.setattr(
        replay,
        "_run_kl_method",
        lambda *args, **kwargs: SimpleNamespace(
            status="ok",
            skip_reason=None,
            found_clusters=2,
            labels=np.array([0, 0, 1, 1]),
        ),
    )
    monkeypatch.setattr(
        replay,
        "_build_node_decisions",
        lambda *, case_id, data_role, method_id, replicate, data_seed, result: (
            _node_decisions(case_id, data_role, method_id)
        ),
    )

    outputs = replay.run_root_tie_rank_generated_neighborhood_replay(
        replay.RootTieRankGeneratedNeighborhoodReplayConfig(
            output_dir=tmp_path / "out",
            proposal_feasibility_rows_path=proposal_path,
            generated_matrix_dir=matrix_dir,
        )
    )

    assert set(outputs) == {
        "run_rows",
        "node_decisions",
        "distribution_rows",
        "pvalue_rows",
        "pvalue_summary",
        "pvalue_case_summary",
        "pvalue_tau_s_sensitivity",
        "measurability_rows",
        "measurability_summary",
        "topology_rows",
        "topology_summary",
        "manifest",
    }
    for path in outputs.values():
        assert path.exists()
