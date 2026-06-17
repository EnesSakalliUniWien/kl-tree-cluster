from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration import (
    root_tie_rank_conditioned_coherent_topology_join as join,
)


def _target() -> dict[str, object]:
    return {
        "case_id": "target",
        "data_role": "observed_target",
        "calibration_role": "observed_target_not_null_support",
        "proposal_family": "observed_target",
        "root_bandwidth_reopen_band": "bandwidth_root_reopen_observed",
        "root_sibling_selected_ratio": 100.0,
        "root_tie_rank_median_fraction": 0.8,
        "root_edge_path_statistic_margin": 1000.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 5.0,
        "alpha_resolution_required_null_count": 99,
        "tail_precision_required_null_count": 1584,
        "additional_null_count_for_alpha_resolution": 99,
        "additional_null_count_for_tail_precision": 1584,
    }


def _generated(case_id: str = "generated") -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "diagnostic_proposal",
        "calibration_role": "diagnostic_proposal_not_null_support",
        "proposal_family": "conditioned_coherent_rank_one_spike_proposal",
        "root_sibling_selected_ratio": 1000.0,
        "root_tie_rank_median_fraction": 0.9,
        "root_edge_path_statistic_margin": 2000.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 6.0,
        "root_active_feature_count": 8,
        "root_full_eigenvalue_count": 8,
        "root_full_component_eigenvalues_json": "[7.0, 3.0, 1.0]",
        "root_projected_eigenvalues_json": "[7.0, 3.0]",
        "root_mp_upper_bound": 2.25,
    }


def _root_topology(case_id: str = "generated") -> dict[str, object]:
    return {
        "case_id": case_id,
        "data_role": "diagnostic_proposal",
        "candidate_scope": "root_non_direct",
        "bandwidth_reference_reopens": True,
        "bandwidth_reference_direct_positive_reopens": True,
        "root_structural_proxy_pass": False,
        "hybrid_strict_support": False,
        "interpolation_best_case_required_tau_s_for_alpha": 10.0,
    }


def test_join_marks_missing_bandwidth_when_replay_has_no_root_frontier() -> None:
    joined = join.build_conditioned_coherent_joined_feasibility_rows(
        base_feasibility_rows=pd.DataFrame.from_records([_target()]),
        conditioned_generated_rows=pd.DataFrame.from_records([_generated()]),
        conditioned_topology_rows=pd.DataFrame.from_records(
            [
                {
                    **_root_topology(),
                    "candidate_scope": "nonroot_non_direct",
                }
            ]
        ),
    )

    generated = joined.loc[joined["case_id"].eq("generated")].iloc[0]

    assert generated["root_frontier_row_count"] == 0
    assert generated["root_bandwidth_reopen_band"] == "bandwidth_reopen_missing"


def test_joined_root_frontier_feeds_spectral_target_panel() -> None:
    joined = join.build_conditioned_coherent_joined_feasibility_rows(
        base_feasibility_rows=pd.DataFrame.from_records([_target()]),
        conditioned_generated_rows=pd.DataFrame.from_records([_generated()]),
        conditioned_topology_rows=pd.DataFrame.from_records([_root_topology()]),
    )

    target_rows = join.build_selected_spectral_generator_target_rows(joined)
    conditioned = target_rows.loc[
        target_rows["proposal_family"].eq(
            "conditioned_coherent_rank_one_spike_proposal"
        )
    ].iloc[0]

    assert conditioned["eligible_generated_count"] == 1
    assert conditioned["spectral_reach_after_current_generator"]


def test_join_preserves_generated_root_spectrum_capture() -> None:
    joined = join.build_conditioned_coherent_joined_feasibility_rows(
        base_feasibility_rows=pd.DataFrame.from_records([_target()]),
        conditioned_generated_rows=pd.DataFrame.from_records([_generated()]),
        conditioned_topology_rows=pd.DataFrame.from_records([_root_topology()]),
    )

    generated = joined.loc[joined["case_id"].eq("generated")].iloc[0]
    assert generated["root_active_feature_count"] == 8
    assert generated["root_full_eigenvalue_count"] == 8
    assert generated["root_full_component_eigenvalues_json"] == "[7.0, 3.0, 1.0]"
    assert generated["root_projected_eigenvalues_json"] == "[7.0, 3.0]"
    assert generated["root_mp_upper_bound"] == 2.25


def test_join_skips_observed_rows_from_combined_generated_input() -> None:
    joined = join.build_conditioned_coherent_joined_feasibility_rows(
        base_feasibility_rows=pd.DataFrame.from_records([_target()]),
        conditioned_generated_rows=pd.DataFrame.from_records(
            [_target(), _generated()]
        ),
        conditioned_topology_rows=pd.DataFrame.from_records([_root_topology()]),
    )

    assert joined["case_id"].astype(str).tolist().count("target") == 1
    target = joined.loc[joined["case_id"].eq("target")].iloc[0]
    generated = joined.loc[joined["case_id"].eq("generated")].iloc[0]
    assert target["root_bandwidth_reopen_band"] == "bandwidth_root_reopen_observed"
    assert generated["root_bandwidth_reopen_band"] == "bandwidth_root_reopen_observed"


def test_join_enriches_observed_targets_with_root_spectrum_capture() -> None:
    joined = join.build_conditioned_coherent_joined_feasibility_rows(
        base_feasibility_rows=pd.DataFrame.from_records([_target()]),
        conditioned_generated_rows=pd.DataFrame.from_records([_generated()]),
        conditioned_topology_rows=pd.DataFrame.from_records([_root_topology()]),
        observed_root_summary_rows=pd.DataFrame.from_records(
            [
                {
                    "case_id": "target",
                    "root_mp_threshold_rows": 32,
                    "root_active_feature_count": 5,
                    "root_full_component_eigenvalues_json": "[4.0, 2.0, 1.0]",
                    "root_selected_eigenvalue_over_mp_upper_bound": 1.25,
                }
            ]
        ),
    )

    target = joined.loc[joined["case_id"].eq("target")].iloc[0]
    assert target["root_mp_threshold_rows"] == 32
    assert target["root_active_feature_count"] == 5
    assert target["root_full_component_eigenvalues_json"] == "[4.0, 2.0, 1.0]"
    assert target["root_selected_eigenvalue_over_mp_upper_bound"] == 1.25


def test_conditioned_join_runner_writes_outputs(tmp_path: Path) -> None:
    base_path = tmp_path / "base.csv"
    generated_path = tmp_path / "generated.csv"
    topology_path = tmp_path / "topology.csv"
    pd.DataFrame.from_records([_target()]).to_csv(base_path, index=False)
    pd.DataFrame.from_records([_generated()]).to_csv(generated_path, index=False)
    pd.DataFrame.from_records([_root_topology()]).to_csv(topology_path, index=False)

    outputs = join.run_conditioned_coherent_topology_join(
        join.RootTieRankConditionedCoherentTopologyJoinConfig(
            output_dir=tmp_path / "out",
            base_feasibility_rows_path=base_path,
            conditioned_generated_rows_path=generated_path,
            conditioned_topology_rows_path=topology_path,
        )
    )

    assert set(outputs) == {"joined_rows", "rows", "summary", "manifest"}
    assert outputs["joined_rows"].exists()
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
