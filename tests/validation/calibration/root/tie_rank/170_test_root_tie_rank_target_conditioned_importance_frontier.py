from __future__ import annotations

import pandas as pd
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_target_conditioned_importance_frontier as frontier,
)


def _target(case_id: str = "target") -> dict[str, object]:
    return {
        "case_id": case_id,
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": 100.0,
        "root_edge_path_statistic_margin": 200.0,
        "root_selected_eigenvalue_over_mp_upper_bound": 4.0,
    }


def _generated(
    *,
    case_id: str,
    conditioning_target_case_id: str = "target",
    selected_ratio: float = 100.0,
    edge_margin: float = 200.0,
    spectral_ratio: float = 2.0,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "base_case_id": "target",
        "data_role": "external_selected_null",
        "calibration_role": "external_null_support",
        "proposal_family": "importance_two_block_external_null",
        "conditioning_target_case_id": conditioning_target_case_id,
        "root_mixed_region_component": "discrete_tie_rank_region",
        "root_tie_rank_median_fraction": 0.8,
        "root_sibling_selected_ratio": selected_ratio,
        "root_edge_path_statistic_margin": edge_margin,
        "root_selected_eigenvalue_over_mp_upper_bound": spectral_ratio,
        "proposal_two_block_delta": 0.1,
        "proposal_spike_feature_fraction": float("nan"),
        "proposal_spike_delta": float("nan"),
        "proposal_block_fraction": float("nan"),
        "importance_log_weight": -10.0,
    }


def test_pre_topology_stratum_key_ignores_bandwidth_and_spectral_tail() -> None:
    first = _target()
    second = {
        **first,
        "root_selected_eigenvalue_over_mp_upper_bound": 99.0,
        "root_bandwidth_reopen_band": "bandwidth_no_root_reopen",
    }

    assert frontier._pre_topology_stratum_key(first) == (frontier._pre_topology_stratum_key(second))


def test_target_rows_keep_only_matching_target_conditioned_candidates() -> None:
    observed = pd.DataFrame.from_records([_target("target"), _target("other")])
    generated = pd.DataFrame.from_records(
        [
            _generated(case_id="hit", spectral_ratio=3.0),
            _generated(case_id="miss_band", selected_ratio=1000.0),
            _generated(
                case_id="wrong_target",
                conditioning_target_case_id="other",
                spectral_ratio=5.0,
            ),
        ]
    )

    rows = frontier.build_target_rows(
        observed_mixed=observed,
        generated_mixed_rows=generated,
        target_case_ids=("target",),
    )

    assert rows["pre_topology_stratum_hit_count"].sum() == 1
    hit = rows.loc[rows["pre_topology_stratum_hit_count"].eq(1)].iloc[0]
    assert hit["best_candidate_case_id"] == "hit"
    assert hit["best_candidate_spectral_ratio"] == 3.0
    assert hit["target_conditioning_status"] == ("pre_topology_stratum_hit_replay_needed")


def test_summary_reports_partial_pre_topology_support() -> None:
    target_rows = pd.DataFrame.from_records(
        [
            {
                "target_case_id": "a",
                "pre_topology_stratum_hit_count": 1,
            },
            {
                "target_case_id": "b",
                "pre_topology_stratum_hit_count": 0,
            },
        ]
    )

    summary = frontier.summarize_target_rows(
        target_rows=target_rows,
        generated_mixed_rows=pd.DataFrame.from_records([_generated(case_id="hit")]),
    ).iloc[0]

    assert summary["target_count"] == 2
    assert summary["pre_topology_supported_target_count"] == 1
    assert summary["summary_status"] == "partial_pre_topology_candidate_support"


def test_correlated_two_factor_importance_generator_records_likelihood_weight() -> None:
    matrix, metadata = frontier.generate_correlated_two_factor_importance_matrix(
        base_case={
            "n_samples": 24,
            "n_features": 12,
            "binary_probability": 0.3,
        },
        seed=123,
        two_block_delta=0.05,
        spike_feature_fraction=0.25,
        spike_delta=0.08,
        factor_correlation=0.5,
    )

    assert matrix.shape == (24, 12)
    assert metadata["proposal_spike_feature_count"] == 3
    assert metadata["proposal_block_fraction"] == 0.5
    assert metadata["importance_log_weight"] == (
        metadata["target_null_log_probability"] - metadata["proposal_log_probability"]
    )
    assert metadata["importance_law_status"] == (
        "target_iid_bernoulli_over_correlated_two_factor_tilted_proposal"
    )


def test_correlated_two_factor_settings_cross_factor_and_spike_grids() -> None:
    settings = frontier._setting_records(
        frontier.TargetConditionedImportanceFrontierConfig(
            output_dir="unused",
            proposal_families=(frontier.IMPORTANCE_CORRELATED_TWO_FACTOR_EXTERNAL_NULL,),
            two_block_delta_grid=(0.1,),
            spike_feature_fraction_grid=(0.2, 0.4),
            spike_delta_grid=(0.3,),
            block_fraction_grid=(0.25, 0.75),
        )
    )

    assert len(settings) == 4
    assert {row["proposal_block_fraction"] for row in settings} == {0.25, 0.75}
    assert {row["proposal_spike_feature_fraction"] for row in settings} == {0.2, 0.4}
