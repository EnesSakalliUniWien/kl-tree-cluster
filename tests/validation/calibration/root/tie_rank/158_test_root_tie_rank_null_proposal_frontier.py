from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.root.tie_rank import (
    root_tie_rank_null_proposal_frontier as panel,
)
from benchmarks.diagnostics.calibration.root.tie_rank.root_tie_rank_calibration_feasibility import (
    build_root_tie_rank_calibration_feasibility_rows,
)


def _frontier_mixed_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "case_id": "target",
                "base_case_id": "target",
                "data_role": "observed_target",
                "calibration_role": "observed_target_not_null_support",
                "proposal_family": "observed_target",
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 10.0,
                "root_tie_rank_median_fraction": 0.82,
                "root_edge_path_statistic_margin": 1500.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.0,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "iid_hit",
                "base_case_id": "target",
                "data_role": "selected_null",
                "calibration_role": "selected_null_candidate_support",
                "proposal_family": panel.IID_MARGINAL_BERNOULLI,
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 8.0,
                "root_tie_rank_median_fraction": 0.81,
                "root_edge_path_statistic_margin": 1200.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.5,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "proposal_hit",
                "base_case_id": "target",
                "data_role": "diagnostic_proposal",
                "calibration_role": "diagnostic_proposal_not_null_support",
                "proposal_family": panel.COLUMN_BETA_BERNOULLI,
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 12.0,
                "root_tie_rank_median_fraction": 0.84,
                "root_edge_path_statistic_margin": 1600.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 3.8,
                "root_bandwidth_reopen_count": 1,
            },
            {
                "case_id": "proposal_miss",
                "base_case_id": "target",
                "data_role": "diagnostic_proposal",
                "calibration_role": "diagnostic_proposal_not_null_support",
                "proposal_family": panel.TWO_BLOCK_TILT_PROPOSAL,
                "root_mixed_region_component": "discrete_tie_rank_region",
                "root_sibling_selected_ratio": 30.0,
                "root_tie_rank_median_fraction": 0.95,
                "root_edge_path_statistic_margin": 80.0,
                "root_selected_eigenvalue_over_mp_upper_bound": 1.5,
                "root_bandwidth_reopen_count": 0,
            },
        ]
    )


def test_proposal_matrix_generators_preserve_roles_and_shapes() -> None:
    base_case = {
        "name": "overlap_probe",
        "n_samples": 20,
        "n_features": 40,
        "n_clusters": 4,
        "feature_sparsity": 0.05,
    }

    iid, iid_metadata = panel.generate_binary_proposal_matrix(
        base_case=base_case,
        proposal_family=panel.IID_MARGINAL_BERNOULLI,
        seed=11,
    )
    beta, beta_metadata = panel.generate_binary_proposal_matrix(
        base_case=base_case,
        proposal_family=panel.COLUMN_BETA_BERNOULLI,
        seed=12,
        beta_concentration=4.0,
    )
    tilted, tilted_metadata = panel.generate_binary_proposal_matrix(
        base_case=base_case,
        proposal_family=panel.TWO_BLOCK_TILT_PROPOSAL,
        seed=13,
        two_block_delta=0.2,
    )
    sparse, sparse_metadata = panel.generate_binary_proposal_matrix(
        base_case=base_case,
        proposal_family=panel.SPARSE_BLOCK_SPIKE_PROPOSAL,
        seed=14,
        spike_feature_fraction=0.10,
        spike_delta=0.3,
    )
    coupled, coupled_metadata = panel.generate_binary_proposal_matrix(
        base_case=base_case,
        proposal_family=panel.COUPLED_EDGE_SPECTRAL_PROPOSAL,
        seed=15,
        two_block_delta=0.2,
        spike_feature_fraction=0.10,
        spike_delta=0.3,
    )

    assert iid.shape == (20, 40)
    assert beta.shape == (20, 40)
    assert tilted.shape == (20, 40)
    assert sparse.shape == (20, 40)
    assert coupled.shape == (20, 40)
    assert iid_metadata["null_feature_probability"] == pytest.approx(0.275)
    assert (
        beta_metadata["generated_feature_probability_max"]
        > beta_metadata["generated_feature_probability_min"]
    )
    assert tilted_metadata["proposal_two_block_delta"] == pytest.approx(0.2)
    assert sparse_metadata["proposal_spike_feature_count"] == 4
    assert sparse_metadata["proposal_spike_delta"] == pytest.approx(0.3)
    assert coupled_metadata["proposal_two_block_delta"] == pytest.approx(0.2)
    assert coupled_metadata["proposal_spike_feature_count"] == 4
    assert coupled_metadata["proposal_spike_delta"] == pytest.approx(0.3)
    assert iid.sum(axis=0).min() >= 1
    assert beta.sum(axis=0).min() >= 1
    assert tilted.sum(axis=0).min() >= 1
    assert sparse.sum(axis=0).min() >= 1
    assert coupled.sum(axis=0).min() >= 1


def test_frontier_counts_proposal_hits_without_counting_them_as_null_support() -> None:
    mixed = _frontier_mixed_rows()
    feasibility = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=mixed,
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    annotated = panel._annotated_feasibility_rows(
        feasibility_rows=feasibility,
        combined_mixed_rows=mixed,
    )
    target = annotated[annotated["case_id"].eq("target")].iloc[0]

    assert target["stratum_calibration_null_support_count"] == 1

    frontier = panel.build_target_frontier_rows(
        observed_case_ids=("target",),
        annotated_feasibility_rows=annotated,
        proposal_families=(
            panel.IID_MARGINAL_BERNOULLI,
            panel.COLUMN_BETA_BERNOULLI,
            panel.TWO_BLOCK_TILT_PROPOSAL,
        ),
    )
    by_family = frontier.set_index("proposal_family")

    assert (
        by_family.loc[
            panel.IID_MARGINAL_BERNOULLI,
            "generated_calibration_support_count",
        ]
        == 1
    )
    assert (
        by_family.loc[
            panel.IID_MARGINAL_BERNOULLI,
            "frontier_hit_status",
        ]
        == "target_stratum_hit_by_calibration_support"
    )
    assert by_family.loc[panel.COLUMN_BETA_BERNOULLI, "generated_row_count"] == 1
    assert (
        by_family.loc[
            panel.COLUMN_BETA_BERNOULLI,
            "generated_calibration_support_count",
        ]
        == 0
    )
    assert (
        by_family.loc[
            panel.COLUMN_BETA_BERNOULLI,
            "generated_exceedance_count",
        ]
        == 1
    )
    assert (
        by_family.loc[
            panel.COLUMN_BETA_BERNOULLI,
            "frontier_hit_status",
        ]
        == "target_stratum_hit_by_diagnostic_proposal_only"
    )
    assert (
        by_family.loc[
            panel.TWO_BLOCK_TILT_PROPOSAL,
            "frontier_hit_status",
        ]
        == "no_target_stratum_hit"
    )


def test_proposal_summary_separates_calibration_and_diagnostic_frontier_hits() -> None:
    mixed = _frontier_mixed_rows()
    feasibility = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=mixed,
        target_alpha=0.25,
        relative_se_target=1.0,
    )
    annotated = panel._annotated_feasibility_rows(
        feasibility_rows=feasibility,
        combined_mixed_rows=mixed,
    )
    families = (
        panel.IID_MARGINAL_BERNOULLI,
        panel.COLUMN_BETA_BERNOULLI,
        panel.TWO_BLOCK_TILT_PROPOSAL,
    )
    frontier = panel.build_target_frontier_rows(
        observed_case_ids=("target",),
        annotated_feasibility_rows=annotated,
        proposal_families=families,
    )
    summary = panel.summarize_proposal_frontier(
        annotated_feasibility_rows=annotated,
        target_frontier_rows=frontier,
        observed_case_ids=("target",),
        proposal_families=families,
    ).set_index("proposal_family")

    assert (
        summary.loc[
            panel.IID_MARGINAL_BERNOULLI,
            "summary_status",
        ]
        == "calibration_candidate_hits_observed_target_strata"
    )
    assert (
        summary.loc[
            panel.COLUMN_BETA_BERNOULLI,
            "summary_status",
        ]
        == "diagnostic_proposal_hits_observed_target_strata_not_calibration"
    )
    assert (
        summary.loc[
            panel.TWO_BLOCK_TILT_PROPOSAL,
            "summary_status",
        ]
        == "generated_no_observed_target_stratum_hits"
    )


def test_annotated_feasibility_preserves_root_spectral_bulk_metadata() -> None:
    mixed = _frontier_mixed_rows()
    support_mask = mixed["case_id"].eq("iid_hit")
    mixed.loc[support_mask, "root_active_feature_count"] = 8
    mixed.loc[support_mask, "root_full_eigenvalue_count"] = 8
    mixed.loc[support_mask, "root_full_component_eigenvalues_json"] = "[3.0, 2.0, 1.0]"
    mixed.loc[support_mask, "root_projected_eigenvalues_json"] = "[3.0, 2.0]"
    mixed.loc[support_mask, "root_mp_upper_bound"] = 2.25
    mixed.loc[support_mask, "root_raw_mp_signal_count"] = 1
    mixed.loc[support_mask, "root_mp_threshold_rows"] = 32
    feasibility = build_root_tie_rank_calibration_feasibility_rows(
        mixed_region_rows=mixed,
        target_alpha=0.25,
        relative_se_target=1.0,
    )

    annotated = panel._annotated_feasibility_rows(
        feasibility_rows=feasibility,
        combined_mixed_rows=mixed,
    )

    row = annotated.loc[annotated["case_id"].eq("iid_hit")].iloc[0]
    assert row["root_active_feature_count"] == 8
    assert row["root_full_eigenvalue_count"] == 8
    assert row["root_full_component_eigenvalues_json"] == "[3.0, 2.0, 1.0]"
    assert row["root_projected_eigenvalues_json"] == "[3.0, 2.0]"
    assert row["root_mp_upper_bound"] == 2.25
    assert row["root_raw_mp_signal_count"] == 1
    assert row["root_mp_threshold_rows"] == 32


def test_build_proposal_mixed_rows_passes_optional_topology_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "generated",
                "base_case_id": "base",
                "data_role": "diagnostic_proposal",
                "calibration_role": "diagnostic_proposal_not_null_support",
                "replicate": 0,
                "proposal_family": panel.COLUMN_BETA_BERNOULLI,
                "root_full_component_eigenvalues_json": "[2.0, 1.0]",
            }
        ]
    )
    merge_margins = pd.DataFrame.from_records([{"case_id": "generated"}])
    topology = pd.DataFrame.from_records(
        [{"case_id": "generated", "candidate_scope": "direct_measurable"}]
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        panel,
        "build_root_selected_tie_cell_burden_rows",
        lambda *, root_summary, merge_margins: pd.DataFrame.from_records(
            [{"case_id": "generated"}]
        ),
    )

    def fake_mixed_rows(
        *,
        root_summary: pd.DataFrame,
        tie_cell_burden_rows: pd.DataFrame,
        topology_frontier_rows: pd.DataFrame | None,
    ) -> pd.DataFrame:
        captured["topology"] = topology_frontier_rows
        return pd.DataFrame.from_records(
            [
                {
                    "case_id": "generated",
                    "root_frontier_row_count": 1,
                    "root_full_component_eigenvalues_json": "[2.0, 1.0]",
                }
            ]
        )

    monkeypatch.setattr(
        panel,
        "build_root_selected_mixed_region_law_rows",
        fake_mixed_rows,
    )

    _tie_rows, mixed_rows = panel.build_proposal_mixed_rows(
        root_rows=root_rows,
        merge_margins=merge_margins,
        topology_frontier_rows=topology,
    )

    assert captured["topology"] is topology
    assert mixed_rows.iloc[0]["root_frontier_row_count"] == 1
    assert mixed_rows.iloc[0]["proposal_family"] == panel.COLUMN_BETA_BERNOULLI
    assert mixed_rows.iloc[0]["root_full_component_eigenvalues_json"] == ("[2.0, 1.0]")
    assert not any(column.endswith("_x") or column.endswith("_y") for column in mixed_rows)


def test_runner_writes_outputs_with_stubbed_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_evaluate(
        _config: panel.RootTieRankNullProposalFrontierConfig,
    ) -> dict[str, pd.DataFrame]:
        frame = pd.DataFrame.from_records([{"case_id": "x"}])
        return {
            "root_rows": frame,
            "merge_margins": frame,
            "tie_rows": frame,
            "mixed_rows": frame,
            "combined_feasibility_rows": frame,
            "combined_feasibility_strata": frame,
            "combined_feasibility_summary": frame,
            "target_support": frame,
            "target_frontier": frame,
            "proposal_summary": frame,
            "failures": pd.DataFrame(),
        }

    monkeypatch.setattr(
        panel,
        "evaluate_root_tie_rank_null_proposal_frontier",
        fake_evaluate,
    )
    outputs = panel.run_root_tie_rank_null_proposal_frontier(
        panel.RootTieRankNullProposalFrontierConfig(output_dir=tmp_path / "out")
    )

    assert outputs["root_rows"].exists()
    assert outputs["target_frontier"].exists()
    assert outputs["proposal_summary"].exists()
    assert outputs["manifest"].exists()
