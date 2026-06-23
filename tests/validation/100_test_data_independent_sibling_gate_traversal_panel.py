from __future__ import annotations

from pathlib import Path

import benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel as traversal_panel
import pandas as pd
from benchmarks.diagnostics.calibration.data_independent_sibling_gate_traversal_panel import (
    DataIndependentSiblingGateTraversalConfig,
    build_traversal_production_components,
    root_feature_subsample_stability,
    run_data_independent_sibling_gate_traversal_panel,
    selected_root_permutation_p_value,
    selected_tree_oracle_cut_ari,
    summarize_root_selective_guard_sensitivity,
    summarize_root_stability_threshold_sensitivity,
    summarize_traversal_rows,
    summarize_traversal_transfer,
)
from scipy.spatial.distance import pdist as scipy_pdist
from tree_break_selection.tree.feature_space import bernoulli_feature_space_from_columns


def test_summarize_traversal_rows_marks_null_and_signal_statuses() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 10.0,
                "sibling_alpha": 0.01,
                "selected_tree_oracle_ari": 1.0,
                "ari": 1.0,
                "found_clusters": 1,
                "exact_cluster_count": True,
                "false_split": False,
                "root_split": False,
                "false_root_split": False,
                "first_split_parent_n": pd.NA,
                "first_split_min_child_n": pd.NA,
                "first_split_child_imbalance": pd.NA,
                "first_split_sibling_p_value": pd.NA,
                "min_split_sibling_p_value": pd.NA,
                "root_selective_p_value": pd.NA,
                "root_stability_subsample_mean_ari": pd.NA,
                "root_stability_subsample_q10_ari": pd.NA,
                "root_stability_guard_blocked": False,
                "split_count": 0,
                "edge_open_count": 10,
                "sibling_open_count": 0,
            },
            {
                "case_id": "unit",
                "data_role": "signal",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 10.0,
                "sibling_alpha": 0.01,
                "selected_tree_oracle_ari": 0.95,
                "ari": 0.9,
                "found_clusters": 2,
                "exact_cluster_count": True,
                "false_split": False,
                "root_split": True,
                "false_root_split": False,
                "first_split_parent_n": 20,
                "first_split_min_child_n": 10,
                "first_split_child_imbalance": 0.0,
                "first_split_sibling_p_value": 0.001,
                "min_split_sibling_p_value": 0.001,
                "root_selective_p_value": pd.NA,
                "root_stability_subsample_mean_ari": pd.NA,
                "root_stability_subsample_q10_ari": pd.NA,
                "root_stability_guard_blocked": False,
                "split_count": 1,
                "edge_open_count": 10,
                "sibling_open_count": 1,
            },
        ]
    )

    summary = summarize_traversal_rows(rows)
    statuses = dict(zip(summary["data_role"], summary["traversal_status"]))

    assert statuses["null"] == "data_independent_traversal_null_candidate"
    assert statuses["signal"] == "data_independent_traversal_signal_retained"
    assert "root_split_rate" in summary.columns
    assert float(
        summary.loc[summary["data_role"].eq("signal"), "root_split_rate"].iloc[0]
    ) == 1.0
    assert summary["root_selective_rejection_rate_at_sibling_alpha"].isna().all()
    assert summary["mean_root_stability_subsample_mean_ari"].isna().all()
    assert set(summary["root_stability_guard_block_rate"]) == {0.0}
    assert set(summary["mean_selected_tree_oracle_ari"]) == {1.0, 0.95}


def test_selected_tree_oracle_cut_ari_scores_true_cluster_cut() -> None:
    data = pd.DataFrame(
        [
            [0, 0],
            [0, 1],
            [10, 10],
            [10, 11],
        ],
        columns=["A", "B"],
    )
    truth = pd.Series([0, 0, 1, 1]).to_numpy()

    assert selected_tree_oracle_cut_ari(data, truth, true_clusters=2) == 1.0


def test_selected_tree_replay_uses_hamming_metric(monkeypatch) -> None:
    calls: list[str] = []

    def recording_pdist(values, metric="euclidean", *args, **kwargs):
        calls.append(str(metric))
        return scipy_pdist(values, metric=metric, *args, **kwargs)

    monkeypatch.setattr(traversal_panel, "pdist", recording_pdist)
    data = pd.DataFrame(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        columns=["A", "B", "C"],
    )
    truth = pd.Series([0, 0, 1, 1]).to_numpy()

    selected_tree_oracle_cut_ari(data, truth, true_clusters=2)

    assert calls == ["hamming"]


def test_selected_root_permutation_p_value_returns_monte_carlo_p_value() -> None:
    data = pd.DataFrame(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        columns=["A", "B", "C"],
    )
    feature_space = bernoulli_feature_space_from_columns(tuple(data.columns))

    result = selected_root_permutation_p_value(
        data,
        feature_space,
        candidate_method="coordinate_bh",
        bootstrap_replicates=2,
        seed=123,
    )

    assert 0.0 <= result["root_observed_p_value"] <= 1.0
    assert 0.0 < result["root_selective_p_value"] <= 1.0
    assert 0.0 <= result["root_selective_null_min_p_value"] <= 1.0


def test_root_feature_subsample_stability_returns_ari_summaries() -> None:
    data = pd.DataFrame(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        columns=["A", "B", "C"],
    )
    feature_space = bernoulli_feature_space_from_columns(tuple(data.columns))

    result = root_feature_subsample_stability(
        data,
        feature_space,
        subsample_replicates=3,
        feature_fraction=2 / 3,
        seed=321,
    )

    assert -1.0 <= result["root_stability_subsample_mean_ari"] <= 1.0
    assert -1.0 <= result["root_stability_subsample_median_ari"] <= 1.0
    assert -1.0 <= result["root_stability_subsample_q10_ari"] <= 1.0


def test_root_feature_subsample_stability_replays_hamming_roots(monkeypatch) -> None:
    calls: list[str] = []

    def recording_pdist(values, metric="euclidean", *args, **kwargs):
        calls.append(str(metric))
        return scipy_pdist(values, metric=metric, *args, **kwargs)

    monkeypatch.setattr(traversal_panel, "pdist", recording_pdist)
    data = pd.DataFrame(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        columns=["A", "B", "C"],
    )
    feature_space = bernoulli_feature_space_from_columns(tuple(data.columns))

    root_feature_subsample_stability(
        data,
        feature_space,
        subsample_replicates=2,
        feature_fraction=2 / 3,
        seed=321,
    )

    assert calls
    assert set(calls) == {"hamming"}


def test_summarize_traversal_transfer_marks_transfer_candidate() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 10.0,
                "n_replicates": 16,
                "mean_ari": 1.0,
                "mean_ari_lower_confidence": 1.0,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.04,
                "false_root_split_rate": 0.0,
                "false_root_split_rate_upper_confidence": 0.04,
                "traversal_status": "data_independent_traversal_null_candidate",
            },
            {
                "case_id": "unit",
                "data_role": "signal",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 10.0,
                "n_replicates": 16,
                "mean_ari": 0.9,
                "mean_ari_lower_confidence": 0.85,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.04,
                "false_root_split_rate": 0.0,
                "false_root_split_rate_upper_confidence": 0.04,
                "traversal_status": "data_independent_traversal_signal_retained",
            },
        ]
    )

    transfer = summarize_traversal_transfer(summary)

    assert transfer.iloc[0]["max_null_false_split_rate"] == 0.0
    assert transfer.iloc[0]["min_signal_mean_ari"] == 0.9
    assert (
        transfer.iloc[0]["data_independent_traversal_transfer_status"]
        == "data_independent_traversal_transfer_candidate"
    )
    assert (
        transfer.iloc[0]["data_independent_traversal_transfer_confidence_status"]
        == "data_independent_traversal_transfer_confidence_candidate"
    )
    assert (
        transfer.iloc[0]["required_zero_false_split_null_replicates_per_case"]
        == 73
    )
    assert (
        transfer.iloc[0]["additional_zero_false_split_null_replicates_per_case"]
        == 57
    )
    assert transfer.iloc[0]["max_observed_null_false_split_count"] == 0
    assert (
        transfer.iloc[0][
            "max_required_null_replicates_given_observed_false_splits"
        ]
        == 73
    )
    assert (
        transfer.iloc[0][
            "max_additional_zero_false_null_replicates_given_observed_false_splits"
        ]
        == 57
    )


def test_summarize_traversal_transfer_sizes_observed_false_support() -> None:
    summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "data_role": "null",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 50.0,
                "n_replicates": 73,
                "mean_ari": 0.98,
                "mean_ari_lower_confidence": 0.95,
                "false_split_rate": 1 / 73,
                "false_split_rate_upper_confidence": 0.073597,
                "false_root_split_rate": 1 / 73,
                "false_root_split_rate_upper_confidence": 0.073597,
                "traversal_status": "data_independent_traversal_null_candidate",
            },
            {
                "case_id": "unit",
                "data_role": "signal",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 50.0,
                "n_replicates": 73,
                "mean_ari": 0.85,
                "mean_ari_lower_confidence": 0.76,
                "false_split_rate": 0.0,
                "false_split_rate_upper_confidence": 0.049992,
                "false_root_split_rate": 0.0,
                "false_root_split_rate_upper_confidence": 0.049992,
                "traversal_status": "data_independent_traversal_signal_retained",
            },
        ]
    )

    transfer = summarize_traversal_transfer(summary)

    assert transfer.iloc[0]["max_observed_null_false_split_count"] == 1
    assert (
        transfer.iloc[0][
            "max_required_null_replicates_given_observed_false_splits"
        ]
        == 110
    )
    assert (
        transfer.iloc[0][
            "max_additional_zero_false_null_replicates_given_observed_false_splits"
        ]
        == 37
    )


def test_build_traversal_production_components_keeps_candidates_diagnostic_only() -> None:
    transfer_summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 500.0,
                "data_independent_traversal_transfer_status": (
                    "data_independent_traversal_transfer_candidate"
                ),
                "data_independent_traversal_transfer_confidence_status": (
                    "data_independent_traversal_transfer_confidence_null_uncertain"
                ),
            },
            {
                "source_family": "categorical_multinomial",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 500.0,
                "data_independent_traversal_transfer_status": (
                    "data_independent_traversal_transfer_null_inflated"
                ),
                "data_independent_traversal_transfer_confidence_status": (
                    "data_independent_traversal_transfer_confidence_signal_uncertain"
                ),
            },
        ]
    )

    components, production_summary = build_traversal_production_components(
        transfer_summary
    )

    decisions = dict(
        zip(
            production_summary["contract_id"],
            production_summary["production_decision"],
        )
    )
    assert components.shape[0] == 4
    assert (
        decisions[
            "data_independent_traversal_transfer:"
            "binary_template:coordinate_bh:penalty=500"
        ]
        == "fail_closed_undefined"
    )
    assert (
        decisions[
            "data_independent_traversal_transfer:"
            "categorical_multinomial:coordinate_bh:penalty=500"
        ]
        == "fail_closed_undefined"
    )


def test_summarize_root_stability_threshold_sensitivity_scores_thresholds() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 50.0,
                "data_role": "null",
                "root_split": True,
                "root_stability_guard_blocked": False,
                "root_stability_subsample_mean_ari": 0.10,
                "false_split": True,
                "ari": 0.0,
            },
            {
                "case_id": "unit",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 50.0,
                "data_role": "signal",
                "root_split": True,
                "root_stability_guard_blocked": False,
                "root_stability_subsample_mean_ari": 0.90,
                "false_split": False,
                "ari": 0.9,
            },
        ]
    )

    summary = summarize_root_stability_threshold_sensitivity(
        rows,
        thresholds=(0.05, 0.20),
    )
    statuses = dict(
        zip(
            summary["sensitivity_threshold"],
            summary["sensitivity_transfer_status"],
        )
    )

    assert statuses[0.05] == "data_independent_traversal_transfer_null_inflated"
    assert statuses[0.20] == "data_independent_traversal_transfer_candidate"


def test_summarize_root_selective_guard_sensitivity_blocks_unselected_roots() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "unit",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 50.0,
                "data_role": "null",
                "sibling_alpha": 0.01,
                "root_split": True,
                "root_stability_guard_blocked": False,
                "root_selective_bootstrap_replicates": 99,
                "root_selective_p_value": 0.02,
                "false_split": True,
                "ari": 0.0,
            },
            {
                "case_id": "unit",
                "source_family": "binary_template",
                "candidate_method": "coordinate_bh",
                "selected_topology_penalty": 50.0,
                "data_role": "signal",
                "sibling_alpha": 0.01,
                "root_split": True,
                "root_stability_guard_blocked": False,
                "root_selective_bootstrap_replicates": 99,
                "root_selective_p_value": 0.01,
                "false_split": False,
                "ari": 0.9,
            },
        ]
    )

    summary = summarize_root_selective_guard_sensitivity(rows)

    assert summary.iloc[0]["max_null_false_split_rate"] == 0.0
    assert summary.iloc[0]["min_signal_mean_ari"] == 0.9
    assert (
        summary.iloc[0]["selected_root_guard_transfer_status"]
        == "data_independent_traversal_transfer_candidate"
    )
    assert summary.iloc[0]["max_selected_root_guard_block_rate"] == 1.0


def test_run_data_independent_sibling_gate_traversal_panel_writes_outputs(
    tmp_path: Path,
) -> None:
    outputs = run_data_independent_sibling_gate_traversal_panel(
        DataIndependentSiblingGateTraversalConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            data_roles=("null", "signal"),
            candidate_methods=("coordinate_bh",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            selected_topology_penalties=(10.0,),
            replicates=1,
            base_seed=20260613,
            root_selective_bootstrap_replicates=1,
            root_stability_subsample_replicates=1,
            root_stability_feature_fraction=0.8,
            root_stability_guard_threshold=-1.0,
        )
    )

    assert set(outputs) == {
        "rows",
        "checkpoint_rows_dir",
        "summary",
        "transfer_summary",
        "root_stability_threshold_sensitivity",
        "root_selective_guard_sensitivity",
        "production_components",
        "production_summary",
        "manifest",
    }
    rows = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_traversal_rows.csv",
        keep_default_na=False,
    )
    summary = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_traversal_summary.csv",
        keep_default_na=False,
    )
    transfer_summary = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_traversal_transfer_summary.csv",
        keep_default_na=False,
    )
    threshold_sensitivity = pd.read_csv(
        tmp_path / "root_stability_threshold_sensitivity.csv",
        keep_default_na=False,
    )
    root_selective_sensitivity = pd.read_csv(
        tmp_path / "root_selective_guard_sensitivity.csv",
        keep_default_na=False,
    )
    production_components = pd.read_csv(
        tmp_path / "production_admissibility_components.csv",
        keep_default_na=False,
    )
    production_summary = pd.read_csv(
        tmp_path / "production_admissibility_summary.csv",
        keep_default_na=False,
    )

    assert not rows.empty
    checkpoint_files = sorted((tmp_path / "checkpoint_rows").glob("*.csv"))
    assert len(checkpoint_files) == 2
    checkpoint_rows = pd.concat(
        [pd.read_csv(path, keep_default_na=False) for path in checkpoint_files],
        ignore_index=True,
    )
    assert checkpoint_rows.shape[0] == rows.shape[0]
    assert set(checkpoint_rows["data_role"]) == {"null", "signal"}
    assert outputs["checkpoint_rows_dir"] == tmp_path / "checkpoint_rows"
    assert set(rows["data_role"]) == {"null", "signal"}
    assert "root_split" in rows.columns
    assert "first_split_sibling_p_value" in rows.columns
    assert "root_selective_p_value" in rows.columns
    assert "root_stability_subsample_mean_ari" in rows.columns
    assert "root_stability_seed" in rows.columns
    assert "root_stability_guard_blocked" in rows.columns
    assert "selected_tree_oracle_ari" in rows.columns
    assert set(pd.to_numeric(rows["root_stability_seed"], errors="coerce")) == {0}
    assert set(rows["tree_distance_metric"]) == {"hamming"}
    assert set(rows["tree_linkage_method"]) == {"average"}
    assert set(rows["root_stability_tree_distance_metric"]) == {"hamming"}
    assert set(rows["root_stability_tree_linkage_method"]) == {"average"}
    assert "mean_selected_tree_oracle_ari" in summary.columns
    root_selective_p = pd.to_numeric(
        rows["root_selective_p_value"],
        errors="coerce",
    )
    root_relevant = rows["root_split"].astype(str).str.lower().isin({"true", "1"})
    root_relevant = root_relevant | rows["root_stability_guard_blocked"].astype(
        str
    ).str.lower().isin({"true", "1"})
    assert pd.to_numeric(rows["root_observed_p_value"], errors="coerce").between(
        0.0,
        1.0,
    ).all()
    assert root_selective_p[root_relevant].between(0.0, 1.0).all()
    assert root_selective_p[~root_relevant].isna().all()
    assert set(summary["candidate_method"]) == {"coordinate_bh"}
    assert set(transfer_summary["selected_topology_penalty"]) == {10.0}
    assert not threshold_sensitivity.empty
    assert not root_selective_sensitivity.empty
    assert not production_components.empty
    assert set(production_summary["production_decision"]) <= {
        "diagnostic_only",
        "fail_closed_undefined",
    }


def test_run_traversal_panel_computes_root_selective_permutation_lazily(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bootstrap_calls: list[int] = []

    def counted_root_selective(*_args, bootstrap_replicates: int, **_kwargs):
        bootstrap_calls.append(int(bootstrap_replicates))
        return {
            "root_observed_p_value": 0.5,
            "root_selective_p_value": pd.NA
            if int(bootstrap_replicates) <= 0
            else 0.5,
            "root_selective_null_min_p_value": pd.NA
            if int(bootstrap_replicates) <= 0
            else 0.5,
            "root_selective_null_q05_p_value": pd.NA
            if int(bootstrap_replicates) <= 0
            else 0.5,
        }

    monkeypatch.setattr(
        traversal_panel,
        "selected_root_permutation_p_value",
        counted_root_selective,
    )

    run_data_independent_sibling_gate_traversal_panel(
        DataIndependentSiblingGateTraversalConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            data_roles=("null", "signal"),
            candidate_methods=("coordinate_bh",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            selected_topology_penalties=(50.0,),
            replicates=1,
            base_seed=20260613,
            root_selective_bootstrap_replicates=5,
            root_stability_subsample_replicates=1,
            root_stability_feature_fraction=0.8,
            root_stability_guard_threshold=0.24,
        )
    )

    rows = pd.read_csv(
        tmp_path / "data_independent_sibling_gate_traversal_rows.csv",
        keep_default_na=False,
    )
    root_relevant = rows["root_split"].astype(str).str.lower().isin({"true", "1"})
    root_relevant = root_relevant | rows["root_stability_guard_blocked"].astype(
        str
    ).str.lower().isin({"true", "1"})

    assert bootstrap_calls.count(0) == rows.shape[0]
    assert bootstrap_calls.count(5) == int(root_relevant.sum())
