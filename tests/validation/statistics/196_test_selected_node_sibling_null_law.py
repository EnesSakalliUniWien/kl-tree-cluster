from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.validation.statistics.selected_node_sibling_null_law import (
    ADAPTIVE_CASE_SUMMARY_NAME,
    ADAPTIVE_DECISIONS_NAME,
    ADAPTIVE_EXTERNAL_AUDIT_NAME,
    ADAPTIVE_MANIFEST_NAME,
    ADAPTIVE_METHOD_SUMMARY_NAME,
    ADAPTIVE_REPORT_NAME,
    CONDITIONS_NAME,
    EXAMPLES_NAME,
    MANIFEST_NAME,
    OCCURRENCES_NAME,
    REPORT_NAME,
    branch_time_multiplier,
    build_adaptive_external_audit,
    build_occurrence_summary,
    build_selected_node_null_conditions,
    build_wald_reaction_examples,
    replay_adaptive_law_benchmark,
    sampling_variance_scale,
    write_selected_node_adaptive_law_replay_artifacts,
    write_selected_node_sibling_null_law_artifacts,
)


def _case_diagnosis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "edge_closed",
                "loss_bucket": "edge_gate_closed_global",
                "null_hypothesis_change": "selected_tree_edge_null_first",
                "full_min_active_sibling_corrected": math.nan,
                "root_active_sibling_corrected_min": math.nan,
                "minimum_diagnostic_sibling_p_value": 0.40,
                "active_vs_diagnostic_p_value_ratio": math.nan,
                "root_branch_length_ratio": 2.0,
                "unstable_subtree_count": 10,
                "large_unstable_subtree_count": 3,
            },
            {
                "case_id": "edge_supported_a",
                "loss_bucket": "sibling_gate_closed_after_edge_open",
                "null_hypothesis_change": "branch_length_conditioned_sibling_null",
                "full_min_active_sibling_corrected": 0.06,
                "root_active_sibling_corrected_min": 0.10,
                "minimum_diagnostic_sibling_p_value": 1e-6,
                "active_vs_diagnostic_p_value_ratio": 60000.0,
                "root_branch_length_ratio": 12.0,
                "unstable_subtree_count": 400,
                "large_unstable_subtree_count": 40,
            },
            {
                "case_id": "edge_supported_b",
                "loss_bucket": "sibling_gate_closed_after_edge_open",
                "null_hypothesis_change": "branch_length_conditioned_sibling_null",
                "full_min_active_sibling_corrected": 0.20,
                "root_active_sibling_corrected_min": 0.15,
                "minimum_diagnostic_sibling_p_value": 1e-9,
                "active_vs_diagnostic_p_value_ratio": 2e8,
                "root_branch_length_ratio": 5.0,
                "unstable_subtree_count": 120,
                "large_unstable_subtree_count": 20,
            },
        ]
    )


def _trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "edge_supported_a",
                "edge_gate_open": True,
                "sibling_p_value_corrected": 0.06,
            },
            {
                "case_id": "edge_supported_a",
                "edge_gate_open": False,
                "sibling_p_value_corrected": math.nan,
            },
            {
                "case_id": "edge_closed",
                "edge_gate_open": False,
                "sibling_p_value_corrected": math.nan,
            },
        ]
    )


def _adaptive_trace() -> pd.DataFrame:
    signature = [f"S{i}" for i in range(10)]
    left = signature[:5]
    right = signature[5:]
    common = {
        "case_id": "unstable_case",
        "test_case": 10,
        "tree_inference": "weighted",
        "run_id": "weighted_r0",
        "trace_type": "full_edge_traversal_trace",
        "left_edge_p_value_bh": 1e-14,
        "right_edge_p_value_bh": 1e-14,
        "sibling_sparse_p_value": 1e-25,
        "sibling_dense_p_value": 1e-25,
        "sibling_fixed_coordinate_bh_p_value": 1e-25,
        "sibling_fixed_global_p_value": 1e-25,
        "left_branch_length": 0.1,
        "right_branch_length": 0.1,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "trace_index": 0,
                "node_id": "R",
                "left_child": "A",
                "right_child": "B",
                "depth": 0,
                "descendant_leaf_signature": str(signature).replace("'", '"'),
                "edge_gate_open": True,
                "sibling_p_value_corrected": 0.15,
            },
            {
                **common,
                "trace_index": 1,
                "node_id": "A",
                "left_child": "",
                "right_child": "",
                "depth": 1,
                "descendant_leaf_signature": str(left).replace("'", '"'),
                "edge_gate_open": False,
                "sibling_p_value_corrected": math.nan,
            },
            {
                **common,
                "trace_index": 2,
                "node_id": "B",
                "left_child": "",
                "right_child": "",
                "depth": 1,
                "descendant_leaf_signature": str(right).replace("'", '"'),
                "edge_gate_open": False,
                "sibling_p_value_corrected": math.nan,
            },
        ]
    )


def _adaptive_case_diagnosis() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "unstable_case",
                "test_case": 10,
                "case_category": "unit",
                "loss_bucket": "sibling_gate_closed_after_edge_open",
                "true_clusters": 2,
                "median_rooted_internal_rf_relative": 0.0,
                "root_branch_length_ratio": 1000.0,
                "large_unstable_subtree_count": 1000,
            }
        ]
    )


def _alpha_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "unstable_case",
                "sibling_alpha": 0.20,
                "best_ari": 0.42,
                "best_nmi": 0.5,
                "best_macro_f1": 0.6,
                "best_found_clusters": 2,
                "min_largest_cluster_fraction": 0.55,
            }
        ]
    )


def test_law_conditions_include_selection_and_multiplicity_layers() -> None:
    conditions = build_selected_node_null_conditions()

    components = set(conditions["law_component"])

    assert {
        "fixed_topology_fixed_subspace",
        "selected_topology_selected_node",
        "nnls_branch_length_scale",
        "local_covariance_eigensystem",
        "multiplicity_projector",
        "adaptive_dimension_mixture",
        "topology_stability_alpha_spending",
    }.issubset(components)
    selected = conditions[conditions["law_component"].eq("selected_topology_selected_node")].iloc[0]
    assert "P_0(W_v >= w_obs | E_sel" in selected["null_object"]
    assert "plain chi_square(k)" in selected["reference_law"]


def test_wald_reaction_examples_show_size_branch_and_multiplicity_effects() -> None:
    examples = build_wald_reaction_examples().set_index("scenario")

    assert branch_time_multiplier(0.20, 0.50) == pytest.approx(1.2)
    assert sampling_variance_scale(20, 160) > sampling_variance_scale(80, 80)

    balanced = examples.loc["balanced_short_branch_same_signal"]
    imbalanced = examples.loc["imbalanced_size_same_signal"]
    long_branch = examples.loc["long_nnls_branch_same_signal"]
    assert balanced["wald_statistic"] > imbalanced["wald_statistic"]
    assert balanced["wald_statistic"] > long_branch["wald_statistic"]

    arbitrary = examples.loc["multiplicity_arbitrary_first_vector"]
    rotated = examples.loc["multiplicity_rotated_first_vector"]
    projector = examples.loc["multiplicity_full_projector"]
    assert arbitrary["wald_statistic"] != pytest.approx(rotated["wald_statistic"])
    assert projector["degrees_of_freedom"] == 3
    assert "orientation invariance" in projector["condition_note"]

    k2 = examples.loc["adaptive_energy_fraction_k2"]
    k3 = examples.loc["same_mass_fixed_k3"]
    assert k3["wald_statistic"] >= k2["wald_statistic"]
    assert k3["chi_square_tail_p_value"] > k2["chi_square_tail_p_value"]


def test_occurrence_summary_identifies_edge_supported_sibling_law() -> None:
    summary = build_occurrence_summary(_case_diagnosis(), _trace())
    by_law = summary.set_index("required_law")

    sibling = by_law.loc["selected_node_sibling_null"]

    assert sibling["case_count"] == 2
    assert sibling["active_sibling_p_min"] == pytest.approx(0.06)
    assert sibling["diagnostic_sibling_p_min"] == pytest.approx(1e-9)
    assert sibling["branch_length_ratio_median"] == pytest.approx(8.5)
    assert sibling["active_sibling_trace_rows"] == 1
    assert "condition on topology" in sibling["interpretation"]


def test_writer_emits_all_selected_node_law_artifacts(tmp_path: Path) -> None:
    case_path = tmp_path / "case_diagnosis.csv"
    trace_path = tmp_path / "trace.csv"
    output_dir = tmp_path / "out"
    _case_diagnosis().to_csv(case_path, index=False)
    _trace().to_csv(trace_path, index=False)

    paths = write_selected_node_sibling_null_law_artifacts(
        output_dir=output_dir,
        case_diagnosis_path=case_path,
        traversal_trace_path=trace_path,
        created_utc="20260709_000000Z",
    )

    assert set(paths) == {"conditions", "examples", "occurrences", "report", "manifest"}
    for filename in [
        CONDITIONS_NAME,
        EXAMPLES_NAME,
        OCCURRENCES_NAME,
        REPORT_NAME,
        MANIFEST_NAME,
    ]:
        assert (output_dir / filename).exists()

    report = (output_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "p_sel(v)" in report
    assert "topology, NNLS branch-time scale, local covariance chart" in report
    assert "Selective Inference for Hierarchical Clustering" in report
    assert "## Requirement Coverage" in report


def test_adaptive_law_replay_blocks_unstable_topology_but_ablation_opens() -> None:
    _decisions, _cells, cases = replay_adaptive_law_benchmark(
        _adaptive_trace(),
        _adaptive_case_diagnosis(),
    )
    by_policy = cases.set_index("policy_id")

    assert (
        by_policy.loc[
            "strict_selected_law_required",
            "replay_status",
        ]
        == "unsupported_missing_exact_selected_node_p_value"
    )
    assert by_policy.loc["adaptive_topology_branch_spending_proxy", "split_cells"] == 0
    assert by_policy.loc["adaptive_no_stability_ablation", "split_cells"] == 1
    assert (
        by_policy.loc[
            "adaptive_topology_branch_spending_proxy",
            "max_local_alpha",
        ]
        < by_policy.loc["adaptive_no_stability_ablation", "max_local_alpha"]
    )


def test_adaptive_external_audit_joins_candidate_split_to_alpha_sweep() -> None:
    _decisions, _cells, cases = replay_adaptive_law_benchmark(
        _adaptive_trace(),
        _adaptive_case_diagnosis(),
    )
    audit = build_adaptive_external_audit(cases, _alpha_sweep())
    ablation = audit[
        audit["policy_id"].eq("adaptive_no_stability_ablation")
        & audit["case_id"].eq("unstable_case")
    ].iloc[0]

    assert ablation["external_audit_status"] == "nearest_available_rerun"
    assert ablation["nearest_sibling_alpha"] == pytest.approx(0.20)
    assert ablation["best_ari"] == pytest.approx(0.42)


def test_adaptive_replay_writer_emits_benchmark_outputs(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.csv"
    case_path = tmp_path / "case.csv"
    alpha_path = tmp_path / "alpha.csv"
    output_dir = tmp_path / "adaptive"
    _adaptive_trace().to_csv(trace_path, index=False)
    _adaptive_case_diagnosis().to_csv(case_path, index=False)
    _alpha_sweep().to_csv(alpha_path, index=False)

    paths = write_selected_node_adaptive_law_replay_artifacts(
        output_dir=output_dir,
        traversal_trace_path=trace_path,
        case_diagnosis_path=case_path,
        alpha_sweep_summary_path=alpha_path,
        created_utc="20260709_000000Z",
    )

    assert set(paths) == {
        "decisions",
        "cell_summary",
        "case_summary",
        "method_summary",
        "external_audit",
        "report",
        "manifest",
    }
    for filename in [
        ADAPTIVE_DECISIONS_NAME,
        ADAPTIVE_CASE_SUMMARY_NAME,
        ADAPTIVE_METHOD_SUMMARY_NAME,
        ADAPTIVE_EXTERNAL_AUDIT_NAME,
        ADAPTIVE_REPORT_NAME,
        ADAPTIVE_MANIFEST_NAME,
    ]:
        assert (output_dir / filename).exists()

    report = (output_dir / ADAPTIVE_REPORT_NAME).read_text(encoding="utf-8")
    assert "Strict selected conditional law required" in report
    assert "diagnostic proxies" in report
