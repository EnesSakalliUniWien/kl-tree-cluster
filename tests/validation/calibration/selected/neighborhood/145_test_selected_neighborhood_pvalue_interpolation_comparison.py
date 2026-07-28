from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.overlap.overlap_conditional_topology_law_panel import (
    build_cached_tree_distances,
)
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_pvalue_interpolation_comparison import (
    build_pvalue_interpolation_comparison_rows,
    compute_holdout_interpolated_p_like,
    run_pvalue_interpolation_comparison,
    summarize_pvalue_interpolation_comparison,
    summarize_region_bandwidths,
    summarize_region_tau_s_ranges,
    summarize_tau_s_range,
    summarize_tau_s_sensitivity,
)


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "case_id": "case",
        "data_role": "signal",
        "method_id": "method",
        "replicate": 0,
        "node_id": "node",
        "parent_id": "root",
        "decision_class": "stable_boundary",
        "traversal_decision": "boundary",
        "sibling_p_value": 0.50,
        "sibling_open": False,
        "sibling_projection_dimension": 2.0,
        "neighborhood_scale": 2.0,
        "topology_support_role": "",
        "topology_signal_role": "",
        "guard_truth_role": "",
    }
    row.update(overrides)
    return row


def test_holdout_interpolation_can_lower_signal_candidate_p_like() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="target", sibling_p_value=0.05),
            _row(
                node_id="signal_anchor",
                sibling_p_value=0.0,
                sibling_open=True,
                decision_class="accepted_internal_split",
                traversal_decision="split",
            ),
            _row(node_id="support_a", sibling_p_value=0.60),
            _row(node_id="support_b", sibling_p_value=0.50),
        ]
    )

    comparison = build_pvalue_interpolation_comparison_rows(
        rows,
        alpha=0.01,
        fallback_tau_s=400.0,
    )
    target = comparison.loc[comparison["node_id"].eq("target")].iloc[0]

    assert target["comparison_class"] == "interpolation_lower_than_direct"
    assert target["behavior_label"] == "signal_extra_open_candidate"
    assert target["interpolated_sibling_null_p_like"] < 0.01
    assert target["signal_anchor_count"] == 1
    assert target["best_case_required_tau_s_for_alpha"] > 0.0


def test_holdout_excludes_target_direct_p_from_signal_anchor() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="target",
                sibling_p_value=0.001,
                sibling_open=True,
                decision_class="accepted_internal_split",
                traversal_decision="split",
            ),
            _row(node_id="support_a", sibling_p_value=0.60),
            _row(node_id="support_b", sibling_p_value=0.50),
        ]
    )

    comparison = build_pvalue_interpolation_comparison_rows(rows, alpha=0.01)
    target = comparison.loc[comparison["node_id"].eq("target")].iloc[0]

    assert target["interpolation_status"] == (
        "interpolated_p_like_no_signal_attenuation_diagnostic_only"
    )
    assert target["signal_anchor_count"] == 0
    assert target["behavior_label"] == "signal_not_caught_by_interpolation"
    assert target["interpolated_sibling_null_p_like"] > 0.10


def test_selected_null_without_signal_anchor_stays_conservative() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(data_role="selected_null", node_id="target", sibling_p_value=0.40),
            _row(data_role="selected_null", node_id="support_a", sibling_p_value=0.60),
            _row(data_role="selected_null", node_id="support_b", sibling_p_value=0.50),
        ]
    )

    comparison = build_pvalue_interpolation_comparison_rows(rows, alpha=0.01)
    target = comparison.loc[comparison["node_id"].eq("target")].iloc[0]

    assert target["behavior_label"] == "selected_null_conservative"
    assert not bool(target["interpolated_significant"])
    assert target["interpolated_sibling_null_p_like"] > 0.10


def test_explicit_selected_nonnull_support_is_excluded() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="target", topology_support_role="selected_nonnull"),
            _row(
                node_id="nonnull_a",
                topology_support_role="selected_nonnull",
                sibling_p_value=0.50,
            ),
            _row(
                node_id="signal_anchor",
                topology_signal_role="signal",
                sibling_p_value=0.001,
            ),
        ]
    )

    comparison = build_pvalue_interpolation_comparison_rows(rows, min_support=1)
    target = comparison.loc[comparison["node_id"].eq("target")].iloc[0]

    assert target["interpolation_status"] == "support_bottleneck"
    assert target["selected_nonnull_excluded_count"] == 2
    assert target["support_anchor_count"] == 0


def test_compute_result_reports_tree_and_weight_details() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(node_id="target", sibling_p_value=0.05),
            _row(node_id="support_a", sibling_p_value=0.60),
            _row(node_id="support_b", sibling_p_value=0.50),
        ]
    )
    cache = build_cached_tree_distances(rows)
    support_mask = rows["node_id"].isin(["target", "support_a", "support_b"])
    signal_mask = pd.Series(False, index=rows.index)

    result = compute_holdout_interpolated_p_like(
        group=rows,
        target_index=0,
        cache=cache,
        support_mask=support_mask,
        signal_mask=signal_mask,
    )

    assert result.status == "interpolated_p_like_no_signal_attenuation_diagnostic_only"
    assert result.support_anchor_count == 2
    assert result.support_weight > 0.0
    assert result.effective_support == pytest.approx(2.0)
    assert result.stable_weighted_p_mean == pytest.approx(0.55)
    assert result.nearest_support_distance == pytest.approx(2.0)

    comparison = build_pvalue_interpolation_comparison_rows(rows)
    target = comparison.loc[comparison["node_id"].eq("target")].iloc[0]
    assert target["interpolated_sibling_null_p_like"] == pytest.approx(result.p_like)
    assert target["effective_support"] == pytest.approx(result.effective_support)
    assert target["stable_weighted_p_mean"] == pytest.approx(result.stable_weighted_p_mean)


def test_holdout_interpolation_uses_branch_length_tree_distances() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="target",
                sibling_p_value=0.05,
                branch_length_to_parent=0.25,
            ),
            _row(
                node_id="support_a",
                sibling_p_value=0.60,
                branch_length_to_parent=1.75,
            ),
            _row(
                node_id="support_b",
                sibling_p_value=0.50,
                branch_length_to_parent=2.75,
            ),
        ]
    )
    cache = build_cached_tree_distances(rows)
    support_mask = rows["node_id"].isin(["target", "support_a", "support_b"])
    signal_mask = pd.Series(False, index=rows.index)

    result = compute_holdout_interpolated_p_like(
        group=rows,
        target_index=0,
        cache=cache,
        support_mask=support_mask,
        signal_mask=signal_mask,
    )

    assert result.tree_distance_status == ("cached_all_pairs_branch_length_tree_distances")
    assert result.nearest_support_distance == pytest.approx(2.0)
    assert result.stable_weighted_p_mean != pytest.approx(0.55)

    comparison = build_pvalue_interpolation_comparison_rows(rows)
    target = comparison.loc[comparison["node_id"].eq("target")].iloc[0]
    assert target["branch_length_to_parent"] == pytest.approx(0.25)
    assert target["tree_distance_status"] == ("cached_all_pairs_branch_length_tree_distances")
    assert target["nearest_support_distance"] == pytest.approx(2.0)


def test_tau_s_sensitivity_reports_recovery_and_false_open_tradeoff() -> None:
    rows = build_pvalue_interpolation_comparison_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    data_role="signal",
                    node_id="signal_target",
                    sibling_p_value=0.001,
                    sibling_open=True,
                    decision_class="accepted_internal_split",
                    traversal_decision="split",
                ),
                _row(
                    data_role="signal",
                    node_id="signal_anchor",
                    sibling_p_value=0.0,
                    sibling_open=True,
                    decision_class="accepted_internal_split",
                    traversal_decision="split",
                ),
                _row(data_role="signal", node_id="support_a", sibling_p_value=0.60),
                _row(data_role="signal", node_id="support_b", sibling_p_value=0.50),
                _row(
                    data_role="selected_null",
                    node_id="null_target",
                    sibling_p_value=0.001,
                    sibling_open=True,
                    decision_class="accepted_internal_split",
                    traversal_decision="split",
                ),
                _row(data_role="selected_null", node_id="support_c", sibling_p_value=0.60),
                _row(data_role="selected_null", node_id="support_d", sibling_p_value=0.50),
            ]
        )
    )

    sensitivity = summarize_tau_s_sensitivity(rows, thresholds=(100.0,))

    assert set(sensitivity["sensitivity_label"]) == {
        "signal_recovered_best_case",
        "selected_null_reopened_best_case",
    }
    assert sensitivity["direct_significant_count"].sum() > 0


def test_tau_s_range_reports_empty_interval_when_null_reopens_first() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "data_role": "signal",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 20.0,
            },
            {
                "data_role": "signal",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 30.0,
            },
            {
                "data_role": "selected_null",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 10.0,
            },
            {
                "data_role": "selected_null",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 12.0,
            },
        ]
    )

    ranges = summarize_tau_s_range(
        rows,
        target_signal_fractions=(0.5,),
        max_selected_null_fractions=(0.5,),
    )
    row = ranges.iloc[0]

    assert row["signal_tau_s_lower_bound"] == pytest.approx(25.0)
    assert row["selected_null_tau_s_upper_bound"] == pytest.approx(11.0)
    assert row["tau_s_range_status"] == ("tau_s_range_empty_selected_null_reopens_first")


def test_tau_s_range_reports_admissible_interval() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "data_role": "signal",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 5.0,
            },
            {
                "data_role": "signal",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 7.0,
            },
            {
                "data_role": "selected_null",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 20.0,
            },
            {
                "data_role": "selected_null",
                "method_id": "method",
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 22.0,
            },
        ]
    )

    ranges = summarize_tau_s_range(
        rows,
        target_signal_fractions=(0.5,),
        max_selected_null_fractions=(0.5,),
    )
    row = ranges.iloc[0]

    assert row["signal_tau_s_lower_bound"] == pytest.approx(6.0)
    assert row["selected_null_tau_s_upper_bound"] == pytest.approx(21.0)
    assert row["admissible_tau_s_width"] == pytest.approx(15.0)
    assert row["tau_s_range_status"] == "tau_s_range_admissible_diagnostic"


def test_region_bandwidths_report_observed_tau_vector() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "direct_significant": True,
                "interpolation_status": ("interpolated_p_like_observed_diagnostic_only"),
                "interpolated_significant": False,
                "tau_b": 2.0,
                "tau_t": 3.0,
                "tau_s": 5.0,
                "h_k": 7.0,
                "distance_to_stopping_edge": 2.0,
                "best_case_required_tau_s_for_alpha": 11.0,
                "effective_support": 2.0,
                "nearest_support_distance": 1.0,
                "nearest_signal_distance": 4.0,
            },
            {
                "case_id": "case_a",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "direct_significant": False,
                "interpolation_status": (
                    "interpolated_p_like_no_signal_attenuation_diagnostic_only"
                ),
                "interpolated_significant": False,
                "tau_b": 4.0,
                "tau_t": 9.0,
                "tau_s": 15.0,
                "h_k": 21.0,
                "distance_to_stopping_edge": 4.0,
                "best_case_required_tau_s_for_alpha": 13.0,
                "effective_support": 3.0,
                "nearest_support_distance": 2.0,
                "nearest_signal_distance": 6.0,
            },
        ]
    )

    summary = summarize_region_bandwidths(rows)
    row = summary.iloc[0]

    assert row["tau_b_median"] == pytest.approx(3.0)
    assert row["tau_t_median"] == pytest.approx(6.0)
    assert row["tau_s_median"] == pytest.approx(10.0)
    assert row["h_k_median"] == pytest.approx(14.0)
    assert row["region_bandwidth_status"] == ("observed_full_bandwidth_vector_diagnostic")


def test_region_tau_s_ranges_are_estimated_per_topology_region() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "case_a",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 6.0,
                "tau_b": 2.0,
                "tau_t": 3.0,
                "tau_s": 5.0,
                "h_k": 7.0,
            },
            {
                "case_id": "case_a",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 8.0,
                "tau_b": 4.0,
                "tau_t": 5.0,
                "tau_s": 9.0,
                "h_k": 11.0,
            },
            {
                "case_id": "case_a",
                "data_role": "selected_null",
                "method_id": "method",
                "replicate": 0,
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 20.0,
                "tau_b": 10.0,
                "tau_t": 12.0,
                "tau_s": 14.0,
                "h_k": 16.0,
            },
            {
                "case_id": "case_a",
                "data_role": "selected_null",
                "method_id": "method",
                "replicate": 0,
                "direct_significant": True,
                "best_case_required_tau_s_for_alpha": 24.0,
                "tau_b": 30.0,
                "tau_t": 32.0,
                "tau_s": 34.0,
                "h_k": 36.0,
            },
        ]
    )

    ranges = summarize_region_tau_s_ranges(
        rows,
        target_signal_fractions=(0.5,),
        max_selected_null_fractions=(0.5,),
    )
    row = ranges.iloc[0]

    assert row["case_id"] == "case_a"
    assert row["signal_tau_s_lower_bound"] == pytest.approx(7.0)
    assert row["selected_null_tau_s_upper_bound"] == pytest.approx(22.0)
    assert row["signal_observed_tau_b_median"] == pytest.approx(3.0)
    assert row["selected_null_observed_tau_b_median"] == pytest.approx(20.0)
    assert row["tau_s_range_status"] == "tau_s_range_admissible_diagnostic"


def test_pvalue_interpolation_comparison_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.csv"
    pd.DataFrame.from_records(
        [
            _row(node_id="target", sibling_p_value=0.05),
            _row(
                node_id="signal_anchor",
                sibling_p_value=0.0,
                sibling_open=True,
                decision_class="accepted_internal_split",
                traversal_decision="split",
            ),
            _row(node_id="support_a", sibling_p_value=0.60),
            _row(node_id="support_b", sibling_p_value=0.50),
        ]
    ).to_csv(input_path, index=False)

    outputs = run_pvalue_interpolation_comparison(
        rows_path=input_path,
        output_dir=tmp_path / "out",
        fallback_tau_s=400.0,
    )
    rows = pd.read_csv(outputs["rows"])
    summary = summarize_pvalue_interpolation_comparison(rows)

    assert not rows.empty
    assert not summary.empty
    assert "median_best_case_required_tau_s_for_alpha" in summary.columns
    assert "tau_b" in rows.columns
    assert "effective_support" in rows.columns
    assert outputs["summary"].exists()
    assert outputs["case_summary"].exists()
    assert outputs["tau_s_sensitivity"].exists()
    assert outputs["tau_s_range"].exists()
    assert outputs["region_bandwidths"].exists()
    assert outputs["region_tau_s_range"].exists()
    assert outputs["manifest"].exists()
