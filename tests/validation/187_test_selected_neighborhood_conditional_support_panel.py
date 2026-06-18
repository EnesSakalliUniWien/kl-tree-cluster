from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.diagnostics.calibration import (
    selected_neighborhood_conditional_support_panel as panel,
)


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "case_id": "overlap_part_4c_small",
        "data_role": "signal",
        "method_id": "method",
        "replicate": 0,
        "node_id": "node",
        "parent_id": "parent",
        "depth": 2,
        "direct_sibling_measurable": False,
        "direct_sibling_open": False,
        "direct_sibling_p_value": 0.20,
        "measurability_action": "fail_closed",
        "measurability_bottleneck": "topology_balance_product_below_floor",
        "interpolation_support_count": 5,
        "interpolation_signal_count": 1,
        "interpolation_selected_nonnull_excluded_count": 0,
        "interpolation_effective_support": 4.0,
        "interpolation_support_weight": 2.0,
        "interpolation_nearest_support_distance": 1.0,
        "interpolation_nearest_signal_distance": 2.0,
        "interpolation_tau_t": 1.0,
        "interpolation_tau_s": 20.0,
        "interpolation_h_k": 1.0,
        "interpolated_pair_null_prior": 0.005,
        "interpolation_best_case_required_tau_s_for_alpha": 10.0,
        "interpolation_behavior_label": "signal_extra_open_candidate",
        "topology_coherent": True,
        "topology_coherence_status": "topology_coherent",
        "topology_balance_product_value": 0.30,
        "structural_outgoing_balance": 0.42,
        "spectral_bottleneck_status": "spectral_flow_observed_diagnostic_only",
    }
    row.update(overrides)
    return row


def _validity(
    case_id: str,
    *,
    validity_status: str,
    tail_status: str,
    usability_status: str,
) -> dict[str, object]:
    return {
        "target_case_id": case_id,
        "root_validity_status": validity_status,
        "root_tail_inference_status": tail_status,
        "selected_root_usability_status": usability_status,
    }


def test_hard_negative_root_invalid_blocks_local_support() -> None:
    rows = pd.DataFrame.from_records(
        [_row(case_id="overlap_extreme_4c", node_id="candidate")]
    )
    root_validity = pd.DataFrame.from_records(
        [
            _validity(
                "overlap_extreme_4c",
                validity_status="root_validity_failed_feature_subsample_replay",
                tail_status="calibrated_selected_root_spectral_tail_available",
                usability_status="fail_closed_root_validity_failed",
            )
        ]
    )

    annotated = panel.build_conditional_support_rows(
        rows,
        root_validity_rows=root_validity,
    )
    candidate = annotated.iloc[0]

    assert bool(candidate["hard_negative_control"])
    assert not bool(candidate["conditional_support_pass"])
    assert not bool(candidate["hard_negative_leak"])
    assert candidate["dominant_blocker"] == (
        "root_validity_failed_hard_negative_control"
    )
    assert candidate["method_action"] == "fail_closed_hard_negative_root_invalid"


def test_valid_root_with_missing_tail_allows_diagnostic_support_only() -> None:
    rows = pd.DataFrame.from_records([_row(node_id="candidate")])
    root_validity = pd.DataFrame.from_records(
        [
            _validity(
                "overlap_part_4c_small",
                validity_status="root_validity_supported_by_stability_and_selection",
                tail_status="fail_closed_selected_root_spectral_tail_support_missing",
                usability_status="fail_closed_valid_root_tail_support_missing",
            )
        ]
    )

    annotated = panel.build_conditional_support_rows(
        rows,
        root_validity_rows=root_validity,
    )
    candidate = annotated.iloc[0]

    assert bool(candidate["root_validity_pass"])
    assert not bool(candidate["root_tail_pass"])
    assert bool(candidate["conditional_support_pass"])
    assert not bool(candidate["promotion_eligible"])
    assert candidate["method_action"] == (
        "conditional_neighborhood_support_root_tail_missing_diagnostic"
    )


def test_selected_null_support_is_reported_as_leak_diagnostic() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                case_id="overlap_mod_4c_small",
                data_role="selected_null",
                node_id="null_candidate",
            )
        ]
    )

    annotated = panel.build_conditional_support_rows(rows)
    candidate = annotated.iloc[0]
    summary = panel.summarize_conditional_support(annotated).iloc[0]

    assert bool(candidate["conditional_support_pass"])
    assert bool(candidate["selected_null_leak"])
    assert candidate["method_action"] == (
        "selected_null_conditional_support_leak_diagnostic"
    )
    assert summary["selected_null_leak_count"] == 1
    assert summary["summary_status"] == "selected_null_leak_observed_diagnostic"


def test_direct_split_is_not_counted_as_neighborhood_support() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                data_role="selected_null",
                direct_sibling_measurable=True,
                direct_sibling_open=True,
                direct_sibling_p_value=0.001,
                measurability_action="split",
                measurability_bottleneck="none",
            )
        ]
    )

    annotated = panel.build_conditional_support_rows(rows)
    candidate = annotated.iloc[0]

    assert candidate["method_action"] == (
        "direct_split_already_measured_not_neighborhood_rescue"
    )
    assert not bool(candidate["conditional_support_pass"])
    assert not bool(candidate["selected_null_leak"])


def test_local_support_bottleneck_fails_closed_before_topology_and_spectral() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                interpolation_support_count=1,
                interpolation_effective_support=1.0,
                interpolation_selected_nonnull_excluded_count=3,
            )
        ]
    )
    root_validity = pd.DataFrame.from_records(
        [
            _validity(
                "overlap_part_4c_small",
                validity_status="root_validity_supported_by_stability_and_selection",
                tail_status="calibrated_selected_root_spectral_tail_available",
                usability_status="usable_selected_root_tail_after_validity_replay",
            )
        ]
    )

    annotated = panel.build_conditional_support_rows(
        rows,
        root_validity_rows=root_validity,
    )
    candidate = annotated.iloc[0]

    assert not bool(candidate["local_support_pass"])
    assert not bool(candidate["conditional_support_pass"])
    assert candidate["dominant_blocker"] == "support_count_below_floor"
    assert candidate["method_action"] == "fail_closed_local_support_missing"


def test_root_non_direct_candidate_cannot_be_neighborhood_rescued() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="root",
                parent_id="",
                measurability_bottleneck="root_selected_topology_requires_root_law",
                topology_coherence_status="root_selected_topology_requires_root_law",
            )
        ]
    )
    root_validity = pd.DataFrame.from_records(
        [
            _validity(
                "overlap_part_4c_small",
                validity_status="root_validity_supported_by_stability_and_selection",
                tail_status="calibrated_selected_root_spectral_tail_available",
                usability_status="usable_selected_root_tail_after_validity_replay",
            )
        ]
    )

    annotated = panel.build_conditional_support_rows(
        rows,
        root_validity_rows=root_validity,
    )
    root = annotated.iloc[0]

    assert root["candidate_scope"] == "root_non_direct"
    assert not bool(root["conditional_support_pass"])
    assert root["method_action"] == "fail_closed_root_neighborhood_rescue_disallowed"


def test_conditional_support_panel_writes_outputs(tmp_path: Path) -> None:
    rows_path = tmp_path / "measurability.csv"
    validity_path = tmp_path / "validity.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame.from_records(
        [
            _row(node_id="candidate"),
            _row(
                case_id="overlap_mod_4c_small",
                data_role="selected_null",
                node_id="null_candidate",
            ),
        ]
    ).to_csv(rows_path, index=False)
    pd.DataFrame.from_records(
        [
            _validity(
                "overlap_part_4c_small",
                validity_status="root_validity_supported_by_stability_and_selection",
                tail_status="fail_closed_selected_root_spectral_tail_support_missing",
                usability_status="fail_closed_valid_root_tail_support_missing",
            )
        ]
    ).to_csv(validity_path, index=False)

    outputs = panel.run_selected_neighborhood_conditional_support(
        panel.SelectedNeighborhoodConditionalSupportConfig(
            output_dir=output_dir,
            measurability_rows_path=rows_path,
            root_validity_rows_path=validity_path,
            root_tail_rows_path=None,
        )
    )

    assert outputs["rows"].exists()
    assert outputs["case_summary"].exists()
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
    written = pd.read_csv(outputs["summary"])
    assert written.iloc[0]["row_count"] == 2
