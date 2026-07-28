from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from benchmarks.diagnostics.calibration.selected.neighborhood import (
    selected_neighborhood_topology_frontier as frontier,
)


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "case_id": "case",
        "data_role": "signal",
        "method_id": "method",
        "replicate": 0,
        "node_id": "node",
        "parent_id": "parent",
        "depth": 1,
        "direct_sibling_measurable": True,
        "direct_sibling_open": False,
        "direct_sibling_p_value": 0.20,
        "measurability_action": "fail_closed",
        "measurability_bottleneck": "direct_measurable_not_significant",
        "topology_coherence_status": "topology_balance_product_below_floor",
        "interpolated_pair_null_prior": 0.08,
        "interpolation_effective_support": 3.0,
        "interpolation_best_case_required_tau_s_for_alpha": 12.0,
        "structural_outgoing_balance": 0.40,
        "topology_balance_product_value": 0.10,
        "spectral_bottleneck_status": "spectral_not_joined",
    }
    row.update(overrides)
    return row


def test_topology_frontier_separates_root_proxy_from_admissible_root_law() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="root",
                parent_id="",
                direct_sibling_measurable=False,
                measurability_bottleneck="root_selected_topology_requires_root_law",
                topology_coherence_status="root_selected_topology_requires_root_law",
                structural_outgoing_balance=0.42,
                topology_balance_product_value=math.nan,
            )
        ]
    )

    annotated = frontier.build_topology_frontier_rows(rows)
    root = annotated.iloc[0]

    assert root["candidate_scope"] == "root_non_direct"
    assert bool(root["root_structural_proxy_pass"])
    assert root["root_selected_law_status"] == "root_selected_region_margin_missing"
    assert not bool(root["hybrid_strict_support"])
    assert root["hybrid_strict_status"] == "root_selected_region_margin_missing"


def test_topology_frontier_joins_root_selected_region_status() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                case_id="overlap_case",
                node_id="root",
                parent_id="",
                direct_sibling_measurable=False,
                measurability_bottleneck="root_selected_topology_requires_root_law",
                topology_coherence_status="root_selected_topology_requires_root_law",
                structural_outgoing_balance=0.42,
                topology_balance_product_value=math.nan,
            )
        ]
    )
    root_summary = pd.DataFrame.from_records(
        [
            {
                "case_id": "overlap_case",
                "root_selected_region_law_status": ("discrete_tie_cell_geometry_required"),
                "root_child_min_merge_margin": 0.0,
                "root_child_tied_minimum_merge_count": 12,
                "root_child_discrete_tie_cell_count": 12,
                "root_sibling_selected_ratio": 42.0,
            }
        ]
    )

    annotated = frontier.build_topology_frontier_rows(
        rows,
        root_selected_region_summary=root_summary,
    )
    root = annotated.iloc[0]

    assert root["root_selected_law_status"] == ("discrete_tie_cell_geometry_required")
    assert root["root_margin_evidence_status"] == ("root_margin_joined_case_family_diagnostic_only")
    assert root["root_child_tied_minimum_merge_count"] == 12
    assert root["root_sibling_selected_ratio"] == 42.0
    assert not bool(root["hybrid_strict_support"])
    assert root["hybrid_strict_status"] == "discrete_tie_cell_geometry_required"


def test_nonroot_frontier_requires_current_floor_and_spectral_support() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="near",
                direct_sibling_measurable=False,
                measurability_bottleneck="topology_balance_product_below_floor",
                topology_balance_product_value=0.19,
                spectral_bottleneck_status="spectral_flow_observed_diagnostic_only",
            ),
            _row(
                node_id="supported",
                direct_sibling_measurable=False,
                measurability_bottleneck="topology_balance_product_below_floor",
                topology_balance_product_value=0.24,
                spectral_bottleneck_status="spectral_flow_observed_diagnostic_only",
            ),
            _row(
                node_id="spectral_missing",
                direct_sibling_measurable=False,
                measurability_bottleneck="topology_balance_product_below_floor",
                topology_balance_product_value=0.24,
                spectral_bottleneck_status="spectral_not_joined",
            ),
        ]
    )

    annotated = frontier.build_topology_frontier_rows(rows)
    by_node = {row["node_id"]: row for _, row in annotated.iterrows()}

    assert bool(by_node["near"]["nonroot_near_frontier"])
    assert not bool(by_node["near"]["hybrid_strict_support"])
    assert by_node["near"]["hybrid_strict_status"] == ("nonroot_topology_below_current_floor")
    assert bool(by_node["supported"]["hybrid_strict_support"])
    assert by_node["supported"]["hybrid_strict_status"] == (
        "hybrid_support_observed_diagnostic_only"
    )
    assert not bool(by_node["spectral_missing"]["hybrid_strict_support"])
    assert by_node["spectral_missing"]["hybrid_strict_status"] == (
        "spectral_transport_unmeasured_or_blocked"
    )


def test_summary_and_sweep_expose_bandwidth_reopening_risk() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                data_role="selected_null",
                node_id="null_root",
                parent_id="",
                direct_sibling_measurable=False,
                measurability_bottleneck="root_selected_topology_requires_root_law",
                topology_coherence_status="root_selected_topology_requires_root_law",
                topology_balance_product_value=math.nan,
                structural_outgoing_balance=0.44,
                interpolation_best_case_required_tau_s_for_alpha=14.0,
            ),
            _row(
                data_role="signal",
                node_id="signal_root",
                parent_id="",
                direct_sibling_measurable=False,
                measurability_bottleneck="root_selected_topology_requires_root_law",
                topology_coherence_status="root_selected_topology_requires_root_law",
                topology_balance_product_value=math.nan,
                structural_outgoing_balance=0.18,
                interpolation_best_case_required_tau_s_for_alpha=22.0,
            ),
        ]
    )

    annotated = frontier.build_topology_frontier_rows(rows, reference_tau_s=20.0)
    summary = frontier.summarize_topology_frontier_rows(annotated)
    sweep = frontier.build_threshold_sweep(
        annotated,
        root_threshold_grid=(0.20, 0.40),
        nonroot_threshold_grid=(0.22,),
    )

    by_role = {row["data_role"]: row for _, row in summary.iterrows()}
    assert by_role["selected_null"]["bandwidth_reference_reopen_count"] == 1
    assert by_role["signal"]["bandwidth_reference_reopen_count"] == 0
    null_root_sweep = sweep[
        sweep["data_role"].eq("selected_null") & sweep["threshold"].eq(0.40)
    ].iloc[0]
    signal_root_sweep = sweep[sweep["data_role"].eq("signal") & sweep["threshold"].eq(0.20)].iloc[0]
    assert null_root_sweep["pass_count"] == 1
    assert signal_root_sweep["pass_count"] == 0


def test_topology_frontier_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "measurability.csv"
    pd.DataFrame.from_records(
        [
            _row(node_id="direct", direct_sibling_open=True, measurability_action="split"),
            _row(
                node_id="root",
                parent_id="",
                direct_sibling_measurable=False,
                measurability_bottleneck="root_selected_topology_requires_root_law",
                topology_coherence_status="root_selected_topology_requires_root_law",
                topology_balance_product_value=math.nan,
            ),
        ]
    ).to_csv(input_path, index=False)

    outputs = frontier.run_selected_neighborhood_topology_frontier(
        frontier.SelectedNeighborhoodTopologyFrontierConfig(
            output_dir=tmp_path / "out",
            measurability_rows_path=input_path,
        )
    )

    assert set(outputs) == {"rows", "summary", "threshold_sweep", "manifest"}
    assert outputs["rows"].exists()
    assert outputs["summary"].exists()
    assert outputs["threshold_sweep"].exists()
    assert outputs["manifest"].exists()
