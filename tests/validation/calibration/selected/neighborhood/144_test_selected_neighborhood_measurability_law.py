from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected.neighborhood.selected_neighborhood_measurability_law import (
    build_measurability_law_rows,
    compute_child_interpolated_null_prior,
    enrich_measurability_input_rows,
    run_selected_neighborhood_measurability_law,
    summarize_measurability_law_rows,
)


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "case_id": "case",
        "data_role": "signal",
        "method_id": "method",
        "replicate": 0,
        "node_id": "node",
        "sibling_p_value": 0.50,
        "sibling_open": False,
        "explicit_guard_blocked": False,
        "root_stability_guard_blocked": False,
        "root_selective_guard_blocked": False,
        "selected_family_guard_blocked": False,
        "interpolated_left_null_prior": math.nan,
        "interpolated_right_null_prior": math.nan,
        "topology_neighborhood_support_count": 0,
        "topology_neighborhood_signal_count": 0,
        "topology_neighborhood_selected_nonnull_excluded_count": 0,
        "topology_neighborhood_support_status": "",
        "topology_neighborhood_tau_b": 1.0,
        "topology_neighborhood_tau_t": 1.0,
        "topology_neighborhood_tau_s": 1.0,
        "topology_neighborhood_h_k": 1.0,
        "balance_product": 0.30,
        "outgoing_edge_norm_balance": 0.96,
    }
    row.update(overrides)
    return row


def test_child_interpolated_null_prior_is_unclipped_and_validated() -> None:
    result = compute_child_interpolated_null_prior(
        ancestor_p_values=[0.40],
        ancestor_weights=[1.0],
        stable_p_values=[0.20],
        stable_weights=[3.0],
        signal_p_values=[0.10],
        signal_distances=[1.0],
        tau_s=2.0,
    )

    interpolated = (1.0 * 0.40 + 3.0 * 0.20) / 4.0
    attenuation = (1.0 - 0.10) * math.exp(-1.0 / 2.0)
    assert result.status == "interpolated_prior_observed_diagnostic_only"
    assert result.prior == pytest.approx(interpolated * (1.0 - attenuation))
    assert 0.0 <= result.prior <= 1.0

    with pytest.raises(ValueError, match="probability"):
        compute_child_interpolated_null_prior(
            stable_p_values=[1.2],
            stable_weights=[1.0],
        )


def test_direct_measurable_sibling_test_takes_precedence() -> None:
    rows = build_measurability_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    node_id="direct_split",
                    sibling_p_value=0.50,
                    sibling_open=True,
                    interpolated_left_null_prior=0.001,
                    interpolated_right_null_prior=0.001,
                    topology_neighborhood_support_count=10,
                ),
                _row(
                    node_id="direct_closed",
                    sibling_p_value=0.50,
                    sibling_open=False,
                    interpolated_left_null_prior=0.001,
                    interpolated_right_null_prior=0.001,
                    topology_neighborhood_support_count=10,
                ),
            ]
        )
    )
    by_node = {row["node_id"]: row for _, row in rows.iterrows()}

    assert by_node["direct_split"]["measurability_action"] == "split"
    assert by_node["direct_split"]["measurability_status"] == "direct_sibling_test_split"
    assert by_node["direct_closed"]["measurability_action"] == "fail_closed"
    assert by_node["direct_closed"]["measurability_bottleneck"] == (
        "direct_measurable_not_significant"
    )


def test_supported_interpolation_can_rescue_guard_blocked_candidate() -> None:
    rows = build_measurability_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    node_id="rescued",
                    selected_family_guard_blocked=True,
                    interpolated_left_null_prior=0.005,
                    interpolated_right_null_prior=0.008,
                    topology_neighborhood_support_count=3,
                    topology_neighborhood_signal_count=1,
                    topology_neighborhood_support_status=(
                        "topology_neighborhood_support_observed_diagnostic_only"
                    ),
                ),
                _row(
                    node_id="weak_prior",
                    selected_family_guard_blocked=True,
                    interpolated_left_null_prior=0.20,
                    interpolated_right_null_prior=0.30,
                    topology_neighborhood_support_count=3,
                    topology_neighborhood_signal_count=1,
                    topology_neighborhood_support_status=(
                        "topology_neighborhood_support_observed_diagnostic_only"
                    ),
                ),
            ]
        ),
        interpolated_alpha=0.01,
        min_interpolation_support=2,
    )
    by_node = {row["node_id"]: row for _, row in rows.iterrows()}

    assert by_node["rescued"]["measurability_action"] == "diagnostic_rescue"
    assert by_node["rescued"]["interpolated_pair_null_prior"] == pytest.approx(0.005)
    assert by_node["weak_prior"]["measurability_action"] == "fail_closed"
    assert by_node["weak_prior"]["measurability_bottleneck"] == ("interpolated_null_not_strong")


def test_interpolation_failures_are_labeled_without_clipping() -> None:
    rows = build_measurability_law_rows(
        pd.DataFrame.from_records(
            [
                _row(
                    node_id="invalid_prior",
                    selected_family_guard_blocked=True,
                    interpolated_pair_null_prior=1.2,
                    topology_neighborhood_support_count=3,
                ),
                _row(
                    node_id="selected_nonnull_only",
                    selected_family_guard_blocked=True,
                    interpolated_pair_null_prior=0.001,
                    topology_neighborhood_support_count=0,
                    topology_neighborhood_selected_nonnull_excluded_count=2,
                ),
                _row(
                    node_id="low_topology",
                    selected_family_guard_blocked=True,
                    interpolated_pair_null_prior=0.001,
                    topology_neighborhood_support_count=3,
                    balance_product=0.10,
                ),
            ]
        )
    )
    by_node = {row["node_id"]: row for _, row in rows.iterrows()}

    assert by_node["invalid_prior"]["measurability_bottleneck"] == (
        "invalid_interpolated_prior_probability_domain"
    )
    assert math.isnan(float(by_node["invalid_prior"]["interpolated_pair_null_prior"]))
    assert by_node["selected_nonnull_only"]["measurability_bottleneck"] == (
        "selection_bottleneck_selected_nonnull_only"
    )
    assert by_node["low_topology"]["measurability_bottleneck"] == (
        "topology_balance_product_below_floor"
    )


def test_optional_interpolation_and_spectral_rows_enrich_candidate_audit() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="candidate",
                parent_id="parent",
                selected_family_guard_blocked=True,
                traversal_decision="pass_through",
                decision_class="selected_family_blocked",
                depth=3,
            )
        ]
    )
    pvalue_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "candidate",
                "interpolated_sibling_null_p_like": 0.001,
                "support_anchor_count": 3,
                "signal_anchor_count": 1,
                "selected_nonnull_excluded_count": 0,
                "support_weight": 1.7,
                "effective_support": 2.6,
                "stable_weighted_p_mean": 0.3,
                "signal_attenuation": 0.9,
                "nearest_support_distance": 2.0,
                "nearest_signal_distance": 1.0,
                "tau_t": 4.0,
                "tau_s": 5.0,
                "h_k": 0.7,
                "best_case_required_tau_s_for_alpha": 11.0,
                "interpolation_status": ("interpolated_p_like_observed_diagnostic_only"),
                "comparison_class": "interpolation_lower_than_direct",
                "behavior_label": "signal_extra_open_candidate",
            }
        ]
    )
    spectral_edges = pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "parent_id": "parent",
                "child_id": "candidate",
                "mp_common_dimension": 2,
                "mp_pair_supported": True,
                "mp_subspace_chordal_distance": 0.25,
                "mp_log_eigenvalue_delta": 0.10,
                "spectral_barrier": 0.20,
                "spectral_flow_affinity": 0.80,
                "flow_status": "mp_flow_observed_diagnostic_only",
            }
        ]
    )

    enriched = enrich_measurability_input_rows(
        rows,
        pvalue_rows=pvalue_rows,
        spectral_flow_edges=spectral_edges,
    )
    law_rows = build_measurability_law_rows(enriched)
    row = law_rows.iloc[0]

    assert row["measurability_action"] == "diagnostic_rescue"
    assert row["interpolation_effective_support"] == pytest.approx(2.6)
    assert row["interpolation_behavior_label"] == "signal_extra_open_candidate"
    assert row["interpolation_tau_t"] == pytest.approx(4.0)
    assert row["topology_neighborhood_tau_t"] == pytest.approx(1.0)
    assert row["spectral_parent_id"] == "parent"
    assert row["spectral_bottleneck_status"] == ("spectral_flow_observed_diagnostic_only")


def test_selected_tree_structural_balance_fallback_completes_missing_topology() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="root",
                parent_id="",
                n_descendant_leaves=400,
                balance_product=math.nan,
            ),
            _row(
                node_id="candidate",
                parent_id="root",
                n_descendant_leaves=200,
                selected_family_guard_blocked=True,
                balance_product=math.nan,
                incoming_branch_balance=math.nan,
                outgoing_balance=math.nan,
                outgoing_edge_norm_balance=0.96,
            ),
            _row(
                node_id="incoming_sibling",
                parent_id="root",
                n_descendant_leaves=200,
                balance_product=math.nan,
            ),
            _row(
                node_id="left_child",
                parent_id="candidate",
                n_descendant_leaves=110,
                balance_product=math.nan,
            ),
            _row(
                node_id="right_child",
                parent_id="candidate",
                n_descendant_leaves=90,
                balance_product=math.nan,
            ),
        ]
    )
    pvalue_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "candidate",
                "interpolated_sibling_null_p_like": 0.001,
                "support_anchor_count": 3,
                "signal_anchor_count": 1,
                "selected_nonnull_excluded_count": 0,
                "support_weight": 1.7,
                "effective_support": 2.6,
                "stable_weighted_p_mean": 0.3,
                "signal_attenuation": 0.9,
                "nearest_support_distance": 2.0,
                "nearest_signal_distance": 1.0,
                "tau_t": 4.0,
                "tau_s": 5.0,
                "h_k": 0.7,
                "interpolation_status": ("interpolated_p_like_observed_diagnostic_only"),
            }
        ]
    )

    enriched = enrich_measurability_input_rows(rows, pvalue_rows=pvalue_rows)
    candidate = enriched.loc[enriched["node_id"].eq("candidate")].iloc[0]
    assert candidate["structural_incoming_branch_balance"] == pytest.approx(0.5)
    assert candidate["structural_outgoing_balance"] == pytest.approx(0.45)
    assert candidate["structural_balance_product"] == pytest.approx(0.225)

    law_rows = build_measurability_law_rows(enriched)
    row = law_rows.loc[law_rows["node_id"].eq("candidate")].iloc[0]
    assert row["topology_balance_product_source"] == ("structural_selected_tree_fallback")
    assert row["topology_coherence_status"] == "topology_coherent"
    assert row["measurability_action"] == "diagnostic_rescue"


def test_root_candidate_requires_root_specific_topology_law() -> None:
    rows = pd.DataFrame.from_records(
        [
            _row(
                node_id="root",
                parent_id="",
                n_descendant_leaves=400,
                selected_family_guard_blocked=True,
                balance_product=math.nan,
                incoming_branch_balance=math.nan,
                outgoing_balance=math.nan,
                outgoing_edge_norm_balance=0.96,
            ),
            _row(
                node_id="left_child",
                parent_id="root",
                n_descendant_leaves=240,
                balance_product=math.nan,
            ),
            _row(
                node_id="right_child",
                parent_id="root",
                n_descendant_leaves=160,
                balance_product=math.nan,
            ),
        ]
    )
    pvalue_rows = pd.DataFrame.from_records(
        [
            {
                "case_id": "case",
                "data_role": "signal",
                "method_id": "method",
                "replicate": 0,
                "node_id": "root",
                "interpolated_sibling_null_p_like": 0.001,
                "support_anchor_count": 3,
                "signal_anchor_count": 1,
                "selected_nonnull_excluded_count": 0,
                "support_weight": 1.7,
                "effective_support": 2.6,
                "stable_weighted_p_mean": 0.3,
                "signal_attenuation": 0.9,
                "nearest_support_distance": 2.0,
                "nearest_signal_distance": 1.0,
                "tau_t": 4.0,
                "tau_s": 5.0,
                "h_k": 0.7,
                "interpolation_status": ("interpolated_p_like_observed_diagnostic_only"),
            }
        ]
    )

    law_rows = build_measurability_law_rows(
        enrich_measurability_input_rows(rows, pvalue_rows=pvalue_rows)
    )
    row = law_rows.loc[law_rows["node_id"].eq("root")].iloc[0]

    assert row["structural_outgoing_balance"] == pytest.approx(0.4)
    assert math.isnan(float(row["structural_incoming_branch_balance"]))
    assert row["topology_coherence_status"] == ("root_selected_topology_requires_root_law")
    assert row["measurability_action"] == "fail_closed"


def test_measurability_law_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.csv"
    pd.DataFrame.from_records(
        [
            _row(node_id="direct", sibling_open=True),
            _row(
                node_id="rescued",
                selected_family_guard_blocked=True,
                interpolated_pair_null_prior=0.001,
                topology_neighborhood_support_count=3,
            ),
        ]
    ).to_csv(input_path, index=False)

    outputs = run_selected_neighborhood_measurability_law(
        rows_path=input_path,
        output_dir=tmp_path / "out",
    )
    law_rows = pd.read_csv(outputs["rows"])
    summary = summarize_measurability_law_rows(law_rows)

    assert set(law_rows["measurability_action"]) == {"split", "diagnostic_rescue"}
    assert not summary.empty
    assert outputs["summary"].exists()
    assert outputs["manifest"].exists()
