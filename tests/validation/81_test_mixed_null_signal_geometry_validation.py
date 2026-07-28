from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from benchmarks.diagnostics.path_b.mixed_null_signal_geometry_validation import (
    _fold_ids,
    build_mixed_null_signal_geometry_panel,
    build_row_aligned_kak_geometry_panel,
    classify_sibling_truth_context,
    evaluate_geometry_models,
    summarize_geometry_models,
    summarize_labeled_calibration,
)
from tree_break_selection.tree.poset_tree import PosetTree


class _SpectralContext:
    def __init__(self) -> None:
        self.principal_component_projections_by_node = {
            "root": np.eye(2),
        }


def test_classify_sibling_truth_context_marks_null_signal_and_mixed() -> None:
    null_context = classify_sibling_truth_context([1, 1, 1], [1, 1])
    signal_context = classify_sibling_truth_context([1, 1, 1], [2, 2, 2])
    mixed_context = classify_sibling_truth_context([1, 2, 1], [2, 3, 3])

    assert null_context["truth_context_label"] == "null_context"
    assert null_context["is_null_context"] is True
    assert signal_context["truth_context_label"] == "signal_context"
    assert signal_context["truth_sibling_separation_score"] == 1.0
    assert mixed_context["truth_context_label"] == "mixed_context"


def test_build_mixed_null_signal_geometry_panel_joins_truth_and_edge_geometry() -> None:
    node_panel = pd.DataFrame(
        {
            "case_id": ["case", "case"],
            "node_id": ["N1", "N2"],
            "depth": [1, 2],
            "leaf_count": [6, 4],
            "test_projection_dimension": [1, 2],
            "raw_mp_signal_count": [1, 2],
            "selected_eigenvalue_gap_ratio": [2.0, 3.0],
            "top_selected_eigenvalue_mass_fraction": [0.8, 0.7],
            "selected_eigenvalue_effective_rank": [1.2, 1.5],
            "sibling_raw_p_value": [0.5, 0.001],
            "sibling_raw_neglog10_p": [0.30103, 3.0],
            "sibling_raw_alpha_margin": [-1.0, 1.0],
            "sibling_bh_p_value": [0.6, 0.002],
            "sibling_neglog10_p": [0.22185, 2.7],
            "sibling_alpha_margin": [-1.1, 0.7],
            "sibling_degrees_of_freedom": [1.0, 2.0],
            "sibling_chi_square_tail_sensitivity": [0.2, 0.4],
            "recursive_sibling_neglog10_gradient": [0.0, 1.0],
            "edge_sibling_connectivity_score": [1.0, 4.0],
            "sibling_bh_different": [False, True],
        }
    )
    edge_panel = pd.DataFrame(
        {
            "case_id": ["case", "case", "case", "case"],
            "parent_id": ["N1", "N1", "N2", "N2"],
            "edge_raw_neglog10_p": [1.0, 2.0, 4.0, 5.0],
            "edge_neglog10_bh_p": [0.5, 1.5, 3.0, 4.0],
            "edge_raw_alpha_margin": [-2.0, -1.0, 1.0, 2.0],
            "edge_alpha_margin": [-2.5, -1.5, 0.5, 1.5],
            "subspace_chordal_distance_normalized": [0.1, 0.2, 0.7, 0.8],
            "edge_raw_minus_parent_sibling_raw_neglog10": [0.5, 1.5, 2.0, 3.0],
            "child_raw_minus_parent_sibling_raw_neglog10": [0.0, 0.1, 1.0, 1.1],
        }
    )
    truth = pd.DataFrame(
        {
            "case_id": ["case", "case"],
            "node_id": ["N1", "N2"],
            "feature_family": ["bernoulli", "bernoulli"],
            "truth_context_label": ["null_context", "signal_context"],
            "is_null_context": [True, False],
            "is_signal_context": [False, True],
            "is_mixed_context": [False, False],
            "parent_leaf_count_truth": [6, 4],
            "parent_size_fraction": [0.6, 0.4],
            "parent_size_bin": ["parent_0.50_1.00", "parent_0.25_0.50"],
            "barycentric_balance": [0.5, 0.5],
            "log_barycentric_leverage": [0.0, 0.0],
            "log_sampling_variance_scale": [-0.4, 0.0],
            "truth_sibling_separation_score": [0.0, 1.0],
        }
    )
    kak = pd.DataFrame(
        {
            "case_id": ["case"],
            "node_id": ["N2"],
            "geometry_parent_radius": [1.5],
            "geometry_angle_to_leading_axis_deg": [45.0],
            "geometry_independent_radius_fraction": [0.5],
            "action_budget_proxy": [0.8],
            "action_budget_proxy_capped": [0.8],
        }
    )

    panel = build_mixed_null_signal_geometry_panel(
        node_panel=node_panel,
        edge_panel=edge_panel,
        truth_labels=truth,
        sibling_alpha=0.01,
        kak_geometry=kak,
    )

    assert panel.shape[0] == 2
    assert panel.loc[panel["node_id"].eq("N2"), "edge_raw_action_count"].iloc[0] == 2
    assert panel.loc[panel["node_id"].eq("N2"), "split_rejected_at_sibling_alpha"].iloc[0]
    assert panel.loc[panel["node_id"].eq("N2"), "action_budget_proxy"].iloc[0] == 0.8


def test_build_row_aligned_kak_geometry_panel_adds_sibling_node_terms() -> None:
    tree = PosetTree()
    tree.add_node("root", is_leaf=False, distribution=np.array([0.5, 0.5]))
    tree.add_node("L", is_leaf=False)
    tree.add_node("R", is_leaf=False)
    for index in range(4):
        tree.add_node(f"s{index}", is_leaf=True, label=f"s{index}")
    tree.add_edges_from(
        [
            ("root", "L"),
            ("root", "R"),
            ("L", "s0"),
            ("L", "s1"),
            ("R", "s2"),
            ("R", "s3"),
        ]
    )
    tree.graph["root"] = "root"
    leaf_data = pd.DataFrame(
        [[0.1, 0.2], [0.2, 0.2], [0.8, 0.7], [0.9, 0.8]],
        index=["s0", "s1", "s2", "s3"],
        columns=["a", "b"],
    )

    kak = build_row_aligned_kak_geometry_panel(
        case_id="case",
        tree=tree,
        leaf_data=leaf_data,
        feature_space=None,
        spectral_context=_SpectralContext(),
    )

    root_row = kak[kak["node_id"].eq("root")]
    assert root_row.shape[0] == 1
    assert root_row["row_aligned_kak_frame"].iloc[0] == "root_selected_pca_whitened_tangent"
    assert np.isfinite(root_row["action_budget_proxy"].iloc[0])


def test_evaluate_geometry_models_reports_recursive_gain_and_missing_kak() -> None:
    rows = []
    for case_idx in range(6):
        for node_idx in range(6):
            is_signal = node_idx >= 3
            edge_action = float(node_idx + case_idx / 10)
            rows.append(
                {
                    "case_id": f"case_{case_idx}",
                    "node_id": f"N{case_idx}_{node_idx}",
                    "feature_family": "bernoulli" if case_idx % 2 else "categorical",
                    "depth": float(node_idx + 1),
                    "depth_bin": "depth_0_1" if node_idx < 2 else "depth_2_3",
                    "parent_size_bin": "parent_0.25_0.50",
                    "case_category": "synthetic",
                    "truth_context_label": "signal_context" if is_signal else "null_context",
                    "is_signal_context": is_signal,
                    "is_null_context": not is_signal,
                    "is_mixed_context": False,
                    "split_rejected_at_sibling_alpha": is_signal,
                    "raw_sibling_rejected_at_alpha": is_signal,
                    "sibling_raw_neglog10_p": 0.2 + edge_action,
                    "sibling_degrees_of_freedom": 1.0,
                    "sibling_chi_square_tail_sensitivity": 0.2 + 0.01 * node_idx,
                    "edge_action": edge_action,
                    "edge_raw_action_count": int(is_signal),
                    "log_parent_leaf_count": 2.0,
                    "log_parent_size_fraction": -1.0,
                    "test_projection_dimension": 1.0,
                    "barycentric_balance": 0.5,
                    "log_barycentric_leverage": 0.0,
                    "log_sampling_variance_scale": -0.5,
                    "raw_mp_signal_count": 1.0,
                    "log_selected_eigenvalue_gap_ratio": 0.1 * node_idx,
                    "top_selected_eigenvalue_mass_fraction": 0.7,
                    "selected_eigenvalue_effective_rank": 1.2,
                    "sibling_raw_alpha_margin": edge_action - 2.0,
                    "sibling_alpha_margin": edge_action - 2.1,
                    "recursive_sibling_neglog10_gradient": float(is_signal),
                    "edge_sibling_connectivity_score": edge_action,
                    "max_edge_raw_minus_parent_sibling_raw_neglog10": edge_action,
                    "max_child_raw_minus_parent_sibling_raw_neglog10": edge_action / 2,
                    "mean_child_subspace_chordal_distance": edge_action / 10,
                    "max_child_subspace_chordal_distance": edge_action / 8,
                    "truth_sibling_separation_score": float(is_signal),
                }
            )
    panel = pd.DataFrame.from_records(rows)

    validation = evaluate_geometry_models(
        panel,
        min_train_rows_per_predictor=1,
        min_test_rows=2,
    )
    summary = summarize_geometry_models(validation)
    calibration = summarize_labeled_calibration(panel)

    assert not validation.empty
    assert validation["model_id"].eq("recursive_pvalue_geometry").any()
    assert validation["model_status"].str.startswith("missing_row_aligned_kak_geometry").any()
    assert summary["model_id"].eq("recursive_pvalue_geometry").any()
    assert calibration["grouping_id"].eq("truth_context").any()


def test_evaluate_geometry_models_scores_sklearn_surfaces_with_kak_terms() -> None:
    rows = []
    for case_idx in range(8):
        for node_idx in range(8):
            is_signal = node_idx >= 4
            edge_action = float(node_idx + 0.2 * case_idx)
            kak_action = 0.1 + 0.15 * node_idx
            rows.append(
                {
                    "case_id": f"case_{case_idx}",
                    "node_id": f"N{case_idx}_{node_idx}",
                    "feature_family": "bernoulli" if case_idx % 2 else "categorical",
                    "depth": float(node_idx + 1),
                    "depth_bin": "depth_0_1" if node_idx < 4 else "depth_2_3",
                    "parent_size_bin": "parent_0.25_0.50",
                    "case_category": "synthetic",
                    "truth_context_label": "signal_context" if is_signal else "null_context",
                    "is_signal_context": is_signal,
                    "is_null_context": not is_signal,
                    "is_mixed_context": False,
                    "split_rejected_at_sibling_alpha": is_signal,
                    "raw_sibling_rejected_at_alpha": is_signal,
                    "sibling_raw_neglog10_p": 0.2 + edge_action,
                    "sibling_degrees_of_freedom": 1.0,
                    "sibling_chi_square_tail_sensitivity": 0.2 + 0.01 * node_idx,
                    "edge_action": edge_action,
                    "edge_raw_action_count": int(is_signal),
                    "log_parent_leaf_count": 2.0,
                    "log_parent_size_fraction": -1.0,
                    "test_projection_dimension": 1.0,
                    "barycentric_balance": 0.5,
                    "log_barycentric_leverage": 0.0,
                    "log_sampling_variance_scale": -0.5,
                    "raw_mp_signal_count": 1.0,
                    "log_selected_eigenvalue_gap_ratio": 0.1 * node_idx,
                    "top_selected_eigenvalue_mass_fraction": 0.7,
                    "selected_eigenvalue_effective_rank": 1.2,
                    "sibling_raw_alpha_margin": edge_action - 2.0,
                    "sibling_alpha_margin": edge_action - 2.1,
                    "recursive_sibling_neglog10_gradient": float(is_signal),
                    "edge_sibling_connectivity_score": edge_action,
                    "max_edge_raw_minus_parent_sibling_raw_neglog10": edge_action,
                    "max_child_raw_minus_parent_sibling_raw_neglog10": edge_action / 2,
                    "mean_child_subspace_chordal_distance": edge_action / 10,
                    "max_child_subspace_chordal_distance": edge_action / 8,
                    "truth_sibling_separation_score": float(is_signal),
                    "geometry_parent_radius": 1.0 + 0.1 * node_idx,
                    "geometry_angle_to_leading_axis_deg": 15.0 + 8.0 * node_idx,
                    "geometry_independent_radius_fraction": 0.1 + 0.08 * node_idx,
                    "geometry_sibling_separation_parent_ratio": 0.2 + 0.1 * node_idx,
                    "geometry_sibling_separation_child_ratio": 0.1 + 0.08 * node_idx,
                    "geometry_abs_common_axis_gap": 0.3 + 0.05 * node_idx,
                    "action_budget_proxy": kak_action,
                    "action_budget_proxy_capped": min(kak_action, 1.0),
                    "angular_shell_risk_score": min(kak_action, 1.0) * (node_idx / 8.0),
                }
            )
    panel = pd.DataFrame.from_records(rows)

    validation = evaluate_geometry_models(
        panel,
        min_train_rows_per_predictor=1,
        min_test_rows=2,
    )
    ok = validation[validation["model_status"].str.startswith("diagnostic_holdout")]

    assert ok["model_id"].eq("sklearn_logistic_edge_sibling_kak_surface").any()
    assert ok["model_id"].eq("sklearn_hist_gradient_edge_sibling_kak_surface").any()
    assert np.isfinite(
        ok.loc[
            ok["model_id"].eq("sklearn_logistic_edge_sibling_kak_surface"),
            "holdout_signal_auc",
        ]
    ).any()


def test_case_hash_modulo_split_uses_stable_checksum() -> None:
    table = pd.DataFrame({"case_id": ["case_a", "case_b", "case_a"]})

    folds = _fold_ids(table, "case_hash_modulo_5")

    assert folds.tolist() == [zlib.crc32(value.encode("utf-8")) % 5 for value in table["case_id"]]
