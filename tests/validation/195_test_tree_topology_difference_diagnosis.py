from __future__ import annotations

import pandas as pd
from benchmarks.validation.tree_topology_difference_diagnosis import (
    DiagnosisInputs,
    build_case_diagnosis,
    build_method_summary,
    build_subtree_highlights,
    build_subtree_support,
    write_topology_difference_diagnosis_artifacts,
)


def _tree_cells() -> pd.DataFrame:
    rows = []
    for case_id, category in [
        ("edge_case", "continuous"),
        ("sibling_case", "overlap"),
    ]:
        for method, total_length in [
            ("average", 10.0),
            ("weighted", 10.1),
            ("centroid", 12.0),
        ]:
            rows.append(
                {
                    "schema_version": "brancharchitect_tree_comparison/v1",
                    "case_id": case_id,
                    "test_case": 1,
                    "case_category": category,
                    "tree_inference": method,
                    "status": "ok",
                    "skip_reason": "",
                    "true_clusters": 3,
                    "found_clusters": 1,
                    "internal_split_count": 5,
                    "total_branch_length": total_length,
                    "mean_branch_length": 0.5,
                    "max_branch_length": 1.0,
                    "largest_cluster_fraction": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _pairwise() -> pd.DataFrame:
    rows = []
    for case_id in ["edge_case", "sibling_case"]:
        rows.extend(
            [
                {
                    "schema_version": "brancharchitect_tree_comparison/v1",
                    "case_id": case_id,
                    "test_case": 1,
                    "case_category": "unit",
                    "left_tree_inference": "average",
                    "right_tree_inference": "weighted",
                    "rooted_internal_rf_relative": 0.0,
                    "rooted_weighted_split_l1": 0.01,
                    "leaf_path_rmse": 0.001,
                    "predicted_label_ari_between_topologies": 1.0,
                },
                {
                    "schema_version": "brancharchitect_tree_comparison/v1",
                    "case_id": case_id,
                    "test_case": 1,
                    "case_category": "unit",
                    "left_tree_inference": "average",
                    "right_tree_inference": "centroid",
                    "rooted_internal_rf_relative": 0.95,
                    "rooted_weighted_split_l1": 3.0,
                    "leaf_path_rmse": 0.03,
                    "predicted_label_ari_between_topologies": 1.0,
                },
                {
                    "schema_version": "brancharchitect_tree_comparison/v1",
                    "case_id": case_id,
                    "test_case": 1,
                    "case_category": "unit",
                    "left_tree_inference": "weighted",
                    "right_tree_inference": "centroid",
                    "rooted_internal_rf_relative": 0.96,
                    "rooted_weighted_split_l1": 3.1,
                    "leaf_path_rmse": 0.031,
                    "predicted_label_ari_between_topologies": 1.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _tree_newick() -> pd.DataFrame:
    rows = []
    for case_id, category in [
        ("edge_case", "continuous"),
        ("sibling_case", "overlap"),
    ]:
        for method, newick in [
            ("average", "((a:1,b:1):0.5,(c:1,d:1):0.5);"),
            ("weighted", "((a:1,b:1):0.5,(c:1,d:1):0.5);"),
            ("centroid", "((a:1,c:1):0.5,(b:1,d:1):0.5);"),
        ]:
            rows.append(
                {
                    "schema_version": "brancharchitect_tree_comparison/v1",
                    "case_id": case_id,
                    "test_case": 1,
                    "case_category": category,
                    "tree_inference": method,
                    "run_id": method,
                    "n_leaves": 4,
                    "n_edges": 6,
                    "branch_length_missing_count": 0,
                    "newick": newick,
                }
            )
    return pd.DataFrame(rows)


def _loss_taxonomy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "edge_case",
                "test_case": 1,
                "case_category": "continuous",
                "loss_bucket": "edge_gate_closed_global",
                "true_clusters": 3,
                "ok_cells": 3,
            },
            {
                "case_id": "sibling_case",
                "test_case": 2,
                "case_category": "overlap",
                "loss_bucket": "sibling_gate_closed_after_edge_open",
                "true_clusters": 3,
                "ok_cells": 3,
            },
        ]
    )


def _pvalue_cells() -> pd.DataFrame:
    rows = []
    for method, root_length, edge_p in [
        ("average", 0.5, 0.50),
        ("weighted", 0.4, 0.45),
        ("centroid", 0.01, 0.08),
    ]:
        rows.append(
            {
                "case_id": "edge_case",
                "tree_inference": method,
                "status": "ok",
                "edge_gate_open": False,
                "sibling_gate_open": False,
                "min_root_edge_p_value_bh": edge_p,
                "sibling_p_value_corrected": float("nan"),
                "sibling_sparse_p_value": 0.20,
                "sibling_dense_p_value": 0.30,
                "sibling_fixed_coordinate_bh_p_value": 0.20,
                "sibling_fixed_global_p_value": 0.30,
                "left_branch_length": root_length,
                "right_branch_length": root_length,
            }
        )
    for method, root_length, sibling_p in [
        ("average", 0.2, 0.12),
        ("weighted", 0.18, 0.08),
        ("centroid", 0.005, 0.06),
    ]:
        rows.append(
            {
                "case_id": "sibling_case",
                "tree_inference": method,
                "status": "ok",
                "edge_gate_open": True,
                "sibling_gate_open": False,
                "min_root_edge_p_value_bh": 1e-8,
                "sibling_p_value_corrected": sibling_p,
                "sibling_sparse_p_value": 1e-6,
                "sibling_dense_p_value": 1e-8,
                "sibling_fixed_coordinate_bh_p_value": 1e-6,
                "sibling_fixed_global_p_value": 1e-8,
                "left_branch_length": root_length,
                "right_branch_length": root_length,
            }
        )
    return pd.DataFrame(rows)


def _trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "sibling_case",
                "trace_type": "full_edge_traversal_trace",
                "actual_visited": True,
                "node_id": "root",
                "edge_gate_open": True,
                "sibling_p_value_corrected": 0.06,
                "sibling_sparse_p_value": 1e-6,
                "sibling_dense_p_value": 1e-8,
                "sibling_fixed_coordinate_bh_p_value": 1e-6,
                "sibling_fixed_global_p_value": 1e-8,
                "left_branch_length": 0.005,
                "right_branch_length": 0.005,
            }
        ]
    )


def test_method_summary_flags_centroid_as_topology_outlier() -> None:
    summary = build_method_summary(_tree_cells(), _pvalue_cells(), _pairwise())

    roles = dict(zip(summary["tree_inference"], summary["method_role"], strict=True))

    assert roles["centroid"] == "nonmonotone_linkage_topology_outlier"
    assert roles["average"] == "stable_linkage_core"


def test_case_diagnosis_separates_edge_and_sibling_null_changes() -> None:
    subtree_support = build_subtree_support(_tree_newick())
    subtree_highlights = build_subtree_highlights(subtree_support)
    diagnosis = build_case_diagnosis(
        _tree_cells(),
        _pairwise(),
        _loss_taxonomy(),
        _pvalue_cells(),
        _trace(),
        pd.DataFrame(),
        pd.DataFrame(),
        subtree_support,
        subtree_highlights,
    )
    by_case = diagnosis.set_index("case_id")

    assert by_case.loc["edge_case", "null_hypothesis_change"].startswith(
        "selected_tree_edge_null_first"
    )
    assert (
        "branch_length_and_topology_conditioned"
        in by_case.loc[
            "sibling_case",
            "null_hypothesis_change",
        ]
    )
    assert by_case.loc["sibling_case", "root_branch_length_ratio"] > 10.0
    assert by_case.loc["sibling_case", "unstable_subtree_count"] > 0


def test_subtree_support_identifies_unstable_leaf_sets() -> None:
    support = build_subtree_support(_tree_newick())
    highlights = build_subtree_highlights(support, max_subtrees_per_case=2)

    unstable = support[support["is_unstable"]]

    assert not unstable.empty
    assert set(unstable["contrast_type"]).issuperset(
        {"nonmonotone_specific_subtree", "stable_core_subtree_missing_in_nonmonotone"}
    )
    assert set(highlights["subtree_difference_rank"]) == {1, 2}
    assert highlights["smaller_side_leaf_count"].min() == 2


def test_writer_emits_diagnosis_artifacts(tmp_path) -> None:
    frames = {
        "tree_cells": _tree_cells(),
        "tree_pairwise": _pairwise(),
        "tree_newick": _tree_newick(),
        "loss_taxonomy": _loss_taxonomy(),
        "pvalue_cells": _pvalue_cells(),
        "traversal_trace": _trace(),
        "alpha_best_by_case": pd.DataFrame(),
        "literature_case_summary": pd.DataFrame(),
    }
    inputs = DiagnosisInputs(
        tree_cells=tmp_path / "tree_cells.csv",
        tree_pairwise=tmp_path / "tree_pairwise.csv",
        tree_newick=tmp_path / "tree_newick.csv",
        loss_taxonomy=tmp_path / "loss.csv",
        pvalue_cells=tmp_path / "pvalues.csv",
        traversal_trace=tmp_path / "trace.csv",
    )

    outputs = write_topology_difference_diagnosis_artifacts(
        frames=frames,
        output_dir=tmp_path / "out",
        inputs=inputs,
    )

    for path in outputs.values():
        assert path.exists()
    report = outputs["report"].read_text()
    assert "Tree Topology Difference Diagnosis" in report
    assert "Subtrees Showing The Differences" in report
    assert "selected-topology, branch-length-conditioned null model" in report
    assert pd.read_csv(outputs["subtree_highlights"]).shape[0] > 0
    recommendations = pd.read_csv(outputs["null_recommendations"])
    assert set(recommendations["recommendation_id"]) == {
        "selected_tree_edge_null",
        "branch_length_conditioned_sibling_null",
        "topology_stability_alpha_spending",
    }
