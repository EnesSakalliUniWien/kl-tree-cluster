from __future__ import annotations

import json
import time
from types import SimpleNamespace

import benchmarks.diagnostics.calibration.selected_family_traversal_panel as selected_panel
import networkx as nx
import pandas as pd
import pytest
from benchmarks.diagnostics.calibration.selected_family_traversal_panel import (
    BASELINE_METHOD_ID,
    SelectedFamilyTraversalPanelConfig,
    _build_node_decisions,
    build_selected_family_production_components,
    run_selected_family_traversal_panel,
    validate_methods,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scripts.analysis.multiscale_umap_overlay import (
    load_overlay_data,
    render_multiscale_umap_overlay,
)


def _annotate_tree_structure(tree: nx.DiGraph, leaves: set[str]) -> None:
    for node in tree.nodes:
        tree.nodes[node]["is_leaf"] = node in leaves
        tree.nodes[node]["label"] = node


def _make_tree() -> PosetTree:
    tree = PosetTree()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "A1"),
            ("A", "A2"),
            ("B", "B1"),
            ("B", "B2"),
        ]
    )
    _annotate_tree_structure(tree, {"A1", "A2", "B1", "B2"})
    return tree


def test_validate_methods_accepts_refined_profile_and_rejects_unknown() -> None:
    assert validate_methods(
        (
            BASELINE_METHOD_ID,
            "fixed_coordinate_conditional_topology_diagnostic_v1",
            "fixed_coordinate_global_passthrough_refined_v1",
            "fixed_coordinate_spectral_transport_passthrough_diagnostic_v1",
            "fixed_coordinate_spectral_transport_passthrough_v1",
        )
    ) == (
        BASELINE_METHOD_ID,
        "fixed_coordinate_conditional_topology_diagnostic_v1",
        "fixed_coordinate_global_passthrough_refined_v1",
        "fixed_coordinate_spectral_transport_passthrough_diagnostic_v1",
        "fixed_coordinate_spectral_transport_passthrough_v1",
    )

    with pytest.raises(ValueError, match="Unknown traversal method"):
        validate_methods(("not_a_method",))


def test_multiscale_node_decisions_distinguish_guard_and_traversal_classes() -> None:
    tree = _make_tree()
    annotations = pd.DataFrame(index=list(tree.nodes))
    annotations["Child_Parent_Divergence_Significant"] = True
    annotations["Sibling_BH_Different"] = False
    annotations["Sibling_Divergence_P_Value"] = 0.5
    annotations["Sibling_Projection_Dimension"] = 3.0
    annotations["Root_Stability_Guard_Blocked"] = False
    annotations["Root_Selective_Permutation_Guard_Blocked"] = False
    annotations["Selective_Permutation_Guard_Blocked"] = False
    annotations["Selective_Permutation_P_Value"] = pd.NA
    annotations["Selective_Permutation_Base_P_Value"] = pd.NA
    annotations["Selective_Permutation_Guard_Refined"] = False
    annotations["Selective_Permutation_Guard_Scope"] = ""
    annotations.loc["root", "Root_Stability_Guard_Blocked"] = True
    annotations.loc["A", "Selective_Permutation_Guard_Blocked"] = True

    result = SimpleNamespace(
        extra={
            "tree": tree,
            "annotations": annotations,
            "decomposition": {
                "cluster_assignments": {
                    0: {"root_node": "A", "leaves": ["A1", "A2"], "size": 2},
                    1: {"root_node": "B1", "leaves": ["B1"], "size": 1},
                    2: {"root_node": "B2", "leaves": ["B2"], "size": 1},
                },
                "traversal_trace": [
                    {
                        "node_id": "root",
                        "decision": "pass_through",
                        "is_leaf": False,
                        "n_children": 2,
                        "n_descendant_leaves": 4,
                        "final_boundary": False,
                    },
                    {
                        "node_id": "A",
                        "decision": "boundary",
                        "is_leaf": False,
                        "n_children": 2,
                        "n_descendant_leaves": 2,
                        "final_boundary": True,
                    },
                    {
                        "node_id": "B",
                        "decision": "split",
                        "is_leaf": False,
                        "n_children": 2,
                        "n_descendant_leaves": 2,
                        "final_boundary": False,
                    },
                    {
                        "node_id": "B1",
                        "decision": "boundary",
                        "is_leaf": True,
                        "n_children": 0,
                        "n_descendant_leaves": 1,
                        "final_boundary": True,
                    },
                ],
            },
        }
    )

    rows = _build_node_decisions(
        case_id="unit",
        data_role="null",
        method_id="fixed_coordinate_global_passthrough_refined_v1",
        replicate=0,
        data_seed=1,
        result=result,
    )
    classes = dict(zip(rows["node_id"], rows["decision_class"]))

    assert classes["root"] == "selected_root_blocked"
    assert classes["A"] == "selected_family_blocked"
    assert classes["B"] == "accepted_internal_split"
    assert classes["B1"] == "leaf_fragment"
    incidence = dict(zip(rows["node_id"], rows["topology_incidence_role"]))
    assert incidence["root"] == "root"
    assert incidence["B"] == "internal"
    assert incidence["B1"] == "leaf"
    assert not bool(
        rows.loc[rows["node_id"].eq("root"), "topology_has_incoming_edge"].iloc[0]
    )
    assert bool(
        rows.loc[rows["node_id"].eq("B"), "topology_has_outgoing_test"].iloc[0]
    )
    assert rows.loc[
        rows["node_id"].eq("B1"),
        "conditional_topology_status",
    ].iloc[0] == "leaf_no_outgoing_test_fail_closed"
    assert rows.loc[
        rows["node_id"].eq("B"),
        "sibling_projection_dimension",
    ].iloc[0] == 3.0


def test_production_components_remain_diagnostic_or_fail_closed() -> None:
    rows = pd.DataFrame.from_records(
        [
            {
                "source_family": "binary_template",
                "method_id": "fixed_coordinate_global_passthrough_refined_v1",
                "data_role": "selected_null",
                "adaptive_projection_avoided": True,
                "false_split": False,
                "ari": 1.0,
            },
            {
                "source_family": "binary_template",
                "method_id": "fixed_coordinate_global_passthrough_refined_v1",
                "data_role": "signal",
                "adaptive_projection_avoided": True,
                "false_split": False,
                "ari": 0.9,
            },
        ]
    )

    components, production = build_selected_family_production_components(rows)

    assert "fixed_profile_transfer_candidate" in set(
        components["component_status"]
    )
    assert set(production["production_decision"]) <= {
        "diagnostic_only",
        "fail_closed_undefined",
    }


def test_run_selected_family_traversal_panel_writes_outputs(tmp_path) -> None:
    outputs = run_selected_family_traversal_panel(
        SelectedFamilyTraversalPanelConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            methods=("fixed_coordinate_guarded_v1",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
        )
    )

    for path in outputs.values():
        assert path.exists(), path

    assert outputs["checkpoint_rows_dir"].is_dir()
    traversal = pd.read_csv(outputs["selected_family_traversal_rows"])
    nodes = pd.read_csv(outputs["multiscale_node_decisions"])
    regions = pd.read_csv(outputs["multiscale_regions"])
    genes = pd.read_csv(outputs["multiscale_gene_assignments"])
    production = pd.read_csv(outputs["production_admissibility_summary"])

    assert traversal.shape[0] == 1
    assert traversal["data_role"].tolist() == ["selected_null"]
    assert set(nodes["decision_class"]).issubset(
        {
            "stable_boundary",
            "selected_root_blocked",
            "selected_family_blocked",
            "spectral_transport_blocked",
            "unstable_passthrough_zone",
            "accepted_internal_split",
            "leaf_fragment",
        }
    )
    assert not regions.empty
    assert not genes.empty
    assert production["production_decision"].isin(
        {"diagnostic_only", "fail_closed_undefined"}
    ).all()


def test_run_selected_family_traversal_panel_resumes_from_checkpoints(
    tmp_path,
    monkeypatch,
) -> None:
    config = SelectedFamilyTraversalPanelConfig(
        output_dir=tmp_path,
        suite="binary",
        case_names=("binary_2clusters",),
        methods=("fixed_coordinate_guarded_v1",),
        data_roles=("null",),
        sibling_alpha=0.01,
        edge_alpha=0.001,
        replicates=1,
        base_seed=20260613,
    )
    run_selected_family_traversal_panel(config)

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("resume should read selected-family checkpoints")

    monkeypatch.setattr(selected_panel, "_run_one", fail_if_recomputed)
    outputs = run_selected_family_traversal_panel(
        SelectedFamilyTraversalPanelConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            methods=("fixed_coordinate_guarded_v1",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
            resume_from_checkpoints=True,
        )
    )

    rows = pd.read_csv(outputs["selected_family_traversal_rows"])
    assert rows.shape[0] == 1
    assert rows.iloc[0]["data_role"] == "selected_null"


def test_run_selected_family_traversal_panel_records_row_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    def slow_row(*_args, **_kwargs):
        time.sleep(2.0)
        raise AssertionError("timeout should fire before slow row returns")

    monkeypatch.setattr(selected_panel, "_run_one", slow_row)
    outputs = run_selected_family_traversal_panel(
        SelectedFamilyTraversalPanelConfig(
            output_dir=tmp_path,
            suite="binary",
            case_names=("binary_2clusters",),
            methods=("fixed_coordinate_guarded_v1",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
            per_row_timeout_seconds=0.1,
        )
    )

    manifest = json.loads(outputs["manifest"].read_text())
    rows = pd.read_csv(outputs["selected_family_traversal_rows"])

    assert rows.empty
    assert manifest["skipped_cases"][0]["case_id"] == "binary_2clusters"
    assert "exceeded" in manifest["skipped_cases"][0]["reason"]


def test_run_selected_family_traversal_panel_records_unsupported_skips(tmp_path) -> None:
    outputs = run_selected_family_traversal_panel(
        SelectedFamilyTraversalPanelConfig(
            output_dir=tmp_path,
            suite="full",
            case_names=("phylo_dna_4taxa_low_mut", "binary_2clusters"),
            methods=("fixed_coordinate_guarded_v1",),
            data_roles=("null",),
            sibling_alpha=0.01,
            edge_alpha=0.001,
            replicates=1,
            base_seed=20260613,
        )
    )

    manifest = json.loads(outputs["manifest"].read_text())
    traversal = pd.read_csv(outputs["selected_family_traversal_rows"])

    assert traversal["case_id"].tolist() == ["binary_2clusters"]
    assert len(manifest["skipped_cases"]) == 1


def test_multiscale_umap_overlay_joins_regions_and_renders_plot(tmp_path) -> None:
    assignments_path = tmp_path / "assignments.csv"
    umap_path = tmp_path / "umap.csv"
    pd.DataFrame.from_records(
        [
            {
                "sample_id": "g1",
                "method_id": "fixed_coordinate_global_passthrough_refined_v1",
                "data_role": "selected_null",
                "replicate": 0,
                "stable_region_id": "flat_cluster_0",
                "zone_region_id": "",
                "final_fragment_cluster_id": 0,
            },
            {
                "sample_id": "g2",
                "method_id": "fixed_coordinate_global_passthrough_refined_v1",
                "data_role": "selected_null",
                "replicate": 0,
                "stable_region_id": "flat_cluster_1",
                "zone_region_id": "zone_N1",
                "final_fragment_cluster_id": 1,
            },
        ]
    ).to_csv(assignments_path, index=False)
    pd.DataFrame.from_records(
        [
            {"gene_symbol": "g1", "UMAP-1": 0.0, "UMAP-2": 1.0},
            {"gene_symbol": "g2", "UMAP-1": 1.0, "UMAP-2": 0.0},
        ]
    ).to_csv(umap_path, index=False)

    overlay = load_overlay_data(
        gene_assignments_path=assignments_path,
        umap_coordinates_path=umap_path,
        method_id="fixed_coordinate_global_passthrough_refined_v1",
        data_role="selected_null",
        replicate=0,
    )
    plot_path = render_multiscale_umap_overlay(
        overlay,
        tmp_path / "overlay.png",
        top_regions=2,
    )

    assert overlay.shape[0] == 2
    assert overlay["has_zone_overlay"].tolist() == [False, True]
    assert plot_path.exists()
