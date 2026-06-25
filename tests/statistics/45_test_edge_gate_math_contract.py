from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator
from tree_break_selection.hierarchy_analysis.decomposition.gates.column_contracts import (
    EDGE_GATE_COLUMNS,
    SIBLING_GATE_COLUMNS,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns
from tree_break_selection.tree.poset_tree import PosetTree


def _minimal_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    for node_id, distribution, leaf_count, is_leaf in [
        ("root", np.array([0.5, 0.5, 0.5, 0.5]), 4, False),
        ("A", np.array([0.2, 0.2, 0.2, 0.2]), 2, False),
        ("B", np.array([0.8, 0.8, 0.8, 0.8]), 2, False),
        ("a0", np.array([0.1, 0.1, 0.1, 0.1]), 1, True),
        ("a1", np.array([0.3, 0.3, 0.3, 0.3]), 1, True),
        ("b0", np.array([0.7, 0.7, 0.7, 0.7]), 1, True),
        ("b1", np.array([0.9, 0.9, 0.9, 0.9]), 1, True),
    ]:
        tree.add_node(
            node_id,
            distribution=distribution,
            leaf_count=leaf_count,
            is_leaf=is_leaf,
            label=node_id,
        )
    tree.add_edge("root", "A", branch_length=10.0)
    tree.add_edge("root", "B", branch_length=10.0)
    tree.add_edge("A", "a0", branch_length=0.1)
    tree.add_edge("A", "a1", branch_length=0.1)
    tree.add_edge("B", "b0", branch_length=0.1)
    tree.add_edge("B", "b1", branch_length=0.1)
    tree.graph["root"] = "root"
    annotations = pd.DataFrame(
        {"leaf_count": {node_id: tree.nodes[node_id]["leaf_count"] for node_id in tree.nodes}}
    )
    leaf_data = pd.DataFrame(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7, 0.7],
            [0.9, 0.9, 0.9, 0.9],
        ],
        index=["a0", "a1", "b0", "b1"],
        columns=[f"PC{i}" for i in range(4)],
        dtype=float,
    )
    return tree, annotations, leaf_data


def _fake_edge_dataframe(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for column in EDGE_GATE_COLUMNS:
        if column.endswith("_Significant") or column.endswith("_Tested"):
            out[column] = True
        elif column.endswith("_Invalid") or column.endswith("_Ancestor_Blocked"):
            out[column] = False
        else:
            out[column] = 0.0
    return out


def _fake_sibling_dataframe(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for column in SIBLING_GATE_COLUMNS:
        if column in {"Sibling_BH_Different"}:
            out[column] = True
        elif column in {
            "Sibling_BH_Same",
            "Sibling_Divergence_Skipped",
            "Sibling_Divergence_Invalid",
        }:
            out[column] = False
        elif (
            column.endswith("_Method")
            or column.endswith("_Calibration")
            or column.endswith("_Role")
        ):
            out[column] = "fixed_coordinate_bh"
        else:
            out[column] = 0.0
    return out


def _append_fake_sibling_columns(annotations_df: pd.DataFrame) -> pd.DataFrame:
    out = annotations_df.copy()
    sibling_columns = _fake_sibling_dataframe(annotations_df.index)
    for column in sibling_columns:
        out[column] = sibling_columns[column]
    return out


def test_fixed_coordinate_sibling_gate_does_not_raise_edge_projection_floor_to_feature_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree, annotations, leaf_data = _minimal_tree()
    feature_space = continuous_feature_space_from_columns(leaf_data.columns)
    captured: dict[str, int] = {}

    def fake_edge_annotation(*args, **kwargs):
        captured["spectral_minimum_dimension"] = int(kwargs["spectral_minimum_dimension"])
        spectral_context = SpectralContext(
            test_projection_dimensions_by_node={"root": captured["spectral_minimum_dimension"]},
            raw_mp_signal_counts_by_node={"root": 0},
            effective_independent_rows_by_node={"root": len(leaf_data)},
            mp_threshold_rows_by_node={"root": len(leaf_data)},
            principal_component_projections_by_node={
                "root": np.eye(captured["spectral_minimum_dimension"], leaf_data.shape[1])
            },
            principal_component_eigenvalues_by_node={
                "root": np.ones(captured["spectral_minimum_dimension"])
            },
        )
        return _fake_edge_dataframe(annotations.index), spectral_context

    monkeypatch.setattr(
        orchestrator,
        "annotate_child_parent_divergence_with_context",
        fake_edge_annotation,
    )
    monkeypatch.setattr(
        orchestrator,
        "annotate_fixed_subspace_sibling_divergence",
        lambda tree, annotations_df, **kwargs: _append_fake_sibling_columns(annotations_df),
    )
    monkeypatch.setattr(
        orchestrator,
        "annotate_fixed_subspace_sibling_evidence_channels",
        lambda tree, annotations_df, **kwargs: annotations_df,
    )

    bundle = run_gate_annotation_pipeline(
        tree,
        annotations,
        leaf_data=leaf_data,
        feature_space=feature_space,
        spectral_minimum_dimension=2,
        sibling_gate_method="fixed_coordinate_bh",
    )

    assert captured["spectral_minimum_dimension"] == 2
    assert bundle.metadata.config.spectral_minimum_dimension == 2


def test_projected_gates_share_adaptive_projection_fraction_and_full_candidate_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree, annotations, leaf_data = _minimal_tree()
    feature_space = continuous_feature_space_from_columns(leaf_data.columns)
    binary_parents = ["root", "A", "B"]
    captured: dict[str, object] = {}

    def fake_edge_annotation(*args, **kwargs):
        captured["spectral_minimum_dimension"] = int(kwargs["spectral_minimum_dimension"])
        captured["spectral_projection_basis_dimension"] = int(
            kwargs["spectral_projection_basis_dimension"]
        )
        captured["edge_adaptive_fraction"] = kwargs["adaptive_projection_dimension_energy_fraction"]
        test_dims = {node_id: 2 for node_id in tree.nodes}
        for leaf_id in ["a0", "a1", "b0", "b1"]:
            test_dims[leaf_id] = 0
        spectral_context = SpectralContext(
            test_projection_dimensions_by_node=test_dims,
            raw_mp_signal_counts_by_node={node_id: 0 for node_id in tree.nodes},
            effective_independent_rows_by_node={node_id: len(leaf_data) for node_id in tree.nodes},
            mp_threshold_rows_by_node={node_id: len(leaf_data) for node_id in tree.nodes},
            principal_component_projections_by_node={
                node_id: np.eye(leaf_data.shape[1]) for node_id in binary_parents
            },
            principal_component_eigenvalues_by_node={
                node_id: np.ones(leaf_data.shape[1]) for node_id in binary_parents
            },
        )
        return _fake_edge_dataframe(annotations.index), spectral_context

    def fake_sibling_annotation(*args, **kwargs):
        captured["sibling_adaptive_fraction"] = kwargs[
            "adaptive_projection_dimension_energy_fraction"
        ]
        return _append_fake_sibling_columns(args[1])

    monkeypatch.setattr(
        orchestrator,
        "annotate_child_parent_divergence_with_context",
        fake_edge_annotation,
    )
    monkeypatch.setattr(
        orchestrator,
        "annotate_sibling_divergence",
        fake_sibling_annotation,
    )
    monkeypatch.setattr(
        orchestrator,
        "annotate_fixed_subspace_sibling_evidence_channels",
        lambda tree, annotations_df, **kwargs: annotations_df,
    )

    bundle = run_gate_annotation_pipeline(
        tree,
        annotations,
        leaf_data=leaf_data,
        feature_space=feature_space,
        spectral_minimum_dimension=2,
        sibling_gate_method="projected_wald_inflation",
        adaptive_projection_dimension_energy_fraction=0.90,
    )

    assert captured["spectral_minimum_dimension"] == 2
    assert captured["spectral_projection_basis_dimension"] == leaf_data.shape[1]
    assert captured["edge_adaptive_fraction"] == 0.90
    assert captured["sibling_adaptive_fraction"] == 0.90
    assert bundle.metadata.config.spectral_minimum_dimension == 2
    assert bundle.metadata.config.adaptive_projection_dimension_energy_fraction == 0.90


def test_branch_length_variance_policy_is_part_of_gate_reuse_metadata() -> None:
    tree, annotations, leaf_data = _minimal_tree()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations,
        leaf_data=leaf_data,
        edge_branch_length_variance_policy="normalized_branch_length",
    )

    assert bundle.metadata.config.edge_branch_length_variance_policy == "normalized_branch_length"
