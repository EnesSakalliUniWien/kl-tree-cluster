from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.tree.poset_tree import PosetTree


def _build_cherry_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    distributions = {
        "top": [0.20, 0.20, 0.20, 0.80, 0.80, 0.80],
        "root": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        "A": [0.12, 0.12, 0.12, 0.88, 0.88, 0.88],
        "B": [0.88, 0.88, 0.88, 0.12, 0.12, 0.12],
        "cal": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        "C": [0.49, 0.49, 0.49, 0.51, 0.51, 0.51],
        "D": [0.51, 0.51, 0.51, 0.49, 0.49, 0.49],
    }
    leaf_counts = {
        "top": 400,
        "root": 200,
        "A": 100,
        "B": 100,
        "cal": 200,
        "C": 100,
        "D": 100,
    }
    leaves = {"A", "B", "C", "D"}
    for node_id, distribution in distributions.items():
        tree.add_node(
            node_id,
            is_leaf=node_id in leaves,
            distribution=np.asarray(distribution, dtype=float),
            label=node_id,
            leaf_count=leaf_counts[node_id],
        )
    tree.add_edges_from(
        [
            ("top", "root", {"branch_length": 0.10}),
            ("top", "cal", {"branch_length": 0.10}),
            ("root", "A", {"branch_length": 0.25}),
            ("root", "B", {"branch_length": 0.20}),
            ("cal", "C", {"branch_length": 0.10}),
            ("cal", "D", {"branch_length": 0.10}),
        ]
    )
    tree.graph["root"] = "top"

    annotations_df = pd.DataFrame({"leaf_count": leaf_counts})
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [0.49, 0.49, 0.49, 0.51, 0.51, 0.51],
            [0.51, 0.51, 0.51, 0.49, 0.49, 0.49],
        ],
        index=["A", "B", "C", "D"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def _gate_bundle():
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df,
        edge_alpha=0.007,
        sibling_alpha=0.123,
        leaf_data=leaf_data,
    )
    return tree, bundle


def test_bundle_is_the_complete_traversal_input() -> None:
    tree, bundle = _gate_bundle()

    result = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=bundle,
    ).decompose_tree()

    assert result["num_clusters"] >= 1
    assert result["independence_analysis"]["edge_alpha"] == 0.007
    assert result["independence_analysis"]["sibling_alpha"] == 0.123


def test_direct_annotations_do_not_invent_alpha_provenance() -> None:
    tree, bundle = _gate_bundle()

    result = TreeDecomposition(
        tree=tree,
        annotations_df=bundle.annotated_df,
    ).decompose_tree()

    assert result["num_clusters"] >= 1
    assert result["independence_analysis"]["edge_alpha"] is None
    assert result["independence_analysis"]["sibling_alpha"] is None


def test_traversal_requires_exactly_one_annotation_input() -> None:
    tree, bundle = _gate_bundle()

    with pytest.raises(ValueError, match="not both"):
        TreeDecomposition(
            tree=tree,
            annotations_df=bundle.annotated_df,
            gate_annotation_bundle=bundle,
        )

    with pytest.raises(ValueError, match="is required"):
        TreeDecomposition(tree=tree)


def test_bundle_metadata_must_identify_the_gate_pipeline() -> None:
    tree, bundle = _gate_bundle()
    invalid_bundle = replace(
        bundle,
        metadata=replace(bundle.metadata, pipeline="other_pipeline"),
    )

    with pytest.raises(ValueError, match="metadata.pipeline"):
        TreeDecomposition(tree=tree, gate_annotation_bundle=invalid_bundle)


@pytest.mark.parametrize(
    "guard_kwargs",
    [
        {"selected_family_passthrough_guard": True},
        {"spectral_transport_passthrough_guard": True},
    ],
)
def test_bundle_metadata_owns_guard_behavior(guard_kwargs: dict[str, bool]) -> None:
    tree, bundle = _gate_bundle()

    with pytest.raises(ValueError, match="metadata owns passthrough guard behavior"):
        TreeDecomposition(
            tree=tree,
            gate_annotation_bundle=bundle,
            **guard_kwargs,
        )
