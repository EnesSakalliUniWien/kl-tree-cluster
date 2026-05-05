from __future__ import annotations

from copy import deepcopy

import kl_clustering_analysis.hierarchy_analysis.tree_decomposition as tree_decomposition_module
import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    GATE_ANNOTATION_METADATA_ATTR,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _build_cherry_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "root",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="root",
        leaf_count=200,
    )
    tree.add_node(
        "A",
        is_leaf=True,
        distribution=np.array([0.12, 0.12, 0.12, 0.88, 0.88, 0.88], dtype=float),
        label="A",
        leaf_count=100,
    )
    tree.add_node(
        "B",
        is_leaf=True,
        distribution=np.array([0.88, 0.88, 0.88, 0.12, 0.12, 0.12], dtype=float),
        label="B",
        leaf_count=100,
    )
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "root": 200,
                "A": 100,
                "B": 100,
            }
        }
    )
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        index=["A", "B"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def test_decompose_reuses_matching_gate_annotations(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    annotated_df = bundle.annotated_df

    assert annotated_df.attrs[GATE_ANNOTATION_METADATA_ATTR] == bundle.metadata

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Gate annotation pipeline should not rerun")

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        fail_if_called,
    )

    result = tree.decompose(annotations_df=annotated_df, leaf_data=leaf_data)

    assert result["num_clusters"] >= 1


def test_decompose_recomputes_stale_gate_annotations(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    stale_df = bundle.annotated_df.copy()
    stale_metadata = deepcopy(stale_df.attrs[GATE_ANNOTATION_METADATA_ATTR])
    stale_metadata["edge"]["alpha"] = -1.0
    stale_df.attrs[GATE_ANNOTATION_METADATA_ATTR] = stale_metadata

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(annotations_df=stale_df, leaf_data=leaf_data)

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_annotations_when_leaf_data_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    changed_leaf_data = leaf_data.copy()
    changed_leaf_data.iloc[:, :] = 1.0 - changed_leaf_data.to_numpy()

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(annotations_df=bundle.annotated_df, leaf_data=changed_leaf_data)

    assert calls == 1
    assert result["num_clusters"] >= 1
