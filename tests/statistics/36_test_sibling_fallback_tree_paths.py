from __future__ import annotations

import numpy as np
import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_spectral_dims,
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


def _build_mixed_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "root",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="root",
        leaf_count=300,
    )
    tree.add_node(
        "I",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="I",
        leaf_count=200,
    )
    tree.add_node(
        "L1",
        is_leaf=True,
        distribution=np.array([0.10, 0.20, 0.20, 0.80, 0.90, 0.80], dtype=float),
        label="L1",
        leaf_count=100,
    )
    tree.add_node(
        "L2",
        is_leaf=True,
        distribution=np.array([0.90, 0.80, 0.80, 0.20, 0.10, 0.20], dtype=float),
        label="L2",
        leaf_count=100,
    )
    tree.add_node(
        "L3",
        is_leaf=True,
        distribution=np.array([0.80, 0.80, 0.80, 0.20, 0.20, 0.20], dtype=float),
        label="L3",
        leaf_count=100,
    )
    tree.add_edge("root", "I")
    tree.add_edge("root", "L3")
    tree.add_edge("I", "L1")
    tree.add_edge("I", "L2")

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "root": 300,
                "I": 200,
                "L1": 100,
                "L2": 100,
                "L3": 100,
            }
        }
    )
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        index=["L1", "L2", "L3"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def test_cherry_with_leaf_data_omits_leaf_pair_parent_from_sibling_dims() -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()

    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    out = bundle.annotated_df

    spectral_dims = out.attrs["_spectral_dims"]
    assert spectral_dims["A"] == 0
    assert spectral_dims["B"] == 0
    assert spectral_dims["root"] > 0

    sibling_dims = derive_sibling_spectral_dims(tree, out)
    assert sibling_dims is None

    assert np.isfinite(out.loc["root", "Sibling_Degrees_of_Freedom"])
    assert np.isfinite(out.loc["root", "Sibling_Divergence_P_Value"])


def test_mixed_parent_with_leaf_data_keeps_internal_parent_in_sibling_dims() -> None:
    tree, annotations_df, leaf_data = _build_mixed_tree()

    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    out = bundle.annotated_df

    spectral_dims = out.attrs["_spectral_dims"]
    assert spectral_dims["L1"] == 0
    assert spectral_dims["L2"] == 0
    assert spectral_dims["L3"] == 0
    assert spectral_dims["I"] > 0
    assert spectral_dims["root"] > 0

    sibling_dims = derive_sibling_spectral_dims(tree, out)
    assert sibling_dims is not None
    assert set(sibling_dims) == {"root"}
    assert sibling_dims["root"] > 0
    assert "I" not in sibling_dims


def test_decompose_without_leaf_data_disables_spectral_metadata_and_merges() -> None:
    tree, annotations_df, _leaf_data = _build_cherry_tree()

    result = tree.decompose(annotations_df=annotations_df.copy(), leaf_data=None)

    assert result["num_clusters"] == 1
    assert tree.annotations_df is not None
    assert tree.annotations_df.attrs["_spectral_dims"] is None
    assert np.isnan(tree.annotations_df.loc["root", "Sibling_Degrees_of_Freedom"])
    assert np.isnan(tree.annotations_df.loc["root", "Sibling_Divergence_P_Value"])
