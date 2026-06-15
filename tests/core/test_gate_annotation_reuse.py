from __future__ import annotations

from dataclasses import replace

import kl_clustering_analysis.hierarchy_analysis.tree_decomposition as tree_decomposition_module
import numpy as np
import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.inflation_correction.empirical_null_inflation_estimation import (
    CalibrationSupportThresholds,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _build_cherry_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "top",
        is_leaf=False,
        distribution=np.array([0.20, 0.20, 0.20, 0.80, 0.80, 0.80], dtype=float),
        label="top",
        leaf_count=400,
    )
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
    tree.add_node(
        "cal",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="cal",
        leaf_count=200,
    )
    tree.add_node(
        "C",
        is_leaf=True,
        distribution=np.array([0.49, 0.49, 0.49, 0.51, 0.51, 0.51], dtype=float),
        label="C",
        leaf_count=100,
    )
    tree.add_node(
        "D",
        is_leaf=True,
        distribution=np.array([0.51, 0.51, 0.51, 0.49, 0.49, 0.49], dtype=float),
        label="D",
        leaf_count=100,
    )
    tree.add_edge("top", "root", branch_length=0.10)
    tree.add_edge("top", "cal", branch_length=0.10)
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)
    tree.add_edge("cal", "C", branch_length=0.10)
    tree.add_edge("cal", "D", branch_length=0.10)
    tree.graph["root"] = "top"

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "top": 400,
                "root": 200,
                "A": 100,
                "B": 100,
                "cal": 200,
                "C": 100,
                "D": 100,
            }
        }
    )
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


def test_decompose_reuses_matching_gate_annotations(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)

    assert bundle.metadata.pipeline == "gate_annotation"
    assert bundle.annotated_df.attrs == {}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Gate annotation pipeline should not rerun")

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        fail_if_called,
    )

    result = tree.decompose(gate_annotation_bundle=bundle, leaf_data=leaf_data)

    assert result["num_clusters"] >= 1


def test_decompose_recomputes_stale_gate_annotations(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    stale_bundle = replace(
        bundle,
        metadata=replace(
            bundle.metadata,
            edge=replace(bundle.metadata.edge, alpha=-1.0),
        ),
    )

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

    result = tree.decompose(gate_annotation_bundle=stale_bundle, leaf_data=leaf_data)

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_when_sibling_gate_method_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["sibling_gate_method"] == "fixed_coordinate_bh"
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(
        gate_annotation_bundle=bundle,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )

    assert calls == 1
    assert result["num_clusters"] >= 1
    fixed_p_values = tree.annotations_df.loc[
        tree.annotations_df["Sibling_Test_Method"].eq("fixed_coordinate_bh"),
        "Sibling_Divergence_P_Value",
    ]
    assert not fixed_p_values.empty
    assert fixed_p_values.between(0.0, 1.0).all()


def test_decompose_recomputes_when_sibling_gate_penalty_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=1.0,
    )

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["sibling_gate_method"] == "fixed_coordinate_bh"
        assert kwargs["sibling_gate_alpha_penalty"] == 50.0
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(
        gate_annotation_bundle=bundle,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
        sibling_gate_alpha_penalty=50.0,
    )

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_when_root_stability_guard_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["sibling_gate_method"] == "fixed_coordinate_bh"
        assert kwargs["root_stability_guard_threshold"] == 0.50
        assert kwargs["root_stability_subsample_replicates"] == 4
        assert kwargs["root_stability_feature_fraction"] == 0.75
        assert kwargs["root_stability_seed"] == 11
        assert kwargs["root_stability_tree_distance_metric"] == "hamming"
        assert kwargs["root_stability_tree_linkage_method"] == "average"
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(
        gate_annotation_bundle=bundle,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
        root_stability_guard_threshold=0.50,
        root_stability_subsample_replicates=4,
        root_stability_feature_fraction=0.75,
        root_stability_seed=11,
    )

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_when_root_selective_guard_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["sibling_gate_method"] == "fixed_coordinate_bh"
        assert kwargs["root_selective_permutation_guard_replicates"] == 2
        assert kwargs["root_selective_permutation_guard_seed"] == 13
        assert kwargs["root_selective_permutation_guard_alpha"] == 0.01
        assert kwargs["root_selective_permutation_guard_tree_distance_metric"] == (
            "hamming"
        )
        assert kwargs["root_selective_permutation_guard_tree_linkage_method"] == (
            "average"
        )
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(
        gate_annotation_bundle=bundle,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
        root_selective_permutation_guard_replicates=2,
        root_selective_permutation_guard_seed=13,
        root_selective_permutation_guard_alpha=0.01,
    )

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_when_sibling_gate_profile_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["sibling_gate_profile"] == "fixed_coordinate_guarded_v1"
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(
        gate_annotation_bundle=bundle,
        leaf_data=leaf_data,
        sibling_gate_profile="fixed_coordinate_guarded_v1",
    )

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_when_selective_root_profile_changes(monkeypatch) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        leaf_data=leaf_data,
        sibling_gate_profile="fixed_coordinate_guarded_v1",
    )

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["sibling_gate_profile"] == "fixed_coordinate_selective_root_v1"
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    result = tree.decompose(
        gate_annotation_bundle=bundle,
        leaf_data=leaf_data,
        sibling_gate_profile="fixed_coordinate_selective_root_v1",
    )

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

    result = tree.decompose(gate_annotation_bundle=bundle, leaf_data=changed_leaf_data)

    assert calls == 1
    assert result["num_clusters"] >= 1


def test_decompose_recomputes_annotations_when_internal_support_enforcement_changes(
    monkeypatch,
) -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()
    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)

    calls = 0

    def counted_pipeline(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["enforce_internal_support_thresholds"] is True
        assert isinstance(kwargs["internal_support_thresholds"], CalibrationSupportThresholds)
        return run_gate_annotation_pipeline(*args, **kwargs)

    monkeypatch.setattr(
        tree_decomposition_module,
        "run_gate_annotation_pipeline",
        counted_pipeline,
    )

    with pytest.raises(ValueError, match="undefined_sparse_context"):
        tree.decompose(
            gate_annotation_bundle=bundle,
            leaf_data=leaf_data,
            enforce_internal_support_thresholds=True,
            internal_support_thresholds=CalibrationSupportThresholds(
                min_supported_records=1,
                min_family_supported_records=1,
                min_stopped_or_null_records=1,
                min_family_effective_sample_size=1.0,
                min_local_effective_sample_size=1.0,
                max_weight_share=1.0,
                max_leave_one_record_delta_log_c=10.0,
            ),
        )

    assert calls == 1
