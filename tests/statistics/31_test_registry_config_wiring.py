from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.column_contracts import (
    EDGE_GATE_COLUMNS,
    SIBLING_GATE_COLUMNS,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    SIBLING_GATE_PROFILES,
    apply_root_selective_permutation_guard,
    apply_root_stability_guard,
    resolve_effective_sibling_alpha,
    resolve_sibling_gate_profile_config,
    run_gate_annotation_pipeline,
    selected_global_sibling_min_permutation_p_value,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.fixed_subspace_annotation import (
    fixed_subspace_sibling_p_value,
)
from kl_clustering_analysis.tree.feature_space import infer_feature_space_from_columns


def _build_small_tree_with_leaf_data() -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    tree = nx.DiGraph()
    tree.graph["root"] = "root"
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)
    tree.add_edge("cal", "C", branch_length=0.10)
    tree.add_edge("cal", "D", branch_length=0.10)

    root_dist = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)
    a_dist = np.array([0.12, 0.12, 0.12, 0.88, 0.88, 0.88], dtype=np.float64)
    b_dist = np.array([0.88, 0.88, 0.88, 0.12, 0.12, 0.12], dtype=np.float64)
    cal_dist = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)
    c_dist = np.array([0.49, 0.49, 0.49, 0.51, 0.51, 0.51], dtype=np.float64)
    d_dist = np.array([0.51, 0.51, 0.51, 0.49, 0.49, 0.49], dtype=np.float64)

    for node, dist, leaf_count, is_leaf in (
        ("root", root_dist, 200, False),
        ("A", a_dist, 100, True),
        ("B", b_dist, 100, True),
        ("cal", cal_dist, 200, False),
        ("C", c_dist, 100, True),
        ("D", d_dist, 100, True),
    ):
        tree.nodes[node]["distribution"] = dist
        tree.nodes[node]["leaf_count"] = leaf_count
        tree.nodes[node]["is_leaf"] = is_leaf
        tree.nodes[node]["label"] = node

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
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
        dtype=np.float64,
    )
    return tree, annotations_df, leaf_data


def _build_passthrough_tree_with_leaf_data() -> tuple[
    nx.DiGraph,
    pd.DataFrame,
    pd.DataFrame,
]:
    tree = nx.DiGraph()
    tree.graph["root"] = "root"
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("B", "C"),
            ("B", "D"),
        ]
    )
    for node in ("root", "A", "B", "C", "D"):
        tree.nodes[node]["is_leaf"] = node in {"A", "C", "D"}
        tree.nodes[node]["label"] = node
        tree.nodes[node]["leaf_count"] = 1 if tree.nodes[node]["is_leaf"] else 3
        tree.nodes[node]["distribution"] = np.array(
            [0.5, 0.5, 0.5, 0.5],
            dtype=np.float64,
        )

    annotations_df = pd.DataFrame(index=["root", "A", "B", "C", "D"])
    leaf_data = pd.DataFrame(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
        ],
        index=["A", "C", "D"],
        dtype=np.float64,
    )
    return tree, annotations_df, leaf_data


def test_pipeline_supports_current_gate_annotation_contract() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
    )
    out = bundle.annotated_df
    for col in EDGE_GATE_COLUMNS:
        assert col in out.columns
    for col in SIBLING_GATE_COLUMNS:
        assert col in out.columns
    assert bundle.metadata.pipeline == "gate_annotation"
    assert bundle.metadata.edge.gate == "edge"
    assert bundle.metadata.sibling.gate == "sibling"
    assert bundle.metadata.config.sibling_gate_method == "projected_wald_inflation"
    assert bundle.metadata.config.sibling_gate_alpha_penalty == 1.0


def test_pipeline_supports_opt_in_fixed_coordinate_sibling_gate() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )
    out = bundle.annotated_df

    for col in EDGE_GATE_COLUMNS:
        assert col in out.columns
    for col in SIBLING_GATE_COLUMNS:
        assert col in out.columns
    assert bundle.metadata.sibling.gate == "sibling"
    assert bundle.metadata.config.sibling_gate_method == "fixed_coordinate_bh"
    assert bundle.metadata.config.sibling_gate_alpha_penalty == 1.0
    tested = out[out["Sibling_Test_Method"].eq("fixed_coordinate_bh")]
    assert not tested.empty
    assert tested["Sibling_Divergence_P_Value"].between(0.0, 1.0).all()
    assert (tested["Sibling_Projection_Dimension"] == "").all() or tested[
        "Sibling_Projection_Dimension"
    ].isna().all()


def test_pipeline_supports_opt_in_fixed_global_sibling_gate() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_global_chi_square",
    )
    out = bundle.annotated_df

    assert bundle.metadata.config.sibling_gate_method == "fixed_global_chi_square"
    tested = out[out["Sibling_Test_Method"].eq("fixed_global_chi_square")]
    assert not tested.empty
    assert tested["Sibling_Divergence_P_Value"].between(0.0, 1.0).all()
    assert tested["Sibling_Degrees_of_Freedom"].gt(0.0).all()


def test_fixed_profile_resolves_guarded_candidate_constants() -> None:
    (
        profile_id,
        method,
        penalty,
        threshold,
        replicates,
        fraction,
        seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_guarded_v1",
    )

    assert profile_id == "fixed_coordinate_guarded_v1"
    assert method == "fixed_coordinate_bh"
    assert penalty == 50.0
    assert threshold == 0.24
    assert replicates == 12
    assert fraction == 0.8
    assert seed == 0
    assert root_selective_replicates == 0
    assert root_selective_seed == 0
    assert root_selective_alpha is None
    assert root_selective_scope == "root"
    assert SIBLING_GATE_PROFILES[profile_id].status == "diagnostic_candidate"


def test_fixed_profile_resolves_selective_root_candidate_constants() -> None:
    (
        profile_id,
        method,
        penalty,
        threshold,
        stability_replicates,
        fraction,
        stability_seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_selective_root_v1",
    )

    assert profile_id == "fixed_coordinate_selective_root_v1"
    assert method == "fixed_coordinate_bh"
    assert penalty == 50.0
    assert threshold == 0.24
    assert stability_replicates == 12
    assert fraction == 0.8
    assert stability_seed == 0
    assert root_selective_replicates == 99
    assert root_selective_seed == 0
    assert root_selective_alpha == 0.01
    assert root_selective_scope == "root"
    assert SIBLING_GATE_PROFILES[profile_id].status == "diagnostic_candidate"


def test_fixed_profile_resolves_selective_traversal_candidate_constants() -> None:
    (
        profile_id,
        method,
        penalty,
        threshold,
        stability_replicates,
        fraction,
        stability_seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_selective_traversal_v1",
    )

    assert profile_id == "fixed_coordinate_selective_traversal_v1"
    assert method == "fixed_coordinate_bh"
    assert penalty == 50.0
    assert threshold == 0.24
    assert stability_replicates == 12
    assert fraction == 0.8
    assert stability_seed == 0
    assert root_selective_replicates == 99
    assert root_selective_seed == 0
    assert root_selective_alpha == 0.01
    assert root_selective_scope == "open_internal"
    assert SIBLING_GATE_PROFILES[profile_id].status == "diagnostic_candidate"


def test_fixed_profile_resolves_selective_passthrough_candidate_constants() -> None:
    (
        profile_id,
        method,
        penalty,
        threshold,
        stability_replicates,
        fraction,
        stability_seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_selective_passthrough_v1",
    )

    assert profile_id == "fixed_coordinate_selective_passthrough_v1"
    assert method == "fixed_coordinate_bh"
    assert penalty == 50.0
    assert threshold == 0.24
    assert stability_replicates == 12
    assert fraction == 0.8
    assert stability_seed == 0
    assert root_selective_replicates == 99
    assert root_selective_seed == 0
    assert root_selective_alpha == 0.01
    assert root_selective_scope == "passthrough_descendant"
    assert SIBLING_GATE_PROFILES[profile_id].status == "diagnostic_candidate"


def test_fixed_profile_resolves_global_passthrough_candidate_constants() -> None:
    (
        profile_id,
        method,
        penalty,
        threshold,
        stability_replicates,
        fraction,
        stability_seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_global_passthrough_v1",
    )

    assert profile_id == "fixed_coordinate_global_passthrough_v1"
    assert method == "fixed_coordinate_bh"
    assert penalty == 50.0
    assert threshold == 0.24
    assert stability_replicates == 12
    assert fraction == 0.8
    assert stability_seed == 0
    assert root_selective_replicates == 99
    assert root_selective_seed == 0
    assert root_selective_alpha == 0.01
    assert root_selective_scope == "global_sibling_min_passthrough_descendant"
    assert SIBLING_GATE_PROFILES[profile_id].status == "diagnostic_candidate"


def test_fixed_profile_resolves_refined_global_passthrough_candidate_constants() -> None:
    (
        profile_id,
        method,
        penalty,
        threshold,
        stability_replicates,
        fraction,
        stability_seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_global_passthrough_refined_v1",
    )

    assert profile_id == "fixed_coordinate_global_passthrough_refined_v1"
    assert method == "fixed_coordinate_bh"
    assert penalty == 50.0
    assert threshold == 0.24
    assert stability_replicates == 12
    assert fraction == 0.8
    assert stability_seed == 0
    assert root_selective_replicates == 99
    assert root_selective_seed == 0
    assert root_selective_alpha == 0.01
    assert root_selective_scope == (
        "global_sibling_min_passthrough_descendant_refined"
    )
    assert SIBLING_GATE_PROFILES[profile_id].status == "diagnostic_candidate"


def test_fixed_profile_allows_explicit_root_selective_guard_when_disabled() -> None:
    (
        _profile_id,
        _method,
        _penalty,
        _threshold,
        _stability_replicates,
        _fraction,
        _stability_seed,
        root_selective_replicates,
        root_selective_seed,
        root_selective_alpha,
        root_selective_scope,
    ) = resolve_sibling_gate_profile_config(
        sibling_gate_profile="fixed_coordinate_guarded_v1",
        root_selective_permutation_guard_replicates=5,
        root_selective_permutation_guard_seed=123,
        root_selective_permutation_guard_alpha=0.01,
    )

    assert root_selective_replicates == 5
    assert root_selective_seed == 123
    assert root_selective_alpha == 0.01
    assert root_selective_scope == "root"


def test_selective_root_profile_rejects_conflicting_guard_replicates() -> None:
    with pytest.raises(
        ValueError,
        match="conflicts with explicit root_selective_permutation_guard_replicates",
    ):
        resolve_sibling_gate_profile_config(
            sibling_gate_profile="fixed_coordinate_selective_root_v1",
            root_selective_permutation_guard_replicates=5,
        )


def test_fixed_profile_rejects_conflicting_explicit_method() -> None:
    with pytest.raises(ValueError, match="conflicts with explicit sibling_gate_method"):
        resolve_sibling_gate_profile_config(
            sibling_gate_profile="fixed_coordinate_guarded_v1",
            sibling_gate_method="fixed_block_bh",
        )


def test_pipeline_profile_avoids_adaptive_sibling_pca(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fixed profile must not resolve parent PCA inputs")

    def stable_root(*_args, **_kwargs):
        return {
            "root_stability_subsample_mean_ari": 1.0,
            "root_stability_subsample_median_ari": 1.0,
            "root_stability_subsample_q10_ari": 1.0,
        }

    monkeypatch.setattr(orchestrator, "_resolve_sibling_gate_inputs", fail_if_called)
    monkeypatch.setattr(
        orchestrator,
        "compute_root_feature_subsample_stability",
        stable_root,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        sibling_gate_profile="fixed_coordinate_guarded_v1",
    )

    assert bundle.metadata.config.sibling_gate_profile_id == (
        "fixed_coordinate_guarded_v1"
    )
    assert bundle.metadata.config.sibling_gate_method == "fixed_coordinate_bh"
    assert bundle.metadata.config.sibling_gate_alpha_penalty == 50.0
    assert bundle.metadata.config.root_stability_guard_threshold == 0.24
    assert bundle.metadata.config.root_stability_subsample_replicates == 12
    assert bundle.annotated_df["Sibling_Test_Method"].eq("fixed_coordinate_bh").any()


def test_fixed_sibling_gate_does_not_resolve_parent_pca_inputs(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fixed sibling gate must not resolve parent PCA inputs")

    monkeypatch.setattr(orchestrator, "_resolve_sibling_gate_inputs", fail_if_called)

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )

    assert bundle.metadata.config.sibling_gate_method == "fixed_coordinate_bh"
    assert bundle.annotated_df["Sibling_Test_Method"].eq("fixed_coordinate_bh").any()


def test_sibling_gate_alpha_penalty_is_passed_as_effective_alpha(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    seen_alpha = None
    original = orchestrator.annotate_fixed_subspace_sibling_divergence

    def wrapped(*args, **kwargs):
        nonlocal seen_alpha
        seen_alpha = kwargs["significance_level_alpha"]
        return original(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "annotate_fixed_subspace_sibling_divergence",
        wrapped,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        sibling_gate_alpha_penalty=50.0,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )

    assert seen_alpha == 0.0002
    assert bundle.metadata.sibling.alpha == 0.01
    assert bundle.metadata.config.sibling_gate_alpha_penalty == 50.0


def test_root_stability_guard_closes_only_unstable_open_root() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
    )
    root = "root"
    annotated = bundle.annotated_df.copy()
    annotated.loc[root, "Sibling_BH_Different"] = True
    annotated.loc[root, "Sibling_BH_Same"] = False
    annotated.loc["A", "Sibling_BH_Different"] = True

    guarded = apply_root_stability_guard(
        tree,
        annotated,
        {
            "root_stability_subsample_mean_ari": 0.20,
            "root_stability_subsample_median_ari": 0.25,
            "root_stability_subsample_q10_ari": 0.10,
        },
        threshold=0.50,
    )

    assert bool(guarded.loc[root, "Sibling_BH_Different"]) is False
    assert bool(guarded.loc[root, "Sibling_BH_Same"]) is True
    assert bool(guarded.loc[root, "Root_Stability_Guard_Blocked"]) is True
    assert bool(guarded.loc["A", "Sibling_BH_Different"]) is True
    assert guarded.loc[root, "Root_Stability_Subsample_Mean_ARI"] == 0.20


def test_root_selective_permutation_guard_closes_unselected_open_root(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_method="fixed_coordinate_bh",
    )
    root = "root"
    annotated = bundle.annotated_df.copy()
    annotated.loc[root, "Sibling_BH_Different"] = True
    annotated.loc[root, "Sibling_BH_Same"] = False
    annotated.loc["A", "Sibling_BH_Different"] = True

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
    )

    assert bool(guarded.loc[root, "Sibling_BH_Different"]) is False
    assert bool(guarded.loc[root, "Sibling_BH_Same"]) is True
    assert bool(guarded.loc[root, "Root_Selective_Permutation_Guard_Blocked"]) is True
    assert bool(guarded.loc["A", "Sibling_BH_Different"]) is True
    assert guarded.loc[root, "Root_Selective_Permutation_P_Value"] == 0.50
    assert guarded.loc[root, "Root_Selective_Permutation_Guard_Alpha"] == 0.01


def test_selective_permutation_guard_closes_open_internal_context(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_method="fixed_coordinate_bh",
    )
    annotated = bundle.annotated_df.copy()
    annotated.loc["root", "Sibling_BH_Different"] = False
    annotated.loc["root", "Sibling_BH_Same"] = True
    annotated.loc["cal", "Sibling_BH_Different"] = True
    annotated.loc["cal", "Sibling_BH_Same"] = False

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
        scope="open_internal",
    )

    assert bool(guarded.loc["root", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["cal", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["cal", "Sibling_BH_Same"]) is True
    assert bool(guarded.loc["cal", "Selective_Permutation_Guard_Blocked"]) is True
    assert guarded.loc["cal", "Selective_Permutation_P_Value"] == 0.50
    assert guarded.loc["cal", "Selective_Permutation_Guard_Scope"] == "open_internal"


def test_selective_permutation_guard_closes_only_passthrough_descendant(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_passthrough_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    annotated = annotations_df.copy()
    annotated["Child_Parent_Divergence_Significant"] = True
    annotated["Sibling_BH_Different"] = False
    annotated["Sibling_BH_Same"] = True
    annotated["Sibling_Divergence_Skipped"] = False
    annotated["Sibling_Divergence_P_Value"] = 1.0
    annotated.loc["B", "Sibling_BH_Different"] = True
    annotated.loc["B", "Sibling_BH_Same"] = False
    annotated.loc["B", "Sibling_Divergence_P_Value"] = 0.001

    calls: list[tuple[object, ...]] = []

    def fake_selected_root(data, *_args, **_kwargs):
        calls.append(tuple(data.index))
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
        scope="passthrough_descendant",
    )

    assert calls == [("C", "D")]
    assert bool(guarded.loc["root", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["B", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["B", "Selective_Permutation_Guard_Blocked"]) is True
    assert guarded.loc["B", "Selective_Permutation_Guard_Scope"] == (
        "passthrough_descendant"
    )


def test_global_selected_family_guard_closes_passthrough_descendant(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_passthrough_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    annotated = annotations_df.copy()
    annotated["Child_Parent_Divergence_Significant"] = True
    annotated["Sibling_BH_Different"] = False
    annotated["Sibling_BH_Same"] = True
    annotated["Sibling_Divergence_Skipped"] = False
    annotated["Sibling_Divergence_P_Value"] = 1.0
    annotated.loc["B", "Sibling_BH_Different"] = True
    annotated.loc["B", "Sibling_BH_Same"] = False
    annotated.loc["B", "Sibling_Divergence_P_Value"] = 0.001

    seen: dict[str, object] = {}

    def fake_selected_family(data, *_args, **kwargs):
        seen["index"] = tuple(data.index)
        seen["observed"] = kwargs["observed_p_value"]
        return {
            "root_observed_p_value": kwargs["observed_p_value"],
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.0001,
            "root_selective_null_q05_p_value": 0.005,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_global_sibling_min_permutation_p_value",
        fake_selected_family,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
        scope="global_sibling_min_passthrough_descendant",
    )

    assert seen == {"index": ("A", "C", "D"), "observed": 0.001}
    assert bool(guarded.loc["root", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["B", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["B", "Selective_Permutation_Guard_Blocked"]) is True
    assert guarded.loc["B", "Selective_Permutation_P_Value"] == 0.50
    assert guarded.loc["B", "Selective_Permutation_Guard_Scope"] == (
        "global_sibling_min_passthrough_descendant"
    )


def test_global_selected_family_guard_keeps_significant_passthrough_descendant(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_passthrough_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    annotated = annotations_df.copy()
    annotated["Child_Parent_Divergence_Significant"] = True
    annotated["Sibling_BH_Different"] = False
    annotated["Sibling_BH_Same"] = True
    annotated["Sibling_Divergence_Skipped"] = False
    annotated["Sibling_Divergence_P_Value"] = 1.0
    annotated.loc["B", "Sibling_BH_Different"] = True
    annotated.loc["B", "Sibling_BH_Same"] = False
    annotated.loc["B", "Sibling_Divergence_P_Value"] = 0.001

    def fake_selected_family(*_args, **kwargs):
        return {
            "root_observed_p_value": kwargs["observed_p_value"],
            "root_selective_p_value": 0.01,
            "root_selective_null_min_p_value": 0.0001,
            "root_selective_null_q05_p_value": 0.005,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_global_sibling_min_permutation_p_value",
        fake_selected_family,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
        scope="global_sibling_min_passthrough_descendant",
    )

    assert bool(guarded.loc["B", "Sibling_BH_Different"]) is True
    assert bool(guarded.loc["B", "Selective_Permutation_Guard_Blocked"]) is False
    assert bool(guarded.loc["B", "Selective_Permutation_Guard_Would_Block"]) is False


def test_refined_global_selected_family_guard_refines_floor_p_value(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_passthrough_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    annotated = annotations_df.copy()
    annotated["Child_Parent_Divergence_Significant"] = True
    annotated["Sibling_BH_Different"] = False
    annotated["Sibling_BH_Same"] = True
    annotated["Sibling_Divergence_Skipped"] = False
    annotated["Sibling_Divergence_P_Value"] = 1.0
    annotated.loc["B", "Sibling_BH_Different"] = True
    annotated.loc["B", "Sibling_BH_Same"] = False
    annotated.loc["B", "Sibling_Divergence_P_Value"] = 0.001

    calls: list[int] = []

    def fake_selected_family(*_args, **kwargs):
        replicates = int(kwargs["bootstrap_replicates"])
        calls.append(replicates)
        selected_p = 0.01 if replicates == 99 else 0.02
        return {
            "root_observed_p_value": kwargs["observed_p_value"],
            "root_selective_p_value": selected_p,
            "root_selective_null_min_p_value": 0.0001,
            "root_selective_null_q05_p_value": 0.005,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_global_sibling_min_permutation_p_value",
        fake_selected_family,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=99,
        seed=123,
        alpha=0.01,
        scope="global_sibling_min_passthrough_descendant_refined",
    )

    assert calls == [99, 999]
    assert bool(guarded.loc["B", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["B", "Selective_Permutation_Guard_Blocked"]) is True
    assert bool(guarded.loc["B", "Selective_Permutation_Guard_Refined"]) is True
    assert guarded.loc["B", "Selective_Permutation_Base_P_Value"] == 0.01
    assert guarded.loc["B", "Selective_Permutation_P_Value"] == 0.02
    assert guarded.loc["B", "Selective_Permutation_Guard_Replicates"] == 999


def test_global_selected_family_p_value_uses_add_one_monte_carlo(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    leaf_data = pd.DataFrame([[0], [1], [0]], index=["a", "b", "c"])
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    null_values = iter([0.01, 0.20, 0.05])

    def fake_null_sample(data, *_args, **_kwargs):
        return data

    def fake_min_p_value(*_args, **_kwargs):
        return next(null_values)

    monkeypatch.setattr(
        orchestrator,
        "_block_permutation_null_sample",
        fake_null_sample,
    )
    monkeypatch.setattr(
        orchestrator,
        "_selected_tree_fixed_sibling_min_p_value",
        fake_min_p_value,
    )

    result = selected_global_sibling_min_permutation_p_value(
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=3,
        seed=123,
        observed_p_value=0.05,
    )

    assert result["root_observed_p_value"] == 0.05
    assert result["root_selective_p_value"] == 0.75
    assert result["root_selective_null_min_p_value"] == 0.01
    assert result["root_selective_null_q05_p_value"] == pytest.approx(0.014)


def test_fast_bernoulli_coordinate_p_value_matches_canonical_contrast() -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, _annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    left = tree.nodes["A"]
    right = tree.nodes["B"]

    fast = orchestrator._fixed_bernoulli_coordinate_sibling_p_value(
        np.asarray(left["distribution"], dtype=float),
        np.asarray(right["distribution"], dtype=float),
        float(left["leaf_count"]),
        float(right["leaf_count"]),
    )
    contrast = build_contrast_covariance(
        np.asarray(left["distribution"], dtype=float),
        np.asarray(right["distribution"], dtype=float),
        float(left["leaf_count"]),
        float(right["leaf_count"]),
        comparison="sibling",
        feature_space=feature_space,
    )
    canonical = fixed_subspace_sibling_p_value(
        contrast.whitened_vector(),
        feature_space,
        method="fixed_coordinate_bh",
    )

    assert fast == pytest.approx(canonical)


def test_fast_categorical_coordinate_p_value_matches_canonical_contrast() -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    feature_space = infer_feature_space_from_columns(
        ("F0_c0", "F0_c1", "F0_c2", "F1_c0", "F1_c1", "F1_c2")
    )
    first = np.array([0.70, 0.20, 0.10, 0.20, 0.50, 0.30], dtype=float)
    second = np.array([0.40, 0.30, 0.30, 0.20, 0.20, 0.60], dtype=float)
    first_sample_size = 11.0
    second_sample_size = 13.0

    fast = orchestrator._fixed_discrete_coordinate_sibling_p_value(
        first,
        second,
        first_sample_size,
        second_sample_size,
        feature_space,
    )
    contrast = build_contrast_covariance(
        first,
        second,
        first_sample_size,
        second_sample_size,
        comparison="sibling",
        feature_space=feature_space,
    )
    canonical = fixed_subspace_sibling_p_value(
        contrast.whitened_vector(),
        feature_space,
        method="fixed_coordinate_bh",
    )

    assert fast == pytest.approx(canonical)


def test_selective_permutation_passthrough_scope_keeps_descendant_after_open_root(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_passthrough_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    annotated = annotations_df.copy()
    annotated["Child_Parent_Divergence_Significant"] = True
    annotated["Sibling_BH_Different"] = False
    annotated["Sibling_BH_Same"] = True
    annotated["Sibling_Divergence_Skipped"] = False
    annotated["Sibling_Divergence_P_Value"] = 1.0
    annotated.loc["root", "Sibling_BH_Different"] = True
    annotated.loc["root", "Sibling_BH_Same"] = False
    annotated.loc["root", "Sibling_Divergence_P_Value"] = 0.001
    annotated.loc["B", "Sibling_BH_Different"] = True
    annotated.loc["B", "Sibling_BH_Same"] = False
    annotated.loc["B", "Sibling_Divergence_P_Value"] = 0.001

    calls: list[tuple[object, ...]] = []

    def fake_selected_root(data, *_args, **_kwargs):
        calls.append(tuple(data.index))
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.01,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
        scope="passthrough_descendant",
    )

    assert calls == [("A", "C", "D")]
    assert bool(guarded.loc["root", "Sibling_BH_Different"]) is True
    assert bool(guarded.loc["B", "Sibling_BH_Different"]) is True
    assert guarded.loc["B", "Selective_Permutation_Guard_Scope"] == ""


def test_selective_permutation_passthrough_scope_skips_guard_blocked_ancestor(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_passthrough_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))
    annotated = annotations_df.copy()
    annotated["Child_Parent_Divergence_Significant"] = True
    annotated["Sibling_BH_Different"] = False
    annotated["Sibling_BH_Same"] = True
    annotated["Sibling_Divergence_Skipped"] = False
    annotated["Sibling_Divergence_P_Value"] = 1.0
    annotated.loc["root", "Sibling_BH_Different"] = True
    annotated.loc["root", "Sibling_BH_Same"] = False
    annotated.loc["root", "Sibling_Divergence_P_Value"] = 0.001
    annotated.loc["B", "Sibling_BH_Different"] = True
    annotated.loc["B", "Sibling_BH_Same"] = False
    annotated.loc["B", "Sibling_Divergence_P_Value"] = 0.001

    calls: list[tuple[object, ...]] = []

    def fake_selected_root(data, *_args, **_kwargs):
        calls.append(tuple(data.index))
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    guarded = apply_root_selective_permutation_guard(
        tree,
        annotated,
        leaf_data,
        feature_space,
        method="fixed_coordinate_bh",
        bootstrap_replicates=5,
        seed=123,
        alpha=0.01,
        scope="passthrough_descendant",
    )

    assert calls == [("A", "C", "D")]
    assert bool(guarded.loc["root", "Sibling_BH_Different"]) is False
    assert bool(guarded.loc["root", "Root_Selective_Permutation_Guard_Blocked"]) is True
    assert bool(guarded.loc["B", "Sibling_BH_Different"]) is True
    assert guarded.loc["B", "Selective_Permutation_Guard_Scope"] == ""


def test_pipeline_runs_opt_in_root_stability_guard(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    def unstable_root(*_args, **_kwargs):
        return {
            "root_stability_subsample_mean_ari": 0.0,
            "root_stability_subsample_median_ari": 0.0,
            "root_stability_subsample_q10_ari": 0.0,
        }

    monkeypatch.setattr(
        orchestrator,
        "compute_root_feature_subsample_stability",
        unstable_root,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        sibling_gate_method="fixed_coordinate_bh",
        root_stability_guard_threshold=0.50,
        root_stability_subsample_replicates=4,
        root_stability_feature_fraction=0.8,
        root_stability_seed=7,
    )

    assert bundle.metadata.config.root_stability_guard_threshold == 0.50
    assert bundle.metadata.config.root_stability_subsample_replicates == 4
    assert bundle.metadata.config.root_stability_feature_fraction == 0.8
    assert bundle.metadata.config.root_stability_seed == 7
    assert bundle.metadata.config.root_stability_tree_distance_metric == "hamming"
    assert bundle.metadata.config.root_stability_tree_linkage_method == "average"
    assert "Root_Stability_Guard_Blocked" in bundle.annotated_df.columns


def test_pipeline_runs_opt_in_root_selective_permutation_guard(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_method="fixed_coordinate_bh",
        root_selective_permutation_guard_replicates=5,
        root_selective_permutation_guard_seed=123,
        root_selective_permutation_guard_alpha=0.01,
    )

    assert bundle.metadata.config.root_selective_permutation_guard_replicates == 5
    assert bundle.metadata.config.root_selective_permutation_guard_seed == 123
    assert bundle.metadata.config.root_selective_permutation_guard_alpha == 0.01
    assert "Root_Selective_Permutation_Guard_Blocked" in bundle.annotated_df.columns


def test_pipeline_selective_root_profile_runs_packaged_guard(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))

    def stable_root(*_args, **_kwargs):
        return {
            "root_stability_subsample_mean_ari": 1.0,
            "root_stability_subsample_median_ari": 1.0,
            "root_stability_subsample_q10_ari": 1.0,
        }

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.50,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "compute_root_feature_subsample_stability",
        stable_root,
    )
    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_profile="fixed_coordinate_selective_root_v1",
    )

    assert bundle.metadata.config.sibling_gate_profile_id == (
        "fixed_coordinate_selective_root_v1"
    )
    assert bundle.metadata.config.sibling_gate_method == "fixed_coordinate_bh"
    assert bundle.metadata.config.root_selective_permutation_guard_replicates == 99
    assert bundle.metadata.config.root_selective_permutation_guard_seed == 0
    assert bundle.metadata.config.root_selective_permutation_guard_alpha == 0.01
    assert bundle.metadata.config.root_selective_permutation_guard_scope == "root"
    assert "Root_Selective_Permutation_Guard_Blocked" in bundle.annotated_df.columns


def test_pipeline_selective_traversal_profile_sets_open_internal_scope(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))

    def stable_root(*_args, **_kwargs):
        return {
            "root_stability_subsample_mean_ari": 1.0,
            "root_stability_subsample_median_ari": 1.0,
            "root_stability_subsample_q10_ari": 1.0,
        }

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.01,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "compute_root_feature_subsample_stability",
        stable_root,
    )
    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_profile="fixed_coordinate_selective_traversal_v1",
    )

    assert bundle.metadata.config.sibling_gate_profile_id == (
        "fixed_coordinate_selective_traversal_v1"
    )
    assert bundle.metadata.config.root_selective_permutation_guard_replicates == 99
    assert bundle.metadata.config.root_selective_permutation_guard_scope == (
        "open_internal"
    )
    assert "Selective_Permutation_Guard_Blocked" in bundle.annotated_df.columns


def test_pipeline_selective_passthrough_profile_sets_passthrough_scope(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))

    def stable_root(*_args, **_kwargs):
        return {
            "root_stability_subsample_mean_ari": 1.0,
            "root_stability_subsample_median_ari": 1.0,
            "root_stability_subsample_q10_ari": 1.0,
        }

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.01,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "compute_root_feature_subsample_stability",
        stable_root,
    )
    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_profile="fixed_coordinate_selective_passthrough_v1",
    )

    assert bundle.metadata.config.sibling_gate_profile_id == (
        "fixed_coordinate_selective_passthrough_v1"
    )
    assert bundle.metadata.config.root_selective_permutation_guard_replicates == 99
    assert bundle.metadata.config.root_selective_permutation_guard_scope == (
        "passthrough_descendant"
    )
    assert "Selective_Permutation_Guard_Blocked" in bundle.annotated_df.columns


def test_pipeline_global_passthrough_profile_sets_global_scope(
    monkeypatch,
) -> None:
    import kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator as orchestrator

    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()
    feature_space = infer_feature_space_from_columns(tuple(leaf_data.columns))

    def stable_root(*_args, **_kwargs):
        return {
            "root_stability_subsample_mean_ari": 1.0,
            "root_stability_subsample_median_ari": 1.0,
            "root_stability_subsample_q10_ari": 1.0,
        }

    def fake_selected_root(*_args, **_kwargs):
        return {
            "root_observed_p_value": 0.001,
            "root_selective_p_value": 0.01,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    def fake_selected_family(*_args, **kwargs):
        return {
            "root_observed_p_value": kwargs.get("observed_p_value", 0.001),
            "root_selective_p_value": 0.01,
            "root_selective_null_min_p_value": 0.01,
            "root_selective_null_q05_p_value": 0.05,
        }

    monkeypatch.setattr(
        orchestrator,
        "compute_root_feature_subsample_stability",
        stable_root,
    )
    monkeypatch.setattr(
        orchestrator,
        "selected_root_permutation_p_value",
        fake_selected_root,
    )
    monkeypatch.setattr(
        orchestrator,
        "selected_global_sibling_min_permutation_p_value",
        fake_selected_family,
    )

    bundle = orchestrator.run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        edge_alpha=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
        feature_space=feature_space,
        sibling_gate_profile="fixed_coordinate_global_passthrough_v1",
    )

    assert bundle.metadata.config.sibling_gate_profile_id == (
        "fixed_coordinate_global_passthrough_v1"
    )
    assert bundle.metadata.config.root_selective_permutation_guard_replicates == 99
    assert bundle.metadata.config.root_selective_permutation_guard_scope == (
        "global_sibling_min_passthrough_descendant"
    )
    assert "Selective_Permutation_Guard_Blocked" in bundle.annotated_df.columns


def test_pipeline_rejects_root_selective_guard_with_adaptive_sibling_gate() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    with pytest.raises(ValueError, match="fixed-subspace sibling gate"):
        run_gate_annotation_pipeline(
            tree,
            annotations_df.copy(),
            edge_alpha=0.01,
            sibling_alpha=0.01,
            leaf_data=leaf_data,
            root_selective_permutation_guard_replicates=5,
        )


def test_pipeline_rejects_inert_root_stability_guard_config() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    with pytest.raises(ValueError, match="root_stability_subsample_replicates"):
        run_gate_annotation_pipeline(
            tree,
            annotations_df.copy(),
            edge_alpha=0.01,
            sibling_alpha=0.01,
            leaf_data=leaf_data,
            sibling_gate_method="fixed_coordinate_bh",
            root_stability_guard_threshold=0.50,
            root_stability_subsample_replicates=0,
        )


def test_resolve_effective_sibling_alpha_rejects_invalid_penalty() -> None:
    assert resolve_effective_sibling_alpha(0.01, 50.0) == 0.0002
    with pytest.raises(ValueError, match="sibling_gate_alpha_penalty"):
        resolve_effective_sibling_alpha(0.01, 0.0)


def test_pipeline_rejects_unknown_sibling_gate_method() -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    with pytest.raises(ValueError, match="Unknown sibling_gate_method"):
        run_gate_annotation_pipeline(
            tree,
            annotations_df.copy(),
            edge_alpha=0.01,
            sibling_alpha=0.01,
            leaf_data=leaf_data,
            sibling_gate_method="adaptive_magic",
        )


def test_fixed_block_gate_aggregates_categorical_feature_blocks() -> None:
    feature_space = infer_feature_space_from_columns(
        ("F0_c0", "F0_c1", "F0_c2", "F1_c0", "F1_c1", "F1_c2")
    )
    z = np.array([2.0, 0.0, 0.0, 0.0], dtype=float)

    coordinate_p = fixed_subspace_sibling_p_value(
        z,
        feature_space,
        method="fixed_coordinate_bh",
    )
    block_p = fixed_subspace_sibling_p_value(
        z,
        feature_space,
        method="fixed_block_bh",
    )

    assert 0.0 < coordinate_p < block_p < 1.0


def test_pipeline_requires_leaf_data_for_spectral_gate_context() -> None:
    tree, annotations_df, _leaf_data = _build_small_tree_with_leaf_data()

    with pytest.raises(ValueError, match="require leaf_data"):
        run_gate_annotation_pipeline(
            tree,
            annotations_df.copy(),
            edge_alpha=0.01,
            sibling_alpha=0.01,
            leaf_data=None,
        )
