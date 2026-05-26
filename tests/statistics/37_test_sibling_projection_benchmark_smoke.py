from __future__ import annotations

from benchmarks.shared.cases.binary import BINARY_CASES
from benchmarks.shared.cases.gaussian import GAUSSIAN_CASES
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist


def _build_case_tree(data_df):
    distance_matrix = pdist(data_df.values, metric="hamming")
    linkage_matrix = linkage(distance_matrix, method="complete")
    return PosetTree.from_linkage(linkage_matrix, data_df.index.tolist())


def _find_case(case_name: str) -> dict:
    for case_group in (BINARY_CASES, GAUSSIAN_CASES):
        for cases in case_group.values():
            for case in cases:
                if case["name"] == case_name:
                    return case
    raise KeyError(f"Unknown benchmark case {case_name!r}")


def _collect_binary_parent_structure(
    case_name: str,
) -> tuple[
    list[tuple[str, list[str], tuple[int, int], tuple[bool, bool]]],
    list[tuple[str, list[str], tuple[int, int], tuple[bool, bool], int, int]],
]:
    case = _find_case(case_name)
    data_df, _labels, _x_original, _metadata = generate_case_data(case)
    tree = _build_case_tree(data_df)
    tree.populate_node_divergences(data_df)

    bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        leaf_data=data_df,
    )
    test_projection_dimensions_by_node = (
        bundle.gate_two_result.spectral_context.test_projection_dimensions_by_node
    )
    sibling_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=bundle.gate_two_result.spectral_context,
        )
    )

    omitted: list[tuple[str, list[str], tuple[int, int], tuple[bool, bool]]] = []
    included: list[tuple[str, list[str], tuple[int, int], tuple[bool, bool], int, int]] = []
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        child_dims = (
            int(test_projection_dimensions_by_node[children[0]]),
            int(test_projection_dimensions_by_node[children[1]]),
        )
        child_is_leaf = (
            bool(tree.nodes[children[0]]["is_leaf"]),
            bool(tree.nodes[children[1]]["is_leaf"]),
        )
        parent_dim = int(test_projection_dimensions_by_node[parent])
        if parent in sibling_projection_dimensions_from_edge_comparisons:
            included.append(
                (
                    parent,
                    children,
                    child_dims,
                    child_is_leaf,
                    parent_dim,
                    int(sibling_projection_dimensions_from_edge_comparisons[parent]),
                )
            )
        else:
            omitted.append((parent, children, child_dims, child_is_leaf))

    return omitted, included


def test_binary_2clusters_contains_leaf_pair_and_internal_pair_sibling_modes() -> None:
    omitted, included = _collect_binary_parent_structure("binary_2clusters")

    assert not omitted
    assert included

    leaf_pair_entries = 0
    internal_entries = 0
    for _parent, _children, child_dims, child_is_leaf, parent_dim, sibling_k in included:
        assert 0 <= sibling_k <= parent_dim
        if child_is_leaf == (True, True):
            assert child_dims == (0, 0)
            leaf_pair_entries += 1
        else:
            internal_entries += 1
        if child_dims == (0, 0):
            assert sibling_k == parent_dim

    assert leaf_pair_entries
    assert internal_entries


def test_gauss_clear_small_contains_leaf_pair_and_internal_pair_sibling_modes() -> None:
    omitted, included = _collect_binary_parent_structure("gauss_clear_small")

    assert not omitted
    assert included

    leaf_pair_entries = 0
    internal_entries = 0
    for _parent, _children, child_dims, child_is_leaf, parent_dim, sibling_k in included:
        assert 0 <= sibling_k <= parent_dim
        if child_is_leaf == (True, True):
            assert child_dims == (0, 0)
            leaf_pair_entries += 1
        else:
            internal_entries += 1
        if child_dims == (0, 0):
            assert sibling_k == parent_dim

    assert leaf_pair_entries
    assert internal_entries
