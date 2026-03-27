from __future__ import annotations

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases.binary import BINARY_CASES
from benchmarks.shared.cases.gaussian import GAUSSIAN_CASES
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_config import (
    derive_sibling_spectral_dims,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


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
    list[tuple[str, list[str], tuple[int, int], tuple[bool, bool], int]],
]:
    case = _find_case(case_name)
    data_df, _labels, _x_original, _metadata = generate_case_data(case)
    tree = _build_case_tree(data_df)
    tree.populate_node_divergences(data_df)

    annotated_df = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        leaf_data=data_df,
    ).annotated_df
    edge_spectral_dims = annotated_df.attrs["_spectral_dims"]
    sibling_dims = derive_sibling_spectral_dims(tree, annotated_df) or {}

    omitted: list[tuple[str, list[str], tuple[int, int], tuple[bool, bool]]] = []
    included: list[tuple[str, list[str], tuple[int, int], tuple[bool, bool], int]] = []
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue

        child_dims = (
            int(edge_spectral_dims.get(children[0], 0)),
            int(edge_spectral_dims.get(children[1], 0)),
        )
        child_is_leaf = (
            bool(tree.nodes[children[0]].get("is_leaf", False)),
            bool(tree.nodes[children[1]].get("is_leaf", False)),
        )
        if parent in sibling_dims:
            included.append(
                (parent, children, child_dims, child_is_leaf, int(sibling_dims[parent]))
            )
        else:
            omitted.append((parent, children, child_dims, child_is_leaf))

    return omitted, included


def test_binary_2clusters_contains_leaf_pair_and_internal_pair_sibling_modes() -> None:
    omitted, included = _collect_binary_parent_structure("binary_2clusters")

    assert omitted
    assert included

    for _parent, _children, child_dims, child_is_leaf in omitted:
        assert child_dims == (0, 0)
        assert child_is_leaf == (True, True)

    for _parent, _children, child_dims, child_is_leaf, sibling_k in included:
        assert sibling_k > 0
        assert max(child_dims) > 0
        assert child_is_leaf != (True, True)


def test_gauss_clear_small_contains_leaf_pair_and_internal_pair_sibling_modes() -> None:
    omitted, included = _collect_binary_parent_structure("gauss_clear_small")

    assert omitted
    assert included

    for _parent, _children, child_dims, child_is_leaf in omitted:
        assert child_dims == (0, 0)
        assert child_is_leaf == (True, True)

    for _parent, _children, child_dims, child_is_leaf, sibling_k in included:
        assert sibling_k > 0
        assert max(child_dims) > 0
        assert child_is_leaf != (True, True)
