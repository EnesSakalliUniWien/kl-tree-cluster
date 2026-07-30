from __future__ import annotations

from benchmarks.shared import tree_consensus
from benchmarks.shared.runners.method_registry import (
    GRAPHTOOLS_ADAPTIVE_K_TREE_STRATEGY_PARAMS,
)
from benchmarks.validation.sweeps import family_metric_nnls_grid
from tree_break_selection.tree.construction import (
    DEFAULT_LINKAGE_TREE_ROOTING,
    DEFAULT_PHYLOGENETIC_TREE_ROOTING,
    DEFAULT_TREE_LINKAGE_METHOD,
    LINKAGE_TREE_BUILDER,
    NEIGHBOR_JOINING_TREE_BUILDER,
    SUPPORTED_LINKAGE_METHODS,
    SUPPORTED_TREE_BUILDERS,
    TREE_CONSENSUS_STRATEGIES,
    TREE_CONSENSUS_STRATEGY_PRIORITY,
)
from tree_break_selection.tree.construction.build import SUPPORTED_TREE_BUILDERS as BUILD_BUILDERS


def test_tree_construction_methods_have_one_canonical_source() -> None:
    assert BUILD_BUILDERS == SUPPORTED_TREE_BUILDERS
    assert tree_consensus.TREE_CONSENSUS_STRATEGIES == TREE_CONSENSUS_STRATEGIES
    assert tree_consensus.TREE_PRIORITY == TREE_CONSENSUS_STRATEGY_PRIORITY
    assert family_metric_nnls_grid.TOPOLOGIES == TREE_CONSENSUS_STRATEGIES


def test_graphtools_strategy_grid_uses_supported_linkage_methods() -> None:
    linkage_params = [
        params
        for params in GRAPHTOOLS_ADAPTIVE_K_TREE_STRATEGY_PARAMS
        if params.get("tree_builder") == LINKAGE_TREE_BUILDER
    ]
    assert {params["tree_linkage_method"] for params in linkage_params} == set(
        SUPPORTED_LINKAGE_METHODS
    )
    assert {params["tree_rooting"] for params in linkage_params} == {
        DEFAULT_LINKAGE_TREE_ROOTING
    }


def test_family_metric_topology_parameters_use_canonical_builder_contract() -> None:
    assert family_metric_nnls_grid._topology_parameters(DEFAULT_TREE_LINKAGE_METHOD) == {
        "tree_builder": LINKAGE_TREE_BUILDER,
        "tree_rooting": DEFAULT_LINKAGE_TREE_ROOTING,
        "tree_linkage_method": DEFAULT_TREE_LINKAGE_METHOD,
    }
    assert family_metric_nnls_grid._topology_parameters(NEIGHBOR_JOINING_TREE_BUILDER) == {
        "tree_builder": NEIGHBOR_JOINING_TREE_BUILDER,
        "tree_rooting": DEFAULT_PHYLOGENETIC_TREE_ROOTING,
        "tree_linkage_method": DEFAULT_TREE_LINKAGE_METHOD,
    }
