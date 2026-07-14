from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
from benchmarks.validation.brancharchitect_tree_comparison import (
    BranchArchitectAdapter,
    cell_row_from_snapshot,
    compare_snapshots,
    load_brancharchitect_adapter,
    snapshot_from_computed,
    write_brancharchitect_tree_comparison_artifacts,
)
from tree_break_selection.tree.poset_tree import PosetTree


def _small_tree(*, scale: float = 1.0) -> PosetTree:
    tree = PosetTree()
    for node, is_leaf in [
        ("root", False),
        ("A", False),
        ("B", False),
        ("a", True),
        ("b", True),
        ("c", True),
        ("d", True),
    ]:
        tree.add_node(node, label=node, is_leaf=is_leaf)
    for parent, child, length in [
        ("root", "A", 1.0 * scale),
        ("root", "B", 1.0 * scale),
        ("A", "a", 0.25 * scale),
        ("A", "b", 0.25 * scale),
        ("B", "c", 0.25 * scale),
        ("B", "d", 0.25 * scale),
    ]:
        tree.add_edge(parent, child, branch_length=length)
    tree.graph["root"] = "root"
    return tree


def _snapshot(*, scale: float = 1.0, run_id: str = "average"):
    full_run_id = (
        "tbs_diffusion_graphtools_adaptive_nnls::"
        f"graphtools_adaptive_k_tree_strategy__tree_linkage_method_{run_id}__r0"
    )
    data = pd.DataFrame(index=["a", "b", "c", "d"])
    computed = SimpleNamespace(
        tree=_small_tree(scale=scale),
        run_id=full_run_id,
        labels=np.array([0, 0, 1, 1]),
        data=data,
    )
    row = SimpleNamespace(
        true_clusters=2,
        found_clusters=2,
        ari=1.0,
        nmi=1.0,
        macro_f1=1.0,
        purity=1.0,
    )
    return snapshot_from_computed(
        case_id="synthetic",
        test_case=1,
        case_category="unit",
        result_row=row,
        computed=computed,
    )


def test_snapshot_exports_weighted_newick_and_cluster_path_separation() -> None:
    snapshot = _snapshot()

    assert snapshot.tree_inference == "average"
    assert snapshot.branch_length_missing_count == 0
    assert snapshot.split_lengths[frozenset({"a", "b"})] == 1.0
    assert snapshot.newick.endswith(";")
    assert ":1" in snapshot.newick

    row = cell_row_from_snapshot(snapshot, max_path_pairs=100)

    assert row["found_clusters"] == 2
    assert row["cluster_sizes"] == "2;2"
    assert row["sampled_intra_cluster_pair_count"] == 2
    assert row["inter_cluster_path_mean"] > row["intra_cluster_path_mean"]
    assert row["cluster_path_separation"] > 0.0


def test_pairwise_same_topology_exposes_branch_length_and_path_difference() -> None:
    left = _snapshot(scale=1.0, run_id="average")
    right = _snapshot(scale=2.0, run_id="weighted")

    row = compare_snapshots(
        left,
        right,
        max_path_pairs=100,
        brancharchitect=BranchArchitectAdapter(status="not_configured", error=""),
        temp_dir=None,
    )

    assert row["rooted_internal_rf"] == 0
    assert row["rooted_internal_rf_relative"] == 0.0
    assert row["rooted_weighted_split_l1"] > 0.0
    assert row["leaf_path_rmse"] > 0.0
    assert row["predicted_label_ari_between_topologies"] == 1.0
    assert row["brancharchitect_status"] == "not_requested"


def test_writer_emits_artifacts_without_brancharchitect(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.validation.brancharchitect_tree_comparison.BUNDLED_BRANCHARCHITECT_PATH",
        tmp_path / "missing-brancharchitect",
    )
    snapshots = [
        _snapshot(scale=1.0, run_id="average"),
        _snapshot(scale=1.2, run_id="weighted"),
    ]

    outputs = write_brancharchitect_tree_comparison_artifacts(
        snapshots=snapshots,
        skip_rows=[],
        output_dir=tmp_path,
        case_names=("synthetic",),
        max_path_pairs=100,
    )

    for path in outputs.values():
        assert path.exists()
    pairwise = pd.read_csv(outputs["tree_pairwise"])
    assert len(pairwise) == 1
    assert pairwise.loc[0, "brancharchitect_status"] == "not_configured"
    assert "BranchArchitect Tree Comparison Gate" in outputs["report"].read_text()


def test_brancharchitect_missing_path_fails_closed() -> None:
    adapter = load_brancharchitect_adapter("/definitely/not/brancharchitect")

    assert adapter.status == "path_missing"
    assert not adapter.has_distances


def test_brancharchitect_installed_package_loads_without_local_path(
    monkeypatch,
    tmp_path,
) -> None:
    package = ModuleType("brancharchitect")
    distances_package = ModuleType("brancharchitect.distances")
    distances = ModuleType("brancharchitect.distances.distances")
    io = ModuleType("brancharchitect.io")
    distances.relative_robinson_foulds_distance = lambda _left, _right: 0.0
    distances.weighted_robinson_foulds_distance = lambda _left, _right: 0.0
    io.read_newick = lambda _path, **_kwargs: object()
    monkeypatch.delenv("TBS_BRANCHARCHITECT_PATH", raising=False)
    monkeypatch.setattr(
        "benchmarks.validation.brancharchitect_tree_comparison.BUNDLED_BRANCHARCHITECT_PATH",
        tmp_path / "uninitialized-submodule",
    )
    monkeypatch.setitem(sys.modules, "brancharchitect", package)
    monkeypatch.setitem(sys.modules, "brancharchitect.distances", distances_package)
    monkeypatch.setitem(sys.modules, "brancharchitect.distances.distances", distances)
    monkeypatch.setitem(sys.modules, "brancharchitect.io", io)

    adapter = load_brancharchitect_adapter()

    assert adapter.status == "distances_available"
    assert adapter.has_distances


def test_deep_ladder_tree_serializes_without_recursion_failure() -> None:
    tree = PosetTree()
    tree.add_node("root", label="root", is_leaf=False)
    parent = "root"
    for index in range(1100):
        child = f"N{index}"
        tree.add_node(child, label=child, is_leaf=False)
        tree.add_edge(parent, child, branch_length=0.001)
        parent = child
    tree.nodes[parent]["is_leaf"] = True
    tree.nodes[parent]["label"] = "leaf"
    tree.graph["root"] = "root"
    computed = SimpleNamespace(
        tree=tree,
        run_id=(
            "tbs_diffusion_graphtools_adaptive_nnls::"
            "graphtools_adaptive_k_tree_strategy__tree_linkage_method_single__r0"
        ),
        labels=np.array([0]),
        data=pd.DataFrame(index=["leaf"]),
    )
    row = SimpleNamespace(
        true_clusters=1,
        found_clusters=1,
        ari=0.0,
        nmi=0.0,
        macro_f1=1.0,
        purity=1.0,
    )

    snapshot = snapshot_from_computed(
        case_id="ladder",
        test_case=1,
        case_category="unit",
        result_row=row,
        computed=computed,
    )

    assert snapshot.newick.endswith(";")
    assert len(snapshot.edge_lengths) == 1100
