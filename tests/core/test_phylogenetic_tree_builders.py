from __future__ import annotations

import subprocess

import networkx as nx
import numpy as np
import pandas as pd
from kl_clustering_analysis.tree.phylogenetic import (
    iqtree3_tree_from_alignment,
    minimum_ancestor_deviation_root,
    neighbor_joining_tree_from_distance,
    poset_tree_from_unrooted_metric_tree,
)


def test_minimum_ancestor_deviation_roots_two_leaf_tree_at_midpoint() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", branch_length=4.0)

    root = minimum_ancestor_deviation_root(graph, leaf_labels=["A", "B"])

    assert root.edge == ("A", "B")
    assert root.distance_from_u == 2.0
    assert root.fraction_from_u == 0.5
    assert root.ancestor_deviation == 0.0


def test_poset_tree_from_unrooted_metric_tree_uses_mad_root() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", branch_length=4.0)

    tree, root = poset_tree_from_unrooted_metric_tree(
        graph,
        leaf_labels=["A", "B"],
        rooting="mad",
    )

    assert tree.graph["root"] == root.root_node == "mad_root"
    assert tree.get_leaves(sort=True) == ["A", "B"]
    assert tree.out_degree(tree.root()) == 2
    assert tree["mad_root"]["A"]["branch_length"] == 2.0
    assert tree["mad_root"]["B"]["branch_length"] == 2.0


def test_neighbor_joining_tree_from_distance_returns_mad_rooted_poset_tree() -> None:
    distances = np.array([5.0, 9.0, 9.0, 10.0, 10.0, 8.0], dtype=float)

    tree, root = neighbor_joining_tree_from_distance(
        distances,
        ["A", "B", "C", "D"],
        rooting="mad",
    )

    assert root.root_node == tree.root()
    assert tree.get_leaves(sort=True) == ["A", "B", "C", "D"]
    assert tree.graph["rooting"] == "mad"
    assert np.isfinite(tree.graph["mad_rooting"]["ancestor_deviation"])


def test_iqtree3_tree_from_alignment_runs_command_and_mad_roots_newick(
    tmp_path,
    monkeypatch,
) -> None:
    data = pd.DataFrame(
        [[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]],
        index=["A sample", "B sample", "C sample", "D sample"],
        columns=["f0", "f1", "f2"],
    )
    captured_cmd: list[str] = []

    def _fake_which(executable: str) -> str:
        assert executable == "iqtree3"
        return "/usr/local/bin/iqtree3"

    def _fake_run(cmd, check, capture_output, text):
        captured_cmd.extend(cmd)
        prefix = cmd[cmd.index("--prefix") + 1]
        treefile = tmp_path / "fixture.treefile"
        treefile.write_text("((A_sample:1,B_sample:1):1,(C_sample:1,D_sample:1):1);\n")
        expected_treefile = tmp_path / "fixture.treefile"
        assert str(expected_treefile) == f"{prefix}.treefile"
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "kl_clustering_analysis.tree.phylogenetic.shutil.which",
        _fake_which,
    )
    monkeypatch.setattr(
        "kl_clustering_analysis.tree.phylogenetic.subprocess.run",
        _fake_run,
    )

    tree, root, metadata = iqtree3_tree_from_alignment(
        data,
        executable="iqtree3",
        model="JC2",
        threads=2,
        rooting="mad",
        work_dir=tmp_path,
        prefix="fixture",
    )

    assert captured_cmd[:5] == [
        "/usr/local/bin/iqtree3",
        "-s",
        str(tmp_path / "fixture.fasta"),
        "-m",
        "JC2",
    ]
    assert metadata["threads"] == 2
    assert root.root_node == tree.root()
    assert tree.get_leaves(sort=True) == [
        "A sample",
        "B sample",
        "C sample",
        "D sample",
    ]
