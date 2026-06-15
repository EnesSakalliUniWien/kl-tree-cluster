"""Phylogenetic tree builders for KL traversal trees.

Neighbor-joining and likelihood programs return unrooted metric trees.  This
module roots those trees with minimum ancestor deviation (MAD) before promoting
them to :class:`PosetTree`.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from kl_clustering_analysis.tree.poset_tree import PosetTree

_MAD_ROOT_NODE = "mad_root"
_EDGE_LENGTH_KEYS = ("branch_length", "length", "weight")
_IQTREE_SYMBOLS = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class MadRootResult:
    """Minimum ancestor-deviation root location on an unrooted tree edge."""

    edge: tuple[object, object]
    fraction_from_u: float
    distance_from_u: float
    ancestor_deviation: float
    ambiguity_index: float | None
    root_node: object | None = None


def _edge_length(attrs: dict[str, object]) -> float:
    for key in _EDGE_LENGTH_KEYS:
        if key in attrs:
            value = float(attrs[key])
            if value < 0:
                raise ValueError("Phylogenetic tree branch lengths must be non-negative.")
            return value
    return 1.0


def _leaf_nodes(graph: nx.Graph, leaf_labels: Iterable[object] | None = None) -> list[object]:
    if leaf_labels is not None:
        labels = list(leaf_labels)
        missing = [label for label in labels if label not in graph]
        if missing:
            raise ValueError(f"Leaf labels are missing from tree: {missing!r}.")
        return labels
    leaves = [node for node, degree in graph.degree() if degree == 1]
    if len(leaves) < 2:
        raise ValueError("MAD rooting requires at least two leaf nodes.")
    return leaves


def minimum_ancestor_deviation_root(
    graph: nx.Graph,
    *,
    leaf_labels: Iterable[object] | None = None,
) -> MadRootResult:
    """Find the minimum ancestor-deviation root on an unrooted weighted tree.

    For a candidate root ``rho`` on an edge, MAD minimizes the summed relative
    pairwise squared deviations
    ``((d(x, rho) - d(y, rho)) / d(x, y)) ** 2`` over leaf pairs.
    """
    if graph.number_of_nodes() == 0 or not nx.is_tree(graph):
        raise ValueError("MAD rooting requires a non-empty undirected tree.")

    leaves = _leaf_nodes(graph, leaf_labels)
    if len(leaves) < 2:
        raise ValueError("MAD rooting requires at least two leaves.")

    weighted = graph.copy()
    for u, v, attrs in weighted.edges(data=True):
        length = _edge_length(attrs)
        attrs["length"] = length
        attrs["branch_length"] = length
        attrs["weight"] = length

    distances = dict(nx.all_pairs_dijkstra_path_length(weighted, weight="length"))
    pair_terms: list[tuple[object, object, float]] = []
    for i, left in enumerate(leaves):
        for right in leaves[i + 1 :]:
            leaf_distance = float(distances[left][right])
            if leaf_distance > 0:
                pair_terms.append((left, right, leaf_distance))
    if not pair_terms:
        raise ValueError("MAD rooting requires positive distances between leaf pairs.")

    candidates: list[MadRootResult] = []
    for u, v, attrs in weighted.edges(data=True):
        length = float(attrs["length"])
        without_edge = weighted.copy()
        without_edge.remove_edge(u, v)
        side_u = set(nx.node_connected_component(without_edge, u))

        quad_a = 0.0
        quad_b = 0.0
        quad_c = 0.0
        for left, right, leaf_distance in pair_terms:
            left_on_u = left in side_u
            right_on_u = right in side_u
            left_base = distances[u][left] if left_on_u else distances[v][left] + length
            right_base = distances[u][right] if right_on_u else distances[v][right] + length
            left_slope = 1.0 if left_on_u else -1.0
            right_slope = 1.0 if right_on_u else -1.0
            slope = left_slope - right_slope
            intercept = left_base - right_base
            inv_sq = 1.0 / (leaf_distance * leaf_distance)
            quad_a += slope * slope * inv_sq
            quad_b += 2.0 * slope * intercept * inv_sq
            quad_c += intercept * intercept * inv_sq

        if quad_a > 0:
            distance_from_u = -quad_b / (2.0 * quad_a)
        else:
            distance_from_u = 0.0
        distance_from_u = min(max(distance_from_u, 0.0), length)
        objective = (
            quad_a * distance_from_u * distance_from_u
            + quad_b * distance_from_u
            + quad_c
        )
        fraction = 0.0 if length == 0 else distance_from_u / length
        candidates.append(
            MadRootResult(
                edge=(u, v),
                fraction_from_u=float(fraction),
                distance_from_u=float(distance_from_u),
                ancestor_deviation=float(objective / len(pair_terms)),
                ambiguity_index=None,
            )
        )

    candidates.sort(key=lambda item: item.ancestor_deviation)
    best = candidates[0]
    ambiguity_index = None
    if len(candidates) > 1 and candidates[1].ancestor_deviation > 0:
        ambiguity_index = best.ancestor_deviation / candidates[1].ancestor_deviation
    return MadRootResult(
        edge=best.edge,
        fraction_from_u=best.fraction_from_u,
        distance_from_u=best.distance_from_u,
        ancestor_deviation=best.ancestor_deviation,
        ambiguity_index=ambiguity_index,
        root_node=None,
    )


def _graph_with_mad_root(graph: nx.Graph, root: MadRootResult) -> tuple[nx.Graph, object]:
    rooted = graph.copy()
    for u, v, attrs in rooted.edges(data=True):
        length = _edge_length(attrs)
        attrs["length"] = length
        attrs["branch_length"] = length
        attrs["weight"] = length

    u, v = root.edge
    edge_length = float(rooted[u][v]["length"])
    distance_from_u = float(root.distance_from_u)
    tolerance = max(1e-12, edge_length * 1e-10)
    if distance_from_u <= tolerance:
        return rooted, u
    if edge_length - distance_from_u <= tolerance:
        return rooted, v

    root_node = _MAD_ROOT_NODE
    suffix = 0
    while root_node in rooted:
        suffix += 1
        root_node = f"{_MAD_ROOT_NODE}_{suffix}"

    rooted.remove_edge(u, v)
    rooted.add_node(root_node)
    rooted.add_edge(
        root_node,
        u,
        length=distance_from_u,
        branch_length=distance_from_u,
        weight=distance_from_u,
    )
    remaining = edge_length - distance_from_u
    rooted.add_edge(
        root_node,
        v,
        length=remaining,
        branch_length=remaining,
        weight=remaining,
    )
    return rooted, root_node


def poset_tree_from_unrooted_metric_tree(
    graph: nx.Graph,
    *,
    leaf_labels: Iterable[object] | None = None,
    rooting: str = "mad",
) -> tuple["PosetTree", MadRootResult | None]:
    """Orient an unrooted metric tree and return a rooted ``PosetTree``."""
    if rooting != "mad":
        raise ValueError(f"Unsupported phylogenetic rooting strategy: {rooting!r}.")
    if graph.number_of_nodes() == 0 or not nx.is_tree(graph):
        raise ValueError("Phylogenetic builders require a non-empty undirected tree.")

    leaves = _leaf_nodes(graph, leaf_labels)
    root_result = minimum_ancestor_deviation_root(graph, leaf_labels=leaves)
    rooted_graph, root_node = _graph_with_mad_root(graph, root_result)

    from kl_clustering_analysis.tree.poset_tree import PosetTree

    tree = PosetTree()
    for node in rooted_graph.nodes:
        tree.add_node(node)
    visited = {root_node}
    queue = [root_node]
    while queue:
        parent = queue.pop(0)
        for child, attrs in rooted_graph[parent].items():
            if child in visited:
                continue
            visited.add(child)
            length = _edge_length(attrs)
            tree.add_edge(parent, child, branch_length=length, weight=length)
            queue.append(child)

    leaf_set = set(leaves)
    for node in tree.nodes:
        is_leaf = node in leaf_set or tree.out_degree(node) == 0
        tree.nodes[node]["is_leaf"] = bool(is_leaf)
        tree.nodes[node]["label"] = str(node)
    tree.nodes[root_node]["is_leaf"] = False
    tree.graph["root"] = root_node
    tree.graph["rooting"] = rooting
    tree.graph["mad_rooting"] = {
        "edge": tuple(str(part) for part in root_result.edge),
        "fraction_from_u": root_result.fraction_from_u,
        "distance_from_u": root_result.distance_from_u,
        "ancestor_deviation": root_result.ancestor_deviation,
        "ambiguity_index": root_result.ambiguity_index,
        "root_node": str(root_node),
    }

    rooted_result = MadRootResult(
        edge=root_result.edge,
        fraction_from_u=root_result.fraction_from_u,
        distance_from_u=root_result.distance_from_u,
        ancestor_deviation=root_result.ancestor_deviation,
        ambiguity_index=root_result.ambiguity_index,
        root_node=root_node,
    )
    return tree, rooted_result


def _skbio_tree_to_graph(tree_node: object) -> tuple[nx.Graph, list[str]]:
    graph = nx.Graph()
    leaf_labels: list[str] = []
    internal_counter = 0
    node_ids: dict[int, str] = {}

    def _node_id(node: object) -> str:
        nonlocal internal_counter
        raw_name = getattr(node, "name", None)
        is_tip = bool(node.is_tip())
        if is_tip and raw_name:
            name = str(raw_name)
            leaf_labels.append(name)
            return name
        key = id(node)
        if key not in node_ids:
            node_ids[key] = f"internal_{internal_counter}"
            internal_counter += 1
        return node_ids[key]

    def _visit(node: object) -> str:
        parent_id = _node_id(node)
        graph.add_node(parent_id)
        for child in node.children:
            child_id = _visit(child)
            length = 0.0 if child.length is None else float(child.length)
            graph.add_edge(
                parent_id,
                child_id,
                branch_length=length,
                length=length,
                weight=length,
            )
        return parent_id

    _visit(tree_node)
    return graph, leaf_labels


def neighbor_joining_tree_from_distance(
    distance_condensed: np.ndarray,
    leaf_names: list[str],
    *,
    rooting: str = "mad",
) -> tuple["PosetTree", MadRootResult]:
    """Build a MAD-rooted neighbor-joining tree from a condensed distance vector."""
    try:
        from skbio import DistanceMatrix
        from skbio.tree import nj
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Neighbor-joining tree construction requires scikit-bio."
        ) from exc

    matrix = squareform(np.asarray(distance_condensed, dtype=float))
    if matrix.shape[0] != len(leaf_names):
        raise ValueError(
            "Distance vector size does not match the number of leaf names: "
            f"{matrix.shape[0]} != {len(leaf_names)}."
        )
    tree_node = nj(DistanceMatrix(matrix, ids=[str(name) for name in leaf_names]))
    graph, leaves = _skbio_tree_to_graph(tree_node)
    tree, root = poset_tree_from_unrooted_metric_tree(
        graph,
        leaf_labels=leaves,
        rooting=rooting,
    )
    return tree, root


def _safe_iqtree_id(index: int, sample_id: object) -> str:
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("_.-")
    if not raw or raw[0].isdigit():
        raw = f"s{index}_{raw}" if raw else f"s{index}"
    return raw[:48]


def _feature_matrix_to_fasta(data_df: pd.DataFrame, path: Path) -> dict[str, str]:
    ids: dict[str, str] = {}
    used: set[str] = set()
    encoded_columns: list[list[str]] = []
    for column in data_df.columns:
        values = data_df[column].tolist()
        unique_values = sorted({value for value in values}, key=lambda value: repr(value))
        if len(unique_values) > len(_IQTREE_SYMBOLS):
            raise ValueError(
                f"Column {column!r} has {len(unique_values)} states; IQ-TREE feature "
                f"encoding supports at most {len(_IQTREE_SYMBOLS)} states per feature."
            )
        mapping = {
            value: _IQTREE_SYMBOLS[state_index]
            for state_index, value in enumerate(unique_values)
        }
        encoded_columns.append([mapping[value] for value in values])

    with path.open("w", encoding="utf-8") as handle:
        for row_index, sample_id in enumerate(data_df.index):
            safe_id = _safe_iqtree_id(row_index, sample_id)
            base_id = safe_id
            suffix = 1
            while safe_id in used:
                suffix += 1
                safe_id = f"{base_id}_{suffix}"
            used.add(safe_id)
            ids[safe_id] = str(sample_id)
            sequence = "".join(column[row_index] for column in encoded_columns)
            handle.write(f">{safe_id}\n{sequence}\n")
    return ids


def _read_newick_tree(path: Path) -> tuple[nx.Graph, list[str]]:
    try:
        from skbio import TreeNode
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("IQ-TREE Newick parsing requires scikit-bio.") from exc

    tree_node = TreeNode.read(str(path))
    graph, leaves = _skbio_tree_to_graph(tree_node)
    for _, _, attrs in graph.edges(data=True):
        length = float(attrs["length"])
        if not math.isfinite(length):
            raise ValueError("Newick tree contains a non-finite branch length.")
    return graph, leaves


def iqtree3_tree_from_alignment(
    data_df: pd.DataFrame,
    *,
    executable: str = "iqtree3",
    model: str = "JC2",
    threads: int = 1,
    rooting: str = "mad",
    work_dir: str | Path | None = None,
    prefix: str = "kl_te_iqtree",
) -> tuple["PosetTree", MadRootResult, dict[str, object]]:
    """Run IQ-TREE 3 on encoded feature rows and return a MAD-rooted tree."""
    executable_path = shutil.which(executable)
    if executable_path is None:
        raise RuntimeError(
            f"IQ-TREE executable {executable!r} was not found on PATH. "
            "Install IQ-TREE 3 or pass iqtree_executable."
        )

    if work_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="kl_te_iqtree_")
        run_dir = Path(temp_context.name)
    else:
        temp_context = None
        run_dir = Path(work_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        alignment_path = run_dir / f"{prefix}.fasta"
        name_map = _feature_matrix_to_fasta(data_df, alignment_path)
        output_prefix = run_dir / prefix
        cmd = [
            executable_path,
            "-s",
            str(alignment_path),
            "-m",
            str(model),
            "-T",
            str(int(threads)),
            "--prefix",
            str(output_prefix),
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        treefile = output_prefix.with_suffix(".treefile")
        if not treefile.exists():
            raise RuntimeError(f"IQ-TREE did not write expected treefile: {treefile}.")

        graph, leaves = _read_newick_tree(treefile)
        relabel_map = {safe: original for safe, original in name_map.items() if safe in graph}
        if relabel_map:
            graph = nx.relabel_nodes(graph, relabel_map, copy=True)
            leaves = [name_map.get(leaf, leaf) for leaf in leaves]
        tree, root = poset_tree_from_unrooted_metric_tree(
            graph,
            leaf_labels=leaves,
            rooting=rooting,
        )
        metadata = {
            "command": cmd,
            "alignment_path": str(alignment_path),
            "treefile": str(treefile),
            "model": str(model),
            "threads": int(threads),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        return tree, root, metadata
    finally:
        if temp_context is not None:
            temp_context.cleanup()
