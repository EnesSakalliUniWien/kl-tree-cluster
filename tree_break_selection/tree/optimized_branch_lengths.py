"""Continuous-data branch-length optimization for rooted ``PosetTree`` objects.

The default linkage constructor stores ultrametric merge-height differences as
``branch_length``.  Those heights are useful topology diagnostics, but they are
not calibrated stochastic time.  This module provides a native fixed-topology
non-negative least-squares fit for continuous data:

    D_ij ~= sum_{e in path(i, j)} tau_e,  tau_e >= 0

where ``D_ij`` is a held-out squared standardized continuous distance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import lsq_linear

BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC = "linkage_ultrametric"
BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS = "fixed_topology_nnls"
BRANCH_LENGTH_OPTIMIZATION_METHODS = (
    BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC,
    BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
)

BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN = "squared_standardized_euclidean"


@dataclass(frozen=True)
class BranchLengthOptimizationResult:
    """Summary of one branch-length optimization pass."""

    method: str
    target_metric: str
    status: str
    n_leaves: int
    n_edges: int
    n_pairs_total: int
    n_pairs_used: int
    pair_sample_size: int | None
    random_state: int
    elapsed_sec: float
    cost: float
    optimality: float
    iterations: int
    target_mean: float
    target_median: float
    fitted_mean: float
    fitted_median: float
    residual_rmse: float
    residual_mae: float
    branch_length_mean: float
    branch_length_median: float
    branch_length_min: float
    branch_length_max: float
    solver_message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_branch_length_optimization_method(value: str) -> str:
    """Return a valid branch-length optimization method id."""
    method = str(value)
    if method not in BRANCH_LENGTH_OPTIMIZATION_METHODS:
        raise ValueError(
            "branch_length_optimization_method must be one of "
            f"{BRANCH_LENGTH_OPTIMIZATION_METHODS!r}; got {value!r}."
        )
    return method


def _leaf_nodes_for_data(tree: nx.DiGraph, data_index: pd.Index) -> list[object]:
    label_to_node: dict[str, object] = {}
    for node_id, attrs in tree.nodes(data=True):
        if bool(attrs.get("is_leaf", False)):
            label = str(attrs.get("label", node_id))
            if label in label_to_node:
                raise ValueError(f"Duplicate tree leaf label {label!r}.")
            label_to_node[label] = node_id

    leaf_nodes: list[object] = []
    missing_labels: list[str] = []
    for label in data_index.astype(str):
        node_id = label_to_node.get(str(label))
        if node_id is None:
            missing_labels.append(str(label))
        else:
            leaf_nodes.append(node_id)
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(f"Tree is missing {len(missing_labels)} data leaf label(s): {preview}.")
    return leaf_nodes


def _edge_indices(tree: nx.DiGraph) -> dict[tuple[object, object], int]:
    return {(parent, child): index for index, (parent, child) in enumerate(tree.edges())}


def _root_path_edge_sets(
    tree: nx.DiGraph,
    leaf_nodes: list[object],
    edge_index_by_pair: dict[tuple[object, object], int],
) -> list[frozenset[int]]:
    path_sets: list[frozenset[int]] = []
    root = tree.graph.get("root")
    if root is None:
        roots = [node for node, degree in tree.in_degree() if degree == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected one tree root, got {roots!r}.")
        root = roots[0]

    for leaf in leaf_nodes:
        current = leaf
        edge_ids: list[int] = []
        while current != root:
            parents = list(tree.predecessors(current))
            if len(parents) != 1:
                raise ValueError(
                    f"Node {current!r} must have exactly one parent; found {len(parents)}."
                )
            parent = parents[0]
            edge_ids.append(edge_index_by_pair[(parent, current)])
            current = parent
        path_sets.append(frozenset(edge_ids))
    return path_sets


def _sample_leaf_pairs(
    n_leaves: int,
    *,
    pair_sample_size: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    total_pairs = n_leaves * (n_leaves - 1) // 2
    if pair_sample_size is None or int(pair_sample_size) >= total_pairs:
        left, right = np.triu_indices(n_leaves, k=1)
        return left.astype(np.int32), right.astype(np.int32), total_pairs

    target_size = int(pair_sample_size)
    if target_size <= 0:
        raise ValueError(f"pair_sample_size must be positive; got {pair_sample_size!r}.")
    rng = np.random.default_rng(int(random_state))
    pairs: set[tuple[int, int]] = set()
    batch_size = max(4096, target_size)
    while len(pairs) < target_size:
        left = rng.integers(0, n_leaves, size=batch_size)
        right = rng.integers(0, n_leaves, size=batch_size)
        different = left != right
        left = left[different]
        right = right[different]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        for pair in zip(low.tolist(), high.tolist(), strict=True):
            pairs.add((int(pair[0]), int(pair[1])))
            if len(pairs) >= target_size:
                break

    ordered_pairs = np.array(sorted(pairs), dtype=np.int32)
    return ordered_pairs[:, 0], ordered_pairs[:, 1], total_pairs


def _squared_standardized_targets(
    data: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    matrix = data.to_numpy(dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("Branch-length optimization requires a 2-D continuous data matrix.")
    center = np.mean(matrix, axis=0, keepdims=True)
    scale = np.std(matrix, axis=0, ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    standardized = (matrix - center) / scale
    diffs = standardized[left] - standardized[right]
    return np.sum(diffs * diffs, axis=1) / float(standardized.shape[1])


def _path_incidence_matrix(
    root_path_sets: list[frozenset[int]],
    left: np.ndarray,
    right: np.ndarray,
    n_edges: int,
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_index, (left_index, right_index) in enumerate(zip(left, right, strict=True)):
        path_edges = root_path_sets[int(left_index)] ^ root_path_sets[int(right_index)]
        row_indices.extend([row_index] * len(path_edges))
        col_indices.extend(path_edges)
    data = np.ones(len(row_indices), dtype=np.float64)
    return sparse.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(left), n_edges),
        dtype=np.float64,
    ).tocsr()


def fit_fixed_topology_nnls_branch_lengths(
    tree: nx.DiGraph,
    continuous_data: pd.DataFrame,
    *,
    target_metric: str = BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN,
    pair_sample_size: int | None = 100_000,
    random_state: int = 0,
    solver_tolerance: float = 1e-6,
    max_iterations: int | None = None,
) -> BranchLengthOptimizationResult:
    """Fit non-negative edge lengths on a fixed tree topology.

    The function mutates ``tree`` by replacing each edge's ``branch_length``
    with the fitted value.  The previous value is preserved as
    ``linkage_branch_length`` when present.
    """
    start_sec = perf_counter()
    if target_metric != BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN:
        raise ValueError(
            "Unsupported branch-length target metric "
            f"{target_metric!r}; expected "
            f"{BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN!r}."
        )
    if not isinstance(continuous_data, pd.DataFrame):
        raise TypeError("continuous_data must be a pandas DataFrame.")
    if continuous_data.isna().to_numpy().any():
        raise ValueError("continuous_data must not contain missing values.")

    leaf_nodes = _leaf_nodes_for_data(tree, continuous_data.index)
    edge_index_by_pair = _edge_indices(tree)
    n_edges = len(edge_index_by_pair)
    if n_edges == 0:
        raise ValueError("Branch-length optimization requires a non-empty tree.")

    root_path_sets = _root_path_edge_sets(tree, leaf_nodes, edge_index_by_pair)
    left, right, total_pairs = _sample_leaf_pairs(
        len(leaf_nodes),
        pair_sample_size=pair_sample_size,
        random_state=random_state,
    )
    targets = _squared_standardized_targets(continuous_data, left, right)
    design = _path_incidence_matrix(root_path_sets, left, right, n_edges)

    solution = lsq_linear(
        design,
        targets,
        bounds=(0.0, np.inf),
        method="trf",
        tol=float(solver_tolerance),
        max_iter=max_iterations,
        lsmr_tol="auto",
        verbose=0,
    )
    lengths = np.asarray(solution.x, dtype=float)
    fitted = np.asarray(design @ lengths, dtype=float)
    residuals = fitted - targets

    edge_by_index = {index: edge for edge, index in edge_index_by_pair.items()}
    for index, length in enumerate(lengths):
        parent, child = edge_by_index[index]
        attrs = tree.edges[parent, child]
        if "branch_length" in attrs and "linkage_branch_length" not in attrs:
            attrs["linkage_branch_length"] = attrs["branch_length"]
        attrs["branch_length"] = float(length)
        attrs["branch_length_source"] = BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS

    tree.graph["branch_length_optimization"] = {
        "method": BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
        "target_metric": target_metric,
        "pair_sample_size": None if pair_sample_size is None else int(pair_sample_size),
        "random_state": int(random_state),
    }

    return BranchLengthOptimizationResult(
        method=BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS,
        target_metric=target_metric,
        status="ok" if bool(solution.success) else "solver_not_converged",
        n_leaves=len(leaf_nodes),
        n_edges=n_edges,
        n_pairs_total=total_pairs,
        n_pairs_used=len(left),
        pair_sample_size=None if pair_sample_size is None else int(pair_sample_size),
        random_state=int(random_state),
        elapsed_sec=perf_counter() - start_sec,
        cost=float(solution.cost),
        optimality=float(solution.optimality),
        iterations=int(solution.nit),
        target_mean=float(np.mean(targets)),
        target_median=float(np.median(targets)),
        fitted_mean=float(np.mean(fitted)),
        fitted_median=float(np.median(fitted)),
        residual_rmse=float(np.sqrt(np.mean(residuals * residuals))),
        residual_mae=float(np.mean(np.abs(residuals))),
        branch_length_mean=float(np.mean(lengths)),
        branch_length_median=float(np.median(lengths)),
        branch_length_min=float(np.min(lengths)),
        branch_length_max=float(np.max(lengths)),
        solver_message=str(solution.message),
    )


__all__ = [
    "BRANCH_LENGTH_OPTIMIZATION_FIXED_TOPOLOGY_NNLS",
    "BRANCH_LENGTH_OPTIMIZATION_LINKAGE_ULTRAMETRIC",
    "BRANCH_LENGTH_OPTIMIZATION_METHODS",
    "BRANCH_LENGTH_TARGET_SQUARED_STANDARDIZED_EUCLIDEAN",
    "BranchLengthOptimizationResult",
    "fit_fixed_topology_nnls_branch_lengths",
    "validate_branch_length_optimization_method",
]
