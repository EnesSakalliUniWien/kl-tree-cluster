"""Oracle recoverability diagnostics for benchmark hierarchy trees."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tree_break_selection.core_utils.tree_utils import bottom_up_nodes

FAILURE_CLASS_SOLVED = "solved"
FAILURE_CLASS_TREE_UNRECOVERABLE = "tree_unrecoverable"
FAILURE_CLASS_GATE_UNDER_SPLIT = "gate_under_split"
FAILURE_CLASS_GATE_OVER_SPLIT = "gate_over_split"
FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE = (
    "tree_recoverable_statistical_failure"
)
FAILURE_CLASS_ORACLE_MATCHED_BELOW_SOLVED = "oracle_matched_below_solved"


@dataclass(frozen=True)
class OracleTreeCutResult:
    """Best recoverable flat partition inside a rooted hierarchy."""

    ari: float
    found_clusters: int
    selected_nodes: tuple[object, ...]
    labels: np.ndarray
    dinkelbach_iterations: int


@dataclass(frozen=True)
class _NodeStats:
    truth_counts: np.ndarray
    same_truth_pairs: float
    predicted_pairs: float


@dataclass(frozen=True)
class _CutState:
    score: float
    same_truth_pairs: float
    predicted_pairs: float
    selected_nodes: tuple[object, ...]


def _pair_count(size: int) -> int:
    return comb(int(size), 2) if size >= 2 else 0


def _validate_leaf_truth(
    tree,
    sample_index: pd.Index,
    true_labels: np.ndarray,
) -> pd.Series:
    if len(sample_index) != len(true_labels):
        raise ValueError(
            "sample_index and true_labels must have identical length. "
            f"Got {len(sample_index)} samples and {len(true_labels)} labels."
        )
    if sample_index.has_duplicates:
        raise ValueError("sample_index must not contain duplicate sample ids.")

    leaf_labels = pd.Index(tree.get_leaves(return_labels=True, sort=True))
    expected = pd.Index(sample_index).sort_values()
    if not leaf_labels.equals(expected):
        missing = expected.difference(leaf_labels)
        extras = leaf_labels.difference(expected)
        raise ValueError(
            "Tree leaves must match sample_index labels. "
            f"Missing={list(missing[:5])}, extra={list(extras[:5])}."
        )

    return pd.Series(np.asarray(true_labels), index=sample_index)


def _compute_node_stats(tree, truth_by_sample: pd.Series) -> dict[object, _NodeStats]:
    truth_levels = tuple(pd.unique(truth_by_sample))
    truth_to_position: dict[Hashable, int] = {
        truth_value: index for index, truth_value in enumerate(truth_levels)
    }
    stats_by_node: dict[object, _NodeStats] = {}

    for node in bottom_up_nodes(tree):
        if tree.nodes[node]["is_leaf"]:
            sample_id = tree.nodes[node]["label"]
            truth_value = truth_by_sample.loc[sample_id]
            counts = np.zeros(len(truth_levels), dtype=np.int64)
            counts[truth_to_position[truth_value]] = 1
        else:
            child_counts = [stats_by_node[child].truth_counts for child in tree.successors(node)]
            if not child_counts:
                raise ValueError(f"Internal node {node!r} has no children.")
            counts = np.sum(child_counts, axis=0)

        same_truth_pairs = float(sum(_pair_count(int(count)) for count in counts))
        predicted_pairs = float(_pair_count(int(counts.sum())))
        stats_by_node[node] = _NodeStats(
            truth_counts=counts,
            same_truth_pairs=same_truth_pairs,
            predicted_pairs=predicted_pairs,
        )

    return stats_by_node


def _whole_node_state(
    node: object,
    node_stats: _NodeStats,
    *,
    q: float,
    truth_pair_fraction: float,
    denominator_slope: float,
) -> _CutState:
    score = node_stats.same_truth_pairs - (
        truth_pair_fraction + q * denominator_slope
    ) * node_stats.predicted_pairs
    return _CutState(
        score=float(score),
        same_truth_pairs=node_stats.same_truth_pairs,
        predicted_pairs=node_stats.predicted_pairs,
        selected_nodes=(node,),
    )


def _combine_states(left: _CutState, right: _CutState) -> _CutState:
    return _CutState(
        score=left.score + right.score,
        same_truth_pairs=left.same_truth_pairs + right.same_truth_pairs,
        predicted_pairs=left.predicted_pairs + right.predicted_pairs,
        selected_nodes=left.selected_nodes + right.selected_nodes,
    )


def _better_state(left: _CutState | None, right: _CutState) -> _CutState:
    if left is None:
        return right
    if right.score > left.score:
        return right
    if right.score == left.score and len(right.selected_nodes) < len(left.selected_nodes):
        return right
    return left


def _optimize_additive_cut(
    tree,
    node_stats_by_node: dict[object, _NodeStats],
    *,
    q: float,
    truth_pair_fraction: float,
    denominator_slope: float,
) -> _CutState:
    best_by_node: dict[object, _CutState] = {}
    for node in bottom_up_nodes(tree):
        whole = _whole_node_state(
            node,
            node_stats_by_node[node],
            q=q,
            truth_pair_fraction=truth_pair_fraction,
            denominator_slope=denominator_slope,
        )
        children = list(tree.successors(node))
        if not children:
            best_by_node[node] = whole
            continue

        split: _CutState | None = None
        for child in children:
            split = best_by_node[child] if split is None else _combine_states(
                split,
                best_by_node[child],
            )
        best_by_node[node] = _better_state(whole, split)

    return best_by_node[tree.root()]


def _convolve_exact_k_states(
    left: dict[int, _CutState],
    right: dict[int, _CutState],
    *,
    max_k: int,
) -> dict[int, _CutState]:
    combined: dict[int, _CutState] = {}
    for left_k, left_state in left.items():
        for right_k, right_state in right.items():
            total_k = left_k + right_k
            if total_k > max_k:
                continue
            combined_state = _combine_states(left_state, right_state)
            combined[total_k] = _better_state(combined.get(total_k), combined_state)
    return combined


def _optimize_additive_cut_exact_k(
    tree,
    node_stats_by_node: dict[object, _NodeStats],
    *,
    q: float,
    truth_pair_fraction: float,
    denominator_slope: float,
    exact_k: int,
) -> _CutState:
    states_by_node: dict[object, dict[int, _CutState]] = {}
    for node in bottom_up_nodes(tree):
        states: dict[int, _CutState] = {
            1: _whole_node_state(
                node,
                node_stats_by_node[node],
                q=q,
                truth_pair_fraction=truth_pair_fraction,
                denominator_slope=denominator_slope,
            )
        }
        children = list(tree.successors(node))
        if children:
            split_states: dict[int, _CutState] | None = None
            for child in children:
                child_states = states_by_node[child]
                split_states = (
                    child_states
                    if split_states is None
                    else _convolve_exact_k_states(
                        split_states,
                        child_states,
                        max_k=exact_k,
                    )
                )
            if split_states is not None:
                for k, split_state in split_states.items():
                    states[k] = _better_state(states.get(k), split_state)
        states_by_node[node] = states

    root_states = states_by_node[tree.root()]
    if exact_k not in root_states:
        raise ValueError(
            f"Tree cannot produce an exact cut with {exact_k} clusters. "
            f"Available cluster counts: {sorted(root_states)}."
        )
    return root_states[exact_k]


def _labels_for_selected_nodes(
    tree,
    selected_nodes: tuple[object, ...],
    sample_index: pd.Index,
) -> np.ndarray:
    labels_by_sample: dict[object, int] = {}
    for cluster_id, node in enumerate(selected_nodes):
        for sample_id in tree.get_leaves(node=node, return_labels=True, sort=False):
            if sample_id in labels_by_sample:
                raise ValueError(f"Sample {sample_id!r} appears in multiple oracle clusters.")
            labels_by_sample[sample_id] = cluster_id

    missing = [sample_id for sample_id in sample_index if sample_id not in labels_by_sample]
    if missing:
        raise ValueError(f"Oracle cut did not cover every sample. Missing={missing[:5]}.")
    return np.asarray([labels_by_sample[sample_id] for sample_id in sample_index], dtype=int)


def _ari_from_pair_sums(
    *,
    same_truth_pairs: float,
    predicted_pairs: float,
    total_truth_pairs: float,
    total_pairs: float,
) -> float:
    truth_pair_fraction = total_truth_pairs / total_pairs
    denominator = 0.5 * (predicted_pairs + total_truth_pairs) - (
        predicted_pairs * truth_pair_fraction
    )
    if denominator == 0.0:
        return 1.0
    numerator = same_truth_pairs - predicted_pairs * truth_pair_fraction
    return float(numerator / denominator)


def oracle_subtree_cut(
    tree,
    *,
    sample_index: pd.Index,
    true_labels: np.ndarray,
    exact_k: int | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> OracleTreeCutResult:
    """Return the best ARI achievable by pruning the rooted tree.

    The admissible partitions are rooted subtree cuts: each selected node becomes
    one predicted cluster, selected nodes are disjoint, and together they cover
    all leaves.  When ``exact_k`` is provided, the oracle is restricted to cuts
    with exactly that many selected nodes.
    """
    sample_index = pd.Index(sample_index)
    truth_by_sample = _validate_leaf_truth(tree, sample_index, np.asarray(true_labels))
    if exact_k is not None and exact_k < 1:
        raise ValueError(f"exact_k must be positive when provided, got {exact_k}.")
    if exact_k is not None and exact_k > len(sample_index):
        raise ValueError(
            f"exact_k={exact_k} exceeds the number of samples ({len(sample_index)})."
        )

    if len(sample_index) < 2 or truth_by_sample.nunique(dropna=False) <= 1:
        if exact_k not in (None, 1):
            raise ValueError("Single-class truth only supports exact_k=1 for oracle ARI.")
        labels = np.zeros(len(sample_index), dtype=int)
        return OracleTreeCutResult(
            ari=1.0,
            found_clusters=1,
            selected_nodes=(tree.root(),),
            labels=labels,
            dinkelbach_iterations=0,
        )

    node_stats_by_node = _compute_node_stats(tree, truth_by_sample)
    root_stats = node_stats_by_node[tree.root()]
    total_pairs = float(_pair_count(len(sample_index)))
    total_truth_pairs = root_stats.same_truth_pairs
    truth_pair_fraction = total_truth_pairs / total_pairs
    denominator_slope = 0.5 - truth_pair_fraction

    q = 0.0
    best_state: _CutState | None = None
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if exact_k is None:
            state = _optimize_additive_cut(
                tree,
                node_stats_by_node,
                q=q,
                truth_pair_fraction=truth_pair_fraction,
                denominator_slope=denominator_slope,
            )
        else:
            state = _optimize_additive_cut_exact_k(
                tree,
                node_stats_by_node,
                q=q,
                truth_pair_fraction=truth_pair_fraction,
                denominator_slope=denominator_slope,
                exact_k=exact_k,
            )
        best_state = state
        next_q = _ari_from_pair_sums(
            same_truth_pairs=state.same_truth_pairs,
            predicted_pairs=state.predicted_pairs,
            total_truth_pairs=total_truth_pairs,
            total_pairs=total_pairs,
        )
        if abs(next_q - q) <= tolerance:
            q = next_q
            break
        q = next_q

    if best_state is None:
        raise RuntimeError("Oracle cut optimization did not evaluate any cut.")

    labels = _labels_for_selected_nodes(tree, best_state.selected_nodes, sample_index)
    ari = float(adjusted_rand_score(truth_by_sample.to_numpy(), labels))
    return OracleTreeCutResult(
        ari=ari,
        found_clusters=len(best_state.selected_nodes),
        selected_nodes=best_state.selected_nodes,
        labels=labels,
        dinkelbach_iterations=iterations,
    )


def classify_tree_recoverability_failure(
    *,
    tbs_ari: float,
    tbs_found_clusters: int,
    true_clusters: int,
    oracle_true_k_subtree_ari: float,
    solved_ari_threshold: float = 0.95,
    recoverable_ari_threshold: float = 0.8,
    oracle_gap_tolerance: float = 1e-9,
) -> str:
    """Classify whether a TBS miss is tree-limited or gate/stopping-limited."""
    metrics = {
        "tbs_ari": tbs_ari,
        "oracle_true_k_subtree_ari": oracle_true_k_subtree_ari,
    }
    for name, value in metrics.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite for failure classification.")

    if not (0.0 <= solved_ari_threshold <= 1.0):
        raise ValueError("solved_ari_threshold must lie in [0, 1].")
    if not (0.0 <= recoverable_ari_threshold <= 1.0):
        raise ValueError("recoverable_ari_threshold must lie in [0, 1].")
    if solved_ari_threshold < recoverable_ari_threshold:
        raise ValueError(
            "solved_ari_threshold must be greater than or equal to "
            "recoverable_ari_threshold."
        )
    if oracle_gap_tolerance < 0.0:
        raise ValueError("oracle_gap_tolerance must be non-negative.")

    found = int(tbs_found_clusters)
    truth = int(true_clusters)
    if found < 1:
        raise ValueError(f"tbs_found_clusters must be positive, got {found}.")
    if truth < 1:
        raise ValueError(f"true_clusters must be positive, got {truth}.")

    if float(tbs_ari) >= solved_ari_threshold:
        return FAILURE_CLASS_SOLVED
    if float(oracle_true_k_subtree_ari) < recoverable_ari_threshold:
        return FAILURE_CLASS_TREE_UNRECOVERABLE
    if float(tbs_ari) >= float(oracle_true_k_subtree_ari) - oracle_gap_tolerance:
        return FAILURE_CLASS_ORACLE_MATCHED_BELOW_SOLVED
    if found < truth:
        return FAILURE_CLASS_GATE_UNDER_SPLIT
    if found > truth:
        return FAILURE_CLASS_GATE_OVER_SPLIT
    return FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE


__all__ = [
    "FAILURE_CLASS_GATE_OVER_SPLIT",
    "FAILURE_CLASS_GATE_UNDER_SPLIT",
    "FAILURE_CLASS_ORACLE_MATCHED_BELOW_SOLVED",
    "FAILURE_CLASS_SOLVED",
    "FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE",
    "FAILURE_CLASS_TREE_UNRECOVERABLE",
    "OracleTreeCutResult",
    "classify_tree_recoverability_failure",
    "oracle_subtree_cut",
]
