"""
Generate synthetic phylogenetic sequence data via branch-wise JC evolution.

This generator simulates categorical sequences on a random binary tree:
1. Sample a root sequence (with per-feature root priors).
2. Evolve sequences along each branch using Jukes-Cantor substitutions.
3. Sample observations at leaves (taxa) with optional within-taxon drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from benchmarks.shared.evolution import (
    evolve_sequence,
    generate_dirichlet_distributions,
)


@dataclass
class PhyloNode:
    """A node in the random phylogenetic tree."""

    id: str
    parent: Optional["PhyloNode"] = None
    left: Optional["PhyloNode"] = None
    right: Optional["PhyloNode"] = None
    # Incoming edge length from parent to this node (root has 0.0).
    branch_length: float = 0.0
    sequence: Optional[np.ndarray] = None
    is_leaf: bool = False
    depth: int = 0


def _generate_random_tree(
    n_leaves: int,
    random_state: np.random.RandomState,
    branch_length_mean: float = 1.0,
) -> tuple[PhyloNode, list[PhyloNode]]:
    """Generate a random binary tree with branch lengths on edges."""
    leaves = [PhyloNode(id=f"T{i}", is_leaf=True, depth=0) for i in range(n_leaves)]
    nodes = leaves.copy()
    internal_id = 0

    while len(nodes) > 1:
        idx1, idx2 = random_state.choice(len(nodes), size=2, replace=False)
        child_left, child_right = nodes[idx1], nodes[idx2]

        # Assign branch lengths to child edges from the new parent.
        child_left.branch_length = float(random_state.exponential(branch_length_mean))
        child_right.branch_length = float(random_state.exponential(branch_length_mean))

        parent = PhyloNode(
            id=f"I{internal_id}",
            left=child_left,
            right=child_right,
            branch_length=0.0,
            depth=max(child_left.depth, child_right.depth) + 1,
        )
        child_left.parent = parent
        child_right.parent = parent
        internal_id += 1

        nodes = [node for i, node in enumerate(nodes) if i not in (idx1, idx2)]
        nodes.append(parent)

    root = nodes[0]
    root.branch_length = 0.0
    return root, leaves


def _sample_root_sequence(
    n_features: int,
    n_categories: int,
    root_concentration: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Sample a root sequence using per-feature Dirichlet priors."""
    root_distributions = generate_dirichlet_distributions(
        n_features=n_features,
        n_categories=n_categories,
        concentration=root_concentration,
        random_state=random_state,
    )
    return np.array(
        [
            random_state.choice(n_categories, p=root_distributions[feature_idx])
            for feature_idx in range(n_features)
        ],
        dtype=int,
    )


def _evolve_tree_sequences(
    root: PhyloNode,
    n_features: int,
    n_categories: int,
    mutation_rate: float,
    root_concentration: float,
    random_state: np.random.RandomState,
) -> None:
    """Evolve sequences from root to leaves."""
    root.sequence = _sample_root_sequence(
        n_features=n_features,
        n_categories=n_categories,
        root_concentration=root_concentration,
        random_state=random_state,
    )

    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.sequence is None:
            continue

        for child in (node.left, node.right):
            if child is None:
                continue
            effective_branch_length = max(0.0, mutation_rate * float(child.branch_length))
            child.sequence = evolve_sequence(
                ancestor=node.sequence,
                branch_length=effective_branch_length,
                n_categories=n_categories,
                random_state=random_state,
            )
            queue.append(child)


def _sample_from_leaves(
    leaves: list[PhyloNode],
    samples_per_leaf: int,
    n_categories: int,
    mutation_rate: float,
    random_state: np.random.RandomState,
    within_taxon_branch_length: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, dict[str, np.ndarray]]:
    """Sample observed sequences from leaves with mild within-taxon drift."""
    sample_dict: dict[str, np.ndarray] = {}
    cluster_assignments: dict[str, int] = {}
    sample_distributions: list[np.ndarray] = []
    sequences_by_leaf: dict[str, list[np.ndarray]] = {leaf.id: [] for leaf in leaves}

    eye = np.eye(n_categories, dtype=np.float32)
    within_effective = max(0.0, mutation_rate * within_taxon_branch_length)

    sample_idx = 0
    for leaf_idx, leaf in enumerate(leaves):
        if leaf.sequence is None:
            raise ValueError(f"Leaf {leaf.id} has no sequence; tree evolution did not complete.")

        for _ in range(samples_per_leaf):
            if within_effective > 0:
                sample_seq = evolve_sequence(
                    ancestor=leaf.sequence,
                    branch_length=within_effective,
                    n_categories=n_categories,
                    random_state=random_state,
                )
            else:
                sample_seq = leaf.sequence.copy()

            sample_name = f"S{sample_idx}"
            sample_dict[sample_name] = sample_seq
            cluster_assignments[sample_name] = leaf_idx
            sample_distributions.append(eye[sample_seq])
            sequences_by_leaf[leaf.id].append(sample_seq)
            sample_idx += 1

    leaf_distributions: dict[str, np.ndarray] = {}
    for leaf_id, seqs in sequences_by_leaf.items():
        if not seqs:
            continue
        stacked = np.stack(seqs, axis=0)  # (n_samples_leaf, n_features)
        one_hot = eye[stacked]  # (n_samples_leaf, n_features, n_categories)
        leaf_distributions[leaf_id] = one_hot.mean(axis=0)

    distributions_array = (
        np.stack(sample_distributions, axis=0)
        if sample_distributions
        else np.empty((0, 0, n_categories), dtype=np.float32)
    )
    return sample_dict, cluster_assignments, distributions_array, leaf_distributions


def _get_tree_structure(root: PhyloNode) -> dict:
    """Extract nested tree structure metadata."""

    def node_to_dict(node: PhyloNode | None) -> dict | None:
        if node is None:
            return None
        return {
            "id": node.id,
            "is_leaf": node.is_leaf,
            "depth": node.depth,
            "branch_length": float(node.branch_length),
            "left": node_to_dict(node.left),
            "right": node_to_dict(node.right),
        }

    return node_to_dict(root) or {}


def generate_phylogenetic_data(
    n_taxa: int,
    n_features: int,
    n_categories: int = 4,
    samples_per_taxon: int = 10,
    mutation_rate: float = 0.3,
    root_concentration: float = 1.0,
    random_seed: Optional[int] = None,
    within_taxon_branch_length: float = 0.2,
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, dict]:
    """Generate phylogenetic categorical sequence data.

    Returns:
        - sample_dict: sample_id -> sequence array (n_features,)
        - cluster_assignments: sample_id -> taxon id (0..n_taxa-1)
        - distributions: (n_samples, n_features, n_categories) one-hot per sample
        - metadata: tree and generation diagnostics
    """
    if n_taxa < 2:
        raise ValueError("n_taxa must be >= 2.")
    if n_features < 1:
        raise ValueError("n_features must be >= 1.")
    if n_categories < 2:
        raise ValueError("n_categories must be >= 2.")
    if samples_per_taxon < 1:
        raise ValueError("samples_per_taxon must be >= 1.")

    random_state = np.random.RandomState(random_seed)

    root, leaves = _generate_random_tree(n_taxa, random_state)
    _evolve_tree_sequences(
        root=root,
        n_features=n_features,
        n_categories=n_categories,
        mutation_rate=mutation_rate,
        root_concentration=root_concentration,
        random_state=random_state,
    )

    sample_dict, cluster_assignments, distributions, leaf_distributions = _sample_from_leaves(
        leaves=leaves,
        samples_per_leaf=samples_per_taxon,
        n_categories=n_categories,
        mutation_rate=mutation_rate,
        random_state=random_state,
        within_taxon_branch_length=within_taxon_branch_length,
    )

    leaf_sequences = {
        leaf.id: leaf.sequence.copy()
        for leaf in leaves
        if leaf.sequence is not None
    }

    metadata = {
        "n_taxa": n_taxa,
        "n_samples": len(sample_dict),
        "n_features": n_features,
        "n_categories": n_categories,
        "samples_per_taxon": samples_per_taxon,
        "mutation_rate": mutation_rate,
        "within_taxon_branch_length": within_taxon_branch_length,
        "root_concentration": root_concentration,
        "tree_structure": _get_tree_structure(root),
        "leaf_sequences": leaf_sequences,
        "leaf_distributions": leaf_distributions,
    }
    return sample_dict, cluster_assignments, distributions, metadata


__all__ = ["generate_phylogenetic_data"]
