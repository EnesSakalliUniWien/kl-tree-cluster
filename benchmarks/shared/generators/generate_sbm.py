"""Stochastic Block Model (SBM) graph generator utilities.

Provides `generate_sbm` which builds a random graph with planted
community structure using NetworkX's SBM implementation and returns the
graph, ground-truth community labels, adjacency matrix, and metadata.

The function is intentionally lightweight (minimal dependencies) and may
raise ImportError with a helpful message if NetworkX is not available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from networkx.generators.community import stochastic_block_model


def _validate_sizes(sizes: List[int]) -> None:
    if not sizes:
        raise ValueError("`sizes` must be a non-empty list of positive integers")
    if any(int(s) <= 0 for s in sizes):
        raise ValueError("All entries of `sizes` must be positive integers")


def _build_probability_matrix(n_blocks: int, p_intra: float, p_inter: float) -> List[List[float]]:
    return [[p_intra if i == j else p_inter for j in range(n_blocks)] for i in range(n_blocks)]


def generate_sbm(
    sizes: List[int],
    p_intra: float,
    p_inter: float,
    seed: int,
    directed: bool,
    allow_self_loops: bool,
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a graph with planted community structure via an SBM.

    Parameters
    ----------
    sizes : list[int]
        Number of nodes in each block (community). The sum is the number of
        nodes in the returned graph.
    p_intra : float
        Probability of an edge between two nodes in the same block.
    p_inter : float
        Probability of an edge between nodes in different blocks.
    seed : int
        RNG seed for reproducibility.
    directed : bool
        Whether to produce a directed graph.
    allow_self_loops : bool
        Whether to allow self-loops in the graph.

    Returns
    -------
    G : networkx.Graph
        The generated graph instance (undirected if ``directed=False``).
    ground_truth : np.ndarray
        Integer community labels for each node in 0..(n_blocks-1).
    adjacency : np.ndarray
        N x N adjacency matrix (0/1) as float array.
    metadata : dict
        Dictionary with keys: sizes, p_intra, p_inter, n_nodes, n_blocks,
        directed, seed.

    Notes
    -----
    This is a thin wrapper around :func:`networkx.generators.community.stochastic_block_model`.
    """
    _validate_sizes(sizes)
    if not (0.0 <= p_inter <= 1.0 and 0.0 <= p_intra <= 1.0):
        raise ValueError("p_intra and p_inter must be probabilities in [0, 1]")

    n_blocks = len(sizes)
    probs = _build_probability_matrix(n_blocks, float(p_intra), float(p_inter))

    # Build graph
    G = stochastic_block_model(
        sizes, probs, directed=directed, seed=seed, selfloops=allow_self_loops
    )

    # Ground truth labels: node ordering in stochastic_block_model respects block order
    ground_truth = []
    for block_idx, size in enumerate(sizes):
        ground_truth.extend([block_idx] * int(size))
    ground_truth = np.asarray(ground_truth, dtype=int)

    # Shared benchmark runners consume numeric matrices.
    A = nx.to_numpy_array(G, dtype=float)

    metadata: Dict[str, Any] = {
        "sizes": sizes,
        "p_intra": p_intra,
        "p_inter": p_inter,
        "n_nodes": int(sum(sizes)),
        "n_blocks": n_blocks,
        "directed": bool(directed),
        "seed": seed,
    }

    return G, ground_truth, A, metadata
