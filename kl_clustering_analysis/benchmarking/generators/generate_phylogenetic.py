"""
Generates synthetic phylogenetic data by simulating trait evolution along a tree.

This module provides `generate_phylogenetic_data` to create datasets that mimic
how categorical traits (like DNA sequences, amino acids, or discrete phenotypes)
evolve along a phylogenetic tree.

The simulation:
1. Generates a random binary tree with specified number of leaves (taxa)
2. Assigns a root distribution for each feature
3. Evolves traits down the tree with mutation probability at each branch
4. Samples observed data at the leaves

This creates hierarchically structured data where closely related taxa
have more similar trait distributions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhyloNode:
    """A node in the phylogenetic tree."""
    id: str
    parent: Optional["PhyloNode"] = None
    left: Optional["PhyloNode"] = None
    right: Optional["PhyloNode"] = None
    branch_length: float = 1.0
    distribution: Optional[np.ndarray] = None  # (n_features, n_categories)
    is_leaf: bool = False
    depth: int = 0


def _generate_random_tree(
    n_leaves: int,
    random_state: np.random.RandomState,
    branch_length_mean: float = 1.0,
) -> Tuple[PhyloNode, List[PhyloNode]]:
    """Generate a random binary tree with n_leaves taxa.
    
    Uses a simple coalescent-like process: repeatedly join random pairs.
    
    Returns:
        (root_node, list_of_leaf_nodes)
    """
    # Create leaf nodes
    leaves = [
        PhyloNode(id=f"T{i}", is_leaf=True, depth=0)
        for i in range(n_leaves)
    ]
    
    # Build tree bottom-up by joining pairs
    nodes = leaves.copy()
    internal_id = 0
    
    while len(nodes) > 1:
        # Pick two random nodes to join
        idx1, idx2 = random_state.choice(len(nodes), size=2, replace=False)
        node1, node2 = nodes[idx1], nodes[idx2]
        
        # Create parent node
        branch_len = random_state.exponential(branch_length_mean)
        parent = PhyloNode(
            id=f"I{internal_id}",
            left=node1,
            right=node2,
            branch_length=branch_len,
            depth=max(node1.depth, node2.depth) + 1,
        )
        node1.parent = parent
        node2.parent = parent
        internal_id += 1
        
        # Remove children, add parent
        nodes = [n for i, n in enumerate(nodes) if i not in (idx1, idx2)]
        nodes.append(parent)
    
    root = nodes[0]
    return root, leaves


def _generate_root_distribution(
    n_features: int,
    n_categories: int,
    concentration: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate root distribution for each feature using Dirichlet.
    
    Args:
        n_features: Number of features/sites
        n_categories: Number of categories per feature
        concentration: Dirichlet concentration (higher = more uniform)
        random_state: Random state
        
    Returns:
        (n_features, n_categories) array of probability distributions
    """
    alpha = np.ones(n_categories) * concentration
    return random_state.dirichlet(alpha, size=n_features)


def _mutate_distribution(
    parent_dist: np.ndarray,
    mutation_rate: float,
    branch_length: float,
    n_categories: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Evolve distribution along a branch.
    
    Uses a simple model: with probability (mutation_rate * branch_length),
    each feature's distribution shifts toward a random category.
    
    Args:
        parent_dist: (n_features, n_categories) parent distribution
        mutation_rate: Base mutation probability per unit branch length
        branch_length: Length of this branch
        n_categories: Number of categories
        random_state: Random state
        
    Returns:
        (n_features, n_categories) child distribution
    """
    n_features = parent_dist.shape[0]
    child_dist = parent_dist.copy()
    
    # Probability of mutation at this branch
    p_mutate = min(1.0, mutation_rate * branch_length)
    
    for f in range(n_features):
        if random_state.random() < p_mutate:
            # Mutation: shift distribution toward a random category
            shift_cat = random_state.randint(n_categories)
            shift_amount = random_state.uniform(0.1, 0.5)
            
            # Create shifted distribution
            new_dist = child_dist[f] * (1 - shift_amount)
            new_dist[shift_cat] += shift_amount
            child_dist[f] = new_dist / new_dist.sum()  # Renormalize
    
    return child_dist


def _evolve_tree(
    root: PhyloNode,
    n_features: int,
    n_categories: int,
    mutation_rate: float,
    root_concentration: float,
    random_state: np.random.RandomState,
) -> None:
    """Evolve distributions down the tree from root to leaves.
    
    Modifies nodes in-place to add distribution attribute.
    """
    # Set root distribution
    root.distribution = _generate_root_distribution(
        n_features, n_categories, root_concentration, random_state
    )
    
    # BFS traversal to evolve down the tree
    queue = [root]
    while queue:
        node = queue.pop(0)
        
        for child in [node.left, node.right]:
            if child is not None:
                child.distribution = _mutate_distribution(
                    node.distribution,
                    mutation_rate,
                    child.branch_length if hasattr(child, 'branch_length') else 1.0,
                    n_categories,
                    random_state,
                )
                queue.append(child)


def _sample_from_leaves(
    leaves: List[PhyloNode],
    samples_per_leaf: int,
    random_state: np.random.RandomState,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], np.ndarray]:
    """Sample observed data from leaf distributions.
    
    Args:
        leaves: List of leaf nodes with distributions
        samples_per_leaf: Number of samples to draw per leaf
        random_state: Random state
        
    Returns:
        (sample_dict, cluster_assignments, distributions_array)
    """
    sample_dict: Dict[str, np.ndarray] = {}
    cluster_assignments: Dict[str, int] = {}
    distributions: List[np.ndarray] = []
    
    sample_idx = 0
    for leaf_idx, leaf in enumerate(leaves):
        dist = leaf.distribution  # (n_features, n_categories)
        n_features = dist.shape[0]
        
        for _ in range(samples_per_leaf):
            # Sample categories from distribution
            sample = np.array([
                random_state.choice(dist.shape[1], p=dist[f])
                for f in range(n_features)
            ])
            
            sample_name = f"S{sample_idx}"
            sample_dict[sample_name] = sample
            cluster_assignments[sample_name] = leaf_idx
            distributions.append(dist)
            sample_idx += 1
    
    return sample_dict, cluster_assignments, np.array(distributions)


def _get_tree_structure(root: PhyloNode) -> Dict:
    """Extract tree structure for metadata."""
    def node_to_dict(node):
        if node is None:
            return None
        return {
            'id': node.id,
            'is_leaf': node.is_leaf,
            'depth': node.depth,
            'left': node_to_dict(node.left),
            'right': node_to_dict(node.right),
        }
    return node_to_dict(root)


def generate_phylogenetic_data(
    n_taxa: int,
    n_features: int,
    n_categories: int = 4,
    samples_per_taxon: int = 10,
    mutation_rate: float = 0.3,
    root_concentration: float = 1.0,
    random_seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], np.ndarray, Dict]:
    """Generate synthetic phylogenetic data by simulating evolution along a tree.
    
    Creates data that mimics how categorical traits evolve along a phylogeny.
    Closely related taxa (leaves) will have more similar trait distributions.
    
    Args:
        n_taxa: Number of taxa (leaves in the tree). Each taxon becomes a cluster.
        n_features: Number of features/sites to simulate.
        n_categories: Number of categories per feature (e.g., 4 for DNA, 20 for amino acids).
        samples_per_taxon: Number of samples to draw from each taxon's distribution.
        mutation_rate: Probability of distribution shift per unit branch length.
            Higher = more divergence between taxa.
        root_concentration: Dirichlet concentration for root distribution.
            Higher = more uniform root, Lower = more concentrated.
        random_seed: Optional seed for reproducibility.
        
    Returns:
        Tuple of:
        - sample_dict: Dict mapping sample names to category arrays (n_features,)
        - cluster_assignments: Dict mapping sample names to taxon (cluster) IDs
        - distributions: (n_samples, n_features, n_categories) probability arrays
        - metadata: Dict with tree structure and parameters
        
    Example:
        >>> samples, labels, dists, meta = generate_phylogenetic_data(
        ...     n_taxa=8, n_features=100, n_categories=4,
        ...     samples_per_taxon=20, mutation_rate=0.5
        ... )
        >>> print(f"Generated {len(samples)} samples from {meta['n_taxa']} taxa")
    """
    random_state = np.random.RandomState(random_seed)
    
    # Generate tree
    root, leaves = _generate_random_tree(n_taxa, random_state)
    
    # Evolve distributions down the tree
    _evolve_tree(
        root, n_features, n_categories,
        mutation_rate, root_concentration, random_state
    )
    
    # Sample from leaves
    sample_dict, cluster_assignments, distributions = _sample_from_leaves(
        leaves, samples_per_taxon, random_state
    )
    
    # Build metadata
    metadata = {
        'n_taxa': n_taxa,
        'n_samples': len(sample_dict),
        'n_features': n_features,
        'n_categories': n_categories,
        'samples_per_taxon': samples_per_taxon,
        'mutation_rate': mutation_rate,
        'root_concentration': root_concentration,
        'tree_structure': _get_tree_structure(root),
        'leaf_distributions': {leaf.id: leaf.distribution for leaf in leaves},
    }
    
    return sample_dict, cluster_assignments, distributions, metadata


__all__ = ["generate_phylogenetic_data"]
