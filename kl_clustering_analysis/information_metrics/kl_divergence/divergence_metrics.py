from typing import Dict, Any, TYPE_CHECKING, Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
from scipy.special import rel_entr
from scipy.cluster.hierarchy import inconsistent as scipy_inconsistent

from ... import config

if TYPE_CHECKING:
    import networkx as nx


def _kl_categorical_general(
    q_params: npt.NDArray[np.float64],
    p_params: npt.NDArray[np.float64],
    eps: float,
) -> npt.NDArray[np.float64]:
    """
    KL divergence for Categorical distributions (including Bernoulli).

    Handles:
    1. Bernoulli (1D input): Expands to [p, 1-p] and computes sum.
    2. Categorical (2D input): Computes sum over last dimension (classes).
    """
    # Ensure inputs are at least 1D
    q_in = np.asarray(q_params)
    p_in = np.asarray(p_params)

    # Case 1: Bernoulli (scalar per feature) -> Expand to 2 states [p, 1-p]
    # Check if last dimension is NOT the class dimension (i.e., it's just feature probabilities)
    if q_in.ndim == 1 or (q_in.ndim == 2 and q_in.shape[1] == 1):
        q = np.clip(q_in.reshape(-1), eps, 1.0 - eps)
        p = np.clip(p_in.reshape(-1), eps, 1.0 - eps)
        # KL = p*log(p/q) + (1-p)*log((1-p)/(1-q))
        # This is mathematically equivalent to expanding to [p, 1-p] and summing
        return rel_entr(q, p) + rel_entr(1.0 - q, 1.0 - p)

    # Case 2: Categorical (N_features x K_classes)
    # Clip probabilities to avoid log(0)
    q_full = np.clip(q_in, eps, 1.0)
    p_full = np.clip(p_in, eps, 1.0)

    # Sum KL across the class dimension (last axis)
    return np.sum(rel_entr(q_full, p_full), axis=-1)


def _kl_poisson(
    q_params: npt.NDArray[np.float64],
    p_params: npt.NDArray[np.float64],
    eps: float,
) -> npt.NDArray[np.float64]:
    """KL divergence for Poisson distributions."""
    lam1 = np.maximum(q_params, eps)
    lam2 = np.maximum(p_params, eps)
    # KL(Poi(lam1) || Poi(lam2)) = lam1 * log(lam1/lam2) + lam2 - lam1
    return rel_entr(lam1, lam2) + lam2 - lam1


# Registry of available KL divergence implementations
_KL_REGISTRY: Dict[str, Callable] = {
    "categorical": _kl_categorical_general,
    "poisson": _kl_poisson,
}


def calculate_kl_divergence_vector(
    query_distribution: npt.NDArray[np.float64],
    reference_distribution: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Element-wise KL divergence: D_KL(Q||P) = sum_i q_i * log(q_i / p_i).

    Uses scipy.special.rel_entr for numerically stable computation.
    Handles general categorical distributions (not limited to binary/Bernoulli).
    Automatically handles edge cases like 0*log(0) = 0.

    Parameters
    ----------
    query_distribution
        Query distribution (probabilities, can be binary or categorical)
    reference_distribution
        Reference distribution (probabilities, can be binary or categorical)

    Returns
    -------
    np.ndarray
        Element-wise relative entropy values

    Notes
    -----
    For Bernoulli (binary) features in [0,1], this computes:
        q*log(q/p) + (1-q)*log((1-q)/(1-p))
    For general categorical distributions, computes:
        sum_i q_i * log(q_i / p_i)
    """
    query_probs = np.asarray(query_distribution, dtype=np.float64).reshape(-1)
    reference_probs = np.asarray(reference_distribution, dtype=np.float64).reshape(-1)
    return rel_entr(query_probs, reference_probs)


def calculate_kl_divergence_per_feature(
    query_params: npt.NDArray[np.float64],
    reference_params: npt.NDArray[np.float64],
    distribution_type: str = "categorical",
    *,
    eps: float = config.EPSILON,
) -> npt.NDArray[np.float64]:
    """
    Per-feature KL divergence for specified distribution.

    Computes element-wise KL divergence based on the distribution type.

    Parameters
    ----------
    query_params
        Distribution parameters for query (e.g. probabilities for Bernoulli)
    reference_params
        Distribution parameters for reference
    distribution_type
        Type of distribution. Supported: 'categorical', 'poisson'.
    eps
        Numerical stability epsilon
    """
    q = np.asarray(query_params, dtype=np.float64).reshape(-1)
    p = np.asarray(reference_params, dtype=np.float64).reshape(-1)

    if distribution_type in _KL_REGISTRY:
        return _KL_REGISTRY[distribution_type](q, p, eps=eps)

    raise NotImplementedError(
        f"Distribution '{distribution_type}' not supported. "
        f"Available: {list(_KL_REGISTRY.keys())}"
    )


def _calculate_leaf_distribution(
    tree: "nx.DiGraph",
    node_id: str,
    leaf_data: Dict[Any, npt.NDArray[np.float64]],
) -> None:
    """
    Set distribution and leaf count for a leaf node.
    """
    label = tree.nodes[node_id].get("label", node_id)
    feature_probabilities = np.asarray(leaf_data[label], dtype=np.float64).reshape(-1)
    tree.nodes[node_id]["distribution"] = feature_probabilities
    tree.nodes[node_id]["leaf_count"] = 1


def _calculate_hierarchy_node_distribution(tree: "nx.DiGraph", node_id: str) -> None:
    """
    Weighted mean of children distributions using children's leaf counts.
    """
    children = list(tree.successors(node_id))
    weighted_distribution_sum = 0.0
    total_descendant_leaves = 0
    for child_id in children:
        child_leaves = int(tree.nodes[child_id]["leaf_count"])
        child_distribution = np.asarray(
            tree.nodes[child_id]["distribution"], dtype=np.float64
        )
        weighted_distribution_sum += child_distribution * child_leaves
        total_descendant_leaves += child_leaves
    tree.nodes[node_id]["leaf_count"] = total_descendant_leaves
    tree.nodes[node_id]["distribution"] = (
        weighted_distribution_sum / total_descendant_leaves
    )


def _populate_distributions(
    tree: "nx.DiGraph",
    root: str,
    leaf_data: pd.DataFrame,
) -> None:
    """
    Populate 'distribution' and 'leaf_count' for all nodes bottom-up.

    Traverses in postorder so children are processed before parents.
    """
    # Convert DataFrame to dict for efficient lookups
    leaf_feature_data = {
        idx: row.values.astype(np.float64) for idx, row in leaf_data.iterrows()
    }

    # Process nodes bottom-up (leaves first, then parents)
    for node_id in nx.dfs_postorder_nodes(tree, source=root):
        is_leaf = tree.nodes[node_id].get("is_leaf", False)

        if is_leaf:
            # Leaf node: use data directly from leaf_data
            _calculate_leaf_distribution(tree, node_id, leaf_feature_data)
        else:
            # Internal node: weighted average of children distributions
            _calculate_hierarchy_node_distribution(tree, node_id)


def _populate_local_kl(
    tree: "nx.DiGraph", distribution_type: str = "categorical"
) -> None:
    """
    Compute LOCAL KL divergence for each edge: KL(child||parent).
    """
    for parent_id, child_id in tree.edges():
        parent_distribution = np.asarray(
            tree.nodes[parent_id]["distribution"], dtype=np.float64
        )
        child_distribution = np.asarray(
            tree.nodes[child_id]["distribution"], dtype=np.float64
        )
        kl_per_feature = calculate_kl_divergence_per_feature(
            child_distribution, parent_distribution, distribution_type=distribution_type
        )
        tree.nodes[child_id]["kl_divergence_per_column_local"] = kl_per_feature
        tree.nodes[child_id]["kl_divergence_local"] = float(np.sum(kl_per_feature))


def _populate_global_kl(
    tree: "nx.DiGraph", root: str, distribution_type: str = "categorical"
) -> None:
    """
    Compute GLOBAL KL divergence for all nodes: KL(node||root).

    Root's global KL is set to NaN (self-comparison is meaningless).
    """
    # Get root distribution as global reference
    root_distribution = np.asarray(tree.nodes[root]["distribution"], dtype=np.float64)

    # Calculate KL divergence for each node relative to root
    for node_id in tree.nodes():
        node_distribution = np.asarray(
            tree.nodes[node_id]["distribution"], dtype=np.float64
        )

        # Compute per-feature KL divergence: KL(node||root) for Bernoulli probabilities.
        kl_divergence_per_feature = calculate_kl_divergence_per_feature(
            node_distribution, root_distribution, distribution_type=distribution_type
        )

        # Store both per-feature and total divergence
        tree.nodes[node_id]["kl_divergence_per_column_global"] = (
            kl_divergence_per_feature
        )
        tree.nodes[node_id]["kl_divergence_global"] = float(
            np.sum(kl_divergence_per_feature)
        )

    # Root self-comparison is undefined: set to NaN
    tree.nodes[root]["kl_divergence_per_column_global"] = None
    tree.nodes[root]["kl_divergence_global"] = np.nan


def _extract_hierarchy_statistics(tree: "nx.DiGraph") -> pd.DataFrame:
    """
    Collect distributions and KL metrics into a DataFrame indexed by node_id.
    
    Also includes branch length information for diagnostic purposes:
    - height: merge distance from linkage (internal nodes only)
    - branch_length: distance from this node to its parent
    - sibling_branch_sum: sum of branch lengths to both children (internal nodes)
    """
    # Pre-compute parent heights for branch length calculation
    parent_map = {}
    for node_id in tree.nodes():
        for child_id in tree.successors(node_id):
            parent_map[child_id] = node_id
    
    node_records = []
    for node_id in tree.nodes():
        node_attrs = tree.nodes[node_id]
        
        # Get height (merge distance, only meaningful for internal nodes)
        height = node_attrs.get("height", 0.0)
        is_leaf = node_attrs.get("is_leaf", False)
        
        # Compute branch length to parent
        branch_length = np.nan
        if node_id in parent_map:
            parent_id = parent_map[node_id]
            parent_height = tree.nodes[parent_id].get("height", 0.0)
            node_height = height if not is_leaf else 0.0
            branch_length = parent_height - node_height
        
        # Compute sibling branch sum (for internal nodes: sum of branches to children)
        sibling_branch_sum = np.nan
        children = list(tree.successors(node_id))
        if len(children) == 2:
            left_id, right_id = children
            left_height = tree.nodes[left_id].get("height", 0.0) if not tree.nodes[left_id].get("is_leaf", False) else 0.0
            right_height = tree.nodes[right_id].get("height", 0.0) if not tree.nodes[right_id].get("is_leaf", False) else 0.0
            sibling_branch_sum = (height - left_height) + (height - right_height)
        
        node_records.append(
            {
                "node_id": node_id,
                "distribution": node_attrs.get("distribution", None),
                "leaf_count": node_attrs.get("leaf_count", 0),
                "is_leaf": is_leaf,
                "height": height,
                "branch_length": branch_length,
                "sibling_branch_sum": sibling_branch_sum,
                "kl_divergence_global": node_attrs.get("kl_divergence_global", np.nan),
                "kl_divergence_per_column_global": node_attrs.get(
                    "kl_divergence_per_column_global", None
                ),
                "kl_divergence_local": node_attrs.get("kl_divergence_local", np.nan),
                "kl_divergence_per_column_local": node_attrs.get(
                    "kl_divergence_per_column_local", None
                ),
            }
        )
    return pd.DataFrame.from_records(node_records).set_index("node_id", drop=True)


def compute_node_divergences(
    tree: "nx.DiGraph",
    leaf_data: pd.DataFrame,
    distribution_type: str = "categorical",
) -> pd.DataFrame:
    """
    Populate tree nodes with distributions and KL divergences, return summary DataFrame.

    Populates each node with:
    - distribution: weighted mean of leaf/child distributions
    - leaf_count: number of descendant leaves
    - kl_divergence_global: KL(node||root)
    - kl_divergence_local: KL(child||parent)
    - per-column versions of both KL metrics

    Assumes each feature is a Bernoulli probability in [0,1].

    Note: Root's global KL is set to NaN (self-comparison is meaningless).
    """
    root_node_id = tree.graph.get("root") or next(
        n for n, d in tree.in_degree() if d == 0
    )
    _populate_distributions(tree, root_node_id, leaf_data)
    _populate_global_kl(tree, root_node_id, distribution_type=distribution_type)
    _populate_local_kl(tree, distribution_type=distribution_type)
    return _extract_hierarchy_statistics(tree)
