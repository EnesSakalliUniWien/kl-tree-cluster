from typing import Dict, Any, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
from scipy.special import rel_entr

if TYPE_CHECKING:
    import networkx as nx


def calculate_kl_divergence_vector(
    q_dist: npt.NDArray[np.float64], p_dist: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Element-wise KL divergence: D_KL(Q||P) = sum_i q_i * log(q_i / p_i).

    Uses scipy.special.rel_entr for numerically stable computation.
    Handles general categorical distributions (not limited to binary/Bernoulli).
    Automatically handles edge cases like 0*log(0) = 0.

    Parameters
    ----------
    q_dist
        Query distribution (probabilities, can be binary or categorical)
    p_dist
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
    q = np.asarray(q_dist, dtype=np.float64).reshape(-1)
    p = np.asarray(p_dist, dtype=np.float64).reshape(-1)
    return rel_entr(q, p)


def _calculate_leaf_distribution(
    tree: "nx.DiGraph",
    node_id: str,
    leaf_data: Dict[Any, npt.NDArray[np.float64]],
) -> None:
    """
    Set distribution and leaf count for a leaf node.
    """
    label = tree.nodes[node_id].get("label", node_id)
    arr = np.asarray(leaf_data[label], dtype=np.float64).reshape(-1)
    tree.nodes[node_id]["distribution"] = arr
    tree.nodes[node_id]["leaf_count"] = 1


def _calculate_hierarchy_node_distribution(tree: "nx.DiGraph", node_id: str) -> None:
    """
    Weighted mean of children distributions using children's leaf counts.
    """
    children = list(tree.successors(node_id))
    weighted_sum = 0.0
    leaf_count = 0
    for child_id in children:
        child_leaves = int(tree.nodes[child_id]["leaf_count"])
        child_dist = np.asarray(tree.nodes[child_id]["distribution"], dtype=np.float64)
        weighted_sum += child_dist * child_leaves
        leaf_count += child_leaves
    tree.nodes[node_id]["leaf_count"] = leaf_count
    tree.nodes[node_id]["distribution"] = weighted_sum / leaf_count


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
    leaf_data_dict = {
        idx: row.values.astype(np.float64) for idx, row in leaf_data.iterrows()
    }

    # Process nodes bottom-up (leaves first, then parents)
    for node_id in nx.dfs_postorder_nodes(tree, source=root):
        is_leaf = tree.nodes[node_id].get("is_leaf", False)

        if is_leaf:
            # Leaf node: use data directly from leaf_data
            _calculate_leaf_distribution(tree, node_id, leaf_data_dict)
        else:
            # Internal node: weighted average of children distributions
            _calculate_hierarchy_node_distribution(tree, node_id)


def _populate_local_kl(tree: "nx.DiGraph") -> None:
    """
    Compute LOCAL KL divergence for each edge: KL(child||parent).
    """
    for parent_id, child_id in tree.edges():
        parent_dist = np.asarray(
            tree.nodes[parent_id]["distribution"], dtype=np.float64
        )
        child_dist = np.asarray(tree.nodes[child_id]["distribution"], dtype=np.float64)
        per_col = calculate_kl_divergence_vector(child_dist, parent_dist)
        tree.nodes[child_id]["kl_divergence_per_column_local"] = per_col
        tree.nodes[child_id]["kl_divergence_local"] = float(np.sum(per_col))


def _populate_global_kl(tree: "nx.DiGraph", root: str) -> None:
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

        # Compute per-feature KL divergence: KL(node||root)
        kl_per_feature = calculate_kl_divergence_vector(
            node_distribution, root_distribution
        )

        # Store both per-feature and total divergence
        tree.nodes[node_id]["kl_divergence_per_column_global"] = kl_per_feature
        tree.nodes[node_id]["kl_divergence_global"] = float(np.sum(kl_per_feature))

    # Root self-comparison is undefined: set to NaN
    tree.nodes[root]["kl_divergence_per_column_global"] = None
    tree.nodes[root]["kl_divergence_global"] = np.nan


def _extract_hierarchy_statistics(tree: "nx.DiGraph") -> pd.DataFrame:
    """
    Collect distributions and KL metrics into a DataFrame indexed by node_id.
    """
    recs = []
    for node_id in tree.nodes():
        nd = tree.nodes[node_id]
        recs.append(
            {
                "node_id": node_id,
                "distribution": nd.get("distribution", None),
                "leaf_count": nd.get("leaf_count", 0),
                "is_leaf": nd.get("is_leaf", False),
                "kl_divergence_global": nd.get("kl_divergence_global", np.nan),
                "kl_divergence_per_column_global": nd.get(
                    "kl_divergence_per_column_global", None
                ),
                "kl_divergence_local": nd.get("kl_divergence_local", np.nan),
                "kl_divergence_per_column_local": nd.get(
                    "kl_divergence_per_column_local", None
                ),
            }
        )
    return pd.DataFrame.from_records(recs).set_index("node_id", drop=True)


def compute_node_divergences(
    tree: "nx.DiGraph",
    leaf_data: pd.DataFrame,
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
    root = tree.graph.get("root") or next(n for n, d in tree.in_degree() if d == 0)
    _populate_distributions(tree, root, leaf_data)
    _populate_global_kl(tree, root)
    _populate_local_kl(tree)
    return _extract_hierarchy_statistics(tree)
