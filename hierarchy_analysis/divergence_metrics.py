from typing import Dict, Any, TYPE_CHECKING, Union, Tuple, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx

if TYPE_CHECKING:
    import networkx as nx


def calculate_kl_divergence_vector(
    q_dist: npt.NDArray[np.float64], p_dist: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Element-wise Bernoulli KL: D_KL(Q||P) = q log(q/p) + (1-q) log((1-q)/(1-p)).
    Assumes features are probabilities in [0,1]. Uses clipping for numerical stability.
    """
    q = np.asarray(q_dist, dtype=np.float64).reshape(-1)
    p = np.asarray(p_dist, dtype=np.float64).reshape(-1)
    if q.shape != p.shape:
        raise ValueError(f"KL input shapes must match; got {q.shape} vs {p.shape}")
    # Guard: values should be probabilities
    if np.any((q < 0) | (q > 1)) or np.any((p < 0) | (p > 1)):
        raise ValueError("KL inputs must be probabilities in [0,1] for Bernoulli KL.")
    epsilon = 1e-9
    q = np.clip(q, epsilon, 1 - epsilon)
    p = np.clip(p, epsilon, 1 - epsilon)
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))


def _find_hierarchy_root(tree: "nx.DiGraph") -> str:
    """
    Return the unique root (in-degree 0). Raise if none or multiple.
    """
    roots = [node for node in tree.nodes() if tree.in_degree(node) == 0]
    if len(roots) == 0:
        raise ValueError("Tree has no root node (no node with in-degree 0).")
    if len(roots) > 1:
        raise ValueError(f"Tree has multiple roots: {roots}")
    return roots[0]


def _validate_leaf_vector(
    name: Any, vec: Any, expected_len: Optional[int]
) -> Tuple[np.ndarray, int]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(
            f"Leaf '{name}' distribution must be 1-D; got shape {arr.shape}."
        )
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(
            f"All leaf vectors must share the same length; expected {expected_len}, got {arr.size} for leaf '{name}'."
        )
    return arr, arr.size


def _calculate_leaf_distribution(
    tree: "nx.DiGraph",
    node_id: str,
    leaf_data: Dict[Any, npt.NDArray[np.float64]],
    expected_len: Optional[int],
) -> int:
    """
    Set distribution and leaf count for a leaf node. Returns feature dimension.
    """
    node_data = tree.nodes[node_id]
    label = node_data.get("label", node_id)
    if label not in leaf_data:
        raise KeyError(f"Leaf label '{label}' not found in provided leaf_data keys.")
    arr, dim = _validate_leaf_vector(label, leaf_data[label], expected_len)
    tree.nodes[node_id]["distribution"] = arr
    tree.nodes[node_id]["leaf_count"] = 1
    return dim


def _calculate_hierarchy_node_distribution(tree: "nx.DiGraph", node_id: str) -> None:
    """
    Weighted mean of children distributions using children's leaf counts.
    """
    children = list(tree.successors(node_id))
    if len(children) == 0:
        raise ValueError(f"Internal node '{node_id}' has no children.")
    sum_vector = None
    leaf_count = 0
    for child_id in children:
        if (
            "distribution" not in tree.nodes[child_id]
            or "leaf_count" not in tree.nodes[child_id]
        ):
            raise ValueError(
                f"Child '{child_id}' of '{node_id}' missing distribution/leaf_count."
            )
        child_leaves = int(tree.nodes[child_id]["leaf_count"])
        child_dist = np.asarray(tree.nodes[child_id]["distribution"], dtype=np.float64)
        if sum_vector is None:
            sum_vector = np.zeros_like(child_dist, dtype=np.float64)
        if child_dist.shape != sum_vector.shape:
            raise ValueError(
                f"Dimension mismatch among children of '{node_id}': "
                f"expected {sum_vector.shape}, got {child_dist.shape} on child '{child_id}'."
            )
        leaf_count += child_leaves
        sum_vector += child_dist * child_leaves
    if leaf_count <= 0:
        raise ValueError(f"Internal node '{node_id}' accumulated zero leaves.")
    tree.nodes[node_id]["leaf_count"] = leaf_count
    tree.nodes[node_id]["distribution"] = sum_vector / leaf_count


def _calculate_hierarchy_distributions_iterative(
    tree: "nx.DiGraph",
    root: str,
    leaf_data: Union[Dict[Any, npt.NDArray[np.float64]], pd.DataFrame],
) -> None:
    """
    Populate 'distribution' and 'leaf_count' bottom-up (postorder).
    """
    # Normalize leaf_data: keep keys as-is (do not coerce to str)
    if isinstance(leaf_data, pd.DataFrame):
        # Each row is a leaf; index values must match tree node 'label'
        leaf_data_dict = {
            idx: row.values.astype(np.float64) for idx, row in leaf_data.iterrows()
        }
    else:
        leaf_data_dict = leaf_data

    # Infer/validate dimension as we go (first leaf defines expected length)
    expected_len: Optional[int] = None

    nodes_postorder = list(nx.dfs_postorder_nodes(tree, source=root))
    for node_id in nodes_postorder:
        node_data = tree.nodes[node_id]
        if node_data.get("is_leaf", False):
            dim = _calculate_leaf_distribution(
                tree, node_id, leaf_data_dict, expected_len
            )
            if expected_len is None:
                expected_len = dim
        else:
            _calculate_hierarchy_node_distribution(tree, node_id)


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


def calculate_hierarchy_kl_divergence(
    tree: "nx.DiGraph",
    leaf_data: Union[Dict[Any, npt.NDArray[np.float64]], pd.DataFrame],
    set_root_global_kl_to_nan: bool = True,
) -> pd.DataFrame:
    """
    Compute per-node mean distributions, GLOBAL KL (node||root), and LOCAL KL (child||parent).

    Assumes each feature is a Bernoulli probability in [0,1].
    """
    # 1) Find the unique root and populate distributions bottom-up
    root = _find_hierarchy_root(tree)
    _calculate_hierarchy_distributions_iterative(tree, root, leaf_data)

    # 2) GLOBAL KL: internal nodes vs root
    global_ref = np.asarray(tree.nodes[root]["distribution"], dtype=np.float64)
    for node_id in tree.nodes():
        if tree.nodes[node_id].get("is_leaf", False):
            continue  # usually not needed; skip to keep semantics tidy
        node_dist = np.asarray(tree.nodes[node_id]["distribution"], dtype=np.float64)
        if node_id == root:
            # Enforce the convention used downstream
            if set_root_global_kl_to_nan:
                tree.nodes[node_id]["kl_divergence_per_column_global"] = None
                tree.nodes[node_id]["kl_divergence_global"] = np.nan
            else:
                per_col = calculate_kl_divergence_vector(node_dist, global_ref)
                tree.nodes[node_id]["kl_divergence_per_column_global"] = per_col
                tree.nodes[node_id]["kl_divergence_global"] = float(np.sum(per_col))
            continue
        per_col = calculate_kl_divergence_vector(node_dist, global_ref)
        tree.nodes[node_id]["kl_divergence_per_column_global"] = per_col
        tree.nodes[node_id]["kl_divergence_global"] = float(np.sum(per_col))

    # 3) LOCAL KL: each edge child||parent
    for parent_id, child_id in tree.edges():
        parent_dist = np.asarray(
            tree.nodes[parent_id]["distribution"], dtype=np.float64
        )
        child_dist = np.asarray(tree.nodes[child_id]["distribution"], dtype=np.float64)
        per_col = calculate_kl_divergence_vector(child_dist, parent_dist)
        tree.nodes[child_id]["kl_divergence_per_column_local"] = per_col
        tree.nodes[child_id]["kl_divergence_local"] = float(np.sum(per_col))

    # 4) Collect into a DataFrame
    return _extract_hierarchy_statistics(tree)
