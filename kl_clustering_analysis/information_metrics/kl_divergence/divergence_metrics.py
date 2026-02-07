from typing import Dict, Any, TYPE_CHECKING, Callable
import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
from scipy.special import rel_entr

from ... import config
from ...tree.distributions import populate_distributions

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
    Element-wise KL divergence: D_KL(Q||P).

    Adapts to input shape:
    - 1D input (n_features,): Treated as Bernoulli (binary).
      Returns KL(Q||P) + KL(1-Q||1-P).
    - 2D input (n_features, n_classes): Treated as Categorical.
      Returns sum_k Q_k * log(Q_k/P_k) along axis 1.

    Parameters
    ----------
    query_distribution
        Query distribution.
    reference_distribution
        Reference distribution.

    Returns
    -------
    np.ndarray
        Vector of KL divergences per feature.
    """
    return _kl_categorical_general(
        query_distribution, 
        reference_distribution, 
        eps=config.EPSILON
    )


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
    """
    node_records = []
    for node_id in tree.nodes():
        node_attrs = tree.nodes[node_id]
        is_leaf = node_attrs.get("is_leaf", False)

        node_records.append(
            {
                "node_id": node_id,
                "distribution": node_attrs.get("distribution", None),
                "leaf_count": node_attrs.get("leaf_count", 0),
                "is_leaf": is_leaf,
                "kl_divergence_global": node_attrs.get("kl_divergence_global", np.nan),
                "kl_divergence_per_column_global": node_attrs.get(
                    "kl_divergence_per_column_global", None
                ),
                "kl_divergence_local": node_attrs.get("kl_divergence_local", np.nan),
                "kl_divergence_local_composite": node_attrs.get(
                    "kl_divergence_local_composite", np.nan
                ),
                "composite_score_weight": node_attrs.get(
                    "composite_score_weight", np.nan
                ),
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
    lambda_factor: float = 0.2,
) -> pd.DataFrame:
    """
    Populate tree nodes with distributions and KL divergences, return summary DataFrame.

    Populates each node with:
    - distribution: weighted mean of leaf/child distributions
    - leaf_count: number of descendant leaves
    - kl_divergence_global: KL(node||root)
    - kl_divergence_local: KL(child||parent)
    - kl_divergence_local_composite: KL + lambda * (mean_KL/mean_BL) * branch_length
    - per-column versions of both KL metrics

    Parameters
    ----------
    tree
        A directed tree (e.g., PosetTree) with structure and optional branch lengths.
    leaf_data
        DataFrame where index matches leaf labels and columns are features.
    distribution_type
        Type of distribution for KL computation ('categorical' or 'poisson').
    lambda_factor
        Weight factor for branch length integration. Default 0.2 means branch length
        accounts for ~20% of the composite score on average.

    Note: Root's global KL is set to NaN (self-comparison is meaningless).
    """
    root_node_id = tree.graph.get("root") or next(
        n for n, d in tree.in_degree() if d == 0
    )
    populate_distributions(
        tree,
        leaf_data,
    )
    _populate_global_kl(tree, root_node_id, distribution_type=distribution_type)
    _populate_local_kl(tree, distribution_type=distribution_type)

    # 3. Compute Composite Score (KL + dynamic_weight * BranchLength)
    # Collect stats for dynamic weighting
    kl_values = []
    bl_values = []

    nodes_iter = [n for n in tree.nodes if n != root_node_id]  # skip root for local KL

    for node in nodes_iter:
        kl = tree.nodes[node].get("kl_divergence_local")
        # Get branch length from parent edge
        parents = list(tree.predecessors(node))
        if parents:
            parent = parents[0]
            bl = tree.edges[parent, node].get("branch_length", 0.0)

            if kl is not None and not np.isnan(kl):
                kl_values.append(kl)
                bl_values.append(bl if bl > 0 else 0.0)

    if kl_values and bl_values:
        mean_kl = np.mean(kl_values)
        mean_bl = np.mean(bl_values)

        # Avoid division by zero
        if mean_bl < 1e-6:
            dynamic_weight = 0.0
        else:
            # Scale branch length to have same magnitude as KL, then apply lambda factor
            # dynamic_weight = lambda_factor * (mean_kl / mean_bl)
            # This ensures (dynamic_weight * mean_bl) = lambda_factor * mean_kl
            dynamic_weight = lambda_factor * (mean_kl / mean_bl)

        # Apply score to all nodes
        for node in nodes_iter:
            kl = tree.nodes[node].get("kl_divergence_local", 0.0)
            if kl is None or np.isnan(kl):
                kl = 0.0

            parents = list(tree.predecessors(node))
            bl = 0.0
            if parents:
                bl = tree.edges[parents[0], node].get("branch_length", 0.0)

            composite_score = kl + dynamic_weight * bl
            tree.nodes[node]["kl_divergence_local_composite"] = composite_score
            tree.nodes[node]["composite_score_weight"] = (
                dynamic_weight  # detailed tracing
            )

    return _extract_hierarchy_statistics(tree)
