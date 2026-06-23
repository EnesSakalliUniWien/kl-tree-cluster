"""Pipeline fixture contracts for gate annotation."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.tree.poset_tree import PosetTree


def _create_test_case_data(
    n_samples: int = 50,
    n_features: int = 20,
    n_clusters: int = 3,
    noise_level: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic binary test data for end-to-end pipeline tests."""
    from sklearn.datasets import make_blobs

    x_continuous, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=noise_level,
        random_state=seed,
    )
    x_binary = (x_continuous > np.median(x_continuous, axis=0)).astype(int)
    x = pd.DataFrame(
        x_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    return x, pd.Series(y_true)


def _build_hierarchical_tree(
    x: pd.DataFrame,
    linkage_method: str = "complete",
    distance_metric: str = "hamming",
) -> tuple[PosetTree, np.ndarray]:
    """Build a PosetTree from a binary feature matrix."""
    distance_matrix = pdist(x.values, metric=distance_metric)
    linkage_matrix = linkage(distance_matrix, method=linkage_method)
    tree = PosetTree.from_linkage(linkage_matrix, x.index.tolist())
    return tree, linkage_matrix


def _run_statistical_analysis(
    tree: PosetTree,
    x: pd.DataFrame,
    *,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
) -> pd.DataFrame:
    """Run the production gate-annotation pipeline on a populated tree."""
    tree.populate_node_divergences(x)
    return run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        leaf_data=x,
    ).annotated_df


def test_pipeline_exposes_active_sibling_gate_calibration() -> None:
    """The pipeline must annotate active sibling gate p-value provenance."""
    x, _y_true = _create_test_case_data(
        n_samples=90,
        n_features=60,
        n_clusters=3,
        noise_level=1.0,
        seed=42,
    )
    tree, _ = _build_hierarchical_tree(x)
    out = _run_statistical_analysis(tree, x, edge_alpha=0.01)

    active = out[out["Sibling_Gate_P_Value_Role"].eq("active_traversal_sibling_gate")]
    assert not active.empty
    assert active["Sibling_Gate_P_Value_Calibration"].eq(
        "empirical_null_inflation"
    ).all()
    assert active["Sibling_Divergence_P_Value"].between(0.0, 1.0).all()
