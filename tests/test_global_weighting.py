import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.global_weighting import (
    compute_global_weight,
    compute_neutral_point,
    estimate_global_weight_strength,
)


def test_global_weight_estimation_bonus_vs_penalty() -> None:
    """Data-driven estimator should give bonuses to strong fractions and penalties to weak ones."""

    child_local_kl = np.array([0.2, 0.05, 0.01, 0.005])
    child_global_kl = np.array([0.25, 0.2, 0.2, 0.25])

    neutral_point = compute_neutral_point(child_local_kl, child_global_kl)
    beta = estimate_global_weight_strength(child_local_kl, child_global_kl)

    strong_fraction_weight = compute_global_weight(
        child_local_kl[0],
        child_global_kl[0],
        beta=beta,
        method="relative",
        neutral_point=neutral_point,
    )
    weak_fraction_weight = compute_global_weight(
        child_local_kl[-1],
        child_global_kl[-1],
        beta=beta,
        method="relative",
        neutral_point=neutral_point,
    )

    assert strong_fraction_weight < 1.0  # bonus for strong local signal
    assert weak_fraction_weight > 1.0  # penalty for weak local signal


def test_child_parent_global_weight_column_is_neutral() -> None:
    """Edge significance currently does not apply global weighting; weights stay at 1.0."""

    tree = nx.DiGraph()
    tree.add_node("root", distribution=np.array([0.5, 0.5]))
    tree.add_node("L", distribution=np.array([0.8, 0.2]))
    tree.add_node("R", distribution=np.array([0.2, 0.8]))
    tree.add_edge("root", "L")
    tree.add_edge("root", "R")

    stats = pd.DataFrame.from_dict(
        {
            "root": {"leaf_count": 2},
            "L": {"leaf_count": 1},
            "R": {"leaf_count": 1},
        },
        orient="index",
    )

    annotated = annotate_child_parent_divergence(
        tree,
        stats,
        significance_level_alpha=0.05,
        fdr_method="flat",
    )

    weights = annotated.loc[["L", "R"], "Child_Parent_Divergence_Global_Weight"].to_numpy()
    assert np.allclose(weights, 1.0)
