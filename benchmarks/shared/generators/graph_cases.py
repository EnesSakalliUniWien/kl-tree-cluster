"""Benchmark case-data generation for graph-derived matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from benchmarks.shared.generators.case_data_contracts import (
    CaseDataResult,
    case_metadata,
)
from benchmarks.shared.generators.generate_sbm import generate_sbm


def generate_sbm_case(test_case: dict, seed: int | None) -> CaseDataResult:
    """Generate an SBM adjacency matrix with an explicit modularity distance."""
    sizes = test_case["sizes"]
    p_intra = test_case["p_intra"]
    p_inter = test_case["p_inter"]
    directed = bool(test_case["directed"])
    allow_self_loops = bool(test_case["allow_self_loops"])

    _graph, ground_truth, adjacency, sbm_meta = generate_sbm(
        sizes=sizes,
        p_intra=p_intra,
        p_inter=p_inter,
        seed=seed,
        directed=directed,
        allow_self_loops=allow_self_loops,
    )

    n_nodes = int(sbm_meta["n_nodes"])
    data_df = pd.DataFrame(
        adjacency.astype(int),
        index=[f"S{j}" for j in range(n_nodes)],
        columns=[f"F{j}" for j in range(n_nodes)],
    )

    sbm_expected = None
    sbm_modularity = None
    sbm_modularity_shifted = None
    sbm_modularity_norm = None

    adj = adjacency.astype(float, copy=False)
    degrees = adj.sum(axis=1)
    m = adj.sum() / 2.0
    if m > 0:
        sbm_expected = np.outer(degrees, degrees) / (2.0 * m)
        sbm_modularity = adj - sbm_expected
        sbm_modularity_shifted = sbm_modularity - sbm_modularity.min()
        sbm_modularity_norm = sbm_modularity_shifted / (sbm_modularity_shifted.max() + 1e-10)
        precomputed_distance_matrix = 1.0 - sbm_modularity_norm
        distance_metric = "sbm_shifted_modularity"
    else:
        precomputed_distance_matrix = 1.0 - adj
        distance_metric = "sbm_adjacency_complement"
    np.fill_diagonal(precomputed_distance_matrix, 0.0)

    precomputed_distance_condensed = squareform(precomputed_distance_matrix)

    metadata = case_metadata(
        test_case=test_case,
        n_samples=n_nodes,
        n_features=n_nodes,
        n_clusters=int(sbm_meta["n_blocks"]),
        noise=float(p_inter),
        generator="sbm",
        source_family="stochastic_block_model",
        feature_representation="graph_adjacency",
        requires_precomputed_tbs_distance=True,
        precomputed_distance_matrix=precomputed_distance_matrix,
        precomputed_distance_condensed=precomputed_distance_condensed,
        distance_metric=distance_metric,
        extra={
            "adjacency": adjacency,
            "sbm_expected": sbm_expected,
            "sbm_modularity": sbm_modularity,
            "sbm_modularity_shifted": sbm_modularity_shifted,
            "sbm_modularity_norm": sbm_modularity_norm,
        },
    )

    return data_df, ground_truth, adjacency, metadata
