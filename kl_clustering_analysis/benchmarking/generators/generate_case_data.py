"""Create test case data for benchmarking.

Exports:
- generate_case_data(test_case: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]

This is the extraction of `_generate_case_data` previously in
`kl_clustering_analysis.benchmarking.pipeline`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.benchmarking.generators.generate_sbm import generate_sbm


from typing import Dict, Any, Tuple, Optional


def _validate_binary_params(test_case: dict) -> Tuple[int, int]:
    """Validate and normalize parameters for the binary generator.

    Returns (n_rows, n_cols).
    """
    n_rows = test_case.get("n_rows", test_case.get("n_samples"))
    n_cols = test_case.get("n_cols", test_case.get("n_features"))
    if n_rows is None or n_cols is None:
        raise ValueError(
            "Binary generator requires 'n_rows'/'n_cols' or 'n_samples'/'n_features'."
        )
    return int(n_rows), int(n_cols)


def _generate_binary_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate the 'binary' style test case using the feature matrix generator."""
    n_rows, n_cols = _validate_binary_params(test_case)
    entropy = test_case.get("entropy_param", 0.5)
    balanced = test_case.get("balanced_clusters", True)
    feature_sparsity = test_case.get("feature_sparsity", None)

    data_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        entropy_param=entropy,
        n_clusters=test_case["n_clusters"],
        random_seed=seed,
        balanced_clusters=balanced,
        feature_sparsity=feature_sparsity,
    )

    original_names = list(data_dict.keys())
    matrix = np.array([data_dict[name] for name in original_names], dtype=int)
    feature_names = [f"F{j}" for j in range(matrix.shape[1])]

    data_df = pd.DataFrame(matrix, index=original_names, columns=feature_names)
    true_labels = np.array(
        [cluster_assignments[name] for name in original_names], dtype=int
    )

    metadata = {
        "n_samples": n_rows,
        "n_features": n_cols,
        "n_clusters": test_case["n_clusters"],
        "noise": entropy,
        "name": test_case.get("name", f"binary_{n_rows}x{n_cols}"),
        "generator": "binary",
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_blobs_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate the default 'blobs' style test case (Gaussian blobs -> median binarized)."""
    n_samples = int(test_case["n_samples"])
    n_features = int(test_case["n_features"])
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    X_bin = (X > np.median(X, axis=0)).astype(int)
    data_df = pd.DataFrame(
        X_bin,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case["n_clusters"],
        "noise": test_case["cluster_std"],
        "name": test_case.get("name", f"blobs_{n_samples}x{n_features}"),
        "generator": "blobs",
    }
    return data_df, y, X, metadata


def _generate_sbm_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a graph test case using the SBM generator.

    Returns a DataFrame where each row corresponds to a node and columns are node indices
    (adjacency rows). Also returns ground-truth labels and the raw adjacency matrix
    as X_original for downstream use.
    """
    sizes = test_case.get("sizes")
    if sizes is None:
        raise ValueError("SBM generator requires a 'sizes' list in the test case")

    p_intra = test_case.get("p_intra", 0.1)
    p_inter = test_case.get("p_inter", 0.01)
    directed = bool(test_case.get("directed", False))
    allow_self_loops = bool(test_case.get("allow_self_loops", False))

    G, ground_truth, A, sbm_meta = generate_sbm(
        sizes=sizes,
        p_intra=p_intra,
        p_inter=p_inter,
        seed=seed,
        directed=directed,
        allow_self_loops=allow_self_loops,
    )

    n_nodes = int(sbm_meta["n_nodes"])
    data_df = pd.DataFrame(
        A.astype(int),
        index=[f"S{j}" for j in range(n_nodes)],
        columns=[f"F{j}" for j in range(n_nodes)],
    )

    metadata = {
        "n_samples": n_nodes,
        "n_features": n_nodes,
        "n_clusters": int(sbm_meta["n_blocks"]),
        "noise": float(p_inter),
        "name": test_case.get("name", f"sbm_{n_nodes}"),
        "generator": "sbm",
        "adjacency": A,
    }

    return data_df, ground_truth, A, metadata


def generate_case_data(
    test_case: dict,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create a binary dataframe, true labels, original features, and metadata for a test case.

    This function dispatches to specialized helpers based on ``test_case['generator']``.
    """
    generator = test_case.get("generator", "blobs")
    seed = test_case.get("seed")

    if generator == "binary":
        return _generate_binary_case(test_case, seed)
    elif generator == "blobs":
        return _generate_blobs_case(test_case, seed)
    elif generator == "sbm":
        return _generate_sbm_case(test_case, seed)
    else:
        raise ValueError(f"Unknown generator: {generator}")
