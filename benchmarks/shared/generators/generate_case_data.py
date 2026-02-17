"""Create test case data for benchmarking.

Exports:
- generate_case_data(test_case: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]

This is the extraction of `_generate_case_data` previously in
`benchmarks.shared.pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.datasets import make_blobs

from benchmarks.shared.generators.generate_categorical_matrix import (
    generate_categorical_feature_matrix,
)
from benchmarks.shared.generators.generate_phylogenetic import generate_phylogenetic_data
from benchmarks.shared.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
from benchmarks.shared.generators.generate_sbm import generate_sbm
from benchmarks.shared.generators.generate_temporal_evolution import (
    generate_temporal_evolution_data,
)


def _one_hot_encode_categorical(
    matrix: np.ndarray,
    n_categories: int,
    sample_names: list[str],
) -> Tuple[pd.DataFrame, int]:
    """One-hot encode a (n_rows, n_cols) category-index matrix into binary.

    Each original feature with K categories becomes K binary indicator columns.
    This converts categorical data into a form compatible with the Bernoulli KL
    pipeline (all values in {0, 1}).

    Returns:
        (data_df, n_binary_features) where data_df has shape (n_rows, n_cols * K).
    """
    n_rows, n_cols = matrix.shape
    n_binary = n_cols * n_categories
    binary = np.zeros((n_rows, n_binary), dtype=int)
    for j in range(n_cols):
        for k in range(n_categories):
            binary[:, j * n_categories + k] = (matrix[:, j] == k).astype(int)
    feature_names = [f"F{j}_c{k}" for j in range(n_cols) for k in range(n_categories)]
    data_df = pd.DataFrame(binary, index=sample_names, columns=feature_names)
    return data_df, n_binary


def _validate_binary_params(test_case: dict) -> Tuple[int, int]:
    """Validate and normalize parameters for the binary generator.

    Returns (n_rows, n_cols).
    """
    n_rows = test_case.get("n_rows", test_case.get("n_samples"))
    n_cols = test_case.get("n_cols", test_case.get("n_features"))
    if n_rows is None or n_cols is None:
        raise ValueError("Binary generator requires 'n_rows'/'n_cols' or 'n_samples'/'n_features'.")
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
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

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
    blobs_result = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=test_case["n_clusters"],
        cluster_std=test_case["cluster_std"],
        random_state=seed,
    )
    X: np.ndarray = blobs_result[0]
    y: np.ndarray = blobs_result[1]
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

    sbm_expected = None
    sbm_modularity = None
    sbm_modularity_shifted = None
    sbm_modularity_norm = None

    adj = A.astype(float, copy=False)
    degrees = adj.sum(axis=1)
    m = adj.sum() / 2.0
    if m > 0:
        sbm_expected = np.outer(degrees, degrees) / (2.0 * m)
        sbm_modularity = adj - sbm_expected
        sbm_modularity_shifted = sbm_modularity - sbm_modularity.min()
        sbm_modularity_norm = sbm_modularity_shifted / (sbm_modularity_shifted.max() + 1e-10)
        precomputed_distance_matrix = 1.0 - sbm_modularity_norm
    else:
        # Fallback for empty graphs (no modularity signal): invert adjacency directly.
        precomputed_distance_matrix = 1.0 - adj
    np.fill_diagonal(precomputed_distance_matrix, 0.0)

    precomputed_distance_condensed = None
    try:
        precomputed_distance_condensed = squareform(precomputed_distance_matrix)
    except ValueError:
        precomputed_distance_condensed = None

    metadata = {
        "n_samples": n_nodes,
        "n_features": n_nodes,
        "n_clusters": int(sbm_meta["n_blocks"]),
        "noise": float(p_inter),
        "name": test_case.get("name", f"sbm_{n_nodes}"),
        "generator": "sbm",
        "adjacency": A,
        "requires_precomputed_kl_distance": True,
        "precomputed_distance_matrix": precomputed_distance_matrix,
        "precomputed_distance_condensed": precomputed_distance_condensed,
        "sbm_expected": sbm_expected,
        "sbm_modularity": sbm_modularity,
        "sbm_modularity_shifted": sbm_modularity_shifted,
        "sbm_modularity_norm": sbm_modularity_norm,
    }

    return data_df, ground_truth, A, metadata


def _validate_categorical_params(test_case: dict) -> Tuple[int, int, int]:
    """Validate and normalize parameters for the categorical generator.

    Returns (n_rows, n_cols, n_categories).
    """
    n_rows = test_case.get("n_rows", test_case.get("n_samples"))
    n_cols = test_case.get("n_cols", test_case.get("n_features"))
    n_categories = test_case.get("n_categories", 3)
    if n_rows is None or n_cols is None:
        raise ValueError(
            "Categorical generator requires 'n_rows'/'n_cols' or 'n_samples'/'n_features'."
        )
    return int(n_rows), int(n_cols), int(n_categories)


def _generate_categorical_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a categorical (multinomial) test case.

    Returns a DataFrame where each cell contains a category index (0 to K-1),
    and the distributions array contains the underlying probability distributions.
    """
    n_rows, n_cols, n_categories = _validate_categorical_params(test_case)
    entropy = test_case.get("entropy_param", 0.5)
    balanced = test_case.get("balanced_clusters", True)
    category_sparsity = test_case.get("category_sparsity", None)

    sample_dict, cluster_assignments, distributions = generate_categorical_feature_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        n_categories=n_categories,
        entropy_param=entropy,
        n_clusters=test_case["n_clusters"],
        random_seed=seed,
        balanced_clusters=balanced,
        category_sparsity=category_sparsity,
    )

    original_names = list(sample_dict.keys())
    # Matrix of sampled categories (n_rows, n_cols)
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)

    # One-hot encode: category indices → binary indicators for Bernoulli KL pipeline
    data_df, n_binary = _one_hot_encode_categorical(matrix, n_categories, original_names)
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = {
        "n_samples": n_rows,
        "n_features": n_binary,
        "n_features_original": n_cols,
        "n_categories": n_categories,
        "n_clusters": test_case["n_clusters"],
        "noise": entropy,
        "name": test_case.get("name", f"categorical_{n_rows}x{n_cols}x{n_categories}"),
        "generator": "categorical",
        "distributions": distributions,  # (n_rows, n_cols, n_categories)
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_phylogenetic_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a phylogenetic simulation test case.

    Simulates trait evolution along a random phylogenetic tree.
    Each taxon (leaf) becomes a cluster, and samples are drawn from
    the evolved distributions at each leaf.
    """
    n_taxa = test_case.get("n_taxa", test_case.get("n_clusters", 4))
    n_features = test_case.get("n_features", test_case.get("n_cols", 50))
    n_categories = test_case.get("n_categories", 4)
    samples_per_taxon = test_case.get("samples_per_taxon", 10)
    mutation_rate = test_case.get("mutation_rate", 0.3)
    root_concentration = test_case.get("root_concentration", 1.0)

    sample_dict, cluster_assignments, distributions, phylo_meta = generate_phylogenetic_data(
        n_taxa=n_taxa,
        n_features=n_features,
        n_categories=n_categories,
        samples_per_taxon=samples_per_taxon,
        mutation_rate=mutation_rate,
        root_concentration=root_concentration,
        random_seed=seed,
    )

    original_names = list(sample_dict.keys())
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)

    # One-hot encode: category indices → binary indicators for Bernoulli KL pipeline
    data_df, n_binary = _one_hot_encode_categorical(matrix, n_categories, original_names)
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = {
        "n_samples": len(original_names),
        "n_features": n_binary,
        "n_features_original": n_features,
        "n_categories": n_categories,
        "n_clusters": n_taxa,
        "n_taxa": n_taxa,
        "samples_per_taxon": samples_per_taxon,
        "mutation_rate": mutation_rate,
        "noise": mutation_rate,  # For compatibility
        "name": test_case.get("name", f"phylo_{n_taxa}taxa_{n_features}feat"),
        "generator": "phylogenetic",
        "distributions": distributions,
        "tree_structure": phylo_meta.get("tree_structure"),
        "leaf_distributions": phylo_meta.get("leaf_distributions"),
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_temporal_evolution_case(
    test_case: dict, seed: Optional[int]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a temporal evolution test case.

    Simulates sequence evolution along a growing branch over time.
    Each time point becomes a cluster, with increasing divergence from ancestor.
    """
    n_time_points = test_case.get("n_time_points", test_case.get("n_clusters", 8))
    n_features = test_case.get("n_features", test_case.get("n_cols", 200))
    n_categories = test_case.get("n_categories", 4)
    samples_per_time = test_case.get("samples_per_time", 20)
    mutation_rate = test_case.get("mutation_rate", 0.3)
    shift_strength = test_case.get("shift_strength", (0.15, 0.5))
    root_concentration = test_case.get("root_concentration", 1.0)

    sample_dict, cluster_assignments, distributions, evo_meta = generate_temporal_evolution_data(
        n_time_points=n_time_points,
        n_features=n_features,
        n_categories=n_categories,
        samples_per_time=samples_per_time,
        mutation_rate=mutation_rate,
        shift_strength=shift_strength,
        root_concentration=root_concentration,
        random_seed=seed,
    )

    original_names = list(sample_dict.keys())
    matrix = np.array([sample_dict[name] for name in original_names], dtype=int)

    # One-hot encode: category indices → binary indicators for Bernoulli KL pipeline
    data_df, n_binary = _one_hot_encode_categorical(matrix, n_categories, original_names)
    true_labels = np.array([cluster_assignments[name] for name in original_names], dtype=int)

    metadata = {
        "n_samples": len(original_names),
        "n_features": n_binary,
        "n_features_original": n_features,
        "n_categories": n_categories,
        "n_clusters": n_time_points,
        "n_time_points": n_time_points,
        "samples_per_time": samples_per_time,
        "mutation_rate": mutation_rate,
        "shift_strength": shift_strength,
        "noise": mutation_rate,  # For compatibility
        "name": test_case.get("name", f"temporal_{n_time_points}tp_{n_features}feat"),
        "generator": "temporal_evolution",
        "distributions": distributions,
        "divergence_from_ancestor": evo_meta.get("divergence_from_ancestor"),
        "divergence_matrix": evo_meta.get("divergence_matrix"),
    }

    return data_df, true_labels, matrix.astype(float), metadata


def _generate_preloaded_case(
    test_case: dict,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a pre-existing data file (TSV/CSV) as a benchmark case.

    Required test_case keys:
        - file_path: path to the data file (absolute or relative to repo root)
        - sep: separator character (default: '\\t')
    Optional:
        - n_clusters: expected number of clusters (for display only; no ground truth)
    """
    file_path = test_case["file_path"]
    sep = test_case.get("sep", "\t")

    # Resolve relative paths against the repo root
    path = Path(file_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        path = repo_root / path

    if not path.exists():
        raise FileNotFoundError(f"Preloaded data file not found: {path}")

    data_df = pd.read_csv(path, sep=sep, index_col=0)
    n_samples, n_features = data_df.shape

    # No ground truth labels for real data
    y = np.full(n_samples, np.nan)
    x_original = data_df.values.copy()

    metadata = {
        "name": test_case.get("name", path.stem),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_clusters": test_case.get("n_clusters"),
        "noise": np.nan,
        "generator": "preloaded",
        "source_file": str(path),
        "sparsity": float(1 - data_df.values.mean()),
    }
    return data_df, y, x_original, metadata


def generate_case_data(
    test_case: dict,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create a binary dataframe, true labels, original features, and metadata for a test case.

    This function dispatches to specialized helpers based on ``test_case['generator']``.
    """
    generator = test_case.get("generator", "blobs")
    seed = test_case.get("seed")

    if generator == "binary":
        data_df, y, x_original, metadata = _generate_binary_case(test_case, seed)
    elif generator == "blobs":
        data_df, y, x_original, metadata = _generate_blobs_case(test_case, seed)
    elif generator == "sbm":
        data_df, y, x_original, metadata = _generate_sbm_case(test_case, seed)
    elif generator == "categorical":
        data_df, y, x_original, metadata = _generate_categorical_case(test_case, seed)
    elif generator == "phylogenetic":
        data_df, y, x_original, metadata = _generate_phylogenetic_case(test_case, seed)
    elif generator == "temporal_evolution":
        data_df, y, x_original, metadata = _generate_temporal_evolution_case(test_case, seed)
    elif generator == "preloaded":
        data_df, y, x_original, metadata = _generate_preloaded_case(test_case)
    else:
        raise ValueError(f"Unknown generator: {generator}")

    # Preserve caller-supplied case metadata for downstream audit/reporting.
    if "category" in test_case:
        metadata.setdefault("category", test_case["category"])
    metadata.setdefault("case_name", metadata.get("name"))

    return data_df, y, x_original, metadata
