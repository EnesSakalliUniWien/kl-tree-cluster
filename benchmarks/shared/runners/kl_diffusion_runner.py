"""KL method runner using diffusion distance + average linkage tree construction.

Builds a k-NN similarity graph, computes diffusion coordinates via the
transition-matrix eigendecomposition, then applies standard average linkage
on the Euclidean diffusion distance. The resulting tree is passed through
the normal KL decomposition pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.types import MethodRunResult
from benchmarks.shared.util.decomposition import (
    _create_report_dataframe,
    _labels_from_decomposition,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _build_diffusion_distance(
    data_df: pd.DataFrame,
    k_neighbors: int = 15,
    diffusion_time: int = 3,
    n_components: int = 30,
) -> np.ndarray:
    """Compute condensed diffusion distance from binary feature matrix."""
    from scipy.linalg import eigh
    from scipy.sparse import lil_matrix
    from sklearn.neighbors import NearestNeighbors

    X = data_df.values.astype(float)
    n = len(X)
    k = min(k_neighbors, n - 1)

    # k-NN graph with Hamming distance
    nn = NearestNeighbors(n_neighbors=k, metric="hamming")
    nn.fit(X)
    knn_dist, knn_idx = nn.kneighbors(X)

    # Symmetrized similarity matrix: sim = 1 - hamming
    W = lil_matrix((n, n), dtype=float)
    for i in range(n):
        for j_pos in range(k):
            j = knn_idx[i, j_pos]
            sim = max(1.0 - knn_dist[i, j_pos], 1e-10)
            W[i, j] = max(W[i, j], sim)
            W[j, i] = max(W[j, i], sim)

    W = W.toarray()
    np.fill_diagonal(W, 0)

    # Symmetric normalization: D^{-1/2} W D^{-1/2}
    D = W.sum(axis=1)
    D[D == 0] = 1e-10
    D_sqrt_inv = 1.0 / np.sqrt(D)
    T_sym = W * D_sqrt_inv[:, None] * D_sqrt_inv[None, :]

    # Eigendecomposition
    n_comps = min(n_components, n - 1)
    eigenvalues, eigenvectors = eigh(T_sym)

    # Top eigenvalues (descending), skip trivial first
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][1 : n_comps + 1]
    eigenvectors = eigenvectors[:, idx][:, 1 : n_comps + 1]

    eigenvalues = np.maximum(eigenvalues, 0)

    # Diffusion coordinates at time t
    diffusion_coords = eigenvectors * (eigenvalues[None, :] ** diffusion_time)

    return pdist(diffusion_coords, metric="euclidean")


def _run_kl_diffusion_method(
    data_df: pd.DataFrame,
    significance_level: float,
    k_neighbors: int = 15,
    diffusion_time: int = 3,
) -> MethodRunResult:
    """Run KL decomposition on a diffusion-distance HAC tree."""

    diff_dist = _build_diffusion_distance(
        data_df,
        k_neighbors=k_neighbors,
        diffusion_time=diffusion_time,
    )

    Z_t = linkage(diff_dist, method="average")
    tree_t = PosetTree.from_linkage(Z_t, leaf_names=data_df.index.tolist())

    decomp_t = tree_t.decompose(
        leaf_data=data_df,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )

    report_t = _create_report_dataframe(decomp_t.get("cluster_assignments", {}))
    labels = np.asarray(_labels_from_decomposition(decomp_t, data_df.index.tolist()))

    return MethodRunResult(
        labels=labels,
        found_clusters=int(decomp_t.get("num_clusters", 0)),
        report_df=report_t,
        status="ok",
        skip_reason=None,
        extra={
            "tree": tree_t,
            "decomposition": decomp_t,
            "annotations": tree_t.annotations_df,
            "linkage_matrix": Z_t,
        },
    )
