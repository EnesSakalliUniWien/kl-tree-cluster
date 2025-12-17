"""Regression test ensuring pbmc3k subsample yields meaningful KL splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.information_metrics import (
    compute_node_divergences,
)
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_independence_cmi,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def _prepare_probabilities(
    n_cells: int = 200, n_genes: int = 300, seed: int = 0, epsilon: float = 1e-3
) -> tuple[pd.DataFrame, sc.AnnData]:
    """Load pbmc3k, subsample, run PCA, and return smoothed binary probabilities."""

    adata = sc.datasets.pbmc3k()
    if n_cells < adata.n_obs:
        rng = np.random.default_rng(seed)
        subset = rng.choice(adata.n_obs, size=n_cells, replace=False)
        adata = adata[subset].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(n_genes, adata.n_vars), subset=True
    )
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=min(25, adata.n_vars))

    df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    binary = (df > 0).astype(np.float32)
    prob = binary * (1 - 2 * epsilon) + epsilon
    return prob, adata


def test_pbmc3k_pipeline_retains_divergent_feature():
    prob_df, adata = _prepare_probabilities()

    # Build hierarchy
    condensed = pdist(prob_df.values, metric="hamming")
    Z = linkage(condensed, method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=prob_df.index.tolist())

    stats_df = compute_node_divergences(tree, prob_df)
    assert not stats_df.empty
    assert float(stats_df["kl_divergence_global"].max()) > 0.0

    n_features = prob_df.shape[1]
    results_df = annotate_child_parent_divergence(
        tree,
        stats_df,
        total_number_of_features=n_features,
        significance_level_alpha=config.SIGNIFICANCE_ALPHA,
    )
    results_df = annotate_sibling_independence_cmi(
        tree,
        results_df,
        significance_level_alpha=config.SIGNIFICANCE_ALPHA,
        n_permutations=20,
        parallel=False,
    )

    clusters = tree.decompose(
        results_df=results_df,
        significance_column="Are_Features_Dependent",
        alpha_local=config.ALPHA_LOCAL,
    )
    assert clusters["num_clusters"] >= 1

    cluster_map = {
        leaf: cid
        for cid, info in clusters["cluster_assignments"].items()
        for leaf in info["leaves"]
    }
    adata.obs["kl_cluster"] = adata.obs_names.map(cluster_map).astype("Int64")
    assert adata.obs["kl_cluster"].isna().sum() == 0

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(25, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata)
    assert "X_umap" in adata.obsm and adata.obsm["X_umap"].shape[1] == 2

    top_feature = stats_df["kl_divergence_global"].idxmax()
    assert top_feature in stats_df.index
    print("Top divergent feature:", top_feature)
