"""
Integration tests for cluster validation using complex synthetic data.

Tests the full pipeline with:
- Balanced binary feature matrices with low entropy
- Unbalanced binary feature matrices with high entropy
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from kl_clustering_analysis.benchmarking import _labels_from_decomposition


def _run_pipeline_on_dataframe(data_df, significance_level=0.05, **kwargs):
    """Minimal pipeline helper for integration tests."""
    Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    decomposition = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=significance_level,
        **kwargs,
    )
    return decomposition, tree


def test_complex_random_feature_matrix_balanced_clusters():
    """Synthetic binary data with low entropy should recover most clusters."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=72,
        n_cols=40,
        entropy_param=0.1,
        n_clusters=4,
        random_seed=314,
        balanced_clusters=True,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

    decomposition, _ = _run_pipeline_on_dataframe(data_df, significance_level=0.05)
    predicted = _labels_from_decomposition(decomposition, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]

    assigned_mask = np.array(predicted) != -1
    assigned_fraction = float(np.mean(assigned_mask))
    assert assigned_fraction > 0.85, "Too many samples left unassigned"

    ari = adjusted_rand_score(
        np.array(true_labels)[assigned_mask], np.array(predicted)[assigned_mask]
    )

    assert decomposition["num_clusters"] >= 3
    # TreeBH is more conservative than flat BH, expect slightly lower ARI
    assert ari > 0.6


def test_complex_random_feature_matrix_unbalanced_clusters():
    """Higher entropy and unbalanced clusters should still yield informative groupings."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=96,
        n_cols=36,
        entropy_param=0.45,
        n_clusters=4,
        random_seed=2024,
        balanced_clusters=False,
    )
    data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

    decomposition, _ = _run_pipeline_on_dataframe(data_df, significance_level=0.05)
    predicted = _labels_from_decomposition(decomposition, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]

    assigned_mask = np.array(predicted) != -1
    assigned_fraction = float(np.mean(assigned_mask))
    assert assigned_fraction > 0.6, "Decomposition discarded too many samples"

    if assigned_mask.any():
        ari = adjusted_rand_score(
            np.array(true_labels)[assigned_mask], np.array(predicted)[assigned_mask]
        )
        assert ari > 0.45

    assigned_clusters = {label for label in predicted if label != -1}
    assert len(assigned_clusters) >= 2, "Expected multiple clusters to be detected"
