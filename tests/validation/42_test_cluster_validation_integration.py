"""
Integration tests for cluster validation using complex synthetic data.

Tests the full pipeline with:
- Balanced binary feature matrices with low entropy
- Unbalanced binary feature matrices with high entropy
"""

import numpy as np
import pandas as pd
from benchmarks.shared.generators import generate_random_feature_matrix
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
)
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.tree.construction import tree_from_linkage


def _run_pipeline_on_dataframe(data_df, significance_level=0.05, **kwargs):
    """Minimal pipeline helper for integration tests."""
    Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
    tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        leaf_data=data_df,
        edge_alpha=DEFAULT_EDGE_ALPHA,
        sibling_alpha=significance_level,
        **kwargs,
    )
    decomposition = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=gate_bundle,
    ).decompose_tree()
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

    assert decomposition["num_clusters"] >= 2
    # The strict sibling calibration can stop before recovering every planted
    # cluster in this noisy fixture, so the integration threshold stays modest.
    assert ari > 0.4


def test_complex_random_feature_matrix_unbalanced_clusters():
    """Moderately higher entropy and unbalanced clusters should stay informative."""
    data_dict, true_clusters = generate_random_feature_matrix(
        n_rows=150,
        n_cols=36,
        entropy_param=0.25,
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
        # ARI threshold accounts for entropy=0.25 noise and unbalanced clusters.
        assert ari > 0.25

    assigned_clusters = {label for label in predicted if label != -1}
    assert len(assigned_clusters) >= 2, "Expected multiple clusters to be detected"
