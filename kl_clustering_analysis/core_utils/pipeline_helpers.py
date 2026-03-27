"""Helpers for building small end-to-end clustering test cases."""

import pandas as pd

from .. import config
from ..hierarchy_analysis.decomposition.gates.orchestrator import run_gate_annotation_pipeline
from ..tree.poset_tree import PosetTree


def create_test_case_data(n_samples=50, n_features=20, n_clusters=3, noise_level=1.0, seed=42):
    """Create synthetic test data using the same method as the validation suite."""
    import numpy as np
    from sklearn.datasets import make_blobs

    print(
        f"Generating test data: {n_samples} samples, {n_features} features, {n_clusters} clusters"
    )

    # Use make_blobs like the validation suite, then binarize
    X_continuous, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=noise_level,
        random_state=seed,
    )

    # Binarize by median threshold (same as validation suite)
    X_binary = (X_continuous > np.median(X_continuous, axis=0)).astype(int)

    # Convert to DataFrame
    X = pd.DataFrame(
        X_binary,
        index=[f"S{j}" for j in range(n_samples)],
        columns=[f"F{j}" for j in range(n_features)],
    )
    y_true = pd.Series(y_true)

    print(f"Generated data shape: {X.shape}")
    print(f"True cluster sizes: {y_true.value_counts().sort_index()}")

    return X, y_true


def build_hierarchical_tree(X, linkage_method="complete", distance_metric="hamming"):
    """Build hierarchical tree from feature matrix using scipy linkage."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    print("Building hierarchical tree...")

    # Compute pairwise distances
    distance_matrix = pdist(X.values, metric=distance_metric)

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=linkage_method)

    # Create PosetTree from linkage matrix
    tree = PosetTree.from_linkage(linkage_matrix, X.index.tolist())

    print(f"Tree built with {len(tree.nodes())} nodes")

    return tree, linkage_matrix


def run_statistical_analysis(tree, X):
    """Run the production gate-annotation pipeline on a populated tree."""
    print("Running statistical analysis...")

    tree.populate_node_divergences(X)
    annotations_df = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        alpha_local=config.EDGE_ALPHA,
        sibling_alpha=config.SIBLING_ALPHA,
        leaf_data=X,
        sibling_method=config.SIBLING_TEST_METHOD,
        sibling_whitening=config.SIBLING_WHITENING,
    ).annotated_df

    print("Statistical analysis complete.")

    return annotations_df
