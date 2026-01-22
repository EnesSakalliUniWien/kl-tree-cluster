"""
MNIST Benchmark for KL Divergence Clustering.

Downloads a subset of MNIST and runs the KL clustering algorithm on it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def load_mnist_subset(
    n_samples: int = 1000, seed: int = 42, use_pca: bool = False, n_components: int = 50
) -> tuple:
    """Load a subset of MNIST for benchmarking.

    Parameters
    ----------
    n_samples : int
        Number of samples to use (default 1000 for speed)
    seed : int
        Random seed for reproducibility
    use_pca : bool
        Whether to apply PCA for dimensionality reduction
    n_components : int
        Number of PCA components if use_pca=True

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Pixel values normalized to [0, 1]
    y : ndarray of shape (n_samples,)
        Digit labels (0-9)
    """
    print(f"Loading MNIST dataset (subset of {n_samples} samples)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    # Take a stratified subset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    # Normalize to [0, 1]
    X_subset = X_subset / 255.0

    # Optional PCA
    if use_pca:
        print(f"  Applying PCA to reduce to {n_components} components...")
        pca = PCA(n_components=n_components, random_state=seed)
        X_subset = pca.fit_transform(X_subset)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    print(f"  Shape: {X_subset.shape}")
    print(f"  Unique labels: {np.unique(y_subset)}")
    print(f"  Label distribution: {np.bincount(y_subset)}")

    return X_subset, y_subset


def run_kl_clustering(
    X: np.ndarray,
    verbose: bool = True,
    binarize_threshold: float = 0.1,
    distance_metric: str = "rogerstanimoto",
    linkage_method: str = "average",
) -> np.ndarray:
    """Run KL divergence clustering on data.

    Parameters
    ----------
    X : ndarray
        Input data matrix
    verbose : bool
        Whether to print progress
    binarize_threshold : float
        Threshold for binarization (pixels > threshold become 1)
    distance_metric : str
        Distance metric for hierarchical clustering
    linkage_method : str
        Linkage method for hierarchical clustering

    Returns
    -------
    labels : ndarray
        Cluster assignments
    sample_names : list
        Sample names in order
    """
    if verbose:
        print("\nRunning KL Divergence clustering...")
        print(f"  Data shape: {X.shape}")

    # Binarize using threshold (not median!)
    # For MNIST: pixels > 0.1 means "ink present"
    X_binary = (X > binarize_threshold).astype(int)

    if verbose:
        # Show sparsity
        sparsity = 1 - X_binary.mean()
        print(
            f"  Binarized with threshold={binarize_threshold}, sparsity={sparsity:.1%}"
        )

    # Create DataFrame with sample names
    sample_names = [f"Sample_{j}" for j in range(X.shape[0])]
    data = pd.DataFrame(X_binary, index=sample_names)

    if verbose:
        print(f"  Binarized data shape: {data.shape}")

    # Build hierarchical tree with specified method
    Z = linkage(
        pdist(data.values, metric=distance_metric),
        method=linkage_method,
    )

    if verbose:
        print(f"  Built hierarchy with {distance_metric} + {linkage_method}")

    # Create PosetTree
    tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

    # Decompose
    significance_level = 0.05
    results = tree.decompose(
        leaf_data=data,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )

    # Extract cluster labels
    cluster_assignments = results.get("cluster_assignments", {})
    n_clusters = results.get("num_clusters", 0)

    if verbose:
        print(f"  Found {n_clusters} clusters")

    # Convert to label array
    label_map = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id

    labels = np.array([label_map.get(name, -1) for name in sample_names])

    return labels


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute clustering evaluation metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Predicted cluster labels

    Returns
    -------
    metrics : dict
        Dictionary with ARI, NMI scores
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    n_true = len(np.unique(y_true))
    n_pred = len(np.unique(y_pred))

    return {
        "ARI": ari,
        "NMI": nmi,
        "n_true_clusters": n_true,
        "n_pred_clusters": n_pred,
    }


def main():
    """Run MNIST benchmark."""
    print("=" * 60)
    print("MNIST Benchmark for KL Divergence Clustering")
    print("=" * 60)

    # Try ALL linkage methods with best distance metrics
    linkage_methods = [
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    ]
    binary_distances = [
        "jaccard",
        "dice",
        "rogerstanimoto",
        "hamming",
        "sokalsneath",
        "russellrao",
        "yule",
        "kulsinski",
    ]

    # Test key combinations
    configs = []
    for linkage in ["single", "complete", "average", "weighted", "ward"]:
        for distance in ["jaccard", "dice", "rogerstanimoto", "hamming"]:
            configs.append({"distance": distance, "linkage": linkage})

    results = []
    for cfg in configs:
        print("\n" + "=" * 60)
        print(f"Config: {cfg['distance']} + {cfg['linkage']}")
        print("=" * 60)

        # Load data
        X, y_true = load_mnist_subset(n_samples=2000, seed=42, use_pca=False)

        # Run clustering with specified tree method
        try:
            y_pred = run_kl_clustering(
                X,
                verbose=True,
                binarize_threshold=0.0,  # Any ink = 1
                distance_metric=cfg["distance"],
                linkage_method=cfg["linkage"],
            )

            # Evaluate
            metrics = evaluate_clustering(y_true, y_pred)
            print(f"\n  True clusters (digits):     {metrics['n_true_clusters']}")
            print(f"  Predicted clusters:         {metrics['n_pred_clusters']}")
            print(f"  Adjusted Rand Index (ARI):  {metrics['ARI']:.4f}")
            print(f"  Normalized Mutual Info:     {metrics['NMI']:.4f}")

            results.append(
                {
                    "distance": cfg["distance"],
                    "linkage": cfg["linkage"],
                    "n_clusters": metrics["n_pred_clusters"],
                    "ARI": metrics["ARI"],
                    "NMI": metrics["NMI"],
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "distance": cfg["distance"],
                    "linkage": cfg["linkage"],
                    "n_clusters": 0,
                    "ARI": 0,
                    "NMI": 0,
                    "error": str(e),
                }
            )

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Distance':<18} {'Linkage':<10} {'Clusters':<10} {'ARI':<10} {'NMI':<10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.get("ARI", 0), reverse=True):
        print(
            f"{r['distance']:<18} {r['linkage']:<10} {r['n_clusters']:<10} {r.get('ARI', 0):.4f}     {r.get('NMI', 0):.4f}"
        )

    return results


if __name__ == "__main__":
    main()
