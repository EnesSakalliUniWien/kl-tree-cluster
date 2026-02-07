"""Compare sparse vs orthonormal projection methods on clustering accuracy.

This script runs the same clustering pipeline with both projection methods
to quantify the impact of exact χ² distribution on clustering results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure project root is in path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import kl_clustering_analysis.config as cfg
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    _PROJECTION_CACHE,
    _PROJECTOR_CACHE,
    generate_projection_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def load_penguins():
    """Load Palmer Penguins dataset."""
    url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv"
    df = pd.read_csv(url).dropna()
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    X = df[feature_cols].values
    species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    y = df["species"].map(species_map).values
    return X, y, "Penguins"


def load_digits_data():
    """Load sklearn digits dataset."""
    digits = load_digits()
    return digits.data, digits.target, "Digits"


def binarize(X, threshold=0.1):
    """Binarize data using threshold on normalized values."""
    X_norm = X / X.max() if X.max() > 0 else X
    return (X_norm > threshold).astype(int)


def run_clustering(X_bin, method="orthonormal"):
    """Run KL clustering with specified projection method."""
    # Set projection method
    original_method = cfg.PROJECTION_METHOD
    cfg.PROJECTION_METHOD = method

    # Clear caches to ensure fresh projections
    _PROJECTION_CACHE.clear()
    _PROJECTOR_CACHE.clear()

    try:
        sample_names = [f"S{i}" for i in range(X_bin.shape[0])]
        data = pd.DataFrame(X_bin, index=sample_names)

        # Build tree
        Z = linkage(pdist(data.values, metric="hamming"), method="weighted")
        tree = PosetTree.from_linkage(Z, leaf_names=sample_names)

        # Decompose
        results = tree.decompose(leaf_data=data, alpha_local=0.05, sibling_alpha=0.05)

        # Extract labels
        cluster_assignments = results.get("cluster_assignments", {})
        label_map = {}
        for cluster_id, info in cluster_assignments.items():
            for leaf in info["leaves"]:
                label_map[leaf] = cluster_id

        labels = np.array([label_map.get(name, -1) for name in sample_names])
        n_clusters = results.get("num_clusters", 0)

        return labels, n_clusters
    finally:
        # Restore original method
        cfg.PROJECTION_METHOD = original_method


def evaluate(y_true, y_pred):
    """Compute clustering metrics."""
    mask = y_pred >= 0
    if mask.sum() == 0:
        return 0.0, 0.0, 0

    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    n_clusters = len(np.unique(y_pred[mask]))
    return ari, nmi, n_clusters


def test_orthonormality():
    """Verify orthonormal projection properties."""
    print("\n" + "=" * 70)
    print("VERIFYING PROJECTION MATRIX PROPERTIES")
    print("=" * 70)

    n_features, k = 100, 20

    # Test orthonormal
    R_orth = generate_projection_matrix(
        n_features, k, random_state=42, method="orthonormal"
    )
    RRT_orth = R_orth @ R_orth.T
    orth_error = np.max(np.abs(RRT_orth - np.eye(k)))

    print(f"\nOrthonormal projection (k={k}, d={n_features}):")
    print(f"  R @ R.T = I?  Max error: {orth_error:.2e}")
    print(f"  Is orthonormal: {orth_error < 1e-10}")

    # Test sparse
    _PROJECTION_CACHE.clear()
    _PROJECTOR_CACHE.clear()
    R_sparse = generate_projection_matrix(
        n_features, k, random_state=42, method="sparse"
    )
    R_sparse_dense = (
        R_sparse.toarray() if hasattr(R_sparse, "toarray") else np.asarray(R_sparse)
    )
    RRT_sparse = R_sparse_dense @ R_sparse_dense.T
    sparse_error = np.max(np.abs(RRT_sparse - np.eye(k)))

    print(f"\nSparse projection (k={k}, d={n_features}):")
    print(f"  R @ R.T = I?  Max error: {sparse_error:.2e}")
    print(f"  Is orthonormal: {sparse_error < 1e-10}")
    nnz = np.count_nonzero(R_sparse_dense)
    print(f"  Sparsity: {100 * (1 - nnz / R_sparse_dense.size):.1f}% zeros")


def compare_methods():
    """Compare sparse vs orthonormal on real datasets."""
    print("\n" + "=" * 70)
    print("COMPARING PROJECTION METHODS ON CLUSTERING")
    print("=" * 70)

    results = []

    # Test datasets
    datasets = [
        ("Penguins", *load_penguins()[:2]),
        ("Digits", *load_digits_data()[:2]),
    ]

    for name, X, y in datasets:
        print(
            f"\n--- {name} (n={len(y)}, d={X.shape[1]}, k_true={len(np.unique(y))}) ---"
        )

        # Binarize
        X_bin = binarize(X, threshold=0.1 if name == "Digits" else 0.0)

        for method in ["orthonormal", "sparse"]:
            labels, n_clusters = run_clustering(X_bin, method=method)
            ari, nmi, n_found = evaluate(y, labels)

            print(f"  {method:12s}: ARI={ari:.4f}, NMI={nmi:.4f}, clusters={n_found}")

            results.append(
                {
                    "dataset": name,
                    "method": method,
                    "ari": ari,
                    "nmi": nmi,
                    "n_clusters": n_found,
                    "n_true": len(np.unique(y)),
                }
            )

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        orth = subset[subset["method"] == "orthonormal"].iloc[0]
        sparse = subset[subset["method"] == "sparse"].iloc[0]

        ari_diff = orth["ari"] - sparse["ari"]
        nmi_diff = orth["nmi"] - sparse["nmi"]

        print(f"\n{dataset}:")
        print(f"  Orthonormal: ARI={orth['ari']:.4f}, NMI={orth['nmi']:.4f}")
        print(f"  Sparse:      ARI={sparse['ari']:.4f}, NMI={sparse['nmi']:.4f}")
        print(f"  Difference:  ΔARI={ari_diff:+.4f}, ΔNMI={nmi_diff:+.4f}")

        if abs(ari_diff) < 0.01:
            print(f"  → Methods produce SIMILAR results")
        elif ari_diff > 0:
            print(f"  → Orthonormal is BETTER by {ari_diff:.4f} ARI")
        else:
            print(f"  → Sparse is BETTER by {-ari_diff:.4f} ARI")

    return df


def main():
    print("\n" + "=" * 70)
    print("PROJECTION METHOD COMPARISON")
    print("Sparse (JL approximation) vs Orthonormal (exact χ²)")
    print("=" * 70)
    print(f"\nCurrent config: PROJECTION_METHOD = {cfg.PROJECTION_METHOD}")

    # Test matrix properties
    test_orthonormality()

    # Compare on clustering
    df = compare_methods()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The orthonormal projection guarantees R @ R.T = I (identity),
which means the test statistic T = ||R @ z||² follows χ²(k) exactly.

Sparse projection is faster but R @ R.T ≠ I, leading to approximate
χ² distribution and potentially different p-values.

For most cases, the difference is small, but for borderline significant
splits, the exact p-values from orthonormal projection may matter.
""")


if __name__ == "__main__":
    main()
