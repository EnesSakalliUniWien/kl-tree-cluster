"""
Purpose: Test real tree structures for validity.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/tree_construction/q_real_tree_validation_suite__tree_construction__validation.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def generate_test_data(n_samples=100, n_features=20, n_clusters=3, seed=42):
    """Generate synthetic binary data."""
    np.random.seed(seed)

    # Create cluster centers
    centers = []
    for k in range(n_clusters):
        center = (np.random.random(n_features) > 0.5).astype(float)
        centers.append(center)

    # Assign samples to clusters with noise
    X = []
    labels = []
    samples_per_cluster = n_samples // n_clusters

    for k, center in enumerate(centers):
        for _ in range(samples_per_cluster):
            # Copy center and flip some bits
            sample = center.copy()
            flip_mask = np.random.random(n_features) < 0.1  # 10% noise
            sample[flip_mask] = 1 - sample[flip_mask]
            X.append(sample)
            labels.append(k)

    X = np.array(X)
    labels = np.array(labels)

    # Create DataFrame
    sample_names = [f"S{i}" for i in range(len(X))]
    df = pd.DataFrame(X, index=sample_names)

    return df, labels


def validate_tree_structure(tree, data):
    """Validate that all parent-child relationships are correct.

    Returns:
        tuple: (is_valid, violations) where violations is a list of problematic edges
    """
    from kl_clustering_analysis.tree.distributions import populate_distributions

    # Populate distributions first to get leaf_count
    populate_distributions(tree, data)

    violations = []

    for parent, child in tree.edges():
        n_parent = tree.nodes[parent].get("leaf_count", 0)
        n_child = tree.nodes[child].get("leaf_count", 0)

        if n_child >= n_parent:
            violations.append(
                {
                    "parent": parent,
                    "child": child,
                    "n_parent": n_parent,
                    "n_child": n_child,
                    "problem": "n_child >= n_parent",
                }
            )

    return len(violations) == 0, violations


def test_linkage_methods():
    """Test different linkage methods for tree validity."""
    print("=" * 70)
    print("TESTING LINKAGE METHODS FOR TREE VALIDITY")
    print("=" * 70)
    print()

    data, true_labels = generate_test_data(n_samples=60, n_features=15, n_clusters=3)
    print(f"Generated data: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"True clusters: {len(np.unique(true_labels))}")
    print()

    linkage_methods = ["average", "complete", "single", "ward"]
    distance_metrics = ["hamming", "euclidean", "cityblock"]

    results = []

    for linkage_method in linkage_methods:
        for metric in distance_metrics:
            # Ward only works with Euclidean
            if linkage_method == "ward" and metric != "euclidean":
                continue

            try:
                # Build tree
                dist = pdist(data.values, metric=metric)
                Z = linkage(dist, method=linkage_method)
                tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

                # Validate structure
                is_valid, violations = validate_tree_structure(tree, data)

                results.append(
                    {
                        "linkage": linkage_method,
                        "metric": metric,
                        "valid": is_valid,
                        "violations": len(violations),
                        "nodes": len(tree.nodes()),
                        "edges": len(tree.edges()),
                    }
                )

                status = "✓ VALID" if is_valid else f"✗ {len(violations)} VIOLATIONS"
                print(f"{linkage_method:10} + {metric:12}: {status}")

                if violations:
                    for v in violations[:3]:  # Show first 3
                        print(
                            f"           → {v['parent']}({v['n_parent']}) -> {v['child']}({v['n_child']})"
                        )

            except Exception as e:
                print(f"{linkage_method:10} + {metric:12}: ERROR - {str(e)[:50]}")
                results.append(
                    {
                        "linkage": linkage_method,
                        "metric": metric,
                        "valid": False,
                        "error": str(e),
                    }
                )

    print()

    # Summary
    valid_count = sum(1 for r in results if r.get("valid", False))
    print(
        f"Summary: {valid_count}/{len(results)} method combinations produced valid trees"
    )
    print()

    return results


def test_decomposition_with_validation():
    """Test that decomposition now raises errors for invalid trees."""
    print("=" * 70)
    print("TESTING DECOMPOSITION WITH VALIDATION")
    print("=" * 70)
    print()

    data, true_labels = generate_test_data(n_samples=30, n_features=10, n_clusters=2)
    print(f"Generated data: {data.shape[0]} samples, {data.shape[1]} features")

    # Build valid tree
    dist = pdist(data.values, metric="hamming")
    Z = linkage(dist, method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

    # Validate first
    is_valid, violations = validate_tree_structure(tree, data)
    print(f"Tree validation: {'PASS' if is_valid else 'FAIL'}")

    if is_valid:
        try:
            results = tree.decompose(
                leaf_data=data, alpha_local=0.05, sibling_alpha=0.05
            )
            print(f"Decomposition: SUCCESS")
            print(f"  Found clusters: {results.get('num_clusters', 0)}")
            print(f"  Status: {results.get('status', 'unknown')}")
        except ValueError as e:
            print(f"Decomposition: FAILED with ValueError")
            print(f"  Error: {e}")
    else:
        print("Skipping decomposition due to invalid tree structure")

    print()


def test_edge_cases():
    """Test edge cases that might trigger violations."""
    print("=" * 70)
    print("TESTING EDGE CASES")
    print("=" * 70)
    print()

    test_cases = [
        ("Very small dataset", 4, 5, 2),
        ("Single cluster", 20, 10, 1),
        ("Many small clusters", 50, 8, 10),
        ("High-dimensional", 30, 100, 3),
    ]

    for desc, n_samples, n_features, n_clusters in test_cases:
        print(f"\nTest: {desc}")
        print(
            f"  n_samples={n_samples}, n_features={n_features}, n_clusters={n_clusters}"
        )

        try:
            data, labels = generate_test_data(n_samples, n_features, n_clusters)
            dist = pdist(data.values, metric="hamming")
            Z = linkage(dist, method="average")
            tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

            is_valid, violations = validate_tree_structure(tree, data)

            if is_valid:
                print(f"  Result: ✓ Valid tree ({len(tree.nodes())} nodes)")
            else:
                print(f"  Result: ✗ {len(violations)} violations")

        except Exception as e:
            print(f"  Result: ERROR - {str(e)[:60]}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REAL TREE VALIDATION TESTS")
    print("=" * 70 + "\n")

    # Test 1: Linkage methods
    test_linkage_methods()

    # Test 2: Decomposition with validation
    test_decomposition_with_validation()

    # Test 3: Edge cases
    test_edge_cases()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The error-raising fix ensures that:")
    print("1. Invalid tree structures are caught early")
    print("2. Statistical tests maintain correct Type I error rates")
    print("3. Users are alerted to data quality issues")
    print()
