"""
Debug script to test the error-raising behavior for invalid tree structures.

This script verifies that:
1. Valid tree structures (n_child < n_parent) work correctly
2. Invalid tree structures (n_child >= n_parent) raise ValueError
3. Edge cases (n_child == n_parent, n_child > n_parent) are handled properly
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_standardized_z,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    sibling_divergence_test,
)


def test_valid_tree_structure():
    """Test that valid tree structures work correctly."""
    print("=" * 60)
    print("TEST 1: Valid tree structures (n_child < n_parent)")
    print("=" * 60)

    child_dist = np.array([0.6, 0.4])
    parent_dist = np.array([0.5, 0.5])

    test_cases = [
        (10, 100, "Small child, large parent"),
        (50, 100, "Half-size child"),
        (99, 100, "Nearly equal sizes"),
        (1, 2, "Minimum valid case"),
    ]

    for n_child, n_parent, description in test_cases:
        try:
            result = _compute_standardized_z(
                child_dist, parent_dist, n_child=n_child, n_parent=n_parent
            )
            print(f"✓ {description}: n_child={n_child}, n_parent={n_parent}")
            print(f"  z-scores: {result}")
        except ValueError as e:
            print(f"✗ {description}: UNEXPECTED ERROR - {e}")
            return False

    print()
    return True


def test_invalid_tree_structure():
    """Test that invalid tree structures raise ValueError."""
    print("=" * 60)
    print("TEST 2: Invalid tree structures (n_child >= n_parent)")
    print("=" * 60)

    child_dist = np.array([0.6, 0.4])
    parent_dist = np.array([0.5, 0.5])

    test_cases = [
        (100, 100, "Equal sizes (n_child == n_parent)"),
        (150, 100, "Child larger than parent"),
        (100, 50, "Child much larger than parent"),
    ]

    all_passed = True
    for n_child, n_parent, description in test_cases:
        try:
            result = _compute_standardized_z(
                child_dist, parent_dist, n_child=n_child, n_parent=n_parent
            )
            print(f"✗ {description}: SHOULD HAVE RAISED ERROR but got result={result}")
            all_passed = False
        except ValueError as e:
            print(f"✓ {description}: Correctly raised ValueError")
            print(f"  Message: {str(e)[:80]}...")

    print()
    return all_passed


def test_branch_length_validation():
    """Test that invalid branch lengths raise ValueError."""
    print("=" * 60)
    print("TEST 3: Invalid branch lengths in sibling test")
    print("=" * 60)

    left_dist = np.array([0.6, 0.4])
    right_dist = np.array([0.5, 0.5])

    # Valid case: positive branch lengths
    try:
        stat, df, pval = sibling_divergence_test(
            left_dist,
            right_dist,
            n_left=50,
            n_right=50,
            branch_length_left=1.0,
            branch_length_right=1.0,
            mean_branch_length=1.0,
        )
        print(f"✓ Valid branch lengths: stat={stat:.4f}, pval={pval:.4f}")
    except ValueError as e:
        print(f"✗ Valid case UNEXPECTED ERROR: {e}")
        return False

    # Invalid cases: zero or negative branch lengths
    invalid_cases = [
        (0.0, 0.0, "Both zero"),
        (-1.0, 1.0, "Left negative"),
        (1.0, -1.0, "Right negative"),
        (-1.0, -1.0, "Both negative"),
    ]

    all_passed = True
    for bl_left, bl_right, description in invalid_cases:
        try:
            stat, df, pval = sibling_divergence_test(
                left_dist,
                right_dist,
                n_left=50,
                n_right=50,
                branch_length_left=bl_left,
                branch_length_right=bl_right,
                mean_branch_length=1.0,
            )
            print(f"✗ {description}: SHOULD HAVE RAISED ERROR")
            all_passed = False
        except ValueError as e:
            print(f"✓ {description}: Correctly raised ValueError")

    print()
    return all_passed


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=" * 60)
    print("TEST 4: Edge cases")
    print("=" * 60)

    child_dist = np.array([0.6, 0.4])
    parent_dist = np.array([0.5, 0.5])

    # Test with very small difference
    try:
        result = _compute_standardized_z(
            child_dist, parent_dist, n_child=999, n_parent=1000
        )
        print(f"✓ n_child=999, n_parent=1000 (diff=1): z-scores computed")
    except ValueError as e:
        print(f"✗ Small difference case: {e}")
        return False

    # Test with extremely small valid difference
    try:
        result = _compute_standardized_z(
            child_dist, parent_dist, n_child=999999, n_parent=1000000
        )
        print(f"✓ n_child=999999, n_parent=1000000: z-scores computed")
    except ValueError as e:
        print(f"✗ Extremely small difference: {e}")
        return False

    print()
    return True


def test_variance_formula_correctness():
    """Verify the variance formula produces correct values."""
    print("=" * 60)
    print("TEST 5: Variance formula correctness")
    print("=" * 60)

    # Known case: theta = 0.5, n_child = 10, n_parent = 100
    # Expected variance = 0.5 * 0.5 * (1/10 - 1/100) = 0.25 * 0.09 = 0.0225
    child_dist = np.array([0.6])  # Single feature for simplicity
    parent_dist = np.array([0.5])

    result = _compute_standardized_z(child_dist, parent_dist, n_child=10, n_parent=100)

    # Expected: z = (0.6 - 0.5) / sqrt(0.0225) = 0.1 / 0.15 = 0.666...
    expected_z = 0.1 / np.sqrt(0.0225)
    print(f"Computed z-score: {result[0]:.6f}")
    print(f"Expected z-score: {expected_z:.6f}")

    if np.isclose(result[0], expected_z, rtol=1e-5):
        print("✓ Variance formula produces correct result")
        return True
    else:
        print("✗ Variance formula produces INCORRECT result")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VARIANCE FORMULA ERROR HANDLING TESTS")
    print("=" * 60 + "\n")

    results = []
    results.append(("Valid tree structures", test_valid_tree_structure()))
    results.append(("Invalid tree structures", test_invalid_tree_structure()))
    results.append(("Branch length validation", test_branch_length_validation()))
    results.append(("Edge cases", test_edge_cases()))
    results.append(
        ("Variance formula correctness", test_variance_formula_correctness())
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Error handling is working correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. Please review the output above.")
        sys.exit(1)
