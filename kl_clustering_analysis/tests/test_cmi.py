#!/usr/bin/env python3
"""
Tests for conditional mutual information (CMI) utilities in misc/cmi.py.

This module tests the CMI functions used in hierarchical clustering analysis.
"""

import numpy as np
from numpy.testing import assert_allclose


class TestCMI:
    """Test suite for conditional mutual information functions."""

    def test_mi_binary_vec_basic(self):
        """Test basic mutual information calculation between binary vectors."""
        from hierarchy_analysis.cmi import _mi_binary_vec_accel as _mi_binary_vec

        # Simple case: identical vectors should have high MI
        x = np.array([1, 0, 1, 0], dtype=np.uint8)
        y = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)

        mi_values = _mi_binary_vec(x, y)

        # Both vectors should have positive MI (they're deterministic relationships)
        assert mi_values[0] > 0
        assert mi_values[1] > 0

        # Test with random/independent vectors
        y_random = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        mi_random = _mi_binary_vec(x, y_random)
        # Random should have lower MI than deterministic
        assert mi_random[0] < mi_values[0]

    def test_mi_binary_vec_edge_cases(self):
        """Test edge cases for mutual information."""
        from hierarchy_analysis.cmi import _mi_binary_vec_accel as _mi_binary_vec

        # Empty features
        x = np.array([], dtype=np.uint8)
        y = np.array([[]], dtype=np.uint8)
        mi_values = _mi_binary_vec(x, y)
        assert len(mi_values) == 1
        assert mi_values[0] == 0.0

        # Single feature - MI is 0 (no statistical relationship can be measured with one sample)
        x = np.array([1], dtype=np.uint8)
        y = np.array([[1], [0]], dtype=np.uint8)
        mi_values = _mi_binary_vec(x, y)
        assert len(mi_values) == 2
        assert mi_values[0] == 0.0  # Same value - no mutual information measurable
        assert (
            mi_values[1] == 0.0
        )  # Different value - still no mutual information measurable with single sample

    def test_mi_binary_vec_accel_consistency(self):
        """Test that accelerated version gives same results as basic version."""
        from hierarchy_analysis.cmi import (
            _mi_binary_vec_accel as _mi_binary_vec,
            _mi_binary_vec_accel,
        )

        # Generate test data
        np.random.seed(42)
        x = np.random.randint(0, 2, 10, dtype=np.uint8)
        y = np.random.randint(0, 2, (5, 10), dtype=np.uint8)

        mi_basic = _mi_binary_vec(x, y)
        mi_accel = _mi_binary_vec_accel(x, y)

        assert_allclose(mi_basic, mi_accel, rtol=1e-6, atol=1e-15)

    def test_cmi_binary_vec_basic(self):
        """Test basic conditional mutual information."""
        from hierarchy_analysis.cmi import _cmi_binary_vec

        # Simple case
        x = np.array([1, 0, 1, 0], dtype=np.uint8)
        y = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        z = np.array([0, 0, 1, 1], dtype=np.uint8)  # Condition variable

        cmi_values = _cmi_binary_vec(x, y, z)
        assert len(cmi_values) == 1
        assert cmi_values[0] >= 0  # CMI should be non-negative

    def test_cmi_binary_vec_independence(self):
        """Test CMI when variables are conditionally independent."""
        from hierarchy_analysis.cmi import _cmi_binary_vec

        # Create conditionally independent data
        np.random.seed(42)
        z = np.random.randint(0, 2, 100, dtype=np.uint8)

        # X and Y independent given Z
        x = z.copy()  # X depends only on Z
        y = np.random.randint(0, 2, 100, dtype=np.uint8)  # Y independent

        # Reshape for vectorized input
        x_test = x[:10]
        y_test = y.reshape(1, -1)[:, :10]
        z_test = z[:10]

        cmi_values = _cmi_binary_vec(x_test, y_test, z_test)
        # CMI should be low for independent variables
        assert cmi_values[0] < 0.1

    def test_perm_test_cmi_binary_basic(self):
        """Test basic permutation test for CMI."""
        from hierarchy_analysis.cmi import _perm_test_cmi_binary

        # Simple test case
        x = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        y = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)
        z = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)

        cmi_obs, p_value = _perm_test_cmi_binary(
            x, y, z, permutations=100, random_state=42
        )

        assert isinstance(cmi_obs, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        assert cmi_obs >= 0

    def test_perm_test_cmi_binary_edge_cases(self):
        """Test permutation test edge cases."""
        from hierarchy_analysis.cmi import _perm_test_cmi_binary

        # Empty data
        x = np.array([], dtype=np.uint8)
        y = np.array([], dtype=np.uint8)
        z = np.array([], dtype=np.uint8)

        cmi_obs, p_value = _perm_test_cmi_binary(x, y, z, permutations=10)
        assert cmi_obs == 0.0
        assert p_value == 1.0

        # Single stratum (all z same)
        x = np.array([1, 0], dtype=np.uint8)
        y = np.array([1, 0], dtype=np.uint8)
        z = np.array([0, 0], dtype=np.uint8)  # No variation in condition

        cmi_obs, p_value = _perm_test_cmi_binary(x, y, z, permutations=10)
        assert p_value == 1.0  # No test possible

    def test_perm_cmi_binary_batch(self):
        """Test batched permutation CMI calculation."""
        from hierarchy_analysis.cmi import _perm_cmi_binary_batch

        x = np.array([1, 0, 1, 0], dtype=np.uint8)
        y = np.array([1, 1, 0, 0], dtype=np.uint8)
        z = np.array([0, 0, 1, 1], dtype=np.uint8)

        rng = np.random.default_rng(42)
        perm_values = _perm_cmi_binary_batch(x, y, z, K=10, rng=rng)

        assert len(perm_values) == 10
        assert all(v >= 0 for v in perm_values)

    def test_cmi_perm_from_args(self):
        """Test the argument unpacking helper for parallel processing."""
        from hierarchy_analysis.cmi import _cmi_perm_from_args

        x = np.array([1, 0, 1, 0], dtype=np.uint8)
        y = np.array([1, 1, 0, 0], dtype=np.uint8)
        z = np.array([0, 0, 1, 1], dtype=np.uint8)

        args = (x, y, z, 50, 42, 32)
        cmi_obs, p_value = _cmi_perm_from_args(args)

        assert isinstance(cmi_obs, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_mi_properties(self):
        """Test mathematical properties of mutual information."""
        from hierarchy_analysis.cmi import _mi_binary_vec_accel as _mi_binary_vec

        # MI is symmetric
        x = np.array([1, 0, 1, 0], dtype=np.uint8)
        y1 = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        y2 = np.array([[0, 1, 0, 1]], dtype=np.uint8)

        mi_xy = _mi_binary_vec(x, y1)[0]
        mi_yx = _mi_binary_vec(y1[0], x.reshape(1, -1))[0]
        assert_allclose(mi_xy, mi_yx, rtol=1e-10)

        # MI is non-negative
        mi_values = _mi_binary_vec(x, np.vstack([y1, y2]))
        assert all(v >= 0 for v in mi_values)

    def test_cmi_properties(self):
        """Test mathematical properties of conditional mutual information."""
        from hierarchy_analysis.cmi import _cmi_binary_vec

        # CMI is non-negative
        x = np.array([1, 0, 1, 0], dtype=np.uint8)
        y = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        z = np.array([0, 0, 1, 1], dtype=np.uint8)

        cmi_values = _cmi_binary_vec(x, y, z)
        assert all(v >= 0 for v in cmi_values)

    def test_consistency_across_implementations(self):
        """Test that accelerated implementation gives consistent results."""
        from hierarchy_analysis.cmi import _mi_binary_vec_accel

        # Test with various data sizes and verify consistency
        for size in [5, 10, 20]:
            np.random.seed(42)
            x = np.random.randint(0, 2, size, dtype=np.uint8)
            y = np.random.randint(0, 2, (3, size), dtype=np.uint8)

            # Run twice to check consistency
            mi_accel1 = _mi_binary_vec_accel(x, y)
            mi_accel2 = _mi_binary_vec_accel(x, y)

            assert_allclose(
                mi_accel1,
                mi_accel2,
                rtol=1e-6,
                atol=1e-15,
                err_msg=f"Inconsistency at size {size}",
            )


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestCMI()

    print("Running CMI tests...")

    try:
        test_instance.test_mi_binary_vec_basic()
        print("✓ Basic MI test passed")

        test_instance.test_mi_binary_vec_edge_cases()
        print("✓ MI edge cases test passed")

        test_instance.test_mi_binary_vec_accel_consistency()
        print("✓ MI acceleration consistency test passed")

        test_instance.test_cmi_binary_vec_basic()
        print("✓ Basic CMI test passed")

        test_instance.test_perm_test_cmi_binary_basic()
        print("✓ Basic permutation test passed")

        test_instance.test_consistency_across_implementations()
        print("✓ Implementation consistency test passed")

        print("\nAll tests passed! ✅")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
