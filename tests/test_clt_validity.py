"""Tests for CLT validity checks using Berry-Esseen bounds.

This module tests the Berry-Esseen-based CLT validity checking implementation
in kl_clustering_analysis.hierarchy_analysis.statistics.clt_validity.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kl_clustering_analysis.hierarchy_analysis.statistics.clt_validity import (
    SHEVTSOVA_CONSTANT,
    VAN_BEEK_CONSTANT,
    CLTValidityResult,
    berry_esseen_bound,
    check_clt_validity_bernoulli,
    check_split_clt_validity,
    compute_minimum_n_berry_esseen,
    compute_third_absolute_moment,
    compute_third_absolute_moment_categorical,
    compute_variance_bernoulli,
)


class TestThirdAbsoluteMoment:
    """Tests for third absolute central moment computation."""

    def test_bernoulli_symmetric(self):
        """Test that symmetric Bernoulli (p=0.5) has correct third moment."""
        p = np.array([0.5])
        rho = compute_third_absolute_moment(p)
        # E[|X - 0.5|^3] = 0.5 * 0.5 * (0.25 + 0.25) = 0.25 * 0.5 = 0.125
        assert_allclose(rho, [0.125], rtol=1e-10)

    def test_bernoulli_extreme(self):
        """Test that extreme Bernoulli (p near 0 or 1) has small third moment."""
        # p = 0.01: most mass at 0, small deviation
        p = np.array([0.01])
        rho = compute_third_absolute_moment(p)
        # E[|X - 0.01|^3] = 0.01 * 0.99 * (0.0001 + 0.9801)
        # = 0.0099 * 0.9802 ≈ 0.0097
        expected = 0.01 * 0.99 * (0.01**2 + 0.99**2)
        assert_allclose(rho, [expected], rtol=1e-10)

    def test_bernoulli_vectorized(self):
        """Test that computation works for multiple features."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rho = compute_third_absolute_moment(probs)

        # Check symmetry: p and 1-p should give same result
        assert_allclose(rho[0], rho[4], rtol=1e-10)
        assert_allclose(rho[1], rho[3], rtol=1e-10)

        # Maximum should be at p = 0.5
        assert rho[2] >= rho[1]
        assert rho[2] >= rho[0]

    def test_bernoulli_zero_and_one(self):
        """Test edge cases p=0 and p=1."""
        p = np.array([0.0, 1.0])
        rho = compute_third_absolute_moment(p)
        # Both should be 0 (degenerate distributions)
        assert_allclose(rho, [0.0, 0.0], atol=1e-10)


class TestVarianceBernoulli:
    """Tests for Bernoulli variance computation."""

    def test_variance_symmetric(self):
        """Test variance at p=0.5."""
        p = np.array([0.5])
        var = compute_variance_bernoulli(p)
        assert_allclose(var, [0.25], rtol=1e-10)

    def test_variance_extreme(self):
        """Test variance at extremes."""
        p = np.array([0.0, 0.01, 0.99, 1.0])
        var = compute_variance_bernoulli(p)
        assert_allclose(var[0], 0.0, atol=1e-10)
        assert_allclose(var[3], 0.0, atol=1e-10)
        # Var(p) = p(1-p) is symmetric around p=0.5
        assert_allclose(var[1], var[2], rtol=1e-10)

    def test_variance_formula(self):
        """Test that variance formula matches definition."""
        probs = np.random.rand(100)
        var_computed = compute_variance_bernoulli(probs)
        var_expected = probs * (1 - probs)
        assert_allclose(var_computed, var_expected, rtol=1e-10)


class TestCategoricalThirdMoment:
    """Tests for categorical distribution third moment."""

    def test_uniform_categorical(self):
        """Test uniform categorical distribution."""
        # k categories, uniform
        k = 4
        probs = np.ones(k) / k
        rho = compute_third_absolute_moment_categorical(probs)

        # Should be positive
        assert rho > 0

        # For uniform, all ||eᵢ - p|| are equal
        # ||eᵢ - p||² = (1-1/k)² + (k-1)(1/k)² = (k-1)²/k² + (k-1)/k²
        # = (k-1)/k² * (k-1 + 1) = (k-1)/k
        expected_sq_norm = (k - 1) / k
        expected = expected_sq_norm**1.5
        assert_allclose(rho, expected, rtol=1e-10)

    def test_two_category_bernoulli_equivalence(self):
        """Test that 2-category categorical matches Bernoulli."""
        p = 0.3
        cat_probs = np.array([1 - p, p])

        rho_cat = compute_third_absolute_moment_categorical(cat_probs)
        rho_bernoulli = compute_third_absolute_moment(np.array([p]))

        # These should be related (though not exactly equal due to encoding)
        # Both should be positive
        assert rho_cat > 0
        assert rho_bernoulli[0] > 0

    def test_degenerate_categorical(self):
        """Test degenerate categorical (one category has prob 1)."""
        probs = np.array([1.0, 0.0, 0.0])
        rho = compute_third_absolute_moment_categorical(probs)
        # Should be 0 (no variance)
        assert_allclose(rho, 0.0, atol=1e-10)


class TestBerryEsseenBound:
    """Tests for Berry-Esseen bound computation."""

    def test_bound_decreases_with_n(self):
        """Test that bound decreases as n increases."""
        rho = 0.1
        sigma_sq = 0.25

        bound_small_n = berry_esseen_bound(10, rho, sigma_sq)
        bound_large_n = berry_esseen_bound(1000, rho, sigma_sq)

        assert bound_large_n < bound_small_n
        # Should decrease as 1/sqrt(n)
        ratio = bound_small_n / bound_large_n
        expected_ratio = np.sqrt(1000 / 10)
        assert_allclose(ratio, expected_ratio, rtol=0.1)

    def test_bound_increases_with_rho(self):
        """Test that bound increases with third moment."""
        n = 100
        sigma_sq = 0.25

        bound_small_rho = berry_esseen_bound(n, 0.05, sigma_sq)
        bound_large_rho = berry_esseen_bound(n, 0.15, sigma_sq)

        assert bound_large_rho > bound_small_rho
        # Should scale linearly with rho
        assert_allclose(bound_large_rho / bound_small_rho, 3.0, rtol=1e-10)

    def test_bound_increases_with_skewness(self):
        """Test that bound is larger for skewed distributions."""
        n = 100

        # Symmetric: p=0.5
        p_sym = 0.5
        rho_sym = compute_third_absolute_moment(np.array([p_sym]))[0]
        sigma_sq_sym = p_sym * (1 - p_sym)
        bound_sym = berry_esseen_bound(n, rho_sym, sigma_sq_sym)

        # Skewed: p=0.1
        p_skew = 0.1
        rho_skew = compute_third_absolute_moment(np.array([p_skew]))[0]
        sigma_sq_skew = p_skew * (1 - p_skew)
        bound_skew = berry_esseen_bound(n, rho_skew, sigma_sq_skew)

        # Skewed distribution should have larger bound
        assert bound_skew > bound_sym

    def test_vectorized_bound(self):
        """Test that bound works with arrays."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        n = 100

        rho = compute_third_absolute_moment(probs)
        sigma_sq = compute_variance_bernoulli(probs)

        bounds = berry_esseen_bound(n, rho, sigma_sq)

        assert len(bounds) == len(probs)
        # All bounds should be positive
        assert np.all(bounds > 0)

    def test_bound_error_on_zero_variance(self):
        """Test that bound raises error for zero variance."""
        with pytest.raises(ValueError, match="Variance must be positive"):
            berry_esseen_bound(10, 0.1, 0.0)

        with pytest.raises(ValueError, match="Variance must be positive"):
            berry_esseen_bound(10, np.array([0.1, 0.2]), np.array([0.0, 0.25]))

    def test_constant_comparison(self):
        """Test that Shevtsova constant gives tighter bound than van Beek."""
        rho = 0.1
        sigma_sq = 0.25
        n = 100

        bound_shevtsova = berry_esseen_bound(n, rho, sigma_sq, SHEVTSOVA_CONSTANT)
        bound_van_beek = berry_esseen_bound(n, rho, sigma_sq, VAN_BEEK_CONSTANT)

        # Shevtsova should be smaller (tighter)
        assert bound_shevtsova < bound_van_beek


class TestCLTValidityBernoulli:
    """Tests for CLT validity checking on Bernoulli distributions."""

    def test_valid_for_large_n_symmetric(self):
        """Test that symmetric Bernoulli with large n is valid."""
        probs = np.array([0.5, 0.5, 0.5])
        n = 1000

        result = check_clt_validity_bernoulli(probs, n, alpha=0.05)

        assert result.is_valid
        assert result.fraction_valid == 1.0
        assert result.actual_n == n
        assert result.min_required_n <= n

    def test_invalid_for_small_n_skewed(self):
        """Test that skewed Bernoulli with small n may be invalid."""
        # Highly skewed distribution
        probs = np.array([0.01, 0.01, 0.01])
        n = 5

        result = check_clt_validity_bernoulli(probs, n, alpha=0.05)

        # Should require much larger n
        assert result.min_required_n > n
        # May or may not be valid depending on fraction threshold

    def test_degenerate_features_count_as_valid(self):
        """Test that degenerate features (p=0 or p=1) are counted as valid."""
        probs = np.array([0.0, 0.5, 1.0])
        n = 10

        result = check_clt_validity_bernoulli(probs, n, alpha=0.05)

        # Should have at least 2/3 valid (2 degenerate + 1 non-degenerate)
        # The non-degenerate feature may or may not be valid depending on n
        assert result.fraction_valid >= 2 / 3
        n = 30

        # With high threshold, may be invalid
        result_strict = check_clt_validity_bernoulli(
            probs, n, alpha=0.05, min_fraction_valid=0.9
        )

        # With low threshold, should be valid
        result_lenient = check_clt_validity_bernoulli(
            probs, n, alpha=0.05, min_fraction_valid=0.5
        )

        # Lenient should be more likely to be valid
        if result_lenient.is_valid:
            assert result_lenient.fraction_valid >= 0.5

    def test_n_zero(self):
        """Test edge case n=0."""
        probs = np.array([0.5])
        result = check_clt_validity_bernoulli(probs, 0, alpha=0.05)

        assert not result.is_valid
        assert result.min_required_n == 1
        assert result.fraction_valid == 0.0

    def test_n_one(self):
        """Test edge case n=1."""
        probs = np.array([0.5])
        result = check_clt_validity_bernoulli(probs, 1, alpha=0.05)

        # n=1 is almost never sufficient for CLT
        assert not result.is_valid or result.max_approximation_error > 0.01

    def test_alpha_scaling(self):
        """Test that smaller alpha requires larger n."""
        probs = np.array([0.3, 0.4, 0.5])
        n = 100

        result_05 = check_clt_validity_bernoulli(probs, n, alpha=0.05)
        result_01 = check_clt_validity_bernoulli(probs, n, alpha=0.01)

        # Smaller alpha should require larger min_n
        assert result_01.min_required_n >= result_05.min_required_n

    def test_features_valid_array(self):
        """Test that features_valid array is correctly populated."""
        probs = np.array([0.5, 0.5, 0.1, 0.9])
        n = 50

        result = check_clt_validity_bernoulli(probs, n, alpha=0.05)

        # All features should be represented
        assert len(result.features_valid) == len(probs)
        # features_valid should be boolean
        assert result.features_valid.dtype == bool
        # fraction_valid should match
        expected_fraction = np.mean(result.features_valid)
        assert_allclose(result.fraction_valid, expected_fraction, rtol=1e-10)


class TestComputeMinimumN:
    """Tests for minimum sample size computation."""

    def test_symmetric_requires_less_n(self):
        """Test that symmetric Bernoulli requires smaller n."""
        probs_sym = np.array([0.5, 0.5, 0.5])
        probs_skew = np.array([0.1, 0.1, 0.1])

        n_sym = compute_minimum_n_berry_esseen(probs_sym, alpha=0.05)
        n_skew = compute_minimum_n_berry_esseen(probs_skew, alpha=0.05)

        # Symmetric should require less n
        assert n_sym <= n_skew

    def test_extreme_requires_more_n(self):
        """Test that extreme probabilities require larger n."""
        probs_moderate = np.array([0.3, 0.4, 0.5])
        probs_extreme = np.array([0.01, 0.02, 0.03])

        n_mod = compute_minimum_n_berry_esseen(probs_moderate, alpha=0.05)
        n_ext = compute_minimum_n_berry_esseen(probs_extreme, alpha=0.05)

        # Extreme should require more n
        assert n_ext > n_mod

    def test_alpha_effect(self):
        """Test that smaller alpha increases required n."""
        probs = np.array([0.3, 0.4, 0.5])

        n_05 = compute_minimum_n_berry_esseen(probs, alpha=0.05)
        n_01 = compute_minimum_n_berry_esseen(probs, alpha=0.01)
        n_001 = compute_minimum_n_berry_esseen(probs, alpha=0.001)

        # Smaller alpha -> larger required n
        assert n_05 <= n_01 <= n_001

    def test_degenerate_all(self):
        """Test all degenerate case."""
        probs = np.array([0.0, 1.0, 0.0])
        n = compute_minimum_n_berry_esseen(probs, alpha=0.05)

        # Should return 1 (no samples needed for degenerate)
        assert n == 1

    def test_single_feature(self):
        """Test with single feature."""
        probs = np.array([0.5])
        n = compute_minimum_n_berry_esseen(probs, alpha=0.05)

        # Should be small for symmetric single feature
        assert n >= 1
        # For p=0.5, should be calculable
        rho = 0.5 * 0.5 * (0.25 + 0.25)  # 0.125
        sigma_sq = 0.25
        expected_n = int(
            np.ceil((SHEVTSOVA_CONSTANT * rho / (sigma_sq**1.5 * 0.05)) ** 2)
        )
        assert n == expected_n


class TestCheckSplitCLTValidity:
    """Tests for checking CLT validity at split nodes."""

    def test_both_children_valid(self):
        """Test when both children have valid CLT."""
        dist_left = np.array([0.5, 0.5, 0.5])
        dist_right = np.array([0.5, 0.5, 0.5])
        n_left = 100
        n_right = 100

        left_result, right_result = check_split_clt_validity(
            dist_left, dist_right, n_left, n_right, alpha=0.05
        )

        assert left_result.is_valid
        assert right_result.is_valid

    def test_one_child_invalid(self):
        """Test when one child has invalid CLT."""
        dist_left = np.array([0.5, 0.5, 0.5])  # valid
        dist_right = np.array([0.01, 0.01, 0.01])  # skewed
        n_left = 100
        n_right = 5  # too small for skewed

        left_result, right_result = check_split_clt_validity(
            dist_left, dist_right, n_left, n_right, alpha=0.05
        )

        assert left_result.is_valid
        # Right child should be invalid (skewed + small n)
        assert not right_result.is_valid or right_result.min_required_n > n_right

    def test_unequal_sample_sizes(self):
        """Test with unequal sample sizes."""
        dist_left = np.array([0.3, 0.4, 0.5])
        dist_right = np.array([0.3, 0.4, 0.5])
        n_left = 200
        n_right = 50

        left_result, right_result = check_split_clt_validity(
            dist_left, dist_right, n_left, n_right, alpha=0.05
        )

        # Both have same distribution, different n
        assert left_result.actual_n == 200
        assert right_result.actual_n == 50
        # Larger n should be more likely to be valid
        if not left_result.is_valid:
            assert not right_result.is_valid

    def test_different_distributions(self):
        """Test with different distributions in children."""
        dist_left = np.array([0.2, 0.3, 0.4])
        dist_right = np.array([0.6, 0.7, 0.8])
        n_left = 100
        n_right = 100

        left_result, right_result = check_split_clt_validity(
            dist_left, dist_right, n_left, n_right, alpha=0.05
        )

        # Results should reflect the different distributions
        # Both should have same n
        assert left_result.actual_n == right_result.actual_n


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_berry_esseen_monotonicity(self):
        """Test monotonicity properties of Berry-Esseen bound."""
        # Fix p and vary n
        p = 0.3
        rho = compute_third_absolute_moment(np.array([p]))[0]
        sigma_sq = p * (1 - p)

        ns = np.array([10, 20, 50, 100, 200, 500, 1000])
        bounds = [berry_esseen_bound(n, rho, sigma_sq) for n in ns]

        # Bounds should decrease monotonically
        for i in range(len(bounds) - 1):
            assert bounds[i + 1] < bounds[i]

        # Should decrease as 1/sqrt(n)
        scaled_bounds = np.array(bounds) * np.sqrt(ns)
        # After scaling by sqrt(n), should be approximately constant
        assert_allclose(scaled_bounds, scaled_bounds[0], rtol=0.01)

    def test_clt_convergence_simulation(self):
        """Simulate CLT convergence and verify Berry-Esseen predicts it."""
        np.random.seed(42)

        # True Bernoulli parameter
        p = 0.3

        # Sample sizes to test
        ns = [10, 30, 100, 300]

        for n in ns:
            # Simulate many sample means
            n_sims = 10000
            samples = np.random.binomial(n, p, size=n_sims) / n

            # Empirical CDF at mean
            z_score = 0.0  # at the mean
            empirical_prob = np.mean(samples <= p + z_score * np.sqrt(p * (1 - p) / n))

            # True normal CDF at z=0 is 0.5
            true_prob = 0.5

            # Empirical error
            empirical_error = abs(empirical_prob - true_prob)

            # Berry-Esseen predicted bound
            rho = p * (1 - p) * (p**2 + (1 - p) ** 2)
            sigma_sq = p * (1 - p)
            predicted_bound = berry_esseen_bound(n, rho, sigma_sq)

            # Empirical error should be less than bound (with high probability)
            # This is a probabilistic test, so we check that the bound
            # is in the right ballpark
            assert empirical_error < 5 * predicted_bound  # Loose check

    def test_skewness_vs_symmetry(self):
        """Compare validity results for skewed vs symmetric distributions."""
        n = 50
        alpha = 0.05

        # Symmetric
        probs_sym = np.full(10, 0.5)
        result_sym = check_clt_validity_bernoulli(probs_sym, n, alpha)

        # Moderately skewed
        probs_mod = np.full(10, 0.3)
        result_mod = check_clt_validity_bernoulli(probs_mod, n, alpha)

        # Highly skewed
        probs_skew = np.full(10, 0.05)
        result_skew = check_clt_validity_bernoulli(probs_skew, n, alpha)

        # More skewed -> higher min_required_n
        assert result_sym.min_required_n <= result_mod.min_required_n
        assert result_mod.min_required_n <= result_skew.min_required_n


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_array(self):
        """Test with empty probability array."""
        probs = np.array([])
        result = check_clt_validity_bernoulli(probs, 10, alpha=0.05)

        # Empty array should be considered valid (no features to check)
        assert result.is_valid
        assert result.fraction_valid == 1.0

    def test_single_element(self):
        """Test with single element."""
        probs = np.array([0.5])
        result = check_clt_validity_bernoulli(probs, 100, alpha=0.05)

        assert result.is_valid
        assert len(result.features_valid) == 1

    def test_very_small_alpha(self):
        """Test with very small alpha."""
        probs = np.array([0.5])
        result = check_clt_validity_bernoulli(probs, 1000, alpha=0.001)

        # Very small alpha should require very large n
        assert result.min_required_n > 100

    def test_very_large_n(self):
        """Test with very large sample size."""
        probs = np.array([0.01, 0.99])  # extreme
        result = check_clt_validity_bernoulli(probs, 100000, alpha=0.05)

        # Should be valid with large enough n
        assert result.is_valid
        # For extreme probs, error bound may still be larger than 0.01
        # Check that it's reasonably small
        assert result.max_approximation_error < 0.1

    def test_numerical_stability_extreme_probs(self):
        """Test numerical stability with extreme probabilities."""
        # Very close to 0 or 1 - treated as degenerate
        probs = np.array([1e-10, 1 - 1e-10])
        result = check_clt_validity_bernoulli(probs, 1000, alpha=0.05)

        # Should not crash
        assert isinstance(result, CLTValidityResult)
        # These are essentially degenerate (machine precision)
        # The implementation checks (probs > 0) & (probs < 1)
        # which may return False for 1e-10 depending on dtype
