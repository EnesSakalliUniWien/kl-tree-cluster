# Categorical Data Handling: Research and Implementation Plan

## Executive Summary

**Status**: The current approach of flattening categorical data ignores the covariance structure of multinomial distributions, leading to inflated test statistics and incorrect Type I error rates.

**Priority**: Medium-High (affects categorical/phylogenetic test cases)

**Recommended Fix**: Implement "drop last category" method for immediate improvement; consider Mahalanobis for high-stakes applications.

---

## 1. Current Implementation (Problematic)

### Location
- `kl_clustering_analysis/hierarchy_analysis/statistics/pooled_variance.py:37-47`
- `kl_clustering_analysis/hierarchy_analysis/statistics/pooled_variance.py:213-215`

### Code
```python
def _flatten_categorical(arr: np.ndarray) -> np.ndarray:
    """Flatten categorical distribution to 1D for Wald test."""
    if _is_categorical(arr):
        return arr.ravel()  # (d, K) -> (d*K,)
    return arr

# In standardize_proportion_difference():
variance_flat = _flatten_categorical(variance)
difference_flat = _flatten_categorical(difference)
z_scores = difference_flat / np.sqrt(variance_flat)
```

### The Problem

**Multinomial Covariance Structure**:
For K categories with probabilities p₁, p₂, ..., pₖ:

$$
\text{Cov}(\hat{p}_j, \hat{p}_k) = \begin{cases} 
\frac{p_j(1-p_j)}{n} & j = k \\ 
-\frac{p_j p_k}{n} & j \neq k 
\end{cases}
$$

**Key Issue**: Categories are **negatively correlated** (probabilities sum to 1).

**Current Approach**: 
- Uses only diagonal variance: $\text{Var}(\hat{p}_k) = p_k(1-p_k)/n$
- Ignores off-diagonal covariance terms
- Flattens (d, K) → (d×K) and treats all dimensions as independent

**Consequence**: 
- Inflated Type I error (false positives)
- Tests count the same divergence multiple times (once per correlated category)

---

## 2. Alternative Approaches

### 2.1 Reduced Degrees of Freedom (Minimal Fix)

**Concept**: Correct the degrees of freedom from d×K to d×(K-1).

**Why**: Each feature with K categories has K-1 degrees of freedom due to the simplex constraint (probabilities sum to 1).

**Implementation**:
```python
# In compute_categorical_effective_df() - ALREADY EXISTS
def compute_categorical_effective_df(n_features: int, n_categories: int) -> int:
    return n_features * (n_categories - 1)  # d * (K - 1)

# Usage in test files:
if is_categorical(z):
    n_features, n_categories = original_shape
    effective_df = compute_categorical_effective_df(n_features, n_categories)
    pval = chi2.sf(stat, df=min(k, effective_df))
else:
    pval = chi2.sf(stat, df=k)
```

**Pros**:
- Minimal code change
- Improves p-value calibration

**Cons**:
- Doesn't fix the covariance structure problem
- Still overestimates test statistic magnitude

**Effort**: Low

---

### 2.2 Drop Last Category (Reference Cell Method)

**Concept**: Don't test all K categories - drop one as reference. The constraint ensures the dropped category captures any remaining information.

**Implementation**:
```python
def _flatten_categorical(arr: np.ndarray) -> np.ndarray:
    """Flatten categorical, dropping last category to handle simplex constraint."""
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, :-1].ravel()  # Drop last category, then flatten
    return arr

# Or for Mahalanobis computation:
def compute_multinomial_wald_drop_last(theta_1, theta_2, n_1, n_2):
    # Drop last category
    theta_1_reduced = theta_1[:, :-1]  # Shape (d, K-1)
    theta_2_reduced = theta_2[:, :-1]
    
    diff = (theta_1_reduced - theta_2_reduced).ravel()  # (d*(K-1),)
    
    # Build block-diagonal covariance (each feature independent)
    Sigma = build_covariance_block_diagonal(theta_1_reduced, theta_2_reduced, n_1, n_2)
    
    # Wald statistic (now full-rank, invertible)
    T = diff.T @ np.linalg.inv(Sigma) @ diff
    df = theta_1_reduced.size  # d * (K-1)
    
    return T, df
```

**Why It Works**:
- Automatically accounts for constraint (if categories 1-(K-1) changed, category K must have changed)
- Reduces dimension from K to K-1, matching true degrees of freedom
- Covariance matrix becomes full-rank (invertible without pseudoinverse)

**Pros**:
- Correct degrees of freedom
- No singular matrix issues
- Simple to implement
- 90% as good as full Mahalanobis

**Cons**:
- Which category to drop? (rotationally dependent)
- Random choice might drop informative category

**Mitigation**: Drop least informative category (lowest variance) or rotate through categories.

**Effort**: Low-Medium

---

### 2.3 Mahalanobis Distance (Full Covariance)

**Concept**: Use complete covariance matrix in Wald statistic.

**Formula**:
$$T = (\hat{\theta}_1 - \hat{\theta}_2)^T \Sigma^{-1} (\hat{\theta}_1 - \hat{\theta}_2) \sim \chi^2_{d(K-1)}$$

**Implementation**:
```python
def compute_multinomial_covariance(pooled_theta: np.ndarray, n: float) -> np.ndarray:
    """
    Compute multinomial covariance matrix for one feature.
    
    For K categories, returns K×K matrix:
        diag: p_k(1-p_k)/n
        off-diag: -p_j*p_k/n
    """
    K = pooled_theta.shape[0]
    Sigma = np.zeros((K, K))
    
    for j in range(K):
        for k in range(K):
            if j == k:
                Sigma[j, k] = pooled_theta[j] * (1 - pooled_theta[j]) / n
            else:
                Sigma[j, k] = -pooled_theta[j] * pooled_theta[k] / n
    
    return Sigma


def mahalanobis_wald_categorical(
    theta_1: np.ndarray,  # Shape (d, K)
    theta_2: np.ndarray,
    n_1: float,
    n_2: float,
) -> tuple[float, int]:
    """
    Mahalanobis Wald test for multinomial distributions.
    
    Uses block-diagonal covariance structure (features independent).
    """
    d, K = theta_1.shape
    pooled = (n_1 * theta_1 + n_2 * theta_2) / (n_1 + n_2)
    
    # Compute pooled variance for each sample
    n_pooled = (n_1 * n_2) / (n_1 + n_2)  # Harmonic mean
    
    T = 0.0
    for i in range(d):
        diff_i = theta_1[i, :] - theta_2[i, :]  # (K,)
        Sigma_i = compute_multinomial_covariance(pooled[i, :], n_pooled)
        
        # Use pseudoinverse for singular matrix
        Sigma_inv_i = np.linalg.pinv(Sigma_i)
        
        T += diff_i @ Sigma_inv_i @ diff_i
    
    df = d * (K - 1)  # Effective degrees of freedom
    return T, df
```

**Why It's Best**:
1. **Accounts for covariance**: Negative correlations between categories properly handled
2. **Optimal power**: Neyman-Pearson lemma - most powerful test for multivariate normal
3. **Correct Type I error**: Exact asymptotic distribution
4. **Invariant**: Rotation-invariant (unlike drop-last-category)

**Challenges**:

1. **Singular Matrix**: Multinomial covariance has rank (K-1), not K
   - **Solution**: Use Moore-Penrose pseudoinverse `np.linalg.pinv()`

2. **Computational Cost**: Inverting d separate K×K matrices
   - **Optimization**: Block-diagonal structure - invert each block independently
   - Cost: O(d × K³) instead of O((dK)³)

3. **High Dimensions**: For d×K > n, covariance estimation is unstable
   - **Solution**: Regularization or shrinkage estimators

**Pros**:
- Statistically optimal
- Correct Type I error
- Handles correlation structure

**Cons**:
- Higher computational cost
- Requires pseudoinverse (more complex)
- Needs careful numerical implementation

**Effort**: High

---

### 2.4 Pearson Chi-Square Test

**Concept**: Use classical Pearson chi-square for categorical comparison.

**Formula**:
$$\chi^2 = \sum_{i=1}^{d} \sum_{k=1}^{K} \frac{(O_{ik} - E_{ik})^2}{E_{ik}}$$

Where O = observed counts, E = expected counts under null.

**Implementation**:
```python
def pearson_chi_square_categorical(
    counts_1: np.ndarray,  # Shape (d, K) - raw counts, not proportions
    counts_2: np.ndarray,
) -> tuple[float, int]:
    """
    Pearson chi-square test for two multinomial samples.
    """
    n_1 = counts_1.sum()
    n_2 = counts_2.sum()
    
    # Observed proportions
    p_1 = counts_1 / n_1
    p_2 = counts_2 / n_2
    
    # Pooled estimate under null
    pooled_counts = (counts_1 + counts_2) / (n_1 + n_2)
    
    # Chi-square statistic
    chi2_stat = 0.0
    for sample_counts, n in [(counts_1, n_1), (counts_2, n_2)]:
        expected = n * pooled_counts
        chi2_stat += np.sum((sample_counts - expected)**2 / expected)
    
    d, K = counts_1.shape
    df = d * (K - 1)
    
    return chi2_stat, df
```

**Pros**:
- Well-established test for categorical data
- Handles covariance through expected frequencies
- No matrix inversion needed

**Cons**:
- Requires raw counts (not proportions)
- Less powerful than Wald for small samples
- Asymptotic approximation (needs large n)

**Effort**: Medium

---

### 2.5 Profile Likelihood Test

**Concept**: Likelihood ratio test with multinomial likelihood.

**Formula**:
$$G = 2 \times (\log L_{\text{alt}} - \log L_{\text{null}}) \sim \chi^2_{d(K-1)}$$

**Implementation**:
```python
def multinomial_log_likelihood(counts: np.ndarray, probs: np.ndarray) -> float:
    """Multinomial log-likelihood."""
    from scipy.stats import multinomial
    n = counts.sum()
    return multinomial.logpmf(counts.ravel(), n, probs.ravel())

def profile_likelihood_test(counts_1, counts_2):
    n_1, n_2 = counts_1.sum(), counts_2.sum()
    
    # MLE under null (pooled)
    p_pooled = (counts_1 + counts_2) / (n_1 + n_2)
    L_null = (multinomial_log_likelihood(counts_1, p_pooled) + 
              multinomial_log_likelihood(counts_2, p_pooled))
    
    # MLE under alternative (separate)
    p_1 = counts_1 / n_1
    p_2 = counts_2 / n_2
    L_alt = (multinomial_log_likelihood(counts_1, p_1) + 
             multinomial_log_likelihood(counts_2, p_2))
    
    G = 2 * (L_alt - L_null)
    df = counts_1.size - counts_1.shape[0]  # d*(K-1)
    
    return G, df
```

**Pros**:
- Most powerful test (Neyman-Pearson)
- Uses full likelihood information

**Cons**:
- Requires iterative optimization (if not closed-form)
- Needs raw counts
- Computationally expensive

**Effort**: High

---

### 2.6 Permutation Test (Non-parametric)

**Concept**: Empirical null distribution via label shuffling.

**Implementation**:
```python
def permutation_test_categorical(data_1, data_2, n_permutations=1000):
    # Observed statistic
    observed_stat, _ = mahalanobis_wald_categorical(data_1, data_2, ...)
    
    # Permutation null
    combined = np.vstack([data_1, data_2])
    n_1 = len(data_1)
    
    perm_stats = []
    for _ in range(n_permutations):
        perm = np.random.permutation(combined)
        perm_1, perm_2 = perm[:n_1], perm[n_1:]
        stat, _ = mahalanobis_wald_categorical(perm_1, perm_2, ...)
        perm_stats.append(stat)
    
    p_value = np.mean(np.array(perm_stats) >= observed_stat)
    return observed_stat, p_value
```

**Pros**:
- No asymptotic approximations
- Exact p-value (for given data)
- Distribution-free

**Cons**:
- Very expensive (1000+ iterations)
- Not practical for large-scale clustering

**Effort**: High (computational)

---

## 3. Implementation Recommendations

### Phase 1: Immediate Fix (This Week)

**Implement "Drop Last Category" Method**

**Rationale**: 
- 90% of Mahalanobis benefit with 10% of effort
- Corrects degrees of freedom
- No singular matrix issues
- Minimal code change

**Files to Modify**:
1. `kl_clustering_analysis/hierarchy_analysis/statistics/pooled_variance.py`
   - Modify `_flatten_categorical()` to drop last category
   
2. `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py`
   - Add categorical df correction in p-value calculation
   
3. `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py`
   - Same df correction

**Testing**:
- Validate on categorical test cases from benchmarks
- Compare Type I error rate to nominal alpha

---

### Phase 2: Validation (Next Week)

**Compare Approaches on Benchmark Data**

Run controlled experiments:
- Generate synthetic categorical data with known ground truth
- Test all methods: current, drop-last, Mahalanobis
- Metrics: Type I error rate, power, ARI on clustering

**Decision Point**:
- If drop-last achieves >90% of Mahalanobis power: keep it
- If gap is significant: implement full Mahalanobis

---

### Phase 3: Advanced (Future)

**Mahalanobis Implementation** (if needed)

Requirements:
- Efficient block-diagonal inversion
- Numerical stability checks
- Fallback to drop-last if matrix inversion fails

**Optimization**:
- Cache covariance inverses when sample sizes match
- Use Cholesky decomposition for stability
- Consider sparse matrix representations for high dimensions

---

## 4. Related Code Locations

| File | Lines | Function |
|------|-------|----------|
| `pooled_variance.py` | 37-47 | `_flatten_categorical()` |
| `pooled_variance.py` | 213-215 | Flattening in `standardize_proportion_difference()` |
| `pooled_variance.py` | 56-66 | `compute_categorical_effective_df()` |
| `edge_significance.py` | ~200 | P-value calculation after projection |
| `sibling_divergence_test.py` | ~170 | P-value calculation after projection |
| `divergence_metrics.py` | 15-40 | `_kl_categorical_general()` |

---

## 5. References

1. Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley.
2. Felsenstein, J. (1985). Phylogenies and the comparative method. *The American Naturalist*, 125(1), 1-15.
3. StatLect. Wald test | Formula, explanation, example. https://www.statlect.com
4. Wikipedia. Multinomial distribution. https://en.wikipedia.org/wiki/Multinomial_distribution

---

## 6. Open Questions

1. **Rotation Dependency**: For drop-last method, should we:
   - Always drop last category (simple)
   - Drop lowest-variance category (optimal)
   - Test all rotations and take maximum (conservative)

2. **Small Samples**: When n/K < 5, asymptotic approximations fail. Do we:
   - Fall back to permutation test?
   - Apply continuity correction?
   - Skip categorical features with low counts?

3. **Mixed Data**: When some features are binary, some categorical:
   - Handle separately and combine p-values?
   - Convert all to categorical (K=2 for binary)?
   - Use different tests for different feature types?

---

**Document Version**: 1.0  
**Date**: 2026-02-13  
**Author**: Research Analysis  
**Status**: Ready for implementation review
