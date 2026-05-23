# Child-Parent Edge Test (Gate 2) - Input/Output Specification

## Overview

The **child-parent edge test** (Gate 2) determines whether a child cluster significantly diverges from its parent distribution in the hierarchy. It uses a **projected Wald chi-square test** with spectral dimension estimation.

---

## INPUTS

### Primary Inputs

| Parameter | Type | Shape | Description | Example Value |
|-----------|------|-------|-------------|---------------|
| `child_dist` | `np.ndarray` | `(d,)` | Bernoulli distribution for child node | `[0.52, 0.48, 0.0, ...]` |
| `parent_dist` | `np.ndarray` | `(d,)` | Bernoulli distribution for parent node | `[0.51, 0.49, 0.02, ...]` |
| `n_child` | `int` | scalar | Number of leaves in child subtree | `5` |
| `n_parent` | `int` | scalar | Number of leaves in parent subtree | `6` |

### Optional Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectral_k` | `int` | `max(k_signal, 2)` | Projection dimension (estimated via Marchenko-Pastur) |
| `branch_length` | `float` | `None` | Edge branch length for variance scaling |
| `mean_branch_length` | `float` | `None` | Mean branch length in tree (for normalization) |
| `pca_projection` | `np.ndarray` | `None` | Pre-computed PCA basis vectors |
| `pca_eigenvalues` | `np.ndarray` | `None` | PCA eigenvalues for whitening |

---

## OUTPUTS

| Output | Type | Description | Interpretation |
|--------|------|-------------|----------------|
| `test_statistic` | `float` | Wald chi-square statistic T | Larger = more divergence |
| `degrees_of_freedom` | `float` | Effective df (projection dimension k) | Typically 1-30 |
| `p_value` | `float` | P(χ²(df) > T) | Small = significant divergence |
| `is_invalid` | `bool` | True if test could not be computed | Check if k=0 or numerical issues |

---

## MATHEMATICAL FORMULATION

### Step 1: Compute Standardized Z-Scores

```python
# Nested sampling factor
nested_factor = 1/n_child - 1/n_parent

# Bernoulli variance under null hypothesis
variance = parent_dist * (1 - parent_dist) * nested_factor

# Standardized differences (z-scores)
z = (child_dist - parent_dist) / sqrt(variance)
```

**Numerical Example:**
```
n_child = 5, n_parent = 6
nested_factor = 1/5 - 1/6 = 0.0333

parent_dist = [0.5, 0.5, 0.5, ...]  (d=50)
child_dist  = [0.5, 0.5, 0.6, ...]

variance = 0.5 * 0.5 * 0.0333 = 0.00833

z = (0.6 - 0.5) / sqrt(0.00833) = 0.1 / 0.0913 = 1.095
```

### Step 2: Estimate Spectral Dimension (k)

```python
# Collect data matrix for parent subtree
X_parent = leaf_data[parent_leaves, :]  # Shape: (n_parent, d)

# Eigendecomposition of correlation matrix
eigenvalues = eigendecompose(X_parent)

# Marchenko-Pastur signal counting
sigma2 = median(eigenvalues[eigenvalues > 0])
gamma = d_active / n_parent
lambda_plus = sigma2 * (1 + sqrt(gamma))^2

# Count eigenvalues above MP threshold
k = count(eigenvalues > lambda_plus)
k = max(k, 2)  # Fixed Gate 2 floor, then capped by active feature count
```

**Numerical Example:**
```
X_parent.shape = (6, 50)  # n=6, d=50
eigenvalues = [5.18, 0.38, 0.20, 0.17, 0.06, 0.0]

sigma2 = median([5.18, 0.38, 0.20, 0.17, 0.06]) = 0.185
gamma = 50/6 = 8.33
lambda_plus = 0.185 * (1 + sqrt(8.33))^2 = 2.59

k = count([5.18, 0.38, ...] > 2.59) = 1
k = max(1, 2) = 2  # Apply floor
```

### Step 3: Build Projection Matrix

```python
# Use the PCA basis from the spectral context.
R = pca_projection[:k, :]  # Shape: (k, d)
```

**Numerical Example:**
```
k = 2, d = 50
R.shape = (2, 50)

R = [[ 0.12, -0.08,  0.15, ...,  0.03],
     [-0.05,  0.11, -0.09, ..., -0.07]]
```

### Step 4: Project Z-Scores

```python
# Project onto k-dimensional subspace
projected_z = R @ z  # Shape: (k,)
```

**Numerical Example:**
```
z.shape = (50,)
R.shape = (2, 50)

projected_z = R @ z = [-0.356, -0.468]  # Shape: (2,)
```

### Step 5: Compute Wald Statistic and P-Value

```python
# Test statistic: T = Σ (projected_z)²
T = sum(projected_z ** 2)

# P-value from chi-squared distribution
p = P(χ²(k) > T) = 1 - CDF_χ²(T, k)
```

**Numerical Example:**
```
projected_z = [-0.356, -0.468]

T = (-0.356)² + (-0.468)² = 0.127 + 0.219 = 0.346

p = P(χ²(2) > 0.346) = 0.841
```

---

## DECISION RULE

```python
alpha = EDGE_ALPHA  # Default: 0.001

if p_value < alpha:
    # REJECT null hypothesis
    # Child significantly diverges from parent
    Gate 2 PASSES → Edge is "significant"
else:
    # FAIL TO REJECT null hypothesis
    # Child does NOT significantly diverge
    Gate 2 FAILS → Edge is "not significant"
```

**Example Decision:**
```
p_value = 0.839
alpha   = 0.001

0.839 >= 0.001 → FAIL TO REJECT H₀ → Gate 2 FAILS
```

---

## COMPLETE NUMERICAL EXAMPLE

### Input Data
```
Edge: N50 → N48

child_dist.shape  = (50,)   # d=50 features
parent_dist.shape = (50,)
n_child  = 5
n_parent = 6

child_dist.mean()  = 0.516
parent_dist.mean() = 0.513
```

### Intermediate Calculations
```
nested_factor = 1/5 - 1/6 = 0.0333

variance.mean() = 0.000370
variance.std()  = 0.001256

z_scores.mean() = 0.0392
z_scores.std()  = 0.3645
z_scores.range  = [-0.490, 2.450]

First 10 z-scores: [0.0, 0.0, 0.490, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### Spectral Analysis
```
Data matrix: (6, 50)  # n=6 samples, d=50 features
Mode: DUAL (n×n Gram matrix)

Eigenvalues: [5.18, 0.38, 0.20, 0.17, 0.06, 0.00]

Marchenko-Pastur threshold: λ+ = 2.59
Signal eigenvalues: [5.18]  # Only 1 above threshold
k = max(1, 2) = 2  # Apply minimum dimension floor
```

### Projection
```
Projection matrix R: (2, 50)

projected_z = R @ z = [-0.356, -0.468]
```

### Test Results
```
Test statistic:  T = 0.352
Degrees of freedom: df = 2.0
P-value:  p = 0.839

Decision: p (0.839) >= α (0.001)
          → FAIL TO REJECT H₀
          → Gate 2 FAILS (not significant)
```

---

## FULL TREE ANNOTATION

When running `annotate_child_parent_divergence()` on a full tree:

### Input
```python
tree: PosetTree with distributions populated
annotations_df: pd.DataFrame with 'leaf_count' column
leaf_data: pd.DataFrame (n_samples, d_features)
significance_level_alpha: float (default: 0.001)
```

### Output Columns in `annotations_df`

| Column Name | Type | Description |
|-------------|------|-------------|
| `Child_Parent_Divergence_P_Value` | float | Raw chi-square p-value |
| `Child_Parent_Divergence_P_Value_BH` | float | Tree-BH corrected p-value |
| `Child_Parent_Divergence_Significant` | bool | True if p < alpha (after correction) |
| `Child_Parent_Divergence_df` | float | Degrees of freedom (k) |
| `Child_Parent_Divergence_Invalid` | bool | True if test was invalid |
| `Child_Parent_Divergence_Tested` | bool | True if edge was tested |
| `Child_Parent_Divergence_Ancestor_Blocked` | bool | True if blocked by ancestor |

### Example Output (Sample of Internal Nodes)
```
Node   P_Value    P_Value_BH    df    Significant
N40    0.0833     NaN           1.0   False
N41    0.0833     NaN           1.0   False
N42    0.0618     NaN           2.0   False
N43    0.1342     NaN           2.0   False
N44    0.1342     NaN           2.0   False
N45    0.0266     NaN           2.0   False
N46    0.1133     NaN           2.0   False
N47    0.1133     NaN           2.0   False
N48    0.2354     NaN           2.0   False
N49    0.0618     NaN           2.0   False

Summary:
  Significant edges: 4 / 39 (10.3%)
  Mean p-value: 0.114
  Median p-value: 0.034
```

---

## KEY DESIGN DECISIONS

### 1. Why Projected Wald Test?

**Problem:** High-dimensional data (d=100-10000) with limited samples (n=10-100)

**Solution:** Project onto k-dimensional subspace where k << d
- Reduces noise from irrelevant features
- Maintains statistical power
- Avoids curse of dimensionality

### 2. Why Marchenko-Pastur for k Estimation?

**Problem:** How to choose projection dimension k?

**Solution:** Use random matrix theory
- Separates signal eigenvalues from noise
- Adapts to data structure automatically
- Robust to varying signal strengths

### 3. Why Tree-BH Correction?

**Problem:** Multiple testing across tree edges

**Solution:** Tree-structured Benjamini-Hochberg
- Accounts for hierarchical dependencies
- More powerful than naive BH correction
- Controls FDR at specified level

### 4. Why Nested Factor (1/n_child - 1/n_parent)?

**Problem:** Child is subset of parent (nested samples)

**Solution:** Correct variance for nested structure
- Var(child - parent) = Var(child) + Var(parent) - 2*Cov(child, parent)
- For nested samples: Cov = Var(parent) * (n_child/n_parent)
- Simplifies to: variance * (1/n_child - 1/n_parent)

---

## COMMON ISSUES

### Issue 1: All P-Values = 1.0

**Cause:** spectral_k = 0 (no signal eigenvalues detected)

**Fix:** Check data quality or increase sample size.

### Issue 2: Too Many Significant Edges

**Cause:** EDGE_ALPHA too liberal, or overfitting in spectral estimation

**Fix:** Lower EDGE_ALPHA (e.g., 0.001 → 0.0001), or use more conservative k estimation

### Issue 3: Invalid Tests

**Cause:** Numerical issues (e.g., zero variance features)

**Fix:** Check for degenerate distributions, add regularization to variance computation

---

## RELATED FILES

- `kl_clustering_analysis/hierarchy_analysis/statistics/child_parent_divergence/child_parent_projected_wald/`
- `kl_clustering_analysis/hierarchy_analysis/statistics/child_parent_divergence/child_parent_projected_wald/child_parent_projected_wald_test.py`
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_projection_basis.py`
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/projected_wald/projected_wald_kernel.py`
- `kl_clustering_analysis/hierarchy_analysis/statistics/projection/spectral/marchenko_pastur.py`

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-28
