# Edge Calibration & Projection Dimension Analysis

**Date**: 2026-02-16  
**Data**: 626 × 456 binary GO-term matrix (97.5% zeros)  
**Scripts**: `scripts/diagnose_edge_calibration_and_k.py`, `scripts/diagnose_z_vector_sparsity.py`

---

## 1. Edge Test Calibration: Is Gate 2 Inflated or Deflated?

### Background

The sibling test (Gate 3) uses `cousin_adjusted_wald` to calibrate for post-selection inflation: it identifies "null-like" sibling pairs, fits a regression `log(T/k) = β₀ + β₁·log(BL) + β₂·log(n_parent)`, and deflates focal pairs by the predicted inflation factor ĉ.

The edge test (Gate 2) has **no calibration at all**. On synthetic benchmarks it showed ~15.9% Type I error at nominal α=5% (documented in `edge_significance_and_v1_v2_problems.md`). The question: can we apply the same cousin-style calibration to Gate 2?

### Empirical Findings

Leaf edges (n_child = 1) are the natural "null-like" set for edge calibration — a single observation against its parent has minimal expected divergence.

| Metric     | Leaf edges (empirical) | Expected under H₀ (χ²(456)/456) |
| ---------- | ---------------------- | ------------------------------- |
| Median T/k | **0.044**              | 1.0                             |
| P5 T/k     | 0.000                  | 0.894                           |
| P25 T/k    | 0.022                  | 0.955                           |
| P50 T/k    | 0.044                  | 0.999                           |
| P75 T/k    | 0.079                  | 1.044                           |
| P95 T/k    | 1.796                  | 1.111                           |

The test statistic at leaf edges is **22× smaller than expected** under the null. The raw Type I error rate is 6.7% (42/626 significant at α=0.05), close to the nominal 5% — so there is no post-selection inflation to speak of on this data.

### Why T/k ≈ 0.04 Instead of 1.0

The z-vector is computed from proportions of binary features with 97.5% zeros:

- θ_child,j ∈ {0, 1} (single leaf = single binary observation)
- θ_parent,j ≈ 0.025 (overall rate in sparse data)
- Nested variance: θ_P(1−θ_P)(1/n_c − 1/n_p) ≈ 0.025 × 0.975 × 1.0 = 0.024
- z_j = (0 − 0.025) / √0.024 ≈ −0.16 for the 97.5% of features that are zero

With 97.5% of z_j ≈ −0.16, the sum ∑z_j² ≈ 456 × 0.026 ≈ 12, giving T/k ≈ 0.026. The z-scores are **not N(0,1)** — they're concentrated near zero because the Bernoulli variance θ(1−θ) is tiny for rare features (θ ≈ 0.025).

The CLT assumption z ~ N(0, I_d) breaks down for sparse binary data. Under H₀ (child ⊂ parent, same distribution), the marginal z_j follows:

$$z_j = \frac{\hat\theta_{c,j} - \hat\theta_{p,j}}{\sqrt{\hat\theta_p(1-\hat\theta_p)(1/n_c - 1/n_p)}}$$

When θ ≈ 0.025 and n_child = 1, the numerator takes only values {−0.025, 0.975} and the denominator ≈ 0.155. The distribution is a two-point mass, not Gaussian.

### Calibration Regression (For Reference)

Despite the deflation, the log-linear regression fit well:

```
log(T/k) = −0.014 + 0.941·log(BL) + 0.660·log(n_parent)
R² = 0.983
median ĉ = 0.047
```

This means applying cousin-style calibration would divide T by ĉ ≈ 0.047 (i.e., **multiply** by ~21). This just compensates for the broken variance formula — it's papering over a fundamentally wrong CLT assumption.

### Calibration Verdict

**Cousin-style calibration is NOT the right fix for Gate 2.** The edge test's problem on sparse data is not post-selection inflation but **CLT violation** — the z-scores are not N(0,1). Calibration would mask the underlying issue. The variance formula itself needs fixing, or the test statistic needs to be replaced.

Contrast with Gate 3 (sibling test): there the z-scores are better calibrated because both siblings have n > 1, making the CLT approximation more reasonable. The sibling test's inflation IS genuine post-selection bias, correctly addressed by cousin_adjusted_wald.

---

## 2. Projection Dimension: JL Lemma Is Misapplied

### Current Implementation

`compute_projection_dimension(n_samples, n_features, eps=0.3)` uses sklearn's `johnson_lindenstrauss_min_dim` to compute k from n_child:

| n_child | k_JL | k/d  | Note                        |
| ------- | ---- | ---- | --------------------------- |
| 1       | 10   | 0.02 | Floored at PROJECTION_MIN_K |
| 2       | 77   | 0.17 |                             |
| 3       | 122  | 0.27 |                             |
| 5       | 178  | 0.39 |                             |
| 10      | 255  | 0.56 |                             |
| 20      | 332  | 0.73 |                             |
| 50      | 434  | 0.95 |                             |
| 100+    | 456  | 1.00 | Saturated (k = d)           |

### Why JL Is Wrong Here

The JL lemma states: for **n points** in ℝ^d, random projection to k dimensions preserves **all pairwise distances** within (1±ε) if k ≥ 8 ln(n)/ε².

The edge test projects **one z-vector** and tests ‖Rz‖² ~ χ²(k). There are no "n points" whose pairwise distances need preserving. The parameter n_child enters the JL formula but has **no statistical meaning** in this context.

### Power Analysis: Random vs Informed Projection

The correct question is: what k maximizes power for detecting ‖μ‖² > 0 when z ~ N(μ, I_d)?

For a k × d orthonormal projection R, the projected statistic T_k = ‖Rz‖² ~ χ²(k, λ_k), where λ_k is the projected noncentrality.

**Random (uninformed) projection:**
- λ_k = (k/d) · λ (signal diluted proportionally)
- Power(k) ≈ Φ(λ√k / (d√2) − z_α) — monotonically increasing in k
- **k = d is always optimal** — projection only hurts

**Informed (signal-subspace) projection:**
- λ_k = λ (full signal preserved in k signal directions)
- Power(k) ≈ Φ(λ/√(2k) − z_α) — decreasing in k
- **k = s (signal dimensionality) is optimal**

### Empirical Confirmation

Power-vs-k analysis on real edges confirms the theory:

Root edge (N1250 → N1249, n_child=624, strong signal):
| k   | T_k    | p-value    |
| --- | ------ | ---------- |
| 10  | 52.0   | 1.2×10⁻⁷   |
| 50  | 392.4  | 1.2×10⁻⁵⁴  |
| 100 | 672.4  | 1.2×10⁻⁸⁵  |
| 200 | 1263.0 | 1.2×10⁻¹⁵³ |
| 456 | 2899.0 | 0          |

Root edge (N1250 → N1207, n_child=2, same signal):
| k   | T_k    | p-value   |
| --- | ------ | --------- |
| 10  | 23.4   | 9.5×10⁻³  |
| 50  | 349.1  | 1.9×10⁻⁴⁶ |
| 100 | 514.3  | 5.4×10⁻⁵⁷ |
| 456 | 2899.0 | 0         |

Both edges produce T = 2899 at k = d = 456 (mathematical identity: both children see the same divergence from the parent, just from opposite sides).

---

## 3. Epsilon: Cannot Be Meaningfully Inferred

### Current Setting

ε = 0.3 is hardcoded in `config.PROJECTION_EPS`. It controls the JL distortion tolerance: distances preserved within (1±ε).

### Data-Driven Back-Inference

If we wanted JL to produce k = effective_rank = 112, we'd need:

$$\varepsilon = \sqrt{\frac{8 \ln n}{k}} = \sqrt{\frac{8 \ln 626}{112}} \approx 0.678$$

This is a nearly meaningless distortion guarantee (68% error in distance preservation).

### Variance-to-Dimension Tradeoff

| k   | Variance captured | Effective ε | JL ε for k (n=626) |
| --- | ----------------- | ----------- | ------------------ |
| 10  | 34.8%             | 0.652       | 2.270              |
| 50  | 68.4%             | 0.316       | 1.015              |
| 100 | 84.4%             | 0.156       | 0.718              |
| 133 | 90.0%             | 0.100       | 0.623              |
| 200 | 96.2%             | 0.039       | 0.508              |
| 456 | 100.0%            | 0.000       | 0.336              |

### Verdict

**ε is a distortion guarantee, not a power parameter.** For the single-vector test used in Gate 2, ε has no direct statistical meaning. If JL is replaced with data-driven k selection, ε becomes irrelevant. Tuning ε is treating the wrong knob.

---

## 4. Data-Driven Dimension Estimates

### Eigenvalue Spectrum

The sample covariance of the 626 × 456 binary matrix has a gently decaying eigenvalue spectrum (no sharp elbow):

- Top eigenvalue: λ₁ = 0.816 (7.7% of total variance)
- 50% variance: k = 23 components
- 75% variance: k = 66 components
- 90% variance: k = 133 components
- 95% variance: k = 183 components
- 99% variance: k = 279 components

### Dimension Estimates

| Method                           | k       | Rationale                                                                                          |
| -------------------------------- | ------- | -------------------------------------------------------------------------------------------------- |
| Effective rank (Shannon entropy) | **112** | exp(−Σ p_i log p_i) where p_i = λ_i/Σλ_j. Continuous dimensionality (Roy & Vetterli, 2007)         |
| Marchenko-Pastur threshold       | **133** | Eigenvalues above MP upper bound (1+√(d/n))² · σ̂². Signal eigenvalues vs noise bulk                |
| Parallel analysis (Horn 1965)    | **1**   | Eigenvalues exceeding 95th percentile of random reference. Too conservative for sparse binary data |
| Full dimension                   | **456** | No projection — always optimal for random (uninformed) projection                                  |

Effective rank (112) and MP threshold (133) are consistent, suggesting ~120 "meaningful" dimensions.

---

## 5. Z-Vector Sparsity Structure

From `diagnose_z_vector_sparsity.py`, the root edge z-vector reveals:

- ‖z‖² = 2899 (expected under H₀: 456)
- max|z| = 17.66, median|z| = 0.16
- **2.6%** of features (12/456) capture **50%** of T
- **7.9%** of features (36/456) capture **90%** of T
- **87%** of features have |z_j| < 1 — pure noise

This is a textbook **sparse normal means** problem: a few features carry strong signal, most are noise. The sum-of-squares test T = Σz_j² is suboptimal because 400 noise features each contribute ~z_j² ≈ 0.026 (deflated) to both the statistic and the null threshold.

### Test Statistic Comparison (z-vector sparsity diagnostic)

On 17 representative edges at α=0.05:

| Method                           | Edges significant | Note                     |
| -------------------------------- | ----------------- | ------------------------ |
| Wald χ²(d)                       | 14/17             | Current approach         |
| Higher Criticism                 | 15/17             | Catches sparse signal    |
| Power Enhancement (Fan-Liao-Yao) | **15/17**         | Catches edge Wald misses |
| Cauchy Combination (ACAT)        | 12/17             | Worse — noise dominates  |
| Oracle (invalid,                 | z                 | >2 only)                 | 16/17 | Upper bound |

Critical edge: **N1216 → N1202** (n=14 leaves)
- Wald χ²(456): T = 153.3, **p = 1.0** (MISS)
- Power Enhancement: J = 6.51, **p = 3.8×10⁻¹¹** (HIT)
- Oracle (11 features with |z|>2): T = 119.5, **p = 2.3×10⁻²⁰** (HIT)

---

## 6. Recommendations

### 6.1 Projection Dimension

**Replace JL-based k with data-driven k:**

For **random** (uninformed) projection: use k = d always. Projection only hurts power.

For **informed** (PCA-based) projection:
```
k* = effective_rank(Σ̂)  or  MP_threshold(λ₁,...,λ_d; n, d)
```
where effective_rank = exp(−Σ p_i log p_i) ≈ 112 for this data. The projection would be onto the top-k eigenvectors of the sample covariance, not random directions.

**n_child should NOT enter the dimension formula.** The JL lemma addresses pairwise distance preservation among n points — irrelevant for a single-vector test.

### 6.2 Epsilon

**ε becomes irrelevant** once JL is replaced with data-driven k. If JL is retained for compatibility, the back-inferred ε ≈ 0.68 (to match effective rank = 112) has no meaningful interpretation as a distortion guarantee.

### 6.3 Edge Calibration (Gate 2)

**Cousin-style calibration is not appropriate for Gate 2.** The leaf-edge T/k = 0.044 (22× below expected) reveals a CLT violation, not post-selection inflation. Calibrating would multiply T by ~21, masking the fundamental problem.

The fix must address one of:
1. **Fix the variance formula** — account for the discrete binary nature of sparse features so z actually IS N(0,1), then use k = d
2. **Replace the test statistic** — use Power Enhancement (Fan, Liao & Yao 2015) which adds a screening component J₀ = Σ_{|z_j|>τ}(z_j² − τ²) that catches sparse signal without requiring each z_j to be N(0,1)
3. **Use informed projection** — PCA-based projection to the signal subspace (k ≈ 112–133) preserves signal while removing noise dimensions

### 6.4 Felsenstein Scaling

With leaf T/k already at 0.044 (raw) vs 0.017 (Felsenstein), the Felsenstein factor further deflates an already-deflated statistic by another ~2.5×. On sparse real data, **Felsenstein makes an under-powered test even weaker**. For data-dependent trees (where branch length = cophenetic distance = signal), Felsenstein scaling is fundamentally misapplied (see `edge_significance_and_v1_v2_problems.md`).

---

## 7. Open Questions

1. **Which fix path?** Fix A (variance formula) requires understanding exact finite-sample distribution of z_j for Bernoulli(θ) with small θ. Fix B (Power Enhancement) has known null N(0,1) but requires threshold τ calibration. Fix C (informed projection) requires computing sample covariance eigenvectors — O(d²n) once per tree.

2. **Interaction with sibling test**: The sibling test uses the same JL-based projection dimension. Does the same analysis apply? Sibling z-scores may be better calibrated (both children have n > 1) but still suffer from the JL misapplication.

3. **Benchmark regression**: Any change to projection dimension or test statistic will affect synthetic benchmark results. Need to verify that well-performing cases (Mean ARI 0.757, Exact K 59/95) are preserved.
