# Gamma GLM Upgrade for Cousin-Weighted Wald Calibration

## Status: IMPLEMENTED (2026-02-16)

The Gamma GLM upgrade has been implemented in `cousin_weighted_wald.py`. The current sibling method `"cousin_weighted_wald"` uses `statsmodels.api.GLM` with a Gamma family and log link by default. This document preserves the original rationale and adds empirical observations.

## Why the log-normal WLS was wrong

Within a purely parametric framework, the main improvement is switching from **log-normal WLS** to a **Gamma GLM with log link**, which matches the actual distribution of $T/k$.

Under the null, $T \sim c \cdot \chi^2(k)$, so $r = T/k$ has:
- $E[r] = c$
- $\text{Var}(r) = 2c^2/k$

The original WLS took $\log(r)$ and fitted OLS — this implicitly assumed **log-normal** errors. But:

1. $E[\log(r)] \neq \log(E[r])$ — Jensen's inequality introduces a downward bias: $\hat{c} = \exp(\hat{\beta}_0 + ...) < E[r]$. The bias is $\approx -1/k$, which is non-negligible for small projection dimensions.

2. The variance of $\log(r)$ is $\text{Var}(\log(r)) \approx 2/k$ (constant in $c$), but WLS treats all residuals with the same error model. The log-transform accidentally makes the variance independent of $c$, which is good — but the mean bias remains.

## The fix: Gamma GLM

The Gamma family has variance function $V(\mu) = \mu^2$, which matches $\text{Var}(r) = 2c^2/k \propto c^2$. With a log link:

$$\log E[T_i/k_i] = \beta_0 + \beta_1 \log(BL_i) + \beta_2 \log(n_i)$$

```python
import statsmodels.api as sm

X = np.column_stack([
    np.ones(n_cal),
    np.log(bl_sums),
    np.log(n_parents.astype(float)),
])

# Gamma GLM with log link, frequency weights
model = sm.GLM(
    ratios,
    X,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    freq_weights=weights,
)
result = model.fit()
beta = result.params
```

This gives you:
- **Unbiased** $\hat{c}$ predictions (no Jensen's bias)
- **Correct variance model** — larger $c$ → larger variance, naturally handled
- **Proper standard errors** on $\beta$ — can assess which covariates matter
- **Deviance residuals** for model diagnostics

## Empirical Observations on Real Data (2026-02-16)

### Gamma GLM calibration on 626 × 456 GO-term matrix

The Gamma GLM was fit on sibling divergence data from a real sparse binary matrix (97.5% zeros):

| Calibration parameter | Value |
|----------------------|-------|
| `calibration_method` | `gamma_glm` |
| `global_c_hat` | **0.376** |
| `calibration_n` | 597 (out of 625 total pairs) |

**Key finding: ĉ < 1 means the calibration is DEFLATING, not inflating.**

Under the post-selection inflation hypothesis, we expect ĉ > 1 (tree construction inflates test statistics). But ĉ = 0.376 means `T_adj = T / 0.376 ≈ 2.66 × T` — the calibration **amplifies** the test statistic, making it easier to reject H₀. This is the opposite of what the documentation and motivation predict.

**Why ĉ < 1 occurs on sparse data:**

The null-like calibration pairs (597/625 — almost all pairs!) have T/k ratios that are systematically below 1.0. This is because:
1. Sparse features (θ ≈ 0.025) produce z-scores near zero for random child subsets — the signal is too weak for z to deviate from 0
2. The Felsenstein variance inflation on long branches further pushes T/k below 1.0
3. Near-zero θ values violate the CLT assumption underlying the Wald test: $z_i = (\hat\theta_c - \hat\theta_p) / \sqrt{Var}$ is NOT normally distributed when θ is near 0 or 1

**Consequence:** The Gamma GLM is doing the right thing given its inputs — the T/k ratios really are systematically below 1. But this means the calibration is correcting a symptom (suppressed T/k) rather than the root cause (sparse data + Felsenstein overcorrection). The deflation partially compensates for the under-powered edge test, but only for the sibling test (Gate 3), not for the edge test (Gate 2) which remains uncalibrated.

### But Gate 2 blocks everything before Gate 3 matters

Even with ĉ = 0.376 boosting sibling test statistics, it doesn't help because Gate 2 (edge significance) fails at the root and at most internal nodes:
- Only 61/624 edges pass Gate 2 after BH correction
- 583/625 sibling tests are **skipped** because Gate 2 failed for both children
- Only 6/625 sibling tests show `Sibling_BH_Different = True`

**The calibration is irrelevant when Gate 2 is the bottleneck.** The pipeline needs Gate 2 calibration, not just Gate 3.

## Additional covariates worth adding

The current 2-feature model can miss structure. Consider:

| Feature                     | Rationale                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| $\log(n_L/n_R)$ (imbalance) | Unbalanced splits inflate $T$ differently — the small child dominates the pooled variance |
| $\log(k)$ (projection dim)  | The JL projection dimension varies across pairs; inflation may interact with $k$          |
| tree depth                  | Deeper nodes have more selection events in their ancestry                                 |

```python
X = np.column_stack([
    np.ones(n_cal),
    np.log(bl_sums),
    np.log(n_parents.astype(float)),
    np.log(imbalance_ratios),      # n_left / n_right, always ≥ 1
    np.log(proj_dims.astype(float)), # k_i
])
```

## Fallback tiers (same structure, better defaults)

The current fallback from regression → median → raw is fine. For the Gamma GLM, the fallback when it fails to converge is the **weighted harmonic mean** of ratios (which is the maximum likelihood estimate for a Gamma with constant shape):

```python
# Gamma MLE for constant c: harmonic-mean-based
global_c = float(np.average(ratios, weights=weights))  # arithmetic mean
# or more robust:
global_c = float(np.exp(np.average(np.log(ratios), weights=weights)))  # geometric mean
```

## Summary of changes

| Original (WLS)             | Current (Gamma GLM)                  |
| -------------------------- | ------------------------------------ |
| `np.log(ratios)` → OLS     | `ratios` → Gamma GLM (log link)      |
| Jensen's bias in $\hat{c}$ | Unbiased via GLM                     |
| 2 covariates               | 2 covariates (BL, n) — imbalance/k TBD |
| `np.linalg.lstsq`          | `sm.GLM(...).fit()`                  |
| Weighted median fallback   | Weighted arithmetic mean fallback    |

## Open Questions (2026-02-16)

1. **Should ĉ be clamped to ≥ 1?** Currently no — ĉ < 1 is allowed and effectively boosts the sibling test. But if the intent is only to correct post-selection *inflation*, values < 1 indicate model misspecification on sparse data.

2. **Should Gate 2 also be calibrated?** The cousin-weighted approach only calibrates Gate 3. Gate 2 has no calibration at all — it relies on raw projected Wald with BH correction. On real data this is the primary bottleneck. Cross-fit permutation calibration (proposed) would address this.

3. **Is 597/625 null-like calibration pairs correct?** When almost ALL pairs are null-like, the calibration set is dominated by pairs where neither child has signal. This means ĉ estimates the "inflation factor" for genuinely null data — which is ĉ < 1 because the Wald test is conservative on sparse data, not because post-selection inflation is absent.
