# Spectral Dimension for Gate 3 — Investigation Report

**Date**: 2026-03-17  
**Experiments**: exp15–exp19  
**Scope**: Determine optimal projection dimension (k) derivation for the Gate 3 sibling divergence test  

---

## 1. Problem Statement

When `SPECTRAL_METHOD = "marchenko_pastur"` was introduced for Gate 2 (edge test), the
per-node spectral dimensions were automatically forwarded to Gate 3 (sibling test) via
`_derive_sibling_spectral_dims()` in `orchestrator.py`. This function computes
`k = min(k_left, k_right)` — the minimum of the two children's Marchenko-Pastur (MP)
signal counts from Gate 2.

**The regression**: MP applied to *within-child* data almost always yields k ≈ 0–2 because
individual children are small, homogeneous subtrees. A χ²(2) test has drastically less power
than the previous JL-based dimension (typically k ≈ 10–30). This caused Gate 3 to
fail to detect true cluster boundaries → under-splitting → K=1 collapses.

**Severity**: Mean ARI dropped from 0.979 (JL baseline) to 0.815 (min-child spectral) across
15 sentinel cases. Seven cases collapsed to K=1 (ARI=0).

---

## 2. Experiment Chronology

### Exp 15 — Spectral Dimension Regression Diagnosis

**Goal**: Confirm min-child spectral k as the regression root cause.  
**Method**: Per-node trace comparing auto-derived spectral dims (Mode A) vs JL fallback (Mode B).  
**Key finding**: Decision flips at true boundaries — nodes that should SPLIT under JL (k ≈ 12)
instead MERGE under spectral (k = 2). The sibling p-value shifts from p < 0.01 to p > 0.5
when k drops from 12 to 2.

| Case                         | Auto-derive ARI | JL fallback ARI |
| ---------------------------- | --------------- | --------------- |
| binary_balanced_low_noise__2 | 0.000           | 1.000           |
| gauss_clear_small            | 0.554           | 1.000           |
| binary_low_noise_12c         | 0.614           | 1.000           |
| binary_perfect_8c            | 0.757           | 1.000           |
| binary_hard_4c               | 0.708           | 0.950           |

### Exp 16 — Strategy Comparison (8 strategies)

**Goal**: Find a better k derivation than both min-child and raw JL.  
**Strategies tested**:

| Strategy          | Formula                    | Mean ARI               |
| ----------------- | -------------------------- | ---------------------- |
| **jl_floor_qrt**  | max(min(k_L, k_R), ⌈JL/4⌉) | **0.991**              |
| **jl_floor_half** | max(min(k_L, k_R), ⌈JL/2⌉) | 0.988                  |
| none (pure JL)    | ⌈8·ln(n)/ε²⌉               | 0.979                  |
| sum_child_2x      | 2 × (k_L + k_R)            | 0.981                  |
| max_child_2x      | 2 × max(k_L, k_R)          | 0.976                  |
| sum_child         | k_L + k_R                  | 0.968                  |
| max_child         | max(k_L, k_R)              | 0.949                  |
| **min_child**     | min(k_L, k_R)              | **0.815** ← regression |

**Finding**: `jl_floor_qrt` = max(spectral_min, JL/4) achieves 0.991, outperforming both
pure JL (0.979) and all pure spectral strategies. The JL/4 floor provides minimum power
while spectral caps prevent noise dilution on high-dimensional data.

### Exp 17 — Literature Strategies (12 additional)

**Goal**: Test strategies from recent high-dimensional testing literature.  
**New strategies** (in addition to exp16 baselines):

| Strategy        | Source                                    | Mean ARI |
| --------------- | ----------------------------------------- | -------- |
| pooled_mp       | MP on pooled L+R data                     | 0.968    |
| effective_rank  | Shannon entropy rank                      | 0.968    |
| ncp_power_opt   | Pick k maximizing χ² power (Steiger 1985) | 0.968    |
| diff_signal     | Significant components in mean difference | 0.968    |
| k_parent        | MP signal count of parent node            | 0.815    |
| k_parent_jl     | max(k_parent_MP, JL/4)                    | 0.991    |
| jl_floor_8th    | max(spectral_min, JL/8)                   | 0.976    |
| jl_spectral_geo | √(JL × spectral_min)                      | 0.968    |

**Finding**: No literature strategy beat `jl_floor_qrt`. Even `k_parent_jl` only ties it —
the JL/4 floor again does the heavy lifting. Pure spectral strategies (pooled_mp,
effective_rank, diff_signal) all land at 0.968, a ceiling for non-floored approaches.

### Exp 18 — Eigenvector Relationship Mapping

**Goal**: Understand eigenspace structure between parent and children at split/merge boundaries.  
**Method**: For every binary parent, compute eigenvalues, effective rank, subspace overlaps
(Grassmann distance), and principal angles between child and parent signal spaces.

**Key discoveries**:

| Feature                | SPLIT (median) | MERGE (median) | Separation |
| ---------------------- | -------------- | -------------- | ---------- |
| Parent k_MP            | 5.0            | 1.0            | 0.770      |
| k_P / (k_L + k_R)      | 1.39           | 0.66           | —          |
| er_P / (er_L + er_R)   | 0.53           | 0.94           | 0.971      |
| Sibling overlap        | 0.31           | 0.78           | —          |
| ∠ LP (principal angle) | 48°            | 22°            | —          |
| ∠ LR (sibling angle)   | 67°            | 33°            | —          |

**Interpretation**: At SPLIT nodes, the parent gains *new* signal dimensions not present in
either child (ratio > 1), and child subspaces are nearly orthogonal (67° apart). At MERGE
nodes, children account for most of the parent's variance (ratio ≈ 1) and overlap strongly.

### Exp 19 — Spectral Equation Laboratory (16 equations, 3 phases)

**Goal**: Exhaustively test spectral-feature-to-k mappings.

#### Phase A: Feature Distributions (1069 nodes: 39 SPLIT, 1030 MERGE)

Best SPLIT/MERGE separators (standardized):

| Feature         | SPLIT median | MERGE median | Separation |
| --------------- | ------------ | ------------ | ---------- |
| Parent λ₁       | 16.7         | 1.7          | **1.524**  |
| √n_parent       | 7.75         | 3.00         | **1.049**  |
| er_P / er_sum   | 0.53         | 1.00         | **0.971**  |
| log(n)·log(d)/2 | 9.86         | 4.10         | **0.874**  |
| JL/4            | 20.0         | 5.0          | 0.776      |
| Parent k_MP     | 5.0          | 1.0          | 0.767      |
| knee_P / k_P    | 0.71         | 3.00         | 0.772      |

#### Phase B: Equation Evaluation

| Equation     | Formula               | Corr w/ JL/4 | k>2 at SPLIT |
| ------------ | --------------------- | ------------ | ------------ |
| jl_qrt       | max(2, JL/4)          | 1.000        | 100%         |
| knee_jl      | max(2, knee_P, JL/4)  | 0.956        | 100%         |
| var90_jl     | max(2, var90_P, JL/4) | 0.854        | 100%         |
| er_jl        | max(2, er_P, JL/4)    | 0.802        | 100%         |
| k_parent     | max(2, k_parent_MP)   | 0.626        | 66.7%        |
| gap_weighted | max(2, k_P × gap)     | 0.737        | 76.9%        |

#### Phase C: Full Benchmark (15 cases)

| Rank | Strategy         | Mean ARI  | Wins/15 | Formula                 |
| ---- | ---------------- | --------- | ------- | ----------------------- |
| 1    | **jl_floor_qrt** | **0.991** | 11      | max(spectral_min, JL/4) |
| 2    | knee_jl          | 0.989     | 11      | max(knee_P, JL/4)       |
| 3    | er_jl            | 0.984     | 12      | max(er_P, JL/4)         |
| 3    | var90_jl         | 0.984     | 12      | max(var90_P, JL/4)      |
| 5    | none (pure JL)   | 0.979     | 11      | ⌈8·ln(n)/ε²⌉            |
| 6    | er_parent        | 0.968     | 11      | round(er_P)             |
| 6    | var90 / var95    | 0.968     | 11      | 90%/95% variance dims   |
| 6    | er_child_sum     | 0.968     | 11      | round(er_L + er_R)      |
| 6    | harm_jl_er       | 0.968     | 11      | 2·JL·er/(JL+er)         |
| 12   | geom_jl_knee     | 0.963     | 10      | √(JL × knee)            |
| 13   | log_n_d          | 0.965     | 10      | log(n)·log(d)/2         |
| 14   | sqrt_n           | 0.952     | 9       | √n_parent               |
| 15   | gap_weighted     | 0.907     | 7       | k_P × spectral_gap      |
| 16   | lam_ratio        | 0.845     | 6       | λ₁ / MP_bound           |
| 17   | k_parent         | 0.815     | 8       | MP count (parent)       |
| 18   | knee_parent      | 0.743     | 5       | elbow k only            |

---

## 3. Conclusions

### Why `jl_floor_qrt` is optimal

The JL quarter-floor `max(spectral_min, ⌈8·ln(n)/ε²⌉/4)` works because it solves two
failure modes simultaneously:

1. **Power floor** (JL/4): Prevents k from falling below the minimum needed for
   statistical power. At small nodes (n < 20), MP spectral counts are unreliable
   (typically 0–2). JL/4 provides k ≈ 5–15 depending on n, ensuring χ²(k) has
   enough degrees of freedom to detect real differences.

2. **Noise cap** (spectral_min): On noisy/high-dimensional data where JL would give
   k ≫ n, the spectral estimate caps k at the true signal dimensionality. This
   prevents adding pure-noise components to the χ² statistic that absorb degrees
   of freedom without contributing signal.

In practice, the floor dominates: JL/4 > spectral_min for ~80% of nodes. The spectral
cap activates mainly at high-n internal nodes where k_MP is large and meaningful.

### Why pure spectral strategies fail

MP eigendecomposition on small within-child samples systematically underestimates the
number of signal dimensions — this is a fundamental sample-size limitation, not a bug
in the MP estimator. At n = 5, the Marchenko-Pastur upper bound is very high, and
only the absolutely dominant eigenvalue (if any) escapes the noise bulk.

The parent's eigenspace is richer than either child's (SPLIT ratio k_P/(k_L+k_R) = 1.39),
but even parent MP counts are too conservative for the power requirements of the χ² test.

### Recommendation

**Keep `jl_floor_qrt` as the production strategy.** No pure spectral or hybrid equation
improves on it. The alternatives `knee_jl` (0.989) and `er_jl/var90_jl` (0.984) are
marginally worse and add computational overhead (eigendecomposition at every node for
elbow detection / variance explained estimation).

If a more "principled" strategy is desired for theoretical motivation, `knee_jl`
(max of elbow-point-k and JL/4) is essentially equivalent and has a clear interpretation:
use the data-driven elbow of the scree plot when it's informative, fall back to JL/4
when the sample is too small for reliable elbow detection.

---

## 4. Experimental Infrastructure

All experiments use common infrastructure from `lab_helpers.py`:
- **15 sentinel cases** spanning failures (K=1 collapses), regression guards (ARI=1.0),
  and intermediates (0.3 ≤ ARI < 0.8)
- `build_tree_and_data()` → generate data, build linkage, populate distributions
- `run_decomposition()` → decompose with strategy overrides
- Results compared via ARI (adjusted Rand index) against ground truth

### Test Case Summary

| Category          | Count | Purpose                                 |
| ----------------- | ----- | --------------------------------------- |
| Failure cases     | 5     | ARI < 0.3 under min-child; confirm fix  |
| Regression guards | 5     | ARI = 1.0 under JL; must not degrade    |
| Intermediates     | 5     | Partial successes; room for improvement |

---

## 5. Open Questions

1. **Production integration**: `_derive_sibling_spectral_dims()` still implements min-child.
   Applying `jl_floor_qrt` requires modifying the orchestrator to combine the spectral
   estimate with a JL/4 floor before passing to the sibling gate.

2. **Full benchmark validation**: The 15 sentinel cases are representative but not exhaustive.
   A full 95-case benchmark run should confirm no regressions before production deployment.

3. **Adaptive floor**: Could the JL fraction (currently 1/4) be tuned based on data
   characteristics (sparsity, effective rank)? Current evidence suggests 1/4 is robust
   across all tested scenarios — further tuning risks overfitting to the benchmark.
