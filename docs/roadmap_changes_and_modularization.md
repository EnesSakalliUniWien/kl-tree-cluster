# Roadmap: Necessary Changes & Modularization

This document maps all identified issues to concrete code changes. Changes are grouped into three tiers and ordered by dependency.

**Last updated**: 2026-02-16, incorporating empirical findings from real-world sparse data (626 × 456 GO-term matrix).

---

## Tier 0 — Critical: Edge Significance (Gate 2) Calibration

> **Added 2026-02-16**: Gate 2 is the primary bottleneck on real sparse data (9.8% detection rate, K=1 collapse). All Tier 1–3 improvements are irrelevant if Gate 2 blocks the traversal at the root.

### 0.1 Felsenstein overcorrection on outlier branches

**Issue**: Average-linkage trees from sparse binary data produce highly asymmetric root splits with long outlier branches (`BL/mean_BL > 4`). The Felsenstein multiplier `1 + BL/mean_BL = 5.29×` kills genuine edge signal, converting significant z-scores (|z| > 17) into non-significant test results (p = 0.14). The Felsenstein formula assumes branch lengths reflect neutral evolutionary time in a fixed phylogeny, but our trees are data-dependent.

**File**: `edge_significance.py` — `_compute_standardized_z()` (line ~55)

**Options**:
1. **Cap the Felsenstein multiplier**: `bl_normalized = min(1 + BL/mean_BL, cap)` where `cap ∈ [2, 3]` prevents extreme variance inflation. Simple but ad-hoc.
2. **Use median instead of mean for normalization**: `mean_BL` is dominated by many short internal edges; `median_BL` would be more robust to outlier branches.
3. **Disable Felsenstein for data-dependent trees**: Since the branch lengths come from average-linkage (not evolutionary inference), the Felsenstein adjustment may be inappropriate entirely. Skip it when the tree source is `from_linkage()` vs. a true phylogeny.
4. **Cross-fit permutation calibration**: Build the tree on one data fold, test on the other. This addresses post-selection bias at its root, making both Felsenstein and the cousin-weighted calibration unnecessary.

**Depends on**: Nothing (standalone fix).

---

### 0.2 Projection dimension saturation (k = d)

**Issue**: With ε = 0.3 and d = 456 features, `compute_projection_dimension(n, d)` returns k = 456 (full dimension) for any n ≥ 100. The "random projection" is an identity transform — no dimensionality reduction occurs. The test becomes T ~ χ²(456) which requires T > 520 to reject at α = 0.05, very conservative for signal spread across many sparse features.

**File**: `random_projection.py` — `compute_projection_dimension()` (line ~65)

**Options**:
1. **Use a more aggressive ε**: ε = 0.5 gives k ≈ 250 for n = 626, providing actual compression. May increase false negatives on weak signals.
2. **Cap k at a fraction of d**: `k = min(k_jl, d // 2)` ensures the projection always compresses. Loses JL guarantee but ensures the projection does something.
3. **Adaptive ε based on sparsity**: When features are very sparse (mean θ < 0.05), the signal-to-noise is low per feature and compression may help by concentrating it.
4. **Use a different test for k ≈ d**: When k = d, the projected Wald reduces to a sum of z² which is equivalent to the standard multivariate Wald test. Consider testing a subset of informative features instead.

**Depends on**: Nothing.

---

### 0.3 BH correction includes untestable edges

**Issue**: The BH correction runs over ALL 1250 edges including ~626 leaf-edges with p = 1.0. This inflates the denominator m by ~2×, halving the effective significance threshold for non-trivial edges. Leaf edges are definitionally untestable (leaves have no children to diverge from a parent's distribution in any meaningful way for clustering) and should be excluded.

**File**: `edge_significance.py` — `annotate_child_parent_divergence()` (line ~310)

**Change**: Filter out leaf edges before BH correction:
```python
# BEFORE: correct over ALL edges
reject_null, p_values_corrected = apply_multiple_testing_correction(
    p_values=p_values_for_correction, ...)

# AFTER: correct only over non-leaf internal edges
is_internal = np.array([tree.out_degree(cid) > 0 for cid in child_ids])
internal_p = p_values_for_correction[is_internal]
internal_reject, internal_corrected = apply_multiple_testing_correction(
    p_values=internal_p, ...)
# Map back to full array
reject_null = np.zeros(n_edges, dtype=bool)
reject_null[is_internal] = internal_reject
p_values_corrected = np.ones(n_edges)
p_values_corrected[is_internal] = internal_corrected
```

**Risk**: This technically changes the hypothesis set and could increase false positives on synthetic benchmarks where the original BH correction was adequate. Need to re-run benchmarks.

**Depends on**: Nothing.

---

### 0.4 Gate 2 calibration (analogous to Gate 3)

**Issue**: Gate 3 (sibling test) has cousin-weighted Wald calibration to correct post-selection inflation. Gate 2 (edge test) has **no calibration at all**. On synthetic benchmarks this causes 15.9% Type I error; on real sparse data it causes 90.2% false negative rate. The edge test needs its own calibration.

**Options**:
1. **Cross-fit permutation**: Split data into two folds. Build tree on fold 1, compute edge statistics on fold 2. This eliminates post-selection bias for both gates simultaneously. Most principled but requires architectural changes.
2. **Cousin-style edge calibration**: Use null-like edges (where children's distributions are not meaningfully different from parent) to estimate the edge-test inflation factor, then deflate all edge statistics. Analogous to what cousin-weighted Wald does for the sibling test.
3. **Permutation null for edge test**: For each edge, permute leaf assignments between child and sibling, recompute T, estimate the null distribution empirically. Computationally expensive but exact.

**Depends on**: Major design decision — affects all downstream components.

---

## Tier 1 — Correctness Fixes (must-do before any benchmark)

### 1.1 Post-hoc merge test asymmetry

**Issue**: Decomposition uses cousin-weighted-calibrated $T_{\text{adj}} = T/\hat{c}$, but post-hoc merge calls raw `sibling_divergence_test()` via `_test_cluster_pair_divergence()` — inflated $T$. Harder to merge than to split.

**Files**:
- `tree_decomposition.py` — `_test_cluster_pair_divergence()` (line ~658)
- `cousin_weighted_wald.py` — `_WeightedCalibrationModel` (line ~112)

**Change**:
1. After annotation, store the fitted calibration model: `self._calibration_model = df.attrs["sibling_divergence_audit"]`
2. In `_test_cluster_pair_divergence()`, deflate the raw stat by `ĉ` before returning:
   ```python
   stat, df, pval = sibling_divergence_test(...)
   c_hat = _predict_c(self._calibration_model, bl_sum, n_ancestor)
   stat_adj = stat / c_hat
   pval_adj = chi2.sf(stat_adj, df)
   return stat_adj, df, pval_adj
   ```
3. Expose `_predict_c` and the model dataclass from `cousin_weighted_wald.py` via `__all__`.

**Depends on**: Nothing.

---

### 1.2 Post-hoc merge descendant removal absorbs unrelated clusters

**Issue**: `cluster_roots -= nx.descendants(tree, lca)` removes ALL cluster roots under the LCA, not just the two being merged. A third cluster under the same LCA is silently absorbed.

**File**: `posthoc_merge.py` (line ~183)

**Change**: Replace blanket descendant removal with targeted removal of only the two merged clusters:
```python
# BEFORE (absorbs everything):
cluster_roots -= nx.descendants(tree, lca)
cluster_roots.discard(lca)
cluster_roots.add(lca)

# AFTER (targeted):
cluster_roots.discard(lc)
cluster_roots.discard(rc)
cluster_roots.add(lca)
```

**Risk**: Must verify the antichain invariant still holds — the LCA must be an ancestor of both `lc` and `rc` but NOT of any other remaining cluster root. Add an assertion:
```python
remaining_under_lca = [r for r in cluster_roots if r != lca and r in nx.descendants(tree, lca)]
assert not remaining_under_lca, f"Antichain violation: {remaining_under_lca} still under {lca}"
```

**Depends on**: Nothing.

---

### 1.3 Remove LCA-blocking in post-hoc merge

**Issue**: If any pair under an LCA rejects $H_0$, ALL merges under that LCA are blocked — even unrelated similar pairs.

**File**: `posthoc_merge.py` (lines ~155–165)

**Change**: Remove the `lca_has_reject` logic entirely. Let BH correction handle multiplicity; block only the specific pairs that reject:
```python
# BEFORE: Skip pairs whose LCA already has a rejection
mergeable_indices = [
    i for i, r in enumerate(reject)
    if (not r) and (not lca_has_reject.get(pairs[i]["lca"], False))
]

# AFTER: Simply use non-rejected pairs
mergeable_indices = [i for i, r in enumerate(reject) if not r]
```

**Depends on**: 1.2 (targeted descendant removal must be in place first, otherwise removing LCA-blocking + blanket descendant removal would be even more destructive).

---

## Tier 2 — Statistical Improvements

### 2.1 ✅ Gamma GLM for calibration regression — DONE (2026-02-14)

Implemented as `cousin_weighted_wald.py` with Gamma GLM via `statsmodels`. The weighted Wald calibration uses `sm.GLM(ratios, X, family=Gamma(link=Log()))` with branch-length sum and n_parent as covariates.

**Empirical finding (2026-02-16)**: On real sparse data (626 × 456 GO-terms), the calibration produces ĉ = 0.376 (DEFLATING, not inflating). 597/625 pairs are classified as null-like, meaning almost all edges fail Gate 2. The calibration model is working correctly given its inputs, but Gate 2 is the upstream bottleneck — see Tier 0.

---

### 2.2 ✅ Add `statsmodels` dependency — DONE

`statsmodels` is already in `requirements.txt` and `pyproject.toml`.

---

## Tier 3 — Modularization

The current `tree_decomposition.py` is 887 lines with 19 methods in a single class. Several concerns are mixed together that should be separated.

### 3.1 Extract calibration model as a first-class object

**Current state**: The calibration model lives inside `cousin_weighted_wald.py` as a private `_WeightedCalibrationModel` dataclass, returned only as audit metadata in `df.attrs`.

**Problem**: Tier 1.1 needs the model to deflate post-hoc merge tests. The current design buries it in DataFrame attrs, making it hard to thread through.

**Change**: 
1. Make `_WeightedCalibrationModel` public → `CalibrationModel`
2. Make `_predict_c` public → `predict_inflation_factor(model, bl_sum, n_parent)`
3. Store the model on `TreeDecomposition` as `self._calibration_model`
4. New file: `statistics/sibling_divergence/calibration_model.py` — holds the dataclass and prediction logic (shared by both annotation and post-hoc merge)

**New module structure**:
```
statistics/sibling_divergence/
    calibration_model.py      # NEW — CalibrationModel, predict_inflation_factor
    cousin_weighted_wald.py   # imports from calibration_model.py
    sibling_divergence_test.py
    cousin_calibrated_test.py
    cousin_adjusted_wald.py
    cousin_tree_guided.py
```

---

### 3.2 Extract annotation dispatch from `TreeDecomposition.__init__`

**Current state**: `_prepare_annotations()` contains a 40-line if/elif chain dispatching on `config.SIBLING_TEST_METHOD`.

**Problem**: Adding a new sibling test method requires editing the decomposer class. The class has 887 lines and mixes annotation orchestration with traversal logic.

**Change**: Create a registry pattern:
```
hierarchy_analysis/
    annotation_dispatch.py    # NEW — sibling_test_registry, prepare_annotations()
```

```python
# annotation_dispatch.py
_SIBLING_ANNOTATORS = {
    "wald": "annotate_sibling_divergence",
    "cousin_ftest": "annotate_sibling_divergence_cousin",
    "cousin_adjusted_wald": "annotate_sibling_divergence_adjusted",
    "cousin_tree_guided": "annotate_sibling_divergence_tree_guided",
    "cousin_weighted_wald": "annotate_sibling_divergence_weighted",
}

def prepare_annotations(tree, results_df, alpha_local, sibling_alpha, method):
    """Run Gate 2 + Gate 3 annotations, returning annotated DataFrame."""
    ...
```

`TreeDecomposition._prepare_annotations()` becomes a one-liner:
```python
def _prepare_annotations(self, df):
    return prepare_annotations(self.tree, df, self.alpha_local, self.sibling_alpha,
                               config.SIBLING_TEST_METHOD)
```

---

### 3.3 Separate post-hoc merge into its own concern

**Current state**: `_maybe_apply_posthoc_merge_with_audit` and `_test_cluster_pair_divergence` live in `TreeDecomposition`, binding the merge logic to the decomposer.

**Problem**: Post-hoc merge needs the calibration model (Tier 1.1), but it shouldn't reach into the decomposer's internals.

**Change**: Extend `posthoc_merge.py` to accept an optional calibration deflation:
```python
def apply_posthoc_merge(
    cluster_roots, alpha, tree, children, root,
    test_divergence,
    calibration_model=None,   # NEW — CalibrationModel or None
):
```

The `test_divergence` callback should already return calibrated stats when `calibration_model` is provided. This keeps the merge module stateless and testable independently.

---

### 3.4 Split `TreeDecomposition` into traversal + gates

**Current state**: One class handles: (a) pre-caching node metadata, (b) annotation orchestration, (c) gate logic, (d) traversal, (e) post-hoc merge, (f) signal localization dispatch.

**Proposed split**:

| New module               | Responsibility                                                           | Methods moved    |
| ------------------------ | ------------------------------------------------------------------------ | ---------------- |
| `tree_decomposition.py`  | Orchestrator: `__init__`, `decompose_tree`, `decompose_tree_v2`          | Stays            |
| `gates.py`               | `_should_split`, `_should_split_v2`                                      | Move out         |
| `annotation_dispatch.py` | `_prepare_annotations`                                                   | Move out (3.2)   |
| `posthoc_merge.py`       | `_maybe_apply_posthoc_merge_with_audit`, `_test_cluster_pair_divergence` | Move merge logic |
| `calibration_model.py`   | `CalibrationModel`, `predict_inflation_factor`                           | New (3.1)        |

This brings `tree_decomposition.py` down from 887 lines to ~400 (traversal + wiring only).

---

## Dependency Graph

```
Tier 0 (Critical — Gate 2)     Tier 1 (Correctness)       Tier 2 (Statistics)        Tier 3 (Modularization)
                                                           
0.1 Felsenstein cap/fix         1.2 Targeted descendant                               3.1 Extract CalibrationModel
0.2 Projection saturation             ▼                                                     ▲
0.3 BH leaf-edge exclusion      1.3 Remove LCA-blocking    2.1 ✅ Gamma GLM (DONE)         │
         ▼                                                  2.2 ✅ statsmodels (DONE)       │
0.4 Gate 2 calibration   ──▶   1.1 Calibrate post-hoc  ──────────────────────────▶  3.1 Extract CalibrationModel
                                     merge test                                          ▼
                                                                                    3.3 Separate post-hoc merge
                                                                                         ▼
                                                                                    3.2 Annotation dispatch
                                                                                         ▼
                                                                                    3.4 Split TreeDecomposition
```

**Recommended execution order**:
1. **0.1** + **0.2** + **0.3** (independent Gate 2 fixes — can be done in parallel)
2. **0.4** (Gate 2 calibration — depends on understanding 0.1–0.3 impact)
3. **1.2** → **1.3** → **1.1** (correctness, bottom-up)
4. **3.1** → **3.3** → **3.2** → **3.4** (modularization)

**Tier 0 is the new top priority.** Items 0.1–0.3 are relatively simple fixes that can be evaluated independently. Together they should substantially improve Gate 2 detection rate on real sparse data. Item 0.4 (full Gate 2 calibration) is a deeper design decision that may subsume 0.1–0.3.

Tier 2 items are complete. Tier 1 and 3 remain as planned.

---

## Test Impact

| Change                          | Tests to update                     | New tests needed                                    |
| ------------------------------- | ----------------------------------- | --------------------------------------------------- |
| 0.1 Felsenstein cap/fix         | `test_edge_significance.py`         | Outlier branch produces capped multiplier           |
| 0.2 Projection saturation       | `test_random_projection.py`         | k < d when d is moderate                            |
| 0.3 BH leaf-edge exclusion      | `test_edge_significance.py`         | Leaf edges excluded from correction set             |
| 0.4 Gate 2 calibration          | `test_edge_significance.py`         | Calibrated edge test on synthetic null              |
| 1.1 Calibrate post-hoc merge    | `test_posthoc_merge.py`             | Assert merge test uses deflated stat                |
| 1.2 Targeted descendant removal | `test_posthoc_merge.py`             | 3-cluster-under-LCA scenario                        |
| 1.3 Remove LCA-blocking         | `test_posthoc_merge.py`             | A≠B, C≈D under same LCA                             |
| 2.1 ✅ Gamma GLM                | `test_clt_validity.py`              | ✅ Done                                             |
| 3.x Modularization              | Import paths only                   | No logic changes                                    |
