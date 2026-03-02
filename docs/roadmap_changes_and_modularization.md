# Roadmap: Necessary Changes & Modularization

This document maps all identified issues to concrete code changes. Changes are grouped into three tiers and ordered by dependency.

**Last updated**: 2026-02-16, incorporating empirical findings from real-world sparse data (626 × 456 GO-term matrix).

---

## Tier 0 — Critical: Edge Significance (Gate 2) Calibration

> **Added 2026-02-16**: Gate 2 is the primary bottleneck on real sparse data (9.8% detection rate, K=1 collapse). All Tier 1–3 improvements are irrelevant if Gate 2 blocks the traversal at the root.

### 0.1 ✅ Felsenstein overcorrection on outlier branches — DONE (2026-02-18)

**Issue**: Average-linkage trees from sparse binary data produce highly asymmetric root splits with long outlier branches (`BL/mean_BL > 4`). The Felsenstein multiplier `1 + BL/mean_BL = 5.29×` kills genuine edge signal, converting significant z-scores (|z| > 17) into non-significant test results (p = 0.14). The Felsenstein formula assumes branch lengths reflect neutral evolutionary time in a fixed phylogeny, but our trees are data-dependent.

**Resolution**: Option 3 — Felsenstein scaling disabled by default via `config.FELSENSTEIN_SCALING = False`. Empirical comparison showed mean ARI 0.694 → 1.000 without it. The `1 + BL/mean_BL` formula remains in `edge_significance.py` → `_compute_standardized_z()` and can be re-enabled for true phylogenetic trees. `annotate_child_parent_divergence()` gates branch-length computation on the config flag: `mean_branch_length = _compute_mean_branch_length(tree) if config.FELSENSTEIN_SCALING else None`.

**Files changed**: `config.py` (new `FELSENSTEIN_SCALING` flag), `edge_significance.py` (gated on flag).

---

### 0.2 ✅ Projection dimension saturation (k = d) — DONE (2026-02-18)

**Issue**: With ε = 0.3 and d = 456 features, `compute_projection_dimension(n, d)` returns k = 456 (full dimension) for any n ≥ 200. The "random projection" is an identity transform — no dimensionality reduction occurs. The test becomes T ~ χ²(456) which requires T > 520 to reject at α = 0.05, very conservative for signal spread across many sparse features.

**Resolution**: Gate 2 (edge test) now uses `SPECTRAL_METHOD = "effective_rank"` (default) which replaces **both** the JL dimension formula **and** the random projection matrix with PCA eigendecomposition of the local correlation matrix. Specifically:
- **Dimension k** = effective rank from eigenvalue spectrum (typically 5–30), not JL formula
- **Projection R** = PCA eigenvectors (data-adaptive), not random orthonormal matrix
- **Test statistic** = $T = \sum (v_i^\top z)^2 / \lambda_i \sim \chi^2(k)$ via eigenvalue whitening (exact under H₀), not $T = \|Rz\|^2$ (JL approximate)

The JL-based `compute_projection_dimension` + random orthonormal R is only a fallback when `spectral_k is None`.

**Gate 2 vs Gate 3 asymmetry**: Both gates use the same projected Wald χ² test structure (compute z → project w = Rz → T = Σ w²). The difference is pragmatic, not mathematical:
- **Gate 2** uses PCA + effective rank: fewer df, lower threshold, exact χ², signal concentrated in top eigenvectors. No calibration model available for Gate 2, so getting the projection right matters more.
- **Gate 3** uses JL + saturated k: more df, higher threshold, but captures signal everywhere. The weighted Wald calibration model (ĉ deflation) corrects the resulting inflation. `_sibling_spectral_dims = None` is set explicitly in `tree_decomposition.py`.

This asymmetry is a design smell — if cross-fit permutation calibration (0.4) is implemented for both gates, it could unify them on the same projection approach.

**Information cap** (d ≥ 4n → k ≤ n) protects small subtrees in the JL path: n=10/d=456 → k=10, n=50/d=456 → k=50, n=100/d=2000 → k=100.

**Files**: `config.py` (`SPECTRAL_METHOD = "effective_rank"`), `edge_significance.py` (PCA projection + whitening for Gate 2, JL fallback), `tree_decomposition.py` (sibling test hardcoded to JL), `random_projection.py` (JL dimension + information cap).

---

### 0.3 ❌ BH correction includes untestable edges — INVALID (2026-02-18)

**Original issue**: The BH correction runs over ALL 1250 edges including ~626 leaf-edges with p = 1.0. This inflates the denominator m by ~2×, halving the effective significance threshold for non-trivial edges.

**Why this is invalid**: The premise that leaf edges get p = 1.0 is **wrong**. Empirical analysis (`scripts/diagnose_edge_calibration_and_k.py`, results in `docs/edge_calibration_and_projection_analysis.md`) shows:

- Leaf edges (n_child = 1) get **real test statistics**, not p = 1.0
- 6.7% of leaf edges are significant at α = 0.05 (close to nominal 5%)
- The actual problem is **CLT violation**, not BH inflation: with sparse binary data (97.5% zeros, θ ≈ 0.025), z-scores for n_child = 1 are two-point masses {−0.16, 6.3}, not N(0,1). The test statistic T/k ≈ 0.04 (22× smaller than the expected 1.0 under χ²)
- Excluding leaf edges from BH would cherry-pick the correction set, increasing false positives on internal edges without addressing the underlying calibration problem

**Root cause**: The edge test's CLT assumption breaks for sparse binary data with small n_child. The variance formula θ(1−θ)(1/n_c − 1/n_p) is correct but the z-scores are not Gaussian. This is fundamentally different from Gate 3's post-selection inflation (which IS correctly addressed by cousin-weighted Wald).

**Superseded by**: 0.4 (Gate 2 calibration), which addresses both the CLT violation and any post-selection bias at the source.

---

### 0.4 Gate 2 calibration (analogous to Gate 3)

**Issue**: Gate 3 (sibling test) has cousin-weighted Wald calibration to correct post-selection inflation. Gate 2 (edge test) has **no calibration at all**. On synthetic benchmarks this causes 15.9% Type I error; on real sparse data it causes 90.2% false negative rate. The edge test needs its own calibration.

**Options**:
1. **Cousin-style edge calibration**: Use null-like edges (where children's distributions are not meaningfully different from parent) to estimate the edge-test inflation factor, then deflate all edge statistics. Analogous to what cousin-weighted Wald does for the sibling test.
2. **Permutation null for edge test**: For each edge, permute leaf assignments between child and sibling, recompute T, estimate the null distribution empirically. Computationally expensive but exact.

**Depends on**: Major design decision — affects all downstream components.

---

## Tier 1 — Correctness Fixes (must-do before any benchmark)

### 1.1 ✅ Post-hoc merge test asymmetry — DONE

**Issue**: Decomposition uses cousin-weighted-calibrated $T_{\text{adj}} = T/\hat{c}$, but post-hoc merge calls raw `sibling_divergence_test()` via `_test_cluster_pair_divergence()` — inflated $T$. Harder to merge than to split.

**Resolution**: `TreeDecomposition.__init__` now extracts the calibration model from `results_df.attrs.get("_calibration_model")` and stores it as `self._calibration_model`. `_test_cluster_pair_divergence()` deflates the raw stat by `ĉ` when a model exists, dispatching to the correct prediction function based on model type (`WeightedCalibrationModel` vs `CalibrationModel`). This ensures post-hoc merge operates on the same deflated scale as the decomposition.

**Files changed**: `tree_decomposition.py` (lines ~165-175 extraction, lines ~561-580 deflation), `cousin_weighted_wald.py` (public `WeightedCalibrationModel` + `predict_weighted_inflation_factor`), `cousin_adjusted_wald.py` (public `CalibrationModel` + `predict_inflation_factor`), `sibling_divergence/__init__.py` (exports both models and prediction functions).

**Tests**: 14 tests in `tests/localization/32_test_posthoc_merge_calibration.py` — verifies deflation with model, no deflation without model, ĉ=1 identity, and full v1 pipeline integration.

---

### 1.2 ✅ Post-hoc merge descendant removal — targeted removal — DONE

**Issue**: `cluster_roots -= nx.descendants(tree, lca)` removes ALL cluster roots under the LCA, not just the two being merged. A third cluster under the same LCA is silently absorbed.

**Resolution**: Replaced blanket descendant removal with targeted `cluster_roots.discard(lc)` / `cluster_roots.discard(rc)` / `cluster_roots.add(lca)`. Added an antichain guard that skips merges when other cluster roots exist under the LCA to prevent invariant violations.

**Files changed**: `posthoc_merge.py` (lines ~170-180).

**Tests**: `test_posthoc_merge_targeted_removal_preserves_third_cluster` and `test_posthoc_merge_does_not_reintroduce_descendants_after_ancestor_merge` in `tests/localization/31_test_posthoc_merge.py`.

---

### 1.3 ✅ Remove LCA-blocking in post-hoc merge — DONE

**Issue**: If any pair under an LCA rejects $H_0$, ALL merges under that LCA are blocked — even unrelated similar pairs.

**Resolution**: Removed `lca_has_reject` logic. Mergeable pairs are now simply `mergeable_indices = [i for i, r in enumerate(reject) if not r]` — BH correction handles multiplicity, and only specific rejected pairs are blocked.

**Files changed**: `posthoc_merge.py` (lines ~140-145).

**Tests**: `test_posthoc_merge_independent_merges_at_different_boundaries` and `test_posthoc_merge_respects_significant_pairs_at_lca` in `tests/localization/31_test_posthoc_merge.py`.

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

### 3.1 ✅ Extract calibration model as a first-class object — DONE

**Issue**: The calibration model lived inside `cousin_weighted_wald.py` as a private `_WeightedCalibrationModel` dataclass, buried in DataFrame attrs.

**Resolution**: Both models are now public:
- `CalibrationModel` in `cousin_adjusted_wald.py` — legacy regression-based model with `predict_inflation_factor()`
- `WeightedCalibrationModel` in `cousin_weighted_wald.py` — weighted Gamma GLM model with `predict_weighted_inflation_factor()`
- Both exported from `sibling_divergence/__init__.py` with `__all__`
- `TreeDecomposition.__init__` extracts the model from `results_df.attrs.get("_calibration_model")` and stores it as `self._calibration_model`
- `_test_cluster_pair_divergence()` dispatches to the correct prediction function based on model type

**Note**: The dedicated `calibration_model.py` file proposed in the original plan was not created — both models remain in their respective annotation files, which keeps model creation and prediction co-located. The public API via `__init__.py` provides the clean import path.

---

### 3.2 ✅ Extract annotation dispatch from `TreeDecomposition.__init__` — DONE

**Completed 2026-02-18.**

`_prepare_annotations()` replaced with a 4-line delegation to `gate_annotations.compute_gate_annotations()`. The existing `gate_annotations.py` (which was dead code) was replaced with the full annotation dispatch logic including `spectral_dims`/`pca_projections`/`pca_eigenvalues` kwargs for `cousin_weighted_wald`. `tree_decomposition.py` went from 670 → 516 lines (combined with 3.3).

**What was done**:
- Replaced dead `gate_annotations.py` with proper `compute_gate_annotations()` function
- `TreeDecomposition._prepare_annotations()` now delegates to `compute_gate_annotations()`
- Removed direct `annotate_child_parent_divergence` / `annotate_sibling_divergence` imports from `tree_decomposition.py`

---

### 3.3 ✅ Separate post-hoc merge into its own concern — DONE

**Completed 2026-02-18.**

`_test_node_pair_divergence` (~40 lines), `_test_cluster_pair_divergence` (~65 lines), and `_compute_cluster_distribution` (~15 lines) extracted from `TreeDecomposition` — now delegates to `pairwise_testing.test_node_pair_divergence()` and `pairwise_testing.test_cluster_pair_divergence()` via thin wrapper methods (~25 lines total). `_maybe_apply_posthoc_merge_with_audit` stays on `TreeDecomposition` (orchestration).

**What was done**:
- Added `distribution_map` and `leaf_count_map` `@property` methods to `PosetTree` to bridge `pairwise_testing.py` API
- Wired previously-dead `pairwise_testing.py` (260 lines) into `tree_decomposition.py`
- Removed `sanitize_positive_branch_length`, `predict_inflation_factor`, `predict_weighted_inflation_factor`, `sibling_divergence_test` imports from `tree_decomposition.py`
- Removed unused `import networkx as nx`
- Updated test mock path in `11_test_tree_decomposition_distance.py`
- `tree_decomposition.py`: 758 → 516 lines (−242 lines, combined with 3.2 and 3.4)

---

### 3.4 ✅ Split `TreeDecomposition` into traversal + gates (DONE)

**Completed 2026-02-17.**

Gate logic extracted from `TreeDecomposition` into `GateEvaluator` class in `gates.py`. Five inline methods removed from `tree_decomposition.py` (−234 lines, 992 → 757 lines). `gates.py` is 343 lines.

**What was done**:

| Module                  | Responsibility                                                  | Status            |
| ----------------------- | --------------------------------------------------------------- | ----------------- |
| `tree_decomposition.py` | Orchestrator: `__init__`, `decompose_tree`, `decompose_tree_v2` | Stays (757 lines) |
| `gates.py`              | `GateEvaluator` class + free functions for traversal            | ✅ Extracted       |
| `gate_annotations.py`   | `_prepare_annotations` → `compute_gate_annotations()`           | ✅ Done (3.2)      |
| `pairwise_testing.py`   | `_test_node_pair_divergence`, `_test_cluster_pair_divergence`   | ✅ Done (3.3)      |
| `calibration_model.py`  | `CalibrationModel`, `predict_inflation_factor`                  | Future (3.1)      |

**Implementation details**:
- `GateEvaluator.__init__` accepts injected `children_map`, `descendant_leaf_sets`, and `root` (avoids coupling to `PosetTree` internals)
- `GateEvaluator.should_split()` — v1 gate logic (was `_should_split`)
- `GateEvaluator.should_split_v2()` — v2 gate logic with signal localization (was `_should_split_v2`), includes **power guard**: returns `(True, None)` when localization finds zero significant difference pairs
- Free functions: `iterate_worklist`, `process_node`, `process_node_v2` — top-down traversal loop
- Dataclass: `V2TraversalState` — traversal bookkeeping for v2
- `TreeDecomposition.__init__` creates `self._gate = GateEvaluator(...)`
- `decompose_tree()` and `decompose_tree_v2()` delegate to the free functions

**Removed inline methods** (5):
1. `_process_node_for_decomposition` → `process_node`
2. `_iterate_nodes_to_visit` → `iterate_worklist`
3. `_should_split` → `GateEvaluator.should_split`
4. `_check_edge_significance` → `GateEvaluator._check_edge_significance`
5. `_should_split_v2` → `GateEvaluator.should_split_v2`

**Bug fixes included**:
- **Power guard** (Bug A): `should_split_v2` discards powerless localization results (zero difference pairs after BH)
- **`merge_similarity_graphs` p-value direction** (Bug D): Similarity edges now keep higher p-value (= strongest similarity evidence)

**Tests**: 215 passed, 4 skipped, 0 failures. Mock paths updated from `tree_decomposition` to `gates` module. Pipeline verified: `quick_start.py` ARI=1.0, 4 clusters. Full benchmark (96 cases × 9 methods) passed.

---

### 3.5 ✅ Wire `PosetTree` constructors to `io.py` — DONE

**Completed 2026-02-18.**

`from_linkage`, `from_agglomerative`, and `from_undirected_edges` on `PosetTree` were duplicate implementations of the same logic already centralized in `tree/io.py` (which uses `_build_tree_from_merges` and `compute_ultrametric_branch_lengths` from `tree/branch_lengths.py`). Replaced all three with thin delegations to the corresponding `io.py` functions.

**What was done**:
- `PosetTree.from_linkage()` → delegates to `tree_from_linkage()`
- `PosetTree.from_agglomerative()` → delegates to `tree_from_agglomerative()`
- `PosetTree.from_undirected_edges()` → delegates to `tree_from_undirected_edges()`
- Removed `from sklearn.cluster import AgglomerativeClustering` import from `poset_tree.py`
- `poset_tree.py`: 539 → 408 lines (−131 lines)
- Branch-length computation now has a single source of truth in `tree/branch_lengths.py`

---

## Dependency Graph

```
Tier 0 (Critical — Gate 2)     Tier 1 (Correctness)       Tier 2 (Statistics)        Tier 3 (Modularization)
                                                           
0.1 ✅ Felsenstein (DONE)       1.2 ✅ Targeted descendant (DONE)                     3.1 ✅ Extract CalibrationModel (DONE)
0.2 ✅ Projection (DONE)        1.3 ✅ Remove LCA-blocking (DONE)                          │
0.3 ❌ BH exclusion (INVALID)   1.1 ✅ Calibrate post-hoc merge (DONE)                     ▼
                                                            2.1 ✅ Gamma GLM (DONE)   3.3 ✅ Separate post-hoc merge (DONE)
0.4 Gate 2 calibration          2.2 ✅ statsmodels (DONE)                                  ▼
                                                                                    3.2 ✅ Annotation dispatch (DONE)
                                                                                         ▼
                                                                                    3.4 ✅ Split TreeDecomposition (DONE)
                                                                                    3.5 ✅ Wire PosetTree constructors → io.py (DONE)
```

**Recommended execution order** (remaining items):
1. **0.4** (Gate 2 calibration — the sole remaining Tier 0 item and most impactful design decision)

**Progress summary**:
- **Tier 0**: 0.1 ✅, 0.2 ✅, 0.3 ❌ (invalid), 0.4 remaining
- **Tier 1**: 1.1 ✅, 1.2 ✅, 1.3 ✅ — **all complete**
- **Tier 2**: 2.1 ✅, 2.2 ✅ — **all complete**
- **Tier 3**: 3.1 ✅, 3.2 ✅, 3.3 ✅, 3.4 ✅, 3.5 ✅ — **all complete**

---

## Test Impact

| Change                             | Tests to update             | New tests needed                                           |
| ---------------------------------- | --------------------------- | ---------------------------------------------------------- |
| 0.1 ✅ Felsenstein (disabled)       | `test_edge_significance.py` | ✅ Done — `FELSENSTEIN_SCALING = False`                     |
| 0.2 ✅ Projection (spectral k)      | `test_random_projection.py` | ✅ Done — `SPECTRAL_METHOD = "effective_rank"`              |
| 0.3 ❌ BH exclusion (INVALID)       | N/A                         | N/A — superseded by 0.4                                    |
| 0.4 Gate 2 calibration             | `test_edge_significance.py` | Calibrated edge test on synthetic null                     |
| 1.1 ✅ Calibrate post-hoc merge     | ✅ Done                      | ✅ 14 tests in `32_test_posthoc_merge_calibration.py`       |
| 1.2 ✅ Targeted descendant removal  | ✅ Done                      | ✅ 6 tests in `31_test_posthoc_merge.py`                    |
| 1.3 ✅ Remove LCA-blocking          | ✅ Done                      | ✅ Covered in `31_test_posthoc_merge.py`                    |
| 2.1 ✅ Gamma GLM                    | `test_clt_validity.py`      | ✅ Done                                                     |
| 3.1 ✅ Extract CalibrationModel     | ✅ Done                      | ✅ Imports tested in `32_test_posthoc_merge_calibration.py` |
| 3.2 ✅ Annotation dispatch          | Mock path in distance test  | No logic changes; 215 pass                                 |
| 3.3 ✅ Pairwise testing extraction  | Mock path in distance test  | No logic changes; 215 pass                                 |
| 3.5 ✅ PosetTree constructor wiring | None                        | No logic changes; 215 pass                                 |
