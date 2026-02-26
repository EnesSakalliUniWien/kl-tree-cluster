# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-02-17

### Added
- **Spectral dimension estimation** (`spectral_dimension.py`): Per-node eigendecomposition of the local correlation matrix replaces the JL-based projection dimension for Gate 2 (edge test). Uses the effective rank (Shannon entropy of eigenvalue spectrum) as projection dimension `k`, with PCA-based whitened projection `T = Σ (vᵢᵀz)² / λᵢ ~ χ²(k)` for exact null calibration. Controlled by `config.SPECTRAL_METHOD = "effective_rank"` (default).
- **Dual-form eigendecomposition** (`spectral_dimension.py`): When `n_desc < d_active`, computes the `n×n` Gram matrix instead of the `d×d` correlation matrix — O(n²d + n³) vs O(d³). For subtrees with n=10 leaves and d=2000 features, this is 10×10 eigh instead of 2000×2000, eliminating the performance bottleneck on high-dimensional cases.
- **Internal node distributions in spectral decomposition** (`spectral_dimension.py`): The data matrix for eigendecomposition now includes both leaf rows AND internal descendant node distribution vectors. This enriches the covariance estimate, especially for nodes high in the tree where internal descendants capture intermediate subtree structure.
- **Information cap on projection dimension** (`random_projection.py`): `compute_projection_dimension()` now caps k at `n_samples` when `d ≥ 4n` (severely rank-deficient data). Prevents the JL formula from returning `k ≫ n`, which would add pure-noise χ² components that absorb degrees of freedom without contributing signal. Only activates for sibling tests (Gate 3); Gate 2 uses spectral dimensions.
- **Cousin-weighted Wald sibling test** (`cousin_weighted_wald.py`): New sibling test method using weighted regression where ALL sibling pairs contribute to the calibration, weighted by edge p-values `w_i = min(p_edge_L, p_edge_R)`. Now the default (`config.SIBLING_TEST_METHOD = "cousin_weighted_wald"`).
- **Cousin-tree-guided sibling test** (`cousin_tree_guided.py`): Walks up the tree from each focal pair to find topologically nearest null-like relatives, using their median T/k as local ĉ. Adapts to local tree structure without global regression.

### Refactored
- **Step 3.4 — `gates.py` extraction**: Gate logic (`should_split`, `should_split_v2`, `_check_edge_significance`) extracted from `TreeDecomposition` into `GateEvaluator` class in `gates.py` (343 lines). `tree_decomposition.py` reduced from 992 to 757 lines (−234 lines). `decompose_tree()` and `decompose_tree_v2()` now delegate to free functions `iterate_worklist`, `process_node`, `process_node_v2` in `gates.py`. `GateEvaluator` constructor accepts injected `children_map`, `descendant_leaf_sets`, `root` to decouple from `PosetTree` internals. Five dead inline methods removed.
- **Power guard in `should_split_v2`** (`gates.py`): When signal localization finds zero significant difference pairs after BH correction, returns `(True, None)` — trusting the aggregate Gate 3 SPLIT but discarding the powerless localization result. Prevents false cross-boundary merges via similarity-only edges.

### Fixed
- **Benchmark hang on high-dimensional cases**: `gaussian_extreme_noise_2` (n=300, d=2000, K=30) previously hung due to ~598 × O(2000³) eigendecompositions at every internal node. Dual-form optimization and information cap eliminate the hang.
- **Test import error** (`59_test_pipeline_pdf_naming.py`): Updated import from removed `benchmarks.shared.pipeline._resolve_pdf_output_path` to `benchmarks.shared.util.pdf.session.resolve_pdf_output_path`.
- **`merge_similarity_graphs` p-value direction** (`signal_localization.py`): When duplicate similarity edges existed across localization levels, kept the **lower** p-value. For similarity edges, higher p-value = stronger evidence of similarity (fail to reject H₀). Fixed to keep the **higher** p-value. Previously caused false cross-boundary merges by under-reporting similarity strength.

## [Unreleased] - 2026-02-14

### Fixed
- **Cousin-adjusted Wald regression extrapolation** (`cousin_adjusted_wald.py`): The log-linear calibration regression `log(T/k) = β₀ + β₁·log(BL_sum) + β₂·log(n_parent)` could extrapolate ĉ far beyond observed calibration data at the root node (where `n_parent` is largest). The β₂ coefficient (~1.3–1.5) caused predicted ĉ = 3–17× at the root while the maximum observed T/k ratio from null-like pairs was only ~1.4×. This over-deflation caused the root sibling test to always fail → K=1 collapse in ~30/95 benchmark cases.
  - **Fix**: `_predict_c()` now clamps the regression prediction at `model.max_observed_ratio` — the maximum T/k ratio actually observed in null-like calibration pairs. This prevents the regression from extrapolating beyond its training domain while preserving calibration within the observed range.
  - **Impact**: K=1 collapses eliminated for all non-SBM/non-phylogenetic cases. Exact K matches improved from 49/95 to 57/95. Mean ARI improved from 0.63 to 0.75. Zero regressions vs. raw Wald on previously working cases.
  - `_CalibrationModel` gains `max_observed_ratio: float` field; diagnostics dict includes `max_observed_ratio` key.

### Changed
- Remove deprecated `n_permutations` parameter from the decomposition API; callers passing it will no longer be accepted. Tests updated accordingly.
- Pipeline plotting behavior: default no longer writes intermediate PNGs; when `concat_plots_pdf=True` the pipeline collects Figures and writes categorized PDFs (`k_distance_plots.pdf`, `tree_plots.pdf`, `umap_plots.pdf`) instead of emitting PNGs.
- PDF utilities: improved diagnosis, figure classification (manifold plots grouped with UMAP), and robust headless handling (Agg backend).

### Removed
- `kl_clustering_analysis.threshold` package removed — helper functionality that was required by debug scripts is now replaced with a small conservative fallback (debug-only) or should be implemented separately where needed.

### Fixed
- Various import-time and optional-dependency issues (runners now import optional dependencies lazily and return skip results when missing) to improve test/CI stability.
- Resolved multiple test failures and tightened integration behavior for PDF generation.


(For full history, add older entries here in reverse chronological order.)