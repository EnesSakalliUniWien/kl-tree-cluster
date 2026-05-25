# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] - 2026-02-17

### Added
- **Spectral dimension estimation** (`spectral_dimension.py`): Per-node eigendecomposition of the local null-whitened tangent correlation matrix provides the projection dimension and parent PCA basis for both edge and sibling projected-Wald tests.
- **Dual-form eigendecomposition** (`spectral_dimension.py`): When `n_desc < d_active`, computes the `n×n` Gram matrix instead of the `d×d` correlation matrix — O(n²d + n³) vs O(d³). For subtrees with n=10 leaves and d=2000 features, this is 10×10 eigh instead of 2000×2000, eliminating the performance bottleneck on high-dimensional cases.
- **Internal node distributions in spectral decomposition** (`spectral_dimension.py`): The data matrix for eigendecomposition now includes both leaf rows AND internal descendant node distribution vectors. This enriches the covariance estimate, especially for nodes high in the tree where internal descendants capture intermediate subtree structure.

### Refactored
- **Step 3.4 — `gates.py` extraction**: Gate logic (`should_split`, `should_split_v2`, `_check_edge_significance`) extracted from `TreeDecomposition` into `GateEvaluator` class in `gates.py` (343 lines). `tree_decomposition.py` reduced from 992 to 757 lines (−234 lines). `decompose_tree()` and `decompose_tree_v2()` now delegate to free functions `iterate_worklist`, `process_node`, `process_node_v2` in `gates.py`. `GateEvaluator` constructor accepts injected `children_map`, `descendant_leaf_sets`, `root` to decouple from `PosetTree` internals. Five dead inline methods removed.
- **Power guard in `should_split_v2`** (`gates.py`): When signal localization finds zero significant difference pairs after BH correction, returns `(True, None)` — trusting the aggregate Gate 3 SPLIT but discarding the powerless localization result. Prevents false cross-boundary merges via similarity-only edges.

### Fixed
- **Benchmark hang on high-dimensional cases**: `gaussian_extreme_noise_2` (n=300, d=2000, K=30) previously hung due to ~598 × O(2000³) eigendecompositions at every internal node. Dual-form optimization and information cap eliminate the hang.
- **Test import error** (`59_test_pipeline_pdf_naming.py`): Updated import from removed `benchmarks.shared.pipeline._resolve_pdf_output_path` to `benchmarks.shared.util.pdf.session.resolve_pdf_output_path`.
- **`merge_similarity_graphs` p-value direction** (`signal_localization.py`): When duplicate similarity edges existed across localization levels, kept the **lower** p-value. For similarity edges, higher p-value = stronger evidence of similarity (fail to reject H₀). Fixed to keep the **higher** p-value. Previously caused false cross-boundary merges by under-reporting similarity strength.

## [Unreleased] - 2026-02-14

### Changed
- Remove obsolete `n_permutations` parameter from the decomposition API; callers passing it will no longer be accepted. Tests updated accordingly.
- Pipeline plotting behavior: default no longer writes intermediate PNGs; when `concat_plots_pdf=True` the pipeline collects Figures and writes categorized PDFs (`k_distance_plots.pdf`, `tree_plots.pdf`, `umap_plots.pdf`) instead of emitting PNGs.
- PDF utilities: improved diagnosis, figure classification (manifold plots grouped with UMAP), and robust headless handling (Agg backend).

### Removed
- `kl_clustering_analysis.threshold` package removed — helper functionality that was required by debug scripts now lives in those scripts or should be implemented directly where needed.

### Fixed
- Various import-time and optional-dependency issues (runners now import optional dependencies lazily and return skip results when missing) to improve test/CI stability.
- Resolved multiple test failures and tightened integration behavior for PDF generation.


(For full history, add older entries here in reverse chronological order.)
