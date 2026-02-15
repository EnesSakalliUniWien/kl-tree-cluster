# Changelog

All notable changes to this project are documented in this file.

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